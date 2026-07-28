"""
pm_trader_bridge.py

Adapter that wraps the `polymarket-paper-trader` (pm_trader) package into the
same interface our bot's PaperBroker exposes, so the main trading loop doesn't
need to know which paper backend is active.

pm_trader advantages over the local PaperLedger:
  - Level-by-level order book execution with exact Polymarket fee model
  - Slippage tracking in basis points per trade
  - SQLite-backed state (crash-safe, no JSON serialization)
  - Limit order state machine (GTC / GTD)
  - Built-in performance analytics (Sharpe, max drawdown, win rate)
  - Public leaderboard support
  - CLI + MCP server for interactive use

When PAPER_BACKEND=pm_trader, the bot routes all paper trades through this
bridge instead of the local PaperLedger. Switch back by setting
PAPER_BACKEND=local.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class PaperFill:
    """Unified fill result matching the local PaperBroker's PaperFill shape."""
    market_id: str
    side: str
    requested_price: float
    filled_price: float
    size_usd: float
    fee_usd: float
    shares: float = 0.0
    slippage_vs_mid: float = 0.0
    realized_pnl: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


class PMTraderBroker:
    """
    Paper trading broker backed by polymarket-paper-trader.

    Uses pm_trader.Engine for real orderbook fills with exact Polymarket
    fee model and slippage tracking. All state persists to SQLite.

    Risk management (position sizing, circuit breaker) still flows through
    RiskManager -- this class only handles simulated execution.
    """

    def __init__(
        self,
        risk_manager: RiskManager,
        data_dir: str = "data/pm_trader",
        starting_balance: float = 10000.0,
    ):
        from pm_trader.engine import Engine as PMEngine

        self.risk_manager = risk_manager
        self.data_dir = Path(data_dir)
        self._engine = PMEngine(data_dir=self.data_dir)
        self.fills: List[PaperFill] = []

        # Initialize account if not already done
        try:
            account = self._engine.get_account()
            logger.info(
                "pm_trader account loaded: cash=$%.2f (started with $%.2f)",
                account.cash, account.starting_balance,
            )
        except Exception:
            account = self._engine.init_account(balance=starting_balance)
            logger.info("pm_trader account initialized with $%.2f", starting_balance)

    def simulate_order(self, market_id: str, category: str, side: str, price: float) -> PaperFill:
        """
        Simulate an order fill against the live Polymarket orderbook.

        Delegates to pm_trader.Engine.buy() / .sell() which walk the real
        order book level-by-level with exact fee modeling.

        Args:
            market_id: Polymarket slug or condition_id (e.g. "will-bitcoin-hit-100k")
            category: Risk category (for RiskManager sizing)
            side: "YES" / "BUY" for buy, "NO" / "SELL" for sell
            price: Current midpoint price (informational, pm_trader fetches live book)

        Returns:
            PaperFill with execution details
        """
        size_usd = self.risk_manager.max_position_size(category)
        if size_usd <= 0:
            raise RuntimeError("No tradable capital available for this position")

        # Map bot side convention to pm_trader outcome
        side_upper = side.upper()
        if side_upper in ("YES", "BUY"):
            outcome = "yes"
            is_buy = True
        elif side_upper in ("NO", "SELL"):
            outcome = "no"
            is_buy = False
        else:
            outcome = side.lower()
            is_buy = side_upper in ("YES", "BUY")

        try:
            if is_buy:
                result = self._engine.buy(market_id, outcome, size_usd, order_type="fok")
                trade = result.trade
                account = result.account

                fill = PaperFill(
                    market_id=market_id,
                    side=side,
                    requested_price=price,
                    filled_price=trade.avg_price,
                    size_usd=trade.amount_usd,
                    fee_usd=trade.fee,
                    shares=trade.shares,
                    slippage_vs_mid=trade.slippage,
                )
                self.fills.append(fill)

                # Register with risk manager
                self.risk_manager.open_position(
                    market_id, category, side, fill.filled_price, fill.size_usd,
                )

                logger.info(
                    "pm_trader BUY %s: %.4f shares @ avg $%.4f "
                    "(notional=$%.4f, fee=$%.4f, slippage=%.1fbps)",
                    market_id, trade.shares, trade.avg_price,
                    trade.amount_usd, trade.fee, trade.slippage,
                )
                return fill

            else:
                # For sells, check if we have a position
                portfolio = self._engine.get_portfolio()
                pos_shares = 0.0
                for p in portfolio:
                    if p["market_slug"] == market_id and p["outcome"] == outcome:
                        pos_shares = p["shares"]
                        break

                if pos_shares <= 0:
                    logger.warning("SELL requested for %s but no pm_trader position exists", market_id)
                    return PaperFill(
                        market_id=market_id, side=side,
                        requested_price=price, filled_price=price,
                        size_usd=0.0, fee_usd=0.0,
                    )

                result = self._engine.sell(market_id, outcome, pos_shares, order_type="fok")
                trade = result.trade

                fill = PaperFill(
                    market_id=market_id,
                    side=side,
                    requested_price=price,
                    filled_price=trade.avg_price,
                    size_usd=trade.amount_usd,
                    fee_usd=trade.fee,
                    shares=trade.shares,
                    slippage_vs_mid=trade.slippage,
                )
                self.fills.append(fill)

                # Close position in risk manager
                if market_id in self.risk_manager.open_positions:
                    self.risk_manager.close_position(market_id, fill.filled_price)

                logger.info(
                    "pm_trader SELL %s: %.4f shares @ avg $%.4f "
                    "(proceeds=$%.4f, fee=$%.4f, slippage=%.1fbps)",
                    market_id, trade.shares, trade.avg_price,
                    trade.amount_usd, trade.fee, trade.slippage,
                )
                return fill

        except Exception as e:
            logger.warning("pm_trader order failed for %s: %s", market_id, e)
            return PaperFill(
                market_id=market_id, side=side,
                requested_price=price, filled_price=price,
                size_usd=0.0, fee_usd=0.0,
            )

    def summary(self) -> dict:
        """
        Return paper ledger summary matching the local PaperLedger's format.

        Pulls data from pm_trader's SQLite database for consistent reporting.
        """
        try:
            balance_info = self._engine.get_balance()
            account = self._engine.get_account()
            portfolio = self._engine.get_portfolio()
            history = self._engine.get_history(limit=1000)

            total_fees = sum(t.fee for t in history)
            open_positions = {}
            for p in portfolio:
                open_positions[p["market_slug"]] = {
                    "token_id": p["market_slug"],
                    "shares": p["shares"],
                    "cost_basis": p["total_cost"],
                    "avg_entry_price": p["avg_entry_price"],
                    "unrealized_pnl": p["unrealized_pnl"],
                }

            return {
                "balance": round(balance_info["cash"], 4),
                "starting_balance": round(balance_info["starting_balance"], 4),
                "positions_value": round(balance_info["positions_value"], 4),
                "total_value": round(balance_info["total_value"], 4),
                "pnl": round(balance_info["pnl"], 4),
                "open_positions": open_positions,
                "total_trades": len(history),
                "total_fees": round(total_fees, 4),
            }
        except Exception as e:
            logger.error("Failed to get pm_trader summary: %s", e)
            return {
                "balance": 0.0,
                "open_positions": {},
                "total_trades": 0,
                "total_fees": 0.0,
            }

    def get_analytics(self) -> dict:
        """Get full performance analytics from pm_trader."""
        try:
            from pm_trader.analytics import compute_stats
            account = self._engine.get_account()
            history = self._engine.get_history(limit=10000)
            portfolio = self._engine.get_portfolio()
            positions_value = sum(p["current_value"] for p in portfolio)
            return compute_stats(history, account, positions_value)
        except Exception as e:
            logger.error("Failed to get pm_trader analytics: %s", e)
            return {}

    def close(self) -> None:
        """Clean up pm_trader engine resources."""
        try:
            self._engine.close()
        except Exception:
            pass
