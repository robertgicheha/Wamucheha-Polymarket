"""
Paper trading broker. Trades against real, live Polymarket orderbook data with
simulated money -- this is what TRADING_MODE=paper routes through instead of
connectors/polymarket_connector.py's place_order. Run this for 2-4 weeks minimum
before ever setting TRADING_MODE=live.

Uses PaperLedger for realistic orderbook-level fills with slippage and fee
modeling, rather than a flat assumed slippage percentage.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class PaperFill:
    market_id: str
    side: str
    requested_price: float
    filled_price: float  # includes simulated slippage
    size_usd: float
    fee_usd: float
    shares: float = 0.0
    slippage_vs_mid: float = 0.0
    realized_pnl: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


class PaperBroker:
    """
    Paper trading broker backed by PaperLedger for realistic orderbook fills.

    Fetches the real Polymarket orderbook, walks it level-by-level, and
    records slippage + fees. All state persists to a JSON file so numbers
    survive bot restarts.

    Risk management (position sizing, circuit breaker) still flows through
    RiskManager -- this class only handles simulated execution.
    """

    def __init__(
        self,
        risk_manager: RiskManager,
        state_file: str = "data/paper_ledger_state.json",
        starting_balance: float = 1000.0,
        fee_bps: float = 200.0,
    ):
        from backtest.paper_trading import PaperLedger

        self.risk_manager = risk_manager
        self.ledger = PaperLedger(
            state_file=state_file,
            starting_balance=starting_balance,
            fee_bps=fee_bps,
        )
        self.fills: List[PaperFill] = []

    def simulate_order(self, market_id: str, category: str, side: str, price: float) -> PaperFill:
        """
        Simulate an order fill against the live Polymarket orderbook.

        BUY:  market_id is the YES token_id, side is "YES"
              (sells map to NO token_id in the orderbook)
        SELL: market_id is the token_id, side is "YES" or "NO"

        Position sizing comes from RiskManager.max_position_size().
        Actual fills come from PaperLedger (real orderbook walk).
        """
        size_usd = self.risk_manager.max_position_size(category)
        if size_usd <= 0:
            raise RuntimeError("No tradable capital available for this position")

        # Determine the token_id and side for the orderbook
        # For YES tokens, BUY YES = BUY on token_id, SELL YES = SELL on token_id
        # For NO tokens, BUY NO = BUY on token_id, SELL NO = SELL on token_id
        order_side = "BUY" if side.upper() in ("YES", "BUY") else "SELL"

        if order_side == "BUY":
            result = self.ledger.place_order(
                token_id=market_id,
                side="BUY",
                usd_amount=size_usd,
            )
        else:
            # For sells, we need to know how many shares we hold
            pos = self.ledger.state.positions.get(market_id, {})
            shares_held = pos.get("shares", 0.0)
            if shares_held <= 0:
                # No position to sell -- create a zero-size fill to indicate rejection
                logger.warning("SELL requested for %s but no paper position exists", market_id)
                return PaperFill(
                    market_id=market_id,
                    side=side,
                    requested_price=price,
                    filled_price=price,
                    size_usd=0.0,
                    fee_usd=0.0,
                )
            result = self.ledger.place_order(
                token_id=market_id,
                side="SELL",
                shares=shares_held,
            )

        if result.get("status") != "filled":
            logger.warning("Paper order rejected for %s: %s", market_id, result.get("reason"))
            return PaperFill(
                market_id=market_id,
                side=side,
                requested_price=price,
                filled_price=price,
                size_usd=0.0,
                fee_usd=0.0,
            )

        fill = PaperFill(
            market_id=market_id,
            side=side,
            requested_price=price,
            filled_price=result["avg_price"],
            size_usd=result["notional"],
            fee_usd=result["fee"],
            shares=result["shares"],
            slippage_vs_mid=result["slippage_vs_mid"],
            realized_pnl=result.get("realized_pnl"),
        )
        self.fills.append(fill)

        # Register position with risk manager for circuit breaker tracking
        if order_side == "BUY":
            self.risk_manager.open_position(
                market_id, category, side, fill.filled_price, fill.size_usd,
            )
        else:
            # Close position in risk manager
            if market_id in self.risk_manager.open_positions:
                self.risk_manager.close_position(market_id, fill.filled_price)

        return fill

    def summary(self) -> dict:
        """Return paper ledger summary for reporting."""
        return self.ledger.summary()
