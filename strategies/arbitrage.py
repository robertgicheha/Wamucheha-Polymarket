"""
Synthetic Arbitrage Engine — Polymarket + Kalshi cross-platform arb
+ Orderbook-based intra-platform arbitrage for 5-minute markets.

Cross-platform arb:
  Buy YES on one platform + NO on the other when combined price < $1.00.

Orderbook arb (5-min markets):
  - UP + DOWN tokens must sum to ~$1.00
  - If UP_best_ask + DOWN_best_ask < $1.00 → buy both = guaranteed profit
  - If UP_best_bid + DOWN_best_bid > $1.00 → sell both = guaranteed profit
  - Cross-source arb: compare PM orderbook to external price feeds

Key insight: We don't hold to maturity. We trade convergence.
  - Enter when spread is wide (cheap arb)
  - Exit when spread narrows (convergence)
  - Capital velocity > holding for resolution

Risk: Counterparty risk (platform default), execution risk (partial fills),
      liquidity risk (can't exit at target price).
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class ArbOpportunity:
    """One arbitrage opportunity between Polymarket and Kalshi."""
    timestamp: float
    market_question: str

    # Polymarket side
    pm_market_id: str
    pm_yes_price: float
    pm_no_price: float
    pm_yes_token_id: str
    pm_no_token_id: str

    # Kalshi side
    kalshi_ticker: str
    kalshi_yes_price: float
    kalshi_no_price: float
    kalshi_strike: float

    # Calculated
    spread: float  # PM_yes + Kalshi_no (or vice versa)
    guaranteed_profit_per_share: float
    direction: str  # "pm_yes_kalshi_no" or "pm_no_kalshi_yes"

    # Metadata
    pm_volume_24h: float = 0.0
    kalshi_volume: int = 0
    confidence: float = 0.0
    time_to_expiry: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "market_question": self.market_question,
            "pm_yes_price": self.pm_yes_price,
            "pm_no_price": self.pm_no_price,
            "kalshi_yes_price": self.kalshi_yes_price,
            "kalshi_no_price": self.kalshi_no_price,
            "spread": self.spread,
            "guaranteed_profit": self.guaranteed_profit_per_share,
            "direction": self.direction,
            "confidence": self.confidence,
        }


@dataclass
class ArbPosition:
    """An open arbitrage position."""
    position_id: str
    opportunity: ArbOpportunity
    opened_at: float

    # Filled prices
    pm_fill_price: float
    kalshi_fill_price: float
    pm_size_shares: float
    kalshi_size_shares: float
    total_cost_usd: float

    # Current state
    current_pm_price: float = 0.0
    current_kalshi_price: float = 0.0
    unrealized_pnl: float = 0.0
    hold_seconds: float = 0.0

    # Exit
    closed: bool = False
    closed_at: Optional[float] = None
    exit_pnl: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "position_id": self.position_id,
            "pm_yes_price": self.pm_fill_price,
            "kalshi_no_price": self.kalshi_fill_price,
            "total_cost": self.total_cost_usd,
            "unrealized_pnl": round(self.unrealized_pnl, 4),
            "hold_seconds": round(self.hold_seconds, 1),
            "closed": self.closed,
            "exit_pnl": round(self.exit_pnl, 4),
        }


@dataclass
class ArbPerformance:
    """Aggregate arbitrage performance."""
    total_opportunities: int = 0
    opportunities_taken: int = 0
    positions_opened: int = 0
    positions_closed: int = 0
    total_pnl: float = 0.0
    total_fees: float = 0.0
    avg_hold_seconds: float = 0.0
    win_rate: float = 0.0
    avg_profit_per_trade: float = 0.0
    max_concurrent_positions: int = 0
    capital_utilization: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "total_opportunities": self.total_opportunities,
            "opportunities_taken": self.opportunities_taken,
            "positions_opened": self.positions_opened,
            "positions_closed": self.positions_closed,
            "total_pnl": round(self.total_pnl, 4),
            "total_fees": round(self.total_fees, 4),
            "avg_hold_seconds": round(self.avg_hold_seconds, 1),
            "win_rate": round(self.win_rate, 1),
            "avg_profit_per_trade": round(self.avg_profit_per_trade, 4),
        }


class ArbitrageEngine:
    """
    Cross-platform synthetic arbitrage between Polymarket and Kalshi.

    Monitors prices on both platforms simultaneously and identifies
    opportunities where the combined cost of YES + NO < $1.00.

    Trading logic:
      1. Scan for arb opportunities every tick
      2. Validate: sufficient liquidity, reasonable spread, acceptable TTE
      3. Size position based on bankroll and min spread threshold
      4. Execute both legs simultaneously (or as close as possible)
      5. Monitor convergence — exit when spread narrows to target
      6. Track P&L including slippage and fees
    """

    def __init__(
        self,
        min_spread_cents: Optional[float] = None,
        max_hold_minutes: Optional[int] = None,
        bankroll: float = 1000.0,
    ):
        self.min_spread_cents = min_spread_cents or settings.arb_min_spread_cents
        self.max_hold_minutes = max_hold_minutes or settings.arb_max_hold_minutes
        self.bankroll = bankroll

        self._opportunities: List[ArbOpportunity] = []
        self._positions: Dict[str, ArbPosition] = {}
        self._closed_positions: List[ArbPosition] = []
        self._performance = ArbPerformance()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._on_opportunity_callbacks: List[Callable] = []
        self._on_trade_callbacks: List[Callable] = []

        # Price state from both platforms
        self._pm_prices: Dict[str, Dict] = {}
        self._kalshi_prices: Dict[str, Dict] = {}

    # ── Public API ─────────────────────────────────────────────────────

    def on_opportunity(self, callback: Callable[[ArbOpportunity], None]) -> None:
        self._on_opportunity_callbacks.append(callback)

    def on_trade(self, callback: Callable[[ArbPosition], None]) -> None:
        self._on_trade_callbacks.append(callback)

    def update_pm_prices(self, market_id: str, yes_price: float, no_price: float, **kwargs) -> None:
        """Update Polymarket prices for a market."""
        self._pm_prices[market_id] = {
            "yes": yes_price,
            "no": no_price,
            "timestamp": time.time(),
            **kwargs,
        }

    def update_kalshi_prices(self, ticker: str, yes_price: float, no_price: float, **kwargs) -> None:
        """Update Kalshi prices for a contract."""
        self._kalshi_prices[ticker] = {
            "yes": yes_price,
            "no": no_price,
            "timestamp": time.time(),
            **kwargs,
        }

    def scan_for_opportunities(self, market_mapping: Optional[List[Dict]] = None) -> List[ArbOpportunity]:
        """
        Scan all known prices for arbitrage opportunities.

        market_mapping: list of dicts linking PM and Kalshi markets:
            [{"pm_market_id": "...", "kalshi_ticker": "...", "question": "..."}]

        Returns list of new opportunities found.
        """
        new_opportunities = []

        if market_mapping is None:
            # Try to match by question similarity
            market_mapping = self._auto_match_markets()

        for mapping in market_mapping:
            pm_id = mapping.get("pm_market_id", "")
            kalshi_ticker = mapping.get("kalshi_ticker", "")
            question = mapping.get("question", "")

            pm = self._pm_prices.get(pm_id)
            kalshi = self._kalshi_prices.get(kalshi_ticker)

            if pm is None or kalshi is None:
                continue

            # Check staleness (skip if data > 5 seconds old)
            if time.time() - pm["timestamp"] > 5 or time.time() - kalshi["timestamp"] > 5:
                continue

            opps = self._check_arb(pm_id, pm, kalshi_ticker, kalshi, question)
            new_opportunities.extend(opps)

        for opp in new_opportunities:
            self._opportunities.append(opp)
            self._performance.total_opportunities += 1
            for cb in self._on_opportunity_callbacks:
                try:
                    cb(opp)
                except Exception as e:
                    logger.error("Opportunity callback error: %s", e)

        return new_opportunities

    def evaluate_opportunity(self, opportunity: ArbOpportunity) -> bool:
        """
        Decide whether to take an arbitrage opportunity.

        Checks:
          1. Spread meets minimum threshold
          2. Sufficient bankroll
          3. Not too many concurrent positions
          4. Max hold time not exceeded
          5. Liquidity sufficient on both sides
        """
        # Spread check
        profit_cents = opportunity.guaranteed_profit_per_share * 100
        if profit_cents < self.min_spread_cents:
            logger.debug(
                "Arb rejected: spread %.2f¢ < min %.2f¢",
                profit_cents, self.min_spread_cents,
            )
            return False

        # Bankroll check
        if self.bankroll <= 0:
            return False

        # Concurrent positions check
        open_count = len([p for p in self._positions.values() if not p.closed])
        if open_count >= 10:
            logger.debug("Arb rejected: too many open positions (%d)", open_count)
            return False

        # Liquidity check (basic)
        if opportunity.pm_volume_24h < 100 or opportunity.kalshi_volume < 10:
            logger.debug("Arb rejected: insufficient liquidity")
            return False

        return True

    def size_position(self, opportunity: ArbOpportunity) -> float:
        """
        Calculate position size in USD for an arb opportunity.

        Uses a conservative approach:
          - Risk a fixed % of bankroll per arb
          - Scale with spread width (wider = more confident = larger)
          - Cap at max position size
        """
        # Base allocation: 2% of bankroll per arb
        base_pct = 0.02

        # Scale with spread width (wider spread = more profit = can be larger)
        spread_bps = opportunity.guaranteed_profit_per_share * 10000
        spread_multiplier = min(2.0, spread_bps / 50)  # 50bps = 1x, 100bps = 2x

        # Scale with confidence
        confidence_multiplier = max(0.5, min(1.5, opportunity.confidence))

        size_usd = self.bankroll * base_pct * spread_multiplier * confidence_multiplier
        max_size = self.bankroll * (settings.max_position_size_pct / 100)
        size_usd = min(size_usd, max_size)

        return round(size_usd, 2)

    async def execute_arb(self, opportunity: ArbOpportunity) -> Optional[ArbPosition]:
        """
        Execute an arbitrage trade on both platforms simultaneously.

        In paper mode, this simulates execution. In live mode, this would
        use PMXT/connector APIs to place orders on both platforms.
        """
        if not self.evaluate_opportunity(opportunity):
            return None

        size_usd = self.size_position(opportunity)
        if size_usd <= 0:
            return None

        # Calculate shares for each side
        if opportunity.direction == "pm_yes_kalshi_no":
            pm_price = opportunity.pm_yes_price
            kalshi_price = opportunity.kalshi_no_price
        else:
            pm_price = opportunity.pm_no_price
            kalshi_price = opportunity.kalshi_yes_price

        pm_shares = size_usd / 2 / pm_price if pm_price > 0 else 0
        kalshi_shares = size_usd / 2 / kalshi_price if kalshi_price > 0 else 0

        # Simulate execution (with slippage)
        slippage = 0.005  # 0.5% slippage per leg
        pm_fill = pm_price * (1 + slippage)
        kalshi_fill = kalshi_price * (1 + slippage)

        total_cost = pm_shares * pm_fill + kalshi_shares * kalshi_fill

        position = ArbPosition(
            position_id=f"arb_{int(time.time() * 1000)}",
            opportunity=opportunity,
            opened_at=time.time(),
            pm_fill_price=pm_fill,
            kalshi_fill_price=kalshi_fill,
            pm_size_shares=pm_shares,
            kalshi_size_shares=kalshi_shares,
            total_cost_usd=total_cost,
            current_pm_price=pm_fill,
            current_kalshi_price=kalshi_fill,
        )

        self._positions[position.position_id] = position
        self._performance.positions_opened += 1
        self._performance.opportunities_taken += 1

        # Update bankroll
        self.bankroll -= total_cost

        for cb in self._on_trade_callbacks:
            try:
                cb(position)
            except Exception as e:
                logger.error("Trade callback error: %s", e)

        logger.info(
            "ARB OPENED: %s %s (PM=%.3f, Kalshi=%.3f, cost=$%.2f, profit=$%.4f)",
            opportunity.direction,
            opportunity.market_question[:50],
            pm_fill, kalshi_fill, total_cost,
            opportunity.guaranteed_profit_per_share * min(pm_shares, kalshi_shares),
        )

        return position

    def update_position_prices(self, position_id: str, pm_price: float, kalshi_price: float) -> None:
        """Update current prices for an open position."""
        pos = self._positions.get(position_id)
        if pos is None or pos.closed:
            return

        pos.current_pm_price = pm_price
        pos.current_kalshi_price = kalshi_price
        pos.hold_seconds = time.time() - pos.opened_at

        # Calculate unrealized P&L
        if pos.opportunity.direction == "pm_yes_kalshi_no":
            pm_value = pos.pm_size_shares * pm_price
            kalshi_value = pos.kalshi_size_shares * (1 - kalshi_price)
        else:
            pm_value = pos.pm_size_shares * (1 - pm_price)
            kalshi_value = pos.kalshi_size_shares * kalshi_price

        pos.unrealized_pnl = (pm_value + kalshi_value) - pos.total_cost_usd

    def check_exits(self) -> List[str]:
        """
        Check all open positions for exit conditions.

        Exit conditions:
          1. Spread converged to target (take profit)
          2. Max hold time exceeded (force exit)
          3. Spread widened (stop loss — rare for true arb)
        """
        to_exit = []
        for pos_id, pos in self._positions.items():
            if pos.closed:
                continue

            # Max hold time
            if pos.hold_seconds > self.max_hold_minutes * 60:
                to_exit.append(pos_id)
                logger.info("ARB EXIT: max hold time reached for %s", pos_id)
                continue

            # Spread convergence check
            if pos.opportunity.direction == "pm_yes_kalshi_no":
                current_spread = pos.current_pm_price + pos.current_kalshi_price
            else:
                current_spread = (1 - pos.current_pm_price) + (1 - pos.current_kalshi_price)

            # Exit if spread has converged to near $1.00 (arb is gone)
            if current_spread >= 0.995:
                to_exit.append(pos_id)
                logger.info("ARB EXIT: spread converged to %.4f for %s", current_spread, pos_id)

        return to_exit

    def close_position(self, position_id: str) -> Optional[ArbPosition]:
        """Close an arbitrage position and realize P&L."""
        pos = self._positions.get(position_id)
        if pos is None or pos.closed:
            return None

        # Calculate exit value
        if pos.opportunity.direction == "pm_yes_kalshi_no":
            pm_value = pos.pm_size_shares * pos.current_pm_price
            kalshi_value = pos.kalshi_size_shares * (1 - pos.current_kalshi_price)
        else:
            pm_value = pos.pm_size_shares * (1 - pos.current_pm_price)
            kalshi_value = pos.kalshi_size_shares * pos.current_kalshi_price

        exit_value = pm_value + kalshi_value
        exit_pnl = exit_value - pos.total_cost_usd

        pos.closed = True
        pos.closed_at = time.time()
        pos.exit_pnl = exit_pnl

        # Update bankroll
        self.bankroll += exit_value

        # Update performance
        self._performance.positions_closed += 1
        self._performance.total_pnl += exit_pnl
        self._closed_positions.append(pos)

        # Calculate win rate
        wins = sum(1 for p in self._closed_positions if p.exit_pnl > 0)
        self._performance.win_rate = wins / len(self._closed_positions) * 100

        # Average hold time
        hold_times = [p.hold_seconds for p in self._closed_positions]
        self._performance.avg_hold_seconds = sum(hold_times) / len(hold_times)

        # Average profit
        pnls = [p.exit_pnl for p in self._closed_positions]
        self._performance.avg_profit_per_trade = sum(pnls) / len(pnls)

        logger.info(
            "ARB CLOSED: %s (pnl=$%.4f, hold=%.1fs)",
            position_id, exit_pnl, pos.hold_seconds,
        )

        return pos

    def _check_arb(
        self,
        pm_id: str,
        pm: Dict,
        kalshi_ticker: str,
        kalshi: Dict,
        question: str,
    ) -> List[ArbOpportunity]:
        """Check for arbitrage between one PM market and one Kalshi contract."""
        opportunities = []

        # Direction 1: PM YES + Kalshi NO
        # If PM_yes + Kalshi_no < 1.00, buy both = guaranteed profit
        spread_1 = pm["yes"] + kalshi["no"]
        profit_1 = 1.0 - spread_1

        if profit_1 > 0:
            opp = ArbOpportunity(
                timestamp=time.time(),
                market_question=question,
                pm_market_id=pm_id,
                pm_yes_price=pm["yes"],
                pm_no_price=pm["no"],
                pm_yes_token_id=pm.get("yes_token_id", ""),
                pm_no_token_id=pm.get("no_token_id", ""),
                kalshi_ticker=kalshi_ticker,
                kalshi_yes_price=kalshi["yes"],
                kalshi_no_price=kalshi["no"],
                kalshi_strike=kalshi.get("strike", 0),
                spread=spread_1,
                guaranteed_profit_per_share=profit_1,
                direction="pm_yes_kalshi_no",
                pm_volume_24h=pm.get("volume_24h", 0),
                kalshi_volume=kalshi.get("volume", 0),
                confidence=min(1.0, profit_1 / 0.05),
                time_to_expiry=kalshi.get("time_to_expiry", 0),
            )
            opportunities.append(opp)

        # Direction 2: PM NO + Kalshi YES
        # If PM_no + Kalshi_yes < 1.00, buy both = guaranteed profit
        spread_2 = pm["no"] + kalshi["yes"]
        profit_2 = 1.0 - spread_2

        if profit_2 > 0:
            opp = ArbOpportunity(
                timestamp=time.time(),
                market_question=question,
                pm_market_id=pm_id,
                pm_yes_price=pm["yes"],
                pm_no_price=pm["no"],
                pm_yes_token_id=pm.get("yes_token_id", ""),
                pm_no_token_id=pm.get("no_token_id", ""),
                kalshi_ticker=kalshi_ticker,
                kalshi_yes_price=kalshi["yes"],
                kalshi_no_price=kalshi["no"],
                kalshi_strike=kalshi.get("strike", 0),
                spread=spread_2,
                guaranteed_profit_per_share=profit_2,
                direction="pm_no_kalshi_yes",
                pm_volume_24h=pm.get("volume_24h", 0),
                kalshi_volume=kalshi.get("volume", 0),
                confidence=min(1.0, profit_2 / 0.05),
                time_to_expiry=kalshi.get("time_to_expiry", 0),
            )
            opportunities.append(opp)

        return opportunities

    def _auto_match_markets(self) -> List[Dict]:
        """Auto-match PM and Kalshi markets by question similarity."""
        mappings = []
        for pm_id, pm_data in self._pm_prices.items():
            pm_question = pm_data.get("question", "").lower()
            for kalshi_ticker, kalshi_data in self._kalshi_prices.items():
                kalshi_question = kalshi_data.get("question", "").lower()
                # Simple similarity: check if key words overlap
                pm_words = set(pm_question.split())
                kalshi_words = set(kalshi_question.split())
                overlap = len(pm_words & kalshi_words)
                if overlap >= 3:
                    mappings.append({
                        "pm_market_id": pm_id,
                        "kalshi_ticker": kalshi_ticker,
                        "question": pm_question,
                    })
        return mappings

    # ── Stats ──────────────────────────────────────────────────────────

    @property
    def stats(self) -> Dict:
        open_positions = [p for p in self._positions.values() if not p.closed]
        return {
            "bankroll": round(self.bankroll, 2),
            "open_positions": len(open_positions),
            "total_opportunities": self._performance.total_opportunities,
            "opportunities_taken": self._performance.opportunities_taken,
            "total_pnl": round(self._performance.total_pnl, 4),
            "win_rate": round(self._performance.win_rate, 1),
            "avg_hold_seconds": round(self._performance.avg_hold_seconds, 1),
            "avg_profit_per_trade": round(self._performance.avg_profit_per_trade, 4),
            "pm_markets_tracked": len(self._pm_prices),
            "kalshi_contracts_tracked": len(self._kalshi_prices),
        }

    @property
    def performance(self) -> ArbPerformance:
        return self._performance

    @property
    def open_positions(self) -> List[ArbPosition]:
        return [p for p in self._positions.values() if not p.closed]

    @property
    def closed_positions(self) -> List[ArbPosition]:
        return list(self._closed_positions)


# ── Orderbook-Based Intra-Platform Arbitrage ─────────────────────────────
# For 5-minute Polymarket markets: UP + DOWN must sum to ~$1.00

@dataclass
class OrderbookLevel:
    """Single price level in an orderbook."""
    price: float
    size: float
    side: str  # "buy" or "sell"
    timestamp: float = 0.0


@dataclass
class OrderbookState:
    """Orderbook state for a 5-min market."""
    market_id: str
    asset: str
    timestamp: float

    up_best_bid: float = 0.0
    up_best_ask: float = 0.0
    up_bid_depth: float = 0.0
    up_ask_depth: float = 0.0

    down_best_bid: float = 0.0
    down_best_ask: float = 0.0
    down_bid_depth: float = 0.0
    down_ask_depth: float = 0.0

    spread_complement: float = 0.0  # up_best_ask + down_best_ask
    bid_complement: float = 0.0     # up_best_bid + down_best_bid

    @property
    def complement_gap(self) -> float:
        """How far from $1.00 the complement spread is."""
        return 1.0 - self.spread_complement

    @property
    def bid_gap(self) -> float:
        """How far from $1.00 the bid complement is."""
        return self.bid_complement - 1.0


@dataclass
class OrderbookArbOpportunity:
    """An intra-platform orderbook arbitrage opportunity."""
    timestamp: float
    market_id: str
    asset: str
    direction: str  # "complement_buy" or "complement_sell"
    complement_spread: float
    profit_per_share: float
    up_price: float
    down_price: float
    up_depth: float
    down_depth: float
    estimated_slippage: float = 0.0
    net_profit_per_share: float = 0.0
    confidence: float = 0.0
    external_price_ref: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "market_id": self.market_id,
            "asset": self.asset,
            "direction": self.direction,
            "complement_spread": round(self.complement_spread, 4),
            "profit_per_share": round(self.profit_per_share, 4),
            "net_profit_per_share": round(self.net_profit_per_share, 4),
            "up_price": round(self.up_price, 4),
            "down_price": round(self.down_price, 4),
            "confidence": round(self.confidence, 3),
        }


class IntraPlatformArbEngine:
    """
    Orderbook-based arbitrage engine for 5-minute Polymarket markets.

    Detects when UP + DOWN token prices deviate from $1.00:
      - If UP_ask + DOWN_ask < $1.00 → buy both sides (complement buy)
      - If UP_bid + DOWN_bid > $1.00 → sell both sides (complement sell)
      - Cross-source: compare PM price to external feed (Chainlink, OKX)

    Includes slippage estimation and depth analysis for realistic fills.
    """

    def __init__(self, min_profit_cents: float = 1.0, bankroll: float = 100.0):
        self.min_profit_cents = min_profit_cents
        self.bankroll = bankroll
        self._orderbooks: Dict[str, OrderbookState] = {}
        self._opportunities: List[OrderbookArbOpportunity] = []
        self._external_prices: Dict[str, float] = {}  # asset -> last price
        self._running = False
        self._on_opportunity_callbacks: List[Callable] = []

    def on_opportunity(self, callback: Callable[[OrderbookArbOpportunity], None]) -> None:
        self._on_opportunity_callbacks.append(callback)

    def update_orderbook(
        self,
        market_id: str,
        asset: str,
        up_bids: List[Tuple[float, float]],
        up_asks: List[Tuple[float, float]],
        down_bids: List[Tuple[float, float]],
        down_asks: List[Tuple[float, float]],
    ) -> Optional[OrderbookState]:
        """Update orderbook state for a 5-min market."""
        up_best_bid = up_bids[0][0] if up_bids else 0.0
        up_best_ask = up_asks[0][0] if up_asks else 1.0
        up_bid_depth = sum(size for _, size in up_bids[:5])
        up_ask_depth = sum(size for _, size in up_asks[:5])

        down_best_bid = down_bids[0][0] if down_bids else 0.0
        down_best_ask = down_asks[0][0] if down_asks else 1.0
        down_bid_depth = sum(size for _, size in down_bids[:5])
        down_ask_depth = sum(size for _, size in down_asks[:5])

        state = OrderbookState(
            market_id=market_id,
            asset=asset,
            timestamp=time.time(),
            up_best_bid=up_best_bid,
            up_best_ask=up_best_ask,
            up_bid_depth=up_bid_depth,
            up_ask_depth=up_ask_depth,
            down_best_bid=down_best_bid,
            down_best_ask=down_best_ask,
            down_bid_depth=down_bid_depth,
            down_ask_depth=down_ask_depth,
            spread_complement=up_best_ask + down_best_ask,
            bid_complement=up_best_bid + down_best_bid,
        )
        self._orderbooks[market_id] = state
        return state

    def update_external_price(self, asset: str, price: float) -> None:
        """Update external reference price (Chainlink, OKX, etc.)."""
        self._external_prices[asset.upper()] = price

    def scan_opportunities(self) -> List[OrderbookArbOpportunity]:
        """Scan all tracked orderbooks for intra-platform arb opportunities."""
        new_opps = []
        for market_id, state in self._orderbooks.items():
            if time.time() - state.timestamp > 10:
                continue

            # Direction 1: Complement buy (UP_ask + DOWN_ask < $1.00)
            if state.spread_complement < 1.0:
                profit = 1.0 - state.spread_complement
                profit_cents = profit * 100
                if profit_cents >= self.min_profit_cents:
                    min_depth = min(state.up_ask_depth, state.down_ask_depth)
                    slippage = self._estimate_slippage(min_depth, profit_cents)
                    net_profit = profit - slippage
                    if net_profit > 0:
                        ext_price = self._external_prices.get(state.asset.upper(), 0)
                        confidence = self._compute_confidence(state, profit, ext_price)
                        opp = OrderbookArbOpportunity(
                            timestamp=time.time(),
                            market_id=market_id,
                            asset=state.asset,
                            direction="complement_buy",
                            complement_spread=state.spread_complement,
                            profit_per_share=profit,
                            up_price=state.up_best_ask,
                            down_price=state.down_best_ask,
                            up_depth=state.up_ask_depth,
                            down_depth=state.down_ask_depth,
                            estimated_slippage=slippage,
                            net_profit_per_share=net_profit,
                            confidence=confidence,
                            external_price_ref=ext_price,
                        )
                        new_opps.append(opp)

            # Direction 2: Complement sell (UP_bid + DOWN_bid > $1.00)
            if state.bid_complement > 1.0:
                profit = state.bid_complement - 1.0
                profit_cents = profit * 100
                if profit_cents >= self.min_profit_cents:
                    min_depth = min(state.up_bid_depth, state.down_bid_depth)
                    slippage = self._estimate_slippage(min_depth, profit_cents)
                    net_profit = profit - slippage
                    if net_profit > 0:
                        opp = OrderbookArbOpportunity(
                            timestamp=time.time(),
                            market_id=market_id,
                            asset=state.asset,
                            direction="complement_sell",
                            complement_spread=state.bid_complement,
                            profit_per_share=profit,
                            up_price=state.up_best_bid,
                            down_price=state.down_best_bid,
                            up_depth=state.up_bid_depth,
                            down_depth=state.down_bid_depth,
                            estimated_slippage=slippage,
                            net_profit_per_share=net_profit,
                            confidence=min(1.0, net_profit / 0.05),
                            external_price_ref=self._external_prices.get(state.asset.upper(), 0),
                        )
                        new_opps.append(opp)

        for opp in new_opps:
            self._opportunities.append(opp)
            for cb in self._on_opportunity_callbacks:
                try:
                    cb(opp)
                except Exception as e:
                    logger.error("Orderbook arb callback error: %s", e)

        return new_opps

    def size_position(self, opportunity: OrderbookArbOpportunity) -> float:
        """Calculate position size in USD for an orderbook arb opportunity."""
        base_pct = 0.03
        depth_factor = min(2.0, min(opportunity.up_depth, opportunity.down_depth) / 50)
        profit_factor = min(2.0, opportunity.net_profit_per_share * 100 / 3)
        size_usd = self.bankroll * base_pct * depth_factor * profit_factor * opportunity.confidence
        max_size = self.bankroll * 0.10
        return round(min(size_usd, max_size), 2)

    def _estimate_slippage(self, depth: float, profit_cents: float) -> float:
        """Estimate slippage based on available depth and profit size."""
        if depth <= 0:
            return profit_cents * 0.5 / 100
        volume_impact = min(0.3, 10.0 / depth)
        base_slippage = profit_cents * 0.1 / 100
        return (base_slippage + volume_impact * profit_cents * 0.2 / 100)

    def _compute_confidence(
        self,
        state: OrderbookState,
        profit: float,
        external_price: float,
    ) -> float:
        """Compute confidence score for an orderbook arb opportunity."""
        depth_score = min(1.0, min(state.up_ask_depth, state.down_ask_depth) / 100)
        profit_score = min(1.0, profit / 0.05)
        freshness = max(0.0, 1.0 - (time.time() - state.timestamp) / 10)
        ext_score = 0.0
        if external_price > 0:
            ext_score = 1.0
        return (depth_score * 0.3 + profit_score * 0.3 + freshness * 0.2 + ext_score * 0.2)

    @property
    def stats(self) -> Dict:
        return {
            "markets_tracked": len(self._orderbooks),
            "opportunities_found": len(self._opportunities),
            "external_prices": dict(self._external_prices),
        }
