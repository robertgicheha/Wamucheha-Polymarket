"""
Kalshi settlement calculator for BTC contracts.

Kalshi settles BTC prediction markets using a 60-second rolling average
of the final BRTI prints published at contract close.

This module:
  1. Maintains a rolling window of BRTI ticks
  2. Calculates the 60-second average at contract close
  3. Determines YES/NO settlement for a given strike price
  4. Provides real-time predicted settlement price during contract lifecycle

Kalshi BTC contract types:
  - Price contracts: "Will BTC be above $X at time T?"
  - Range contracts: "Will BTC be between $X and $Y at time T?"
  - Touch contracts: "Will BTC touch $X before time T?"
"""
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from data.brti.brti_engine import BRTITick
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class KalshiContract:
    """Represents a Kalshi BTC price contract."""
    ticker: str
    event_ticker: str
    strike_price: float
    contract_type: str  # "price", "range", "touch"
    close_time: float  # unix timestamp when contract closes
    settlement_time: float  # unix timestamp when settlement happens

    # For range contracts
    range_low: Optional[float] = None
    range_high: Optional[float] = None

    # Current state
    last_yes_price: Optional[float] = None
    last_no_price: Optional[float] = None
    volume: int = 0

    @property
    def time_to_close_seconds(self) -> float:
        return max(0, self.close_time - time.time())

    @property
    def is_closable(self) -> bool:
        return time.time() >= self.close_time

    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "event_ticker": self.event_ticker,
            "strike_price": self.strike_price,
            "contract_type": self.contract_type,
            "close_time": self.close_time,
            "settlement_time": self.settlement_time,
            "range_low": self.range_low,
            "range_high": self.range_high,
            "last_yes_price": self.last_yes_price,
            "last_no_price": self.last_no_price,
            "volume": self.volume,
            "time_to_close_seconds": self.time_to_close_seconds,
        }


@dataclass
class SettlementResult:
    """Result of a contract settlement calculation."""
    ticker: str
    strike_price: float
    settlement_price: float  # 60-sec average BRTI
    result: str  # "YES" or "NO"
    yes_payout: float  # 1.0 if YES, 0.0 if NO
    no_payout: float  # 0.0 if YES, 1.0 if NO
    ticks_used: int
    window_start: float
    window_end: float

    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "strike_price": self.strike_price,
            "settlement_price": self.settlement_price,
            "result": self.result,
            "yes_payout": self.yes_payout,
            "no_payout": self.no_payout,
            "ticks_used": self.ticks_used,
            "window_start": self.window_start,
            "window_end": self.window_end,
        }


class KalshiSettlementCalculator:
    """
    Calculates Kalshi settlement prices using rolling BRTI averages.

    Per Kalshi's methodology:
      - Settlement = average of final 60 seconds of BRTI prints
      - For price contracts: YES if settlement ≥ strike
      - For range contracts: YES if settlement in [low, high]
      - Touch contracts settle differently (handled separately)
    """

    def __init__(
        self,
        window_seconds: Optional[int] = None,
        max_contracts: int = 100,
    ):
        self.window_seconds = window_seconds or settings.brti_kalshi_avg_window
        self.max_contracts = max_contracts

        # Rolling window of BRTI ticks
        self._tick_window: Deque[BRTITick] = deque()

        # Active contracts we're tracking
        self._contracts: Dict[str, KalshiContract] = {}

        # Settlement history
        self._settlements: List[SettlementResult] = []
        self._max_settlement_history = 1000

    # ── Tick Ingestion ─────────────────────────────────────────────────

    def on_brti_tick(self, tick: BRTITick) -> None:
        """
        Feed a BRTI tick into the settlement window.

        Called by BRTIEngine.on_tick() — maintains the rolling window.
        """
        self._tick_window.append(tick)

        # Trim old ticks outside the window
        cutoff = tick.timestamp - self.window_seconds
        while self._tick_window and self._tick_window[0].timestamp < cutoff:
            self._tick_window.popleft()

    # ── Settlement Calculation ─────────────────────────────────────────

    def calculate_settlement(self, contract: KalshiContract) -> Optional[SettlementResult]:
        """
        Calculate the settlement result for a contract.

        Uses the rolling 60-second window of BRTI ticks to compute
        the average price and determine YES/NO outcome.
        """
        if not self._tick_window:
            logger.warning("No BRTI ticks available for settlement of %s", contract.ticker)
            return None

        ticks = list(self._tick_window)
        if not ticks:
            return None

        # Calculate 60-second average
        prices = [t.brti_price for t in ticks if t.brti_price > 0]
        if not prices:
            return None

        avg_price = sum(prices) / len(prices)

        # Determine settlement based on contract type
        result, yes_payout, no_payout = self._determine_settlement(
            contract, avg_price
        )

        settlement = SettlementResult(
            ticker=contract.ticker,
            strike_price=contract.strike_price,
            settlement_price=avg_price,
            result=result,
            yes_payout=yes_payout,
            no_payout=no_payout,
            ticks_used=len(prices),
            window_start=ticks[0].timestamp,
            window_end=ticks[-1].timestamp,
        )

        self._settlements.append(settlement)
        if len(self._settlements) > self._max_settlement_history:
            self._settlements = self._settlements[-self._max_settlement_history:]

        logger.info(
            "Settlement for %s: %s (avg=%.2f, strike=%.2f, ticks=%d)",
            contract.ticker, result, avg_price, contract.strike_price, len(prices),
        )

        return settlement

    def _determine_settlement(
        self, contract: KalshiContract, avg_price: float
    ) -> tuple:
        """Determine YES/NO and payouts for a contract type."""
        if contract.contract_type == "price":
            return self._settle_price_contract(contract, avg_price)
        elif contract.contract_type == "range":
            return self._settle_range_contract(contract, avg_price)
        elif contract.contract_type == "touch":
            return self._settle_touch_contract(contract, avg_price)
        else:
            logger.error("Unknown contract type: %s", contract.contract_type)
            return "UNKNOWN", 0.0, 0.0

    def _settle_price_contract(
        self, contract: KalshiContract, avg_price: float
    ) -> tuple:
        """
        Price contract: YES if settlement ≥ strike.

        Most common Kalshi BTC contract type.
        """
        if avg_price >= contract.strike_price:
            return "YES", 1.0, 0.0
        else:
            return "NO", 0.0, 1.0

    def _settle_range_contract(
        self, contract: KalshiContract, avg_price: float
    ) -> tuple:
        """
        Range contract: YES if settlement is between low and high.
        """
        low = contract.range_low or 0.0
        high = contract.range_high or float('inf')

        if low <= avg_price <= high:
            return "YES", 1.0, 0.0
        else:
            return "NO", 0.0, 1.0

    def _settle_touch_contract(
        self, contract: KalshiContract, avg_price: float
    ) -> tuple:
        """
        Touch contract: YES if BTC touched strike at any point.

        Note: Touch contracts need tick-by-tick monitoring, not just
        the 60-second average. This is a simplified version that
        checks if the average is near the strike.
        """
        tolerance = contract.strike_price * 0.001  # 0.1% tolerance
        if abs(avg_price - contract.strike_price) <= tolerance:
            return "YES", 1.0, 0.0
        else:
            return "NO", 0.0, 1.0

    # ── Predicted Settlement ───────────────────────────────────────────

    def get_predicted_settlement(self, contract: KalshiContract) -> Optional[Dict]:
        """
        Get real-time predicted settlement during contract lifecycle.

        Useful for determining current edge before contract closes.
        """
        if not self._tick_window:
            return None

        ticks = list(self._tick_window)
        prices = [t.brti_price for t in ticks if t.brti_price > 0]
        if not prices:
            return None

        avg_price = sum(prices) / len(prices)
        min_price = min(prices)
        max_price = max(prices)
        volatility = self._calculate_volatility(prices)

        # Probability of YES based on current average and strike
        # Simple model: if avg > strike, high probability of YES
        if contract.contract_type == "price":
            if avg_price > contract.strike_price:
                yes_probability = min(1.0, 0.5 + (avg_price - contract.strike_price) / contract.strike_price * 10)
            else:
                yes_probability = max(0.0, 0.5 - (contract.strike_price - avg_price) / contract.strike_price * 10)
        else:
            yes_probability = 0.5  # Unknown for other types

        return {
            "ticker": contract.ticker,
            "strike_price": contract.strike_price,
            "predicted_avg": avg_price,
            "min_price_window": min_price,
            "max_price_window": max_price,
            "volatility": volatility,
            "estimated_yes_probability": yes_probability,
            "ticks_in_window": len(prices),
            "time_to_close_seconds": contract.time_to_close_seconds,
        }

    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calculate short-term volatility as std dev of returns."""
        if len(prices) < 2:
            return 0.0

        import math
        returns = [
            (prices[i] - prices[i-1]) / prices[i-1]
            for i in range(1, len(prices))
            if prices[i-1] > 0
        ]
        if not returns:
            return 0.0

        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / len(returns)
        return math.sqrt(variance)

    # ── Contract Management ────────────────────────────────────────────

    def add_contract(self, contract: KalshiContract) -> None:
        """Add a contract to track for settlement."""
        if len(self._contracts) >= self.max_contracts:
            logger.warning("Max contracts reached, cannot add %s", contract.ticker)
            return
        self._contracts[contract.ticker] = contract
        logger.info("Tracking contract %s (strike=%.2f, type=%s)",
                     contract.ticker, contract.strike_price, contract.contract_type)

    def remove_contract(self, ticker: str) -> None:
        """Remove a tracked contract."""
        self._contracts.pop(ticker, None)

    def get_settlable_contracts(self) -> List[KalshiContract]:
        """Get contracts that are ready to settle."""
        return [c for c in self._contracts.values() if c.is_closable]

    def get_active_contracts(self) -> List[KalshiContract]:
        """Get all active (not yet closed) contracts."""
        return [c for c in self._contracts.values() if not c.is_closable]

    # ── Stats ──────────────────────────────────────────────────────────

    @property
    def stats(self) -> Dict:
        return {
            "tick_window_size": len(self._tick_window),
            "window_seconds": self.window_seconds,
            "active_contracts": len(self.get_active_contracts()),
            "settlable_contracts": len(self.get_settlable_contracts()),
            "total_settlements": len(self._settlements),
            "current_avg_price": self._current_average_price(),
        }

    def _current_average_price(self) -> Optional[float]:
        """Get current rolling average price."""
        if not self._tick_window:
            return None
        prices = [t.brti_price for t in self._tick_window if t.brti_price > 0]
        if not prices:
            return None
        return sum(prices) / len(prices)

    @property
    def recent_settlements(self) -> List[SettlementResult]:
        return list(self._settlements[-10:])
