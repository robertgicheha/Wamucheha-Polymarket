"""
Core BRTI calculation engine.

Replicates the CF Benchmarks BTC Reference Rate methodology:
  1. Receives consolidated orderbook from OrderbookConsolidator
  2. Builds bid-side and ask-side price-volume curves
  3. Computes mid price for each side
  4. Calculates utilized depth — max volume where deviation from mid ≤ 0.5%
  5. Applies exponential weighting (λ = 1/5000) across price levels
  6. Computes BRTI as volume-weighted mid price
  7. Ticks at 1-second intervals

References:
  - CF Benchmarks methodology: https://cfbenchmarks.com/methodology
  - Bürgi, Deng & Whelan (2026) — used for downstream FLB calibration, not here
"""
import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from config.settings import settings
from connectors.exchange_ws_base import OrderbookConsolidator, OrderbookSnapshot

logger = logging.getLogger(__name__)


@dataclass
class PriceVolumePoint:
    """A point on the mid price-volume curve."""
    price: float
    cumulative_volume: float


@dataclass
class BRTITick:
    """One BRTI publication."""
    timestamp: float
    brti_price: float
    bid_volume: float
    ask_volume: float
    utilized_depth_bid: float
    utilized_depth_ask: float
    mid_price_bid: float
    mid_price_ask: float
    spread_bps: float
    exchanges_used: List[str]

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "brti_price": self.brti_price,
            "bid_volume": self.bid_volume,
            "ask_volume": self.ask_volume,
            "utilized_depth_bid": self.utilized_depth_bid,
            "utilized_depth_ask": self.utilized_depth_ask,
            "mid_price_bid": self.mid_price_bid,
            "mid_price_ask": self.mid_price_ask,
            "spread_bps": self.spread_bps,
            "exchanges_used": self.exchanges_used,
        }


class BRTIEngine:
    """
    Core BRTI calculation engine.

    Methodology per CF Benchmarks:
      - Aggregate orderbooks from 4+ exchanges
      - Apply 100 BTC order cap
      - For each side (bid/ask), build a price-volume curve
      - Calculate utilized depth per side
      - Apply exponential weighting to price levels
      - Compute weighted mid price
      - Publish at 1-second frequency
    """

    def __init__(
        self,
        order_size_cap: Optional[float] = None,
        max_volume: Optional[float] = None,
        deviation_threshold: Optional[float] = None,
        lambda_decay: float = 1 / 5000,
        validation_enabled: Optional[bool] = None,
        max_divergence_bps: Optional[float] = None,
        tick_interval: Optional[int] = None,
    ):
        self.order_size_cap = order_size_cap or settings.brti_order_size_cap
        self.max_volume = max_volume or settings.brti_max_volume
        self.deviation_threshold = deviation_threshold or settings.brti_deviation_threshold
        self.lambda_decay = lambda_decay
        self.validation_enabled = (
            validation_enabled
            if validation_enabled is not None
            else settings.brti_validation_enabled
        )
        self.max_divergence_bps = max_divergence_bps or settings.brti_max_divergence_bps
        self.tick_interval = tick_interval or settings.brti_tick_interval_seconds

        self._consolidator = OrderbookConsolidator(order_size_cap=self.order_size_cap)
        self._last_tick: Optional[BRTITick] = None
        self._tick_history: List[BRTITick] = []
        self._max_history = 3600  # 1 hour of 1-sec ticks
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._on_tick_callbacks: List[Callable[[BRTITick], None]] = []
        self._tick_count = 0
        self._validation_failures = 0

    # ── Public API ─────────────────────────────────────────────────────

    def on_tick(self, callback: Callable[[BRTITick], None]) -> None:
        """Register a callback for each BRTI tick."""
        self._on_tick_callbacks.append(callback)

    def update_orderbook(self, snapshot: OrderbookSnapshot) -> None:
        """Feed an orderbook snapshot into the consolidator."""
        self._consolidator.update_snapshot(snapshot)

    def calculate_tick(self) -> Optional[BRTITick]:
        """
        Calculate one BRTI tick from current consolidated orderbook.

        Returns None if no valid data is available.
        """
        consolidated = self._consolidator.get_consolidated()
        if consolidated is None:
            return None

        bids = consolidated["bids"]
        asks = consolidated["asks"]
        exchanges_used = consolidated["exchanges_used"]
        latest_ts = consolidated["timestamp"]

        if len(bids) < 2 or len(asks) < 2:
            return None

        # Build price-volume curves
        bid_curve = self._build_price_volume_curve(bids)
        ask_curve = self._build_price_volume_curve(asks)

        if not bid_curve or not ask_curve:
            return None

        # Calculate utilized depth per side
        utilized_bid = self._calculate_utilized_depth(bid_curve, side="bid")
        utilized_ask = self._calculate_utilized_depth(ask_curve, side="ask")

        # Apply exponential weighting and compute weighted mid price
        weighted_mid_bid = self._weighted_mid_price(bid_curve, utilized_bid)
        weighted_mid_ask = self._weighted_mid_price(ask_curve, utilized_ask)

        if weighted_mid_bid is None or weighted_mid_ask is None:
            return None

        # BRTI = volume-weighted average of bid and ask weighted mid prices
        total_volume = utilized_bid + utilized_ask
        if total_volume <= 0:
            return None

        brti_price = (
            weighted_mid_bid * utilized_bid + weighted_mid_ask * utilized_ask
        ) / total_volume

        # Validation: check bid/ask divergence
        spread_bps = abs(weighted_mid_ask - weighted_mid_bid) / brti_price * 10000

        if self.validation_enabled:
            if spread_bps > self.max_divergence_bps:
                logger.warning(
                    "BRTI validation: spread %.2f bps exceeds max %.2f bps — "
                    "divergence too high, skipping tick",
                    spread_bps, self.max_divergence_bps,
                )
                self._validation_failures += 1
                return None

        tick = BRTITick(
            timestamp=latest_ts,
            brti_price=brti_price,
            bid_volume=utilized_bid,
            ask_volume=utilized_ask,
            utilized_depth_bid=utilized_bid,
            utilized_depth_ask=utilized_ask,
            mid_price_bid=weighted_mid_bid,
            mid_price_ask=weighted_mid_ask,
            spread_bps=spread_bps,
            exchanges_used=exchanges_used,
        )

        self._last_tick = tick
        self._tick_history.append(tick)
        if len(self._tick_history) > self._max_history:
            self._tick_history = self._tick_history[-self._max_history:]
        self._tick_count += 1

        for cb in self._on_tick_callbacks:
            try:
                cb(tick)
            except Exception as e:
                logger.error("BRTI tick callback error: %s", e)

        return tick

    async def start(self, tick_callback: Optional[Callable[[BRTITick], None]] = None) -> None:
        """Start the 1-second tick loop."""
        if tick_callback:
            self.on_tick(tick_callback)
        self._running = True
        self._task = asyncio.create_task(self._tick_loop())
        logger.info(
            "BRTI engine started (tick interval=%ds, lambda=%.6f)",
            self.tick_interval, self.lambda_decay,
        )

    async def stop(self) -> None:
        """Stop the tick loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("BRTI engine stopped (ticks: %d, validation failures: %d)",
                     self._tick_count, self._validation_failures)

    @property
    def last_tick(self) -> Optional[BRTITick]:
        return self._last_tick

    @property
    def tick_history(self) -> List[BRTITick]:
        return list(self._tick_history)

    @property
    def stats(self) -> Dict:
        return {
            "running": self._running,
            "tick_count": self._tick_count,
            "validation_failures": self._validation_failures,
            "last_price": self._last_tick.brti_price if self._last_tick else None,
            "last_timestamp": self._last_tick.timestamp if self._last_tick else None,
            "history_length": len(self._tick_history),
        }

    # ── Price-Volume Curve ─────────────────────────────────────────────

    def _build_price_volume_curve(
        self, levels: List[Dict]
    ) -> List[PriceVolumePoint]:
        """
        Build a cumulative price-volume curve from orderbook levels.

        Each point represents: at this price level, the total cumulative volume
        from the best price to this level.

        Args:
            levels: list of {"price": float, "size": float}, sorted
                    (descending for bids, ascending for asks)
        """
        if not levels:
            return []

        curve = []
        cumulative = 0.0
        best_price = levels[0]["price"]

        for level in levels:
            cumulative += level["size"]
            curve.append(PriceVolumePoint(
                price=level["price"],
                cumulative_volume=cumulative,
            ))

        return curve

    # ── Utilized Depth ─────────────────────────────────────────────────

    def _calculate_utilized_depth(
        self,
        curve: List[PriceVolumePoint],
        side: str = "bid",
    ) -> float:
        """
        Calculate utilized depth per CF Benchmarks methodology.

        Utilized depth is the maximum volume where the price deviation
        from the best price is ≤ 0.5% (the deviation_threshold).

        For bids: levels with price >= best_price * (1 - threshold)
        For asks: levels with price <= best_price * (1 + threshold)
        """
        if not curve:
            return 0.0

        best_price = curve[0].price
        if best_price <= 0:
            return 0.0

        max_volume = 0.0
        for point in curve:
            deviation = abs(point.price - best_price) / best_price

            if deviation <= self.deviation_threshold:
                max_volume = point.cumulative_volume
            else:
                break

        return max_volume

    # ── Weighted Mid Price ─────────────────────────────────────────────

    def _weighted_mid_price(
        self,
        curve: List[PriceVolumePoint],
        utilized_depth: float,
    ) -> Optional[float]:
        """
        Calculate the exponentially weighted mid price.

        For each price level within utilized depth:
          weight = exp(-cumulative_volume / λ)

        BRTI price = Σ(price × weight) / Σ(weight)

        This gives more weight to levels closer to the best price,
        with exponential decay toward deeper levels.
        """
        if not curve or utilized_depth <= 0:
            return None

        weighted_sum = 0.0
        weight_total = 0.0

        for point in curve:
            if point.cumulative_volume > utilized_depth:
                break

            # Cumulative volume at this price level
            volume_at_price = point.cumulative_volume

            # Exponential decay weight — deeper levels get exponentially less weight
            weight = math.exp(-volume_at_price * self.lambda_decay)

            weighted_sum += point.price * weight
            weight_total += weight

        if weight_total <= 0:
            return None

        return weighted_sum / weight_total

    # ── Tick Loop ──────────────────────────────────────────────────────

    async def _tick_loop(self) -> None:
        """Main tick loop — publishes BRTI at configured interval."""
        logger.info("BRTI tick loop started")
        while self._running:
            try:
                tick = self.calculate_tick()
                if tick is None:
                    logger.debug("BRTI tick skipped — no valid data")
            except Exception as e:
                logger.error("BRTI tick error: %s", e, exc_info=True)

            await asyncio.sleep(self.tick_interval)

    # ── Convenience Methods ────────────────────────────────────────────

    def get_mid_spread(self) -> Optional[float]:
        """Get current spread between bid and ask mid prices in bps."""
        if self._last_tick is None:
            return None
        return self._last_tick.spread_bps

    def get_price_history(self, seconds: int = 60) -> List[Tuple[float, float]]:
        """Get (timestamp, price) pairs for the last N seconds."""
        cutoff = time.time() - seconds
        return [
            (t.timestamp, t.brti_price)
            for t in self._tick_history
            if t.timestamp >= cutoff
        ]

    def get_volatility(self, window_seconds: int = 60) -> Optional[float]:
        """
        Calculate recent volatility as std dev of log returns.
        Useful for downstream ML features.
        """
        history = self.get_price_history(window_seconds)
        if len(history) < 10:
            return None

        import numpy as np
        prices = [p for _, p in history if p > 0]
        if len(prices) < 10:
            return None

        log_returns = np.diff(np.log(prices))
        return float(np.std(log_returns))
