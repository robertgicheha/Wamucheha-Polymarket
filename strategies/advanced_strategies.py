"""
Advanced strategies:
  4. Momentum Convergence — trade with trend when model confirms
  5. Mean Reversion — trade against extreme moves when model confirms
  6. Time Decay Harvesting — exploit theta decay near expiry

These strategies use additional signals (momentum, volatility, TTE)
beyond just edge, making them more nuanced than the base strategies.
"""
import math
from typing import Dict, Optional

from strategies.base_strategies import BaseStrategy, TradeDecision
from config.settings import settings


class MomentumConvergenceStrategy(BaseStrategy):
    """
    Trade with the trend when the ML model confirms the direction.

    Logic:
      - If market is trending up AND model says YES is underpriced → buy YES
      - If market is trending down AND model says NO is underpriced → buy NO
      - Size increases with trend strength and model confidence
      - Exits if momentum reverses

    This captures the "momentum premium" — prices that are moving in a
    direction tend to continue (short-term), especially when the model
    agrees with the direction.
    """
    name = "momentum_convergence"
    description = "Trade with trend when ML model confirms direction"

    def __init__(
        self,
        momentum_threshold: float = 0.001,
        momentum_lookback: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.momentum_threshold = momentum_threshold
        self.momentum_lookback = momentum_lookback

    def should_trade(
        self,
        model_prob: float,
        market_price: float,
        bankroll: float,
        tte_seconds: int = 0,
        volatility: float = 0.0,
        extra: Optional[Dict] = None,
    ) -> TradeDecision:
        if not self._validate_price(market_price):
            return self._no_trade(f"Price {market_price:.3f} outside zone")

        extra = extra or {}
        momentum = extra.get("momentum", 0.0)
        price_trend = extra.get("price_trend", 0.0)

        side = self._determine_side(model_prob, market_price)
        edge = self._calculate_edge(model_prob, market_price, side)

        if edge < self.min_edge:
            return self._no_trade(f"Edge {edge:.4f} below min", side)

        # Check momentum alignment
        if side == "YES" and momentum < -self.momentum_threshold:
            return self._no_trade("Momentum opposes YES trade", side)
        if side == "NO" and momentum > self.momentum_threshold:
            return self._no_trade("Momentum opposes NO trade", side)

        # Momentum must exceed threshold
        if abs(momentum) < self.momentum_threshold:
            return self._no_trade(f"Momentum {momentum:.6f} below threshold", side)

        # Size: base Kelly + momentum multiplier
        kelly_frac = self._kelly_fraction(model_prob if side == "YES" else (1 - model_prob), market_price if side == "YES" else (1 - market_price))
        momentum_multiplier = min(2.0, 1.0 + abs(momentum) * 100)
        size_usd = bankroll * kelly_frac * momentum_multiplier

        # Reduce for high volatility (momentum is less reliable)
        if volatility > 0.03:
            size_usd *= max(0.3, 1.0 - (volatility - 0.03) * 15)

        size_usd = max(0.0, round(size_usd, 2))

        return TradeDecision(
            should_trade=size_usd > 0,
            side=side,
            size_usd=size_usd,
            confidence=min(1.0, edge / 0.10 * momentum_multiplier / 2),
            edge=edge,
            strategy_name=self.name,
            reason=f"Momentum={momentum:.6f}, mult={momentum_multiplier:.2f}, edge={edge:.4f}",
        )

    def _kelly_fraction(self, p: float, market_p: float) -> float:
        if market_p <= 0 or market_p >= 1:
            return 0.0
        b = (1 - market_p) / market_p
        if b <= 0:
            return 0.0
        full_kelly = (p * (b + 1) - 1) / b
        return max(0.0, min(full_kelly * self.kelly_cap, settings.max_position_size_pct / 100))


class MeanReversionStrategy(BaseStrategy):
    """
    Trade against extreme moves when the model confirms overreaction.

    Logic:
      - If market has moved far from recent mean AND model confirms
        the move is overdone → trade the reversion
      - Uses Z-score of price vs moving average
      - Larger positions when deviation is more extreme

    This captures the "mean reversion premium" — extreme moves
    tend to partially reverse, especially in prediction markets
    where prices are bounded [0, 1].
    """
    name = "mean_reversion"
    description = "Trade against extreme moves when model confirms overreaction"

    def __init__(
        self,
        z_score_threshold: float = 1.5,
        z_score_max: float = 3.0,
        lookback: int = 50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.z_score_threshold = z_score_threshold
        self.z_score_max = z_score_max
        self.lookback = lookback

    def should_trade(
        self,
        model_prob: float,
        market_price: float,
        bankroll: float,
        tte_seconds: int = 0,
        volatility: float = 0.0,
        extra: Optional[Dict] = None,
    ) -> TradeDecision:
        if not self._validate_price(market_price):
            return self._no_trade(f"Price {market_price:.3f} outside zone")

        extra = extra or {}
        z_score = extra.get("z_score", 0.0)
        price_vs_ma = extra.get("price_vs_ma", 0.0)

        side = self._determine_side(model_prob, market_price)
        edge = self._calculate_edge(model_prob, market_price, side)

        if edge < self.min_edge:
            return self._no_trade(f"Edge {edge:.4f} below min", side)

        # Mean reversion: trade AGAINST the z-score direction
        if side == "YES" and z_score > self.z_score_threshold:
            pass  # Price is above mean, model says buy YES → conflict, skip
        elif side == "NO" and z_score < -self.z_score_threshold:
            pass  # Price is below mean, model says buy NO → conflict, skip
        elif side == "YES" and z_score < -self.z_score_threshold:
            pass  # Price below mean, model says buy YES → aligned, good
        elif side == "NO" and z_score > self.z_score_threshold:
            pass  # Price above mean, model says buy NO → aligned, good
        else:
            return self._no_trade(f"Z-score {z_score:.2f} not extreme enough", side)

        # Check alignment: reversion must be in same direction as model
        if side == "YES" and z_score > 0:
            return self._no_trade("Reversion opposes model for YES", side)
        if side == "NO" and z_score < 0:
            return self._no_trade("Reversion opposes model for NO", side)

        abs_z = min(abs(z_score), self.z_score_max)
        if abs_z < self.z_score_threshold:
            return self._no_trade(f"Z-score {z_score:.2f} below threshold", side)

        # Size scales with extremity
        z_multiplier = abs_z / self.z_score_threshold
        kelly_frac = self._kelly_fraction(
            model_prob if side == "YES" else (1 - model_prob),
            market_price if side == "YES" else (1 - market_price),
        )
        size_usd = bankroll * kelly_frac * min(z_multiplier, 2.5)

        # Time decay: mean reversion needs time to play out
        if 0 < tte_seconds < 120:
            size_usd *= tte_seconds / 120

        size_usd = max(0.0, round(size_usd, 2))

        return TradeDecision(
            should_trade=size_usd > 0,
            side=side,
            size_usd=size_usd,
            confidence=min(1.0, edge / 0.10 * z_multiplier / 2),
            edge=edge,
            strategy_name=self.name,
            reason=f"Z-score={z_score:.2f}, z_mult={z_multiplier:.2f}, edge={edge:.4f}",
        )

    def _kelly_fraction(self, p: float, market_p: float) -> float:
        if market_p <= 0 or market_p >= 1:
            return 0.0
        b = (1 - market_p) / market_p
        if b <= 0:
            return 0.0
        full_kelly = (p * (b + 1) - 1) / b
        return max(0.0, min(full_kelly * self.kelly_cap, settings.max_position_size_pct / 100))


class TimeDecayHarvestingStrategy(BaseStrategy):
    """
    Harvest theta decay near expiry in prediction markets.

    Key insight: As a prediction market contract approaches expiry,
    the price converges to 0 or 1. Options that are deep ITM or OTM
    experience accelerating time decay.

    Strategy:
      - If model is HIGHLY confident (prob > 0.85 or < 0.15)
        and TTE is short (< 5 min), buy the winning side
      - The price will converge to 0 or 1, capturing the spread
      - Size increases as TTE decreases (more certainty)
      - Only trade when the convergence is "free money" (very high edge)

    Risk: If the model is wrong, you lose everything near expiry.
    Mitigation: Only trade with extreme confidence + extreme edge.
    """
    name = "time_decay_harvesting"
    description = "Exploit price convergence near expiry with high-confidence predictions"

    def __init__(
        self,
        min_confidence: float = 0.80,
        max_tte: int = 300,
        min_edge_decay: float = 0.10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_confidence = min_confidence
        self.max_tte = max_tte
        self.min_edge_decay = min_edge_decay

    def should_trade(
        self,
        model_prob: float,
        market_price: float,
        bankroll: float,
        tte_seconds: int = 0,
        volatility: float = 0.0,
        extra: Optional[Dict] = None,
    ) -> TradeDecision:
        if not self._validate_price(market_price):
            return self._no_trade(f"Price {market_price:.3f} outside zone")

        # Only trade near expiry
        if tte_seconds > self.max_tte:
            return self._no_trade(f"TTE {tte_seconds}s > max {self.max_tte}s")

        if tte_seconds <= 0:
            return self._no_trade("Contract already expired")

        side = self._determine_side(model_prob, market_price)
        edge = self._calculate_edge(model_prob, market_price, side)

        # Need very high edge for time decay harvesting
        if edge < self.min_edge_decay:
            return self._no_trade(f"Edge {edge:.4f} below decay min {self.min_edge_decay}", side)

        # Need high confidence
        confidence = min(model_prob, 1 - model_prob)
        if confidence < (1 - self.min_confidence):
            return self._no_trade(
                f"Confidence {confidence:.3f} below {self.min_confidence}", side
            )

        # Size: scales with TTE decay (shorter = more certain = larger)
        time_decay_factor = 1.0 + (self.max_tte - tte_seconds) / self.max_tte
        kelly_frac = self._kelly_fraction(
            model_prob if side == "YES" else (1 - model_prob),
            market_price if side == "YES" else (1 - market_price),
        )

        # Aggressive sizing for high-confidence near-expiry
        size_usd = bankroll * kelly_frac * time_decay_factor * 1.5
        size_usd = min(size_usd, bankroll * (settings.max_position_size_pct / 100))

        # Reduce for volatility (even near expiry, volatility is risky)
        if volatility > 0.015:
            size_usd *= max(0.4, 1.0 - (volatility - 0.015) * 20)

        size_usd = max(0.0, round(size_usd, 2))

        return TradeDecision(
            should_trade=size_usd > 0,
            side=side,
            size_usd=size_usd,
            confidence=min(1.0, edge / 0.15),
            edge=edge,
            strategy_name=self.name,
            reason=f"TTE={tte_seconds}s, decay_factor={time_decay_factor:.2f}, conf={confidence:.3f}",
        )

    def _kelly_fraction(self, p: float, market_p: float) -> float:
        if market_p <= 0 or market_p >= 1:
            return 0.0
        b = (1 - market_p) / market_p
        if b <= 0:
            return 0.0
        full_kelly = (p * (b + 1) - 1) / b
        return max(0.0, min(full_kelly * self.kelly_cap, settings.max_position_size_pct / 100))


# ── Strategy Registry ──────────────────────────────────────────────────

ALL_STRATEGIES = {
    "kelly": lambda: __import__("strategies.base_strategies", fromlist=["KellyStrategy"]).KellyStrategy(),
    "fixed_fractional": lambda: __import__("strategies.base_strategies", fromlist=["FixedFractionalStrategy"]).FixedFractionalStrategy(),
    "target_profit": lambda: __import__("strategies.base_strategies", fromlist=["TargetProfitStrategy"]).TargetProfitStrategy(),
    "momentum_convergence": lambda: __import__("strategies.advanced_strategies", fromlist=["MomentumConvergenceStrategy"]).MomentumConvergenceStrategy(),
    "mean_reversion": lambda: __import__("strategies.advanced_strategies", fromlist=["MeanReversionStrategy"]).MeanReversionStrategy(),
    "time_decay_harvesting": lambda: __import__("strategies.advanced_strategies", fromlist=["TimeDecayHarvestingStrategy"]).TimeDecayHarvestingStrategy(),
}


def get_strategy(name: str) -> BaseStrategy:
    """Get a strategy instance by name."""
    factory = ALL_STRATEGIES.get(name)
    if factory is None:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(ALL_STRATEGIES.keys())}")
    return factory()


def get_all_strategies() -> Dict[str, BaseStrategy]:
    """Get instances of all strategies."""
    return {name: factory() for name, factory in ALL_STRATEGIES.items()}
