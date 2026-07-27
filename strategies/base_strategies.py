"""
Base strategy class and core strategies:
  1. Kelly Criterion — optimal sizing from edge and odds
  2. Fixed Fractional — constant % of bankroll per trade
  3. Target Profit — fixed $ profit target per trade

All strategies share the same interface:
  - should_trade(signal, market_price, bankroll) -> TradeDecision
  - size_position(edge, market_price, bankroll) -> float (USD)
"""
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

from config.settings import settings


@dataclass
class TradeDecision:
    """Output of a strategy's decision function."""
    should_trade: bool
    side: str  # "YES" or "NO"
    size_usd: float
    confidence: float
    edge: float
    strategy_name: str
    reason: str

    def to_dict(self) -> Dict:
        return {
            "should_trade": self.should_trade,
            "side": self.side,
            "size_usd": self.size_usd,
            "confidence": self.confidence,
            "edge": self.edge,
            "strategy_name": self.strategy_name,
            "reason": self.reason,
        }


class BaseStrategy(ABC):
    """Abstract base for all trading strategies."""

    name: str = "base"
    description: str = "Base strategy"

    def __init__(
        self,
        min_edge: float = 0.05,
        min_price: float = 0.08,
        max_price: float = 0.85,
        kelly_cap: float = 0.25,
    ):
        self.min_edge = min_edge or settings.min_edge_threshold
        self.min_price = min_price or settings.min_price_threshold
        self.max_price = max_price or settings.max_price_threshold
        self.kelly_cap = kelly_cap or settings.ml_kelly_fraction

    @abstractmethod
    def should_trade(
        self,
        model_prob: float,
        market_price: float,
        bankroll: float,
        tte_seconds: int = 0,
        volatility: float = 0.0,
        extra: Optional[Dict] = None,
    ) -> TradeDecision:
        raise NotImplementedError

    def _determine_side(self, model_prob: float, market_price: float) -> str:
        """YES if model thinks market is underpriced, NO if overpriced."""
        if model_prob > market_price:
            return "YES"
        elif model_prob < (1 - market_price):
            return "NO"
        return "YES"

    def _validate_price(self, market_price: float) -> bool:
        """Reject trades outside the empirical calibration zone."""
        return self.min_price <= market_price <= self.max_price

    def _calculate_edge(self, model_prob: float, market_price: float, side: str) -> float:
        """Calculate edge based on side."""
        if side == "YES":
            return model_prob - market_price
        else:
            return (1 - model_prob) - (1 - market_price)

    def _no_trade(self, reason: str, side: str = "YES") -> TradeDecision:
        return TradeDecision(
            should_trade=False, side=side, size_usd=0.0,
            confidence=0.0, edge=0.0, strategy_name=self.name, reason=reason,
        )


class KellyStrategy(BaseStrategy):
    """
    Kelly Criterion — optimal fraction of bankroll to bet.

    Full Kelly: f* = (bp - q) / b
      where b = odds, p = model_prob, q = 1 - p

    We use fractional Kelly (default 25%) to account for
    model estimation error and reduce variance.
    """
    name = "kelly"
    description = "Fractional Kelly criterion — optimal sizing from edge and odds"

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
            return self._no_trade(
                f"Price {market_price:.3f} outside zone [{self.min_price}, {self.max_price}]"
            )

        side = self._determine_side(model_prob, market_price)
        edge = self._calculate_edge(model_prob, market_price, side)

        if edge < self.min_edge:
            return self._no_trade(
                f"Edge {edge:.4f} below min {self.min_edge}", side
            )

        # Kelly fraction
        if side == "YES":
            p = model_prob
            b = (1 - market_price) / market_price
        else:
            p = 1 - model_prob
            b = market_price / (1 - market_price)

        if b <= 0:
            return self._no_trade("Invalid odds", side)

        full_kelly = (p * (b + 1) - 1) / b
        if full_kelly <= 0:
            return self._no_trade("Negative Kelly fraction", side)

        kelly_fraction = min(full_kelly * self.kelly_cap, settings.max_position_size_pct / 100)
        size_usd = bankroll * kelly_fraction

        # Adjust for high volatility
        if volatility > 0.02:
            size_usd *= max(0.5, 1.0 - (volatility - 0.02) * 10)

        # Adjust for short TTE (less time for convergence)
        if 0 < tte_seconds < 60:
            size_usd *= tte_seconds / 60

        size_usd = max(0.0, round(size_usd, 2))

        return TradeDecision(
            should_trade=size_usd > 0,
            side=side,
            size_usd=size_usd,
            confidence=min(1.0, edge / 0.10),
            edge=edge,
            strategy_name=self.name,
            reason=f"Kelly f={kelly_fraction:.4f}, edge={edge:.4f}, odds={b:.2f}",
        )


class FixedFractionalStrategy(BaseStrategy):
    """
    Fixed Fractional — risk a constant % of bankroll per trade.
    Simple, robust, no estimation of odds needed.
    """
    name = "fixed_fractional"
    description = "Fixed % of bankroll per trade"

    def __init__(self, risk_pct: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.risk_pct = risk_pct

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

        side = self._determine_side(model_prob, market_price)
        edge = self._calculate_edge(model_prob, market_price, side)

        if edge < self.min_edge:
            return self._no_trade(f"Edge {edge:.4f} below min {self.min_edge}", side)

        size_usd = bankroll * (self.risk_pct / 100)

        # Scale down for high volatility
        if volatility > 0.02:
            size_usd *= max(0.5, 1.0 - (volatility - 0.02) * 10)

        size_usd = max(0.0, round(size_usd, 2))

        return TradeDecision(
            should_trade=size_usd > 0,
            side=side,
            size_usd=size_usd,
            confidence=min(1.0, edge / 0.10),
            edge=edge,
            strategy_name=self.name,
            reason=f"Fixed {self.risk_pct}% = ${size_usd:.2f}, edge={edge:.4f}",
        )


class TargetProfitStrategy(BaseStrategy):
    """
    Target Profit — size positions to achieve a fixed $ profit target.
    Uses implied odds to calculate position size.

    target_profit = size * (payout - 1)  for YES bets
    size = target_profit / (payout - 1)
    """
    name = "target_profit"
    description = "Size to achieve a fixed $ profit target per trade"

    def __init__(self, target_profit_usd: float = 5.0, **kwargs):
        super().__init__(**kwargs)
        self.target_profit_usd = target_profit_usd

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

        side = self._determine_side(model_prob, market_price)
        edge = self._calculate_edge(model_prob, market_price, side)

        if edge < self.min_edge:
            return self._no_trade(f"Edge {edge:.4f} below min {self.min_edge}", side)

        # Calculate size from target profit
        if side == "YES":
            payout_per_dollar = (1 - market_price) / market_price
        else:
            payout_per_dollar = market_price / (1 - market_price)

        if payout_per_dollar <= 0:
            return self._no_trade("Invalid payout", side)

        size_usd = self.target_profit_usd / payout_per_dollar

        # Cap at max position size
        max_size = bankroll * (settings.max_position_size_pct / 100)
        size_usd = min(size_usd, max_size)
        size_usd = max(0.0, round(size_usd, 2))

        return TradeDecision(
            should_trade=size_usd > 0,
            side=side,
            size_usd=size_usd,
            confidence=min(1.0, edge / 0.10),
            edge=edge,
            strategy_name=self.name,
            reason=f"Target ${self.target_profit_usd:.2f}, size=${size_usd:.2f}",
        )
