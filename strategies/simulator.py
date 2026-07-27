"""
Strategy Simulator — backtest all strategies and select by Sharpe/P&L.

Runs each strategy against historical data, computes performance metrics,
and selects the optimal strategy (or strategy blend) based on:
  - Sharpe ratio (risk-adjusted returns)
  - Total P&L
  - Max drawdown
  - Win rate
  - Profit factor

The simulator does NOT select by accuracy — a strategy can have low accuracy
but high Sharpe if it sizes correctly (wins big, loses small).
"""
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from strategies.base_strategies import BaseStrategy, TradeDecision

logger = logging.getLogger(__name__)


@dataclass
class SimulatedTrade:
    """One simulated trade."""
    timestamp: float
    market_id: str
    side: str
    entry_price: float
    exit_price: float
    size_usd: float
    pnl_usd: float
    edge: float
    strategy_name: str
    tte_seconds: int = 0


@dataclass
class StrategyPerformance:
    """Aggregated performance metrics for a strategy."""
    strategy_name: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl_usd: float = 0.0
    total_volume_usd: float = 0.0
    max_drawdown_usd: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    avg_win_usd: float = 0.0
    avg_loss_usd: float = 0.0
    avg_edge: float = 0.0
    avg_holding_seconds: float = 0.0
    expectancy: float = 0.0
    trades: List[SimulatedTrade] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "strategy_name": self.strategy_name,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl_usd": round(self.total_pnl_usd, 2),
            "total_volume_usd": round(self.total_volume_usd, 2),
            "max_drawdown_usd": round(self.max_drawdown_usd, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "profit_factor": round(self.profit_factor, 4),
            "win_rate": round(self.win_rate, 2),
            "avg_win_usd": round(self.avg_win_usd, 2),
            "avg_loss_usd": round(self.avg_loss_usd, 2),
            "avg_edge": round(self.avg_edge, 4),
            "expectancy": round(self.expectancy, 4),
        }


class StrategySimulator:
    """
    Backtest simulator that evaluates strategies against historical data.

    Usage:
      simulator = StrategySimulator(initial_bankroll=1000)
      for strategy in strategies:
          perf = simulator.backtest(strategy, historical_signals, market_data)
      best = simulator.select_best()
    """

    def __init__(
        self,
        initial_bankroll: float = 1000.0,
        risk_free_rate: float = 0.0,
        slippage_pct: float = 0.5,
        fee_pct: float = 0.0,
    ):
        self.initial_bankroll = initial_bankroll
        self.risk_free_rate = risk_free_rate
        self.slippage_pct = slippage_pct
        self.fee_pct = fee_pct
        self.results: Dict[str, StrategyPerformance] = {}

    def backtest(
        self,
        strategy: BaseStrategy,
        signals: List[Dict],
        market_data: Optional[List[Dict]] = None,
    ) -> StrategyPerformance:
        """
        Backtest a strategy against historical signals.

        Args:
            strategy: the strategy to test
            signals: list of dicts with keys:
                - model_prob: float (ML probability)
                - market_price: float (current market price)
                - timestamp: float
                - market_id: str
                - tte_seconds: int (optional)
                - volatility: float (optional)
                - momentum: float (optional)
                - z_score: float (optional)
            market_data: optional price history for exit simulation

        Returns:
            StrategyPerformance with all metrics
        """
        perf = StrategyPerformance(strategy_name=strategy.name)
        bankroll = self.initial_bankroll
        peak_bankroll = self.initial_bankroll
        equity_curve = [bankroll]
        trades = []

        for signal in signals:
            model_prob = signal.get("model_prob", 0.5)
            market_price = signal.get("market_price", 0.5)
            timestamp = signal.get("timestamp", 0.0)
            market_id = signal.get("market_id", "unknown")
            tte = signal.get("tte_seconds", 0)
            vol = signal.get("volatility", 0.0)

            # Get decision from strategy
            decision = strategy.should_trade(
                model_prob=model_prob,
                market_price=market_price,
                bankroll=bankroll,
                tte_seconds=tte,
                volatility=vol,
                extra=signal,
            )

            if not decision.should_trade or decision.size_usd <= 0:
                equity_curve.append(bankroll)
                continue

            # Simulate trade
            exit_price = self._simulate_exit(
                model_prob, market_price, decision.side, tte, signal
            )

            # Apply slippage
            if decision.side == "YES":
                entry_price = market_price * (1 + self.slippage_pct / 100)
            else:
                entry_price = market_price * (1 - self.slippage_pct / 100)
            entry_price = max(0.01, min(0.99, entry_price))

            # Calculate P&L
            if decision.side == "YES":
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price

            pnl_usd = decision.size_usd * pnl_pct
            fee_usd = decision.size_usd * (self.fee_pct / 100)
            pnl_usd -= fee_usd

            bankroll += pnl_usd
            if bankroll > peak_bankroll:
                peak_bankroll = bankroll

            trade = SimulatedTrade(
                timestamp=timestamp,
                market_id=market_id,
                side=decision.side,
                entry_price=entry_price,
                exit_price=exit_price,
                size_usd=decision.size_usd,
                pnl_usd=pnl_usd,
                edge=decision.edge,
                strategy_name=strategy.name,
                tte_seconds=tte,
            )
            trades.append(trade)
            equity_curve.append(bankroll)

        # Compute performance metrics
        perf.trades = trades
        perf.total_trades = len(trades)
        perf.winning_trades = sum(1 for t in trades if t.pnl_usd > 0)
        perf.losing_trades = sum(1 for t in trades if t.pnl_usd <= 0)
        perf.total_pnl_usd = bankroll - self.initial_bankroll
        perf.total_volume_usd = sum(t.size_usd for t in trades)

        if perf.total_trades > 0:
            perf.win_rate = perf.winning_trades / perf.total_trades * 100
            perf.avg_edge = np.mean([t.edge for t in trades])
            wins = [t.pnl_usd for t in trades if t.pnl_usd > 0]
            losses = [t.pnl_usd for t in trades if t.pnl_usd <= 0]
            perf.avg_win_usd = np.mean(wins) if wins else 0.0
            perf.avg_loss_usd = np.mean(losses) if losses else 0.0

            total_wins = sum(wins)
            total_losses = abs(sum(losses))
            perf.profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")
            perf.expectancy = perf.total_pnl_usd / perf.total_trades

        # Drawdown
        equity = np.array(equity_curve)
        peaks = np.maximum.accumulate(equity)
        drawdowns = (peaks - equity) / peaks * 100
        perf.max_drawdown_pct = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
        perf.max_drawdown_usd = float(np.max(peaks - equity)) if len(equity) > 0 else 0.0

        # Sharpe ratio (annualized, assuming ~5min intervals)
        if len(equity) > 1:
            returns = np.diff(equity) / equity[:-1]
            returns = returns[np.isfinite(returns)]
            if len(returns) > 1 and np.std(returns) > 0:
                # Annualize: assume 5min intervals, ~105,120 per year
                periods_per_year = 105120
                mean_return = np.mean(returns) - self.risk_free_rate / periods_per_year
                std_return = np.std(returns)
                perf.sharpe_ratio = float(mean_return / std_return * np.sqrt(periods_per_year))

                # Sortino (downside deviation only)
                downside = returns[returns < 0]
                if len(downside) > 0:
                    downside_std = np.std(downside)
                    if downside_std > 0:
                        perf.sortino_ratio = float(mean_return / downside_std * np.sqrt(periods_per_year))

        self.results[strategy.name] = perf
        logger.info(
            "Backtested %s: %d trades, P&L=$%.2f, Sharpe=%.3f, WinRate=%.1f%%",
            strategy.name, perf.total_trades, perf.total_pnl_usd,
            perf.sharpe_ratio, perf.win_rate,
        )

        return perf

    def _simulate_exit(
        self,
        model_prob: float,
        market_price: float,
        side: str,
        tte: int,
        signal: Dict,
    ) -> float:
        """
        Simulate exit price based on model probability and market dynamics.

        In reality, exit happens when:
        - Contract resolves (price → 0 or 1)
        - Stop-loss triggered
        - Take-profit triggered
        - Strategy-specific exit signal

        For backtesting, we use the model probability as the "true" probability
        and add some noise to simulate market impact.
        """
        # Base exit: model probability (what the model thinks the true value is)
        if side == "YES":
            base_exit = model_prob
        else:
            base_exit = 1 - model_prob

        # Add noise based on volatility
        vol = signal.get("volatility", 0.01)
        noise = np.random.normal(0, vol * 0.1)
        exit_price = base_exit + noise

        # Bound to [0.01, 0.99]
        exit_price = max(0.01, min(0.99, exit_price))

        return exit_price

    def select_best(self, metric: str = "sharpe_ratio") -> Optional[StrategyPerformance]:
        """Select the best strategy by a given metric."""
        if not self.results:
            return None

        if metric == "sharpe_ratio":
            return max(self.results.values(), key=lambda p: p.sharpe_ratio)
        elif metric == "total_pnl_usd":
            return max(self.results.values(), key=lambda p: p.total_pnl_usd)
        elif metric == "sortino_ratio":
            return max(self.results.values(), key=lambda p: p.sortino_ratio)
        elif metric == "profit_factor":
            return max(self.results.values(), key=lambda p: p.profit_factor)
        elif metric == "expectancy":
            return max(self.results.values(), key=lambda p: p.expectancy)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def select_top_n(self, n: int = 3, metric: str = "sharpe_ratio") -> List[StrategyPerformance]:
        """Select top N strategies by metric."""
        if not self.results:
            return []
        sorted_results = sorted(
            self.results.values(),
            key=lambda p: getattr(p, metric, 0),
            reverse=True,
        )
        return sorted_results[:n]

    def blend_strategies(
        self,
        strategies: List[BaseStrategy],
        signals: List[Dict],
        weights: Optional[Dict[str, float]] = None,
    ) -> StrategyPerformance:
        """
        Simulate a blended strategy that allocates across multiple strategies.

        If weights are None, uses equal weights. Otherwise, uses the provided
        weights (must sum to 1.0).
        """
        if weights is None:
            w = 1.0 / len(strategies)
            weights = {s.name: w for s in strategies}

        # Backtest each strategy individually
        for strategy in strategies:
            self.backtest(strategy, signals)

        # Blend the equity curves
        blended_perf = StrategyPerformance(strategy_name="blended")
        if not self.results:
            return blended_perf

        # Use the strategy with most trades as the base
        base_name = max(self.results.keys(), key=lambda k: self.results[k].total_trades)
        base_trades = self.results[base_name].trades

        # For each trade, weight the P&L by strategy weights
        total_pnl = 0.0
        for trade in base_trades:
            strategy_weight = weights.get(trade.strategy_name, 0.0)
            total_pnl += trade.pnl_usd * strategy_weight

        blended_perf.total_pnl_usd = total_pnl
        blended_perf.strategy_name = "blended"
        self.results["blended"] = blended_perf

        return blended_perf

    def get_summary(self) -> Dict:
        """Get a summary of all backtest results."""
        summary = {}
        for name, perf in self.results.items():
            summary[name] = perf.to_dict()
        return summary

    def rank_strategies(self, weights: Optional[Dict[str, float]] = None) -> List[Dict]:
        """
        Rank strategies by a weighted score of multiple metrics.

        Default weights emphasize Sharpe (40%), P&L (30%), and Sortino (30%).
        """
        if weights is None:
            weights = {
                "sharpe_ratio": 0.4,
                "total_pnl_usd": 0.3,
                "sortino_ratio": 0.3,
            }

        rankings = []
        for name, perf in self.results.items():
            score = 0.0
            for metric, weight in weights.items():
                value = getattr(perf, metric, 0)
                score += value * weight
            rankings.append({
                "strategy": name,
                "score": round(score, 4),
                "sharpe": round(perf.sharpe_ratio, 4),
                "pnl": round(perf.total_pnl_usd, 2),
                "trades": perf.total_trades,
                "win_rate": round(perf.win_rate, 1),
            })

        rankings.sort(key=lambda x: x["score"], reverse=True)
        return rankings
