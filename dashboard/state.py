"""
In-process state holder for dashboard and main.py.
Tracks compounding metrics, BRTI price, arbitrage P&L,
TTE model status, and withdrawal history.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from risk.risk_manager import Position, RiskManager, TradeResult, WithdrawalRecord


@dataclass
class DashboardState:
    # Core financials
    bankroll: float = 0.0
    principal: float = 0.0
    profit: float = 0.0
    total_withdrawn: float = 0.0
    peak_bankroll: float = 0.0

    # Compounding metrics
    profit_pct: float = 0.0
    compound_growth_rate: float = 0.0
    drawdown_pct: float = 0.0
    all_time_high_pnl: float = 0.0

    # Trade stats
    total_trades: int = 0
    total_wins: int = 0
    total_losses: int = 0
    win_rate: float = 0.0
    consecutive_losses: int = 0

    # Status
    halted: bool = False
    halt_reason: Optional[str] = None
    mode: str = "paper"
    training_mode: bool = True

    # Positions & history
    open_positions: List[Position] = field(default_factory=list)
    recent_trades: List[TradeResult] = field(default_factory=list)
    withdrawal_history: List[WithdrawalRecord] = field(default_factory=list)

    # BRTI
    brti_price: float = 0.0
    brti_spread_bps: float = 0.0
    brti_exchanges_used: int = 0
    brti_tick_count: int = 0

    # Arbitrage
    arb_pnl: float = 0.0
    arb_open_positions: int = 0
    arb_closed_positions: int = 0
    arb_win_rate: float = 0.0
    arb_opportunities: int = 0

    # TTE / ML
    tte_models_fitted: int = 0
    tte_total_models: int = 900
    last_retrain: Optional[str] = None

    # Strategy
    active_strategy: str = "kelly"

    # Milestone tracking
    last_milestone_pct: float = 0.0
    milestone_alerts: List[str] = field(default_factory=list)

    # Paper trading (training mode)
    paper_balance: float = 0.0
    paper_starting_balance: float = 0.0
    paper_positions_value: float = 0.0
    paper_total_value: float = 0.0
    paper_pnl: float = 0.0
    paper_positions: Dict = field(default_factory=dict)
    paper_total_trades: int = 0
    paper_total_fees: float = 0.0
    paper_backend: str = "pm_trader"
    paper_analytics: Dict = field(default_factory=dict)


_state = DashboardState()


def update_state(
    risk_manager: RiskManager,
    brti_engine=None,
    arb_engine=None,
    tte_orchestrator=None,
) -> None:
    global _state
    from config.settings import settings

    brti_price = 0.0
    brti_spread = 0.0
    brti_exchanges = 0
    brti_ticks = 0
    if brti_engine and brti_engine.last_tick:
        brti_price = brti_engine.last_tick.brti_price
        brti_spread = brti_engine.last_tick.spread_bps
        brti_exchanges = len(brti_engine.last_tick.exchanges_used)
        brti_ticks = brti_engine._tick_count

    arb_pnl = 0.0
    arb_open = 0
    arb_closed = 0
    arb_wr = 0.0
    arb_opps = 0
    if arb_engine:
        arb_stats = arb_engine.stats
        arb_pnl = arb_stats.get("total_pnl", 0)
        arb_open = arb_stats.get("open_positions", 0)
        arb_closed = arb_stats.get("positions_closed", 0)
        arb_wr = arb_stats.get("win_rate", 0)
        arb_opps = arb_stats.get("total_opportunities", 0)

    tte_fitted = 0
    tte_total = 900
    last_retrain = None
    if tte_orchestrator:
        tte_stats = tte_orchestrator.stats
        tte_fitted = tte_stats.get("fitted_tte_models", 0)
        tte_total = tte_stats.get("total_tte_models", 900)
        last_retrain = tte_stats.get("last_full_retrain")

    _state = DashboardState(
        bankroll=risk_manager.bankroll,
        principal=risk_manager.principal,
        profit=risk_manager.total_profit,
        total_withdrawn=risk_manager.total_withdrawn,
        peak_bankroll=risk_manager.peak_bankroll,
        profit_pct=risk_manager.profit_pct,
        compound_growth_rate=risk_manager.compound_growth_rate,
        drawdown_pct=risk_manager.drawdown_pct,
        all_time_high_pnl=risk_manager.all_time_high_pnl,
        total_trades=len(risk_manager.trade_history),
        total_wins=risk_manager.total_wins,
        total_losses=risk_manager.total_losses,
        win_rate=risk_manager.win_rate,
        consecutive_losses=risk_manager.consecutive_losses,
        halted=risk_manager.halted,
        halt_reason=risk_manager.halt_reason,
        mode=settings.trading_mode.upper(),
        training_mode=settings.training_mode,
        open_positions=list(risk_manager.open_positions.values()),
        recent_trades=risk_manager.trade_history[-20:],
        withdrawal_history=risk_manager.withdrawal_history[-10:],
        brti_price=brti_price,
        brti_spread_bps=brti_spread,
        brti_exchanges_used=brti_exchanges,
        brti_tick_count=brti_ticks,
        arb_pnl=arb_pnl,
        arb_open_positions=arb_open,
        arb_closed_positions=arb_closed,
        arb_win_rate=arb_wr,
        arb_opportunities=arb_opps,
        tte_models_fitted=tte_fitted,
        tte_total_models=tte_total,
        last_retrain=last_retrain,
        active_strategy=settings.strategy,
    )


def get_state() -> DashboardState:
    return _state


def update_paper_state(paper_broker) -> None:
    """Update paper trading fields in the dashboard state."""
    global _state
    if paper_broker is None:
        return
    try:
        summary = paper_broker.summary()
        analytics = {}
        if hasattr(paper_broker, "get_analytics"):
            analytics = paper_broker.get_analytics()
        _state.paper_balance = summary.get("balance", 0.0)
        _state.paper_starting_balance = summary.get("starting_balance", 0.0)
        _state.paper_positions_value = summary.get("positions_value", 0.0)
        _state.paper_total_value = summary.get("total_value", 0.0)
        _state.paper_pnl = summary.get("pnl", 0.0)
        _state.paper_positions = summary.get("open_positions", {})
        _state.paper_total_trades = summary.get("total_trades", 0)
        _state.paper_total_fees = summary.get("total_fees", 0.0)
        _state.paper_backend = getattr(paper_broker, "__class__", type).__name__
        _state.paper_analytics = analytics
    except Exception:
        pass
