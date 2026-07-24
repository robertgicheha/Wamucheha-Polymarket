"""
In-process state holder for dashboard and main.py. Tracks compounding metrics,
withdrawal history, and profit milestones. main.py calls update_state() each cycle.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List

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
    halt_reason: str | None = None
    mode: str = "paper"

    # Positions & history
    open_positions: List[Position] = field(default_factory=list)
    recent_trades: List[TradeResult] = field(default_factory=list)
    withdrawal_history: List[WithdrawalRecord] = field(default_factory=list)

    # Milestone tracking
    last_milestone_pct: float = 0.0
    milestone_alerts: List[str] = field(default_factory=list)


_state = DashboardState()


def update_state(risk_manager: RiskManager) -> None:
    global _state
    from config.settings import settings

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
        open_positions=list(risk_manager.open_positions.values()),
        recent_trades=risk_manager.trade_history[-20:],
        withdrawal_history=risk_manager.withdrawal_history[-10:],
    )


def get_state() -> DashboardState:
    return _state
