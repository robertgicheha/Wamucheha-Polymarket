"""
Risk management. This module is deliberately NOT "learned" -- these are hard
constraints. The signal/ML layers can be wrong; this layer's job is to make sure
being wrong doesn't blow up the account.

COMPOUNDING MODE:
  - All profits are reinvested automatically as the bankroll grows
  - Position sizes scale with total bankroll (compound growth)
  - The principal is a tracking reference, NOT a lock on capital
  - You withdraw manually via Telegram command when you choose
  - The bot tracks your growth rate, milestones, and profit history
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from config.settings import settings


@dataclass
class Position:
    market_id: str
    category: str
    side: str  # "YES" or "NO"
    entry_price: float
    size_usd: float
    opened_at: datetime
    stop_loss_price: float


@dataclass
class TradeResult:
    market_id: str
    category: str
    pnl_usd: float
    closed_at: datetime


@dataclass
class WithdrawalRecord:
    amount_usd: float
    destination: str
    timestamp: datetime
    status: str  # "pending", "completed", "failed"


class CircuitBreakerTripped(Exception):
    pass


class RiskManager:
    def __init__(self, bankroll_usd: float):
        self.starting_bankroll = bankroll_usd
        self.bankroll = bankroll_usd
        self.principal = settings.initial_stake_usd
        self.day_start_bankroll = bankroll_usd
        self.day_start_time = datetime.utcnow()
        self.open_positions: Dict[str, Position] = {}
        self.trade_history: List[TradeResult] = []
        self.withdrawal_history: List[WithdrawalRecord] = []
        self.consecutive_losses = 0
        self.halted = False
        self.halt_reason: Optional[str] = None
        self.total_withdrawn: float = 0.0
        self.peak_bankroll: float = bankroll_usd
        self.all_time_high_pnl: float = 0.0

    # ---- Position sizing ----

    def max_position_size(self, category: str) -> float:
        """
        Fractional-Kelly capped position size.
        In compounding mode, uses the FULL bankroll — profits compound into
        larger positions as the account grows.
        """
        if self.halted:
            return 0.0

        # Use full bankroll for compounding (not just profit)
        tradable = self._tradable_capital()
        base_cap = tradable * (settings.max_position_size_pct / 100)

        category_exposure = sum(
            p.size_usd for p in self.open_positions.values() if p.category == category
        )
        category_cap = tradable * (settings.max_exposure_per_category_pct / 100)
        remaining_category_room = max(0.0, category_cap - category_exposure)

        return max(0.0, min(base_cap, remaining_category_room))

    def _tradable_capital(self) -> float:
        """
        In compounding mode (reinvest_profits_only=false), the entire bankroll
        is tradable — profits compound into larger positions.
        In profit-only mode, only profit above principal is tradable.
        """
        if settings.reinvest_profits_only:
            profit = max(0.0, self.bankroll - self.principal)
            return profit
        return self.bankroll

    def kelly_fraction(self, model_prob: float, market_price: float, kelly_cap: float = 0.5) -> float:
        """
        Fractional Kelly sizing as a proportion of bankroll.
        kelly_cap keeps us at a fraction of full Kelly since full Kelly is too
        aggressive for a model with estimation error in its probabilities.
        """
        if market_price <= 0 or market_price >= 1:
            return 0.0
        edge = model_prob - market_price
        if edge <= 0:
            return 0.0
        b = (1 - market_price) / market_price  # payout odds
        full_kelly = (model_prob * (b + 1) - 1) / b
        return max(0.0, min(full_kelly * kelly_cap, settings.max_position_size_pct / 100))

    # ---- Compounding tracking ----

    @property
    def total_profit(self) -> float:
        """Total profit from trading (bankroll - initial principal)."""
        return max(0.0, self.bankroll - self.principal)

    @property
    def profit_pct(self) -> float:
        """Profit as a percentage of initial principal."""
        if self.principal <= 0:
            return 0.0
        return (self.bankroll - self.principal) / self.principal * 100

    @property
    def compound_growth_rate(self) -> float:
        """Annualized compound growth rate estimate (based on time running)."""
        elapsed = (datetime.utcnow() - self.day_start_time).total_seconds()
        if elapsed <= 0 or self.starting_bankroll <= 0:
            return 0.0
        days = elapsed / 86400
        if days < 0.01:
            return 0.0
        growth = self.bankroll / self.starting_bankroll
        if growth <= 0:
            return 0.0
        # Annualize
        annualized = (growth ** (365.25 / days)) - 1
        return annualized * 100

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown from peak bankroll."""
        if self.peak_bankroll <= 0:
            return 0.0
        return (self.peak_bankroll - self.bankroll) / self.peak_bankroll * 100

    @property
    def total_wins(self) -> int:
        return sum(1 for t in self.trade_history if t.pnl_usd >= 0)

    @property
    def total_losses(self) -> int:
        return sum(1 for t in self.trade_history if t.pnl_usd < 0)

    @property
    def win_rate(self) -> float:
        total = len(self.trade_history)
        if total == 0:
            return 0.0
        return self.total_wins / total * 100

    def get_compounding_summary(self) -> Dict:
        """Full compounding status report."""
        return {
            "principal": self.principal,
            "bankroll": self.bankroll,
            "total_profit": self.total_profit,
            "profit_pct": self.profit_pct,
            "compound_growth_rate_annualized": self.compound_growth_rate,
            "peak_bankroll": self.peak_bankroll,
            "drawdown_pct": self.drawdown_pct,
            "total_withdrawn": self.total_withdrawn,
            "total_trades": len(self.trade_history),
            "total_wins": self.total_wins,
            "total_losses": self.total_losses,
            "win_rate": self.win_rate,
            "open_positions_count": len(self.open_positions),
            "consecutive_losses": self.consecutive_losses,
            "halted": self.halted,
        }

    # ---- Circuit breaker ----

    def check_circuit_breaker(self) -> None:
        if datetime.utcnow() - self.day_start_time > timedelta(days=1):
            self.day_start_bankroll = self.bankroll
            self.day_start_time = datetime.utcnow()

        # Update peak bankroll
        if self.bankroll > self.peak_bankroll:
            self.peak_bankroll = self.bankroll

        daily_drawdown_pct = (
            (self.day_start_bankroll - self.bankroll) / self.day_start_bankroll * 100
            if self.day_start_bankroll > 0
            else 0
        )
        if daily_drawdown_pct >= settings.circuit_breaker_daily_drawdown_pct:
            self._halt(
                f"Daily drawdown {daily_drawdown_pct:.1f}% >= "
                f"{settings.circuit_breaker_daily_drawdown_pct}% limit"
            )
        if self.consecutive_losses >= settings.circuit_breaker_max_consecutive_losses:
            self._halt(
                f"{self.consecutive_losses} consecutive losses >= "
                f"{settings.circuit_breaker_max_consecutive_losses} limit"
            )

    def _halt(self, reason: str) -> None:
        if not self.halted:
            self.halted = True
            self.halt_reason = reason
            raise CircuitBreakerTripped(reason)

    def can_open_new_position(self) -> bool:
        return not self.halted and len(self.open_positions) < settings.max_concurrent_positions

    # ---- Stop-loss ----

    def open_position(self, market_id: str, category: str, side: str, entry_price: float, size_usd: float) -> Position:
        stop_loss_price = (
            entry_price * (1 - settings.stop_loss_pct / 100)
            if side == "YES"
            else entry_price * (1 + settings.stop_loss_pct / 100)
        )
        pos = Position(
            market_id=market_id,
            category=category,
            side=side,
            entry_price=entry_price,
            size_usd=size_usd,
            opened_at=datetime.utcnow(),
            stop_loss_price=stop_loss_price,
        )
        self.open_positions[market_id] = pos
        return pos

    def check_stop_losses(self, current_prices: Dict[str, float]) -> List[str]:
        """Returns market_ids that should be closed due to stop-loss trigger."""
        to_close = []
        for market_id, pos in self.open_positions.items():
            price = current_prices.get(market_id)
            if price is None:
                continue
            if pos.side == "YES" and price <= pos.stop_loss_price:
                to_close.append(market_id)
            elif pos.side == "NO" and price >= pos.stop_loss_price:
                to_close.append(market_id)
        return to_close

    def close_position(self, market_id: str, exit_price: float) -> TradeResult:
        pos = self.open_positions.pop(market_id)
        if pos.side == "YES":
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price
        pnl_usd = pos.size_usd * pnl_pct
        self.bankroll += pnl_usd

        # Update peak
        if self.bankroll > self.peak_bankroll:
            self.peak_bankroll = self.bankroll
        if self.bankroll - self.principal > self.all_time_high_pnl:
            self.all_time_high_pnl = self.bankroll - self.principal

        if pnl_usd < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        result = TradeResult(
            market_id=market_id, category=pos.category, pnl_usd=pnl_usd, closed_at=datetime.utcnow()
        )
        self.trade_history.append(result)
        self.check_circuit_breaker()
        return result

    # ---- Manual withdrawal ----

    def can_withdraw(self, amount_usd: float) -> bool:
        """Check if a withdrawal of this amount is possible without endangering open positions."""
        if not settings.withdrawal_enabled:
            return False
        if amount_usd <= 0:
            return False
        if amount_usd > self.total_profit:
            return False
        # Ensure enough remains to cover open positions
        margin = self.bankroll - amount_usd
        total_exposure = sum(p.size_usd for p in self.open_positions.values())
        return margin >= total_exposure

    def request_withdrawal(self, amount_usd: float, destination: str = "") -> WithdrawalRecord:
        """
        Process a manual withdrawal request. Deducts from bankroll.
        Returns a WithdrawalRecord — the actual on-chain transfer happens
        via the polygon_connector separately.
        """
        if not self.can_withdraw(amount_usd):
            raise ValueError(
                f"Cannot withdraw ${amount_usd:.2f}: "
                f"available profit=${self.total_profit:.2f}, "
                f"enabled={settings.withdrawal_enabled}"
            )

        dest = destination or settings.withdrawal_destination_address
        record = WithdrawalRecord(
            amount_usd=amount_usd,
            destination=dest,
            timestamp=datetime.utcnow(),
            status="pending",
        )
        self.withdrawal_history.append(record)
        self.bankroll -= amount_usd
        self.total_withdrawn += amount_usd
        return record

    def complete_withdrawal(self, index: int) -> None:
        """Mark a withdrawal as completed after on-chain confirmation."""
        if 0 <= index < len(self.withdrawal_history):
            self.withdrawal_history[index].status = "completed"

    def fail_withdrawal(self, index: int) -> None:
        """Mark a withdrawal as failed and refund the bankroll."""
        if 0 <= index < len(self.withdrawal_history):
            record = self.withdrawal_history[index]
            if record.status == "pending":
                record.status = "failed"
                self.bankroll += record.amount_usd
                self.total_withdrawn -= record.amount_usd
