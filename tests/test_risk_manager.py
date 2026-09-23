"""
Sanity tests for risk.risk_manager.RiskManager — the hard-constraints layer
that's supposed to catch a wrong model before it blows up the account.
"""
import pytest

from risk.risk_manager import CircuitBreakerTripped, RiskManager


def test_kelly_fraction_zero_when_no_edge():
    rm = RiskManager(bankroll_usd=1000.0)
    assert rm.kelly_fraction(model_prob=0.5, market_price=0.6) == 0.0


def test_kelly_fraction_positive_with_edge():
    rm = RiskManager(bankroll_usd=1000.0)
    f = rm.kelly_fraction(model_prob=0.7, market_price=0.5)
    assert f > 0.0


def test_circuit_breaker_trips_on_consecutive_losses():
    rm = RiskManager(bankroll_usd=1000.0)
    rm.open_position("m1", "crypto", "YES", entry_price=0.5, size_usd=10)
    rm.close_position("m1", exit_price=0.0)  # loss
    rm.open_position("m2", "crypto", "YES", entry_price=0.5, size_usd=10)
    rm.close_position("m2", exit_price=0.0)  # loss

    with pytest.raises(CircuitBreakerTripped):
        rm.open_position("m3", "crypto", "YES", entry_price=0.5, size_usd=10)
        rm.close_position("m3", exit_price=0.0)  # 3rd consecutive loss -> halt

    assert rm.halted is True


def test_max_position_size_respects_category_cap():
    rm = RiskManager(bankroll_usd=1000.0)
    size = rm.max_position_size("crypto")
    assert 0 < size <= 1000.0 * 0.4  # max_exposure_per_category_pct default 40%


def test_halted_account_has_zero_position_size():
    rm = RiskManager(bankroll_usd=1000.0)
    rm.halted = True
    assert rm.max_position_size("crypto") == 0.0


def test_can_withdraw_respects_available_profit():
    rm = RiskManager(bankroll_usd=1000.0)
    rm.principal = 1000.0
    rm.bankroll = 1200.0  # $200 profit
    assert rm.can_withdraw(100.0) is True
    assert rm.can_withdraw(500.0) is False
