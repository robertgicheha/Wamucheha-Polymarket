"""
Tests for the real-prediction wiring in strategies.lifecycle_engine:
  - _tte_prob_to_yes_probability: converts the TTE model's self-relative
    directional read into P(resolve YES) given distance to price_to_beat.
  - _compute_btc_signal: real momentum/z-score/volatility from BRTI history,
    with graceful fallback when ML is disabled/untrained/unavailable.
  - _get_strategy_signal: falls back to the market-price heuristic instead
    of trading blind when no real signal is available.
  - the circuit-breaker gate added to run_market_tick.
"""
import time

import pytest

from config.settings import settings
from strategies.lifecycle_engine import FiveMinuteLifecycleEngine, MarketWindow


def _make_window(**overrides):
    now = time.time()
    defaults = dict(
        condition_id="0xabc",
        question="Will BTC be above $100000 at 12:05pm?",
        asset="btc",
        price_to_beat=100000.0,
        token_id_yes="yes-token",
        token_id_no="no-token",
        start_time=now - 60,
        end_time=now + 240,
    )
    defaults.update(overrides)
    return MarketWindow(**defaults)


# ── _tte_prob_to_yes_probability ────────────────────────────────────────

def test_tte_prob_neutral_at_the_money_with_neutral_signal():
    p = FiveMinuteLifecycleEngine._tte_prob_to_yes_probability(
        tte_prob=0.5, current_price=100000.0, price_to_beat=100000.0,
        vol_per_second=0.0001, tte_seconds=60,
    )
    assert abs(p - 0.5) < 1e-6


def test_tte_prob_favors_yes_when_price_already_above_strike():
    p = FiveMinuteLifecycleEngine._tte_prob_to_yes_probability(
        tte_prob=0.5, current_price=101000.0, price_to_beat=100000.0,
        vol_per_second=0.0005, tte_seconds=60,
    )
    assert p > 0.5


def test_tte_prob_favors_no_when_price_already_below_strike():
    p = FiveMinuteLifecycleEngine._tte_prob_to_yes_probability(
        tte_prob=0.5, current_price=99000.0, price_to_beat=100000.0,
        vol_per_second=0.0005, tte_seconds=60,
    )
    assert p < 0.5


def test_tte_prob_bullish_model_signal_pushes_probability_up():
    kwargs = dict(current_price=100000.0, price_to_beat=100000.0,
                  vol_per_second=0.0005, tte_seconds=60)
    neutral = FiveMinuteLifecycleEngine._tte_prob_to_yes_probability(tte_prob=0.5, **kwargs)
    bullish = FiveMinuteLifecycleEngine._tte_prob_to_yes_probability(tte_prob=0.9, **kwargs)
    bearish = FiveMinuteLifecycleEngine._tte_prob_to_yes_probability(tte_prob=0.1, **kwargs)
    assert bullish > neutral > bearish


def test_tte_prob_handles_degenerate_inputs_without_crashing():
    assert FiveMinuteLifecycleEngine._tte_prob_to_yes_probability(
        tte_prob=0.5, current_price=0.0, price_to_beat=100000.0,
        vol_per_second=0.001, tte_seconds=60,
    ) == 0.5
    assert FiveMinuteLifecycleEngine._tte_prob_to_yes_probability(
        tte_prob=0.5, current_price=100000.0, price_to_beat=0.0,
        vol_per_second=0.001, tte_seconds=60,
    ) == 0.5
    # zero volatility must not raise a division error
    p = FiveMinuteLifecycleEngine._tte_prob_to_yes_probability(
        tte_prob=0.9, current_price=100000.0, price_to_beat=100000.0,
        vol_per_second=0.0, tte_seconds=60,
    )
    assert 0.0 <= p <= 1.0


def test_tte_prob_output_always_clamped():
    p = FiveMinuteLifecycleEngine._tte_prob_to_yes_probability(
        tte_prob=1.0, current_price=200000.0, price_to_beat=1.0,
        vol_per_second=0.0001, tte_seconds=900,
    )
    assert 0.01 <= p <= 0.99


# ── _compute_btc_signal fallback behavior ───────────────────────────────

def test_compute_btc_signal_none_without_brti_engine():
    engine = FiveMinuteLifecycleEngine(bankroll=100.0)
    window = _make_window()
    assert engine._compute_btc_signal(window) is None


def test_compute_btc_signal_none_without_ticks():
    class EmptyBRTI:
        last_tick = None
        tick_history = []

    engine = FiveMinuteLifecycleEngine(bankroll=100.0, brti_engine=EmptyBRTI())
    assert engine._compute_btc_signal(_make_window()) is None


def test_compute_btc_signal_falls_back_when_ml_disabled(monkeypatch):
    monkeypatch.setattr(settings, "ml_prediction_enabled", False)

    class FakeTick:
        brti_price = 100500.0

    class FakeBRTI:
        last_tick = FakeTick()

        def get_price_history(self, seconds):
            base = 100000.0
            return [(time.time() + i, base + i * 10) for i in range(30)]

        def get_volatility(self, seconds):
            return 0.0007

    engine = FiveMinuteLifecycleEngine(
        bankroll=100.0, brti_engine=FakeBRTI(), tte_orchestrator=object(),
    )
    signal = engine._compute_btc_signal(_make_window())
    assert signal is not None
    assert signal["model_prob"] is None  # ML disabled -> caller uses fallback heuristic
    assert signal["momentum"] > 0  # rising synthetic price series


# ── _get_strategy_signal end-to-end fallback ────────────────────────────

def test_get_strategy_signal_uses_heuristic_without_brti_engine():
    engine = FiveMinuteLifecycleEngine(bankroll=1000.0, strategy_name="kelly")
    engine._strategy = engine._create_strategy("kelly")
    window = _make_window(price_to_beat=100000.0)
    window.orderbook_snapshot = {"mid_price": 0.7, "imbalance": 0.1, "spread_bps": 20}
    window.current_yes_price = 0.7
    window.current_no_price = 0.3

    signal = engine._get_strategy_signal(window)
    assert signal is not None
    # No brti_engine -> falls back to heuristic; should not raise and should
    # return a well-formed dict either way (should_trade True or False).
    assert "should_trade" in signal


# ── circuit breaker gate ─────────────────────────────────────────────────

def test_run_market_tick_skips_new_signals_when_risk_manager_halted():
    class HaltedRiskManager:
        halted = True

    engine = FiveMinuteLifecycleEngine(bankroll=1000.0, risk_manager=HaltedRiskManager())
    engine._strategy = engine._create_strategy("kelly")
    window = _make_window()
    engine._market_windows[window.condition_id] = window
    engine.start_market(window.condition_id)

    orderbook = {"mid_price": 0.9, "imbalance": 0.0, "spread_bps": 10}
    engine.run_market_tick(window.condition_id, orderbook=orderbook)

    # Halted -> no new strategy signal should have been computed/stored.
    assert window.strategy_signal is None
    assert window.position_side == ""
