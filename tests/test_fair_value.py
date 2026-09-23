"""Tests for the up/down fair-value and fee model (strategies/fair_value.py)."""
import pytest

from strategies.fair_value import (
    all_in_buy_cost,
    binary_kelly_fraction,
    net_sell_proceeds,
    prob_up,
    taker_fee_per_share,
)

S = 100_000.0
SIGMA = 0.0001  # per-second log vol (~1.7%/day-ish scale for 5m windows)


def test_taker_fee_matches_polymarket_formula():
    # rate 0.07, exponent 1: 0.07 * 0.5 * 0.5 = 1.75c per share at 50c
    assert taker_fee_per_share(0.5, 0.07, 1.0) == pytest.approx(0.0175)
    assert taker_fee_per_share(0.9, 0.07, 1.0) == pytest.approx(0.07 * 0.09)
    assert taker_fee_per_share(0.5, 0.0) == 0.0
    assert taker_fee_per_share(1.0, 0.07) == 0.0


def test_buy_cost_and_sell_proceeds_are_symmetric_around_price():
    assert all_in_buy_cost(0.5, 0.07) == pytest.approx(0.5175)
    assert net_sell_proceeds(0.5, 0.07) == pytest.approx(0.4825)


def test_at_the_money_is_a_coin_flip():
    assert prob_up(S, S, SIGMA, 240) == pytest.approx(0.5)


def test_price_above_strike_favors_up_and_is_monotonic_in_distance():
    p1 = prob_up(S * 1.0005, S, SIGMA, 240)
    p2 = prob_up(S * 1.001, S, SIGMA, 240)
    assert 0.5 < p1 < p2


def test_less_time_left_means_more_certainty():
    far = prob_up(S * 1.0005, S, SIGMA, 280)
    near = prob_up(S * 1.0005, S, SIGMA, 30, running_twap=S * 1.0005)
    assert near > far


def test_last_minute_uses_the_running_twap():
    # Spot just dipped below the strike, but most of the averaging window
    # already printed well above it -> still very likely Up.
    p = prob_up(S * 0.9999, S, SIGMA, 10, running_twap=S * 1.001)
    assert p > 0.9


def test_basis_noise_pulls_toward_one_half():
    tight = prob_up(S * 1.0003, S, SIGMA, 120, basis_bps=0.0)
    noisy = prob_up(S * 1.0003, S, SIGMA, 120, basis_bps=10.0)
    assert 0.5 < noisy < tight


def test_drift_tilt_moves_probability():
    assert prob_up(S, S, SIGMA, 200, drift_z=0.5) > 0.5 > prob_up(S, S, SIGMA, 200, drift_z=-0.5)


def test_degenerate_inputs_do_not_crash():
    assert prob_up(0.0, S, SIGMA, 100) == 0.5
    assert prob_up(S, 0.0, SIGMA, 100) == 0.5
    assert prob_up(S * 1.01, S, 0.0, 100) == pytest.approx(0.99)
    assert prob_up(S * 0.99, S, 0.0, 100) == pytest.approx(0.01)


def test_binary_kelly():
    assert binary_kelly_fraction(0.6, 0.5) == pytest.approx(0.2)
    assert binary_kelly_fraction(0.5, 0.5) == 0.0
    assert binary_kelly_fraction(0.4, 0.5) == 0.0
