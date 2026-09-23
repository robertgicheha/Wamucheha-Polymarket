"""
Fair value and cost model for Polymarket crypto "Up or Down" windows.

Resolution rule (from the market description): the window resolves "Up" if
the Chainlink 60-second TWAP at the END of the window is >= the TWAP at the
START of the window. So the strike ("price to beat") is the 60s average
ending at window start, and the settlement value is the 60s average ending
at window end.

We model the index as driftless Brownian motion in price with per-second
volatility sigma (from our BRTI tick history), which gives a closed form for
the distribution of the final 60s average:

  tau >= W:  the whole averaging window is in the future.
             mean = S, var = S^2 sigma^2 (tau - 2W/3)
  tau <  W:  part of the average is already fixed (running mean K over the
             elapsed W - tau seconds), the rest is the future average of
             length tau.
             mean = ((W - tau) K + tau S) / W
             var  = (tau / W)^2 * S^2 sigma^2 * tau / 3

A settlement-basis term (our exchange index vs Chainlink) is added to the
variance, applied to both the strike and the settlement value.

Fees: Polymarket crypto markets charge TAKER fees only, per share:
    fee = rate * (p * (1 - p)) ** exponent        (rate=0.07, exponent=1 today)
fetched live per market; makers pay nothing.
"""
import math
from typing import Optional

from scipy.stats import norm

TWAP_WINDOW_SECONDS = 60.0
MIN_PROB = 0.01
MAX_PROB = 0.99


def taker_fee_per_share(price: float, rate: float, exponent: float = 1.0) -> float:
    """Taker fee in USDC for one share bought or sold at `price`."""
    if price <= 0 or price >= 1 or rate <= 0:
        return 0.0
    return rate * (price * (1.0 - price)) ** exponent


def all_in_buy_cost(price: float, rate: float, exponent: float = 1.0) -> float:
    """Cost per share of a taker BUY at `price`, including the taker fee."""
    return price + taker_fee_per_share(price, rate, exponent)


def net_sell_proceeds(price: float, rate: float, exponent: float = 1.0) -> float:
    """Proceeds per share of a taker SELL at `price`, after the taker fee."""
    return price - taker_fee_per_share(price, rate, exponent)


def prob_up(
    current_price: float,
    strike: float,
    vol_per_second: float,
    tau_seconds: float,
    running_twap: Optional[float] = None,
    basis_bps: float = 0.0,
    drift_z: float = 0.0,
    window_seconds: float = TWAP_WINDOW_SECONDS,
) -> float:
    """
    P(final 60s TWAP >= strike).

    running_twap: mean of the index since (end - window_seconds), needed only
        when tau < window_seconds; falls back to current_price if unknown.
    drift_z: optional directional tilt in standard deviations (e.g. from an ML
        model), added to the z-score. 0 = pure volatility model.
    """
    if current_price <= 0 or strike <= 0:
        return 0.5
    tau = max(0.0, float(tau_seconds))
    sigma = max(0.0, float(vol_per_second))
    w = window_seconds

    if tau >= w:
        mean = current_price
        var = (current_price * sigma) ** 2 * (tau - 2.0 * w / 3.0)
    else:
        known = running_twap if running_twap and running_twap > 0 else current_price
        mean = ((w - tau) * known + tau * current_price) / w
        var = (tau / w) ** 2 * (current_price * sigma) ** 2 * tau / 3.0

    basis = basis_bps * 1e-4 * strike
    var += 2.0 * basis ** 2  # basis error on both strike and settlement

    if var <= 0:
        if mean > strike:
            return MAX_PROB
        if mean < strike:
            return MIN_PROB
        return 0.5

    z = (mean - strike) / math.sqrt(var) + drift_z
    return float(min(MAX_PROB, max(MIN_PROB, norm.cdf(z))))


def binary_kelly_fraction(prob: float, cost: float) -> float:
    """
    Full-Kelly bankroll fraction for buying a $1-payout share at all-in `cost`
    when the win probability is `prob`: f* = (p - c) / (1 - c).
    """
    if cost <= 0 or cost >= 1 or prob <= cost:
        return 0.0
    return (prob - cost) / (1.0 - cost)
