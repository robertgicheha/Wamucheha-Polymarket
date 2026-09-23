"""
Tests for data.btc_tick_buffer — maps BRTIEngine's live tick history into
the DataFrame schema ml.btc_features.BTCFeatureEngine expects.
"""
import time

import pytest

from data.brti.brti_engine import BRTITick
from data.btc_tick_buffer import build_feature_frame, raw_price_frame, ticks_to_frame


def _make_ticks(n=200, start_price=100000.0, drift=1.0):
    now = time.time()
    ticks = []
    price = start_price
    for i in range(n):
        price += drift
        ticks.append(BRTITick(
            timestamp=now + i,
            brti_price=price,
            bid_volume=5.0,
            ask_volume=5.0,
            utilized_depth_bid=5.0,
            utilized_depth_ask=5.0,
            mid_price_bid=price - 0.5,
            mid_price_ask=price + 0.5,
            spread_bps=1.0,
            exchanges_used=["coinbase", "kraken"],
        ))
    return ticks


class FakeBRTIEngine:
    """Minimal stand-in for BRTIEngine — only the parts btc_tick_buffer uses."""

    def __init__(self, ticks):
        self.tick_history = ticks

    @property
    def last_tick(self):
        return self.tick_history[-1] if self.tick_history else None


def test_ticks_to_frame_maps_expected_columns():
    ticks = _make_ticks(10)
    df = ticks_to_frame(ticks)
    assert len(df) == 10
    for col in ["timestamp", "price", "mid_price", "bid1_price", "ask1_price",
                "bid_depth_5", "ask_depth_5", "spread"]:
        assert col in df.columns
    assert df["price"].iloc[-1] == ticks[-1].brti_price


def test_ticks_to_frame_empty_input():
    df = ticks_to_frame([])
    assert df.empty


def test_build_feature_frame_returns_none_below_minimum_history():
    engine = FakeBRTIEngine(_make_ticks(10))  # below the 60-tick minimum
    assert build_feature_frame(engine) is None


def test_build_feature_frame_computes_features_once_enough_history():
    engine = FakeBRTIEngine(_make_ticks(200))
    features = build_feature_frame(engine)
    assert features is not None
    assert not features.empty
    assert len(features) == 200
    # Sanity: a steadily rising price series should show positive momentum
    assert "momentum_10" in features.columns
    assert features["momentum_10"].iloc[-1] > 0


def test_raw_price_frame_respects_lookback():
    engine = FakeBRTIEngine(_make_ticks(100))
    full = raw_price_frame(engine)
    limited = raw_price_frame(engine, lookback=10)
    assert len(full) == 100
    assert len(limited) == 10
    assert limited["price"].iloc[-1] == full["price"].iloc[-1]
