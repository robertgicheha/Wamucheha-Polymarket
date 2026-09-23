"""
End-to-end paper lifecycle for an up/down window with a fake market feed and
fake BRTI index: discover -> strike capture -> entry on real edge -> resolution
via the feed -> PnL booked and reported to the risk manager.
"""
import time

import pytest

from config.settings import settings
from connectors.polymarket_connector import FeeParams, parse_book
from strategies.lifecycle_engine import FiveMinuteLifecycleEngine, MarketPhase

STRIKE = 100_000.0


class FakeTick:
    def __init__(self, price):
        self.brti_price = price


class FakeBRTI:
    """Flat at the strike before window start, then a constant level."""

    def __init__(self, start, level):
        self.start = start
        self.level = level

    @property
    def last_tick(self):
        return FakeTick(self.level)

    def get_price_history(self, seconds):
        now = time.time()
        return [(now - i, STRIKE if now - i <= self.start else self.level)
                for i in range(int(seconds), -1, -1)]

    def get_volatility(self, seconds):
        return 0.0001


def _book(token, bid, ask, size=200):
    return parse_book({"asset_id": token, "bids": [{"price": str(bid), "size": str(size)}],
                       "asks": [{"price": str(ask), "size": str(size)}]})


class FakeFeed:
    def __init__(self, market, up_book, down_book):
        self.market = market
        self.books = {"up": up_book, "down": down_book}
        self.resolution = None
        self.discovered = False

    def discover(self, now):
        if self.discovered:
            return []
        self.discovered = True
        return [self.market]

    def get_books(self, token_ids):
        return {"up": self.books["up"], "down": self.books["down"]}

    def get_fee_params(self, condition_id):
        return FeeParams(rate=0.07, exponent=1.0)

    def get_resolution(self, slug):
        return self.resolution


class FakeMarket:
    def __init__(self, start, end):
        self.start, self.end = start, end

    def to_discovery_dict(self):
        return {"condition_id": "0xwin", "slug": "btc-updown-5m-1", "question": "Bitcoin Up or Down",
                "asset": "btc", "start_time": self.start, "end_time": self.end,
                "token_id_yes": "up", "token_id_no": "down", "min_order_size": 5}


class RecordingRisk:
    halted = False

    def __init__(self):
        self.closed = []

    def max_position_size(self, category):
        return 1e9

    def record_closed_trade(self, market_id, category, pnl):
        self.closed.append(pnl)


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    monkeypatch.setattr(settings, "min_net_edge", 0.03)
    monkeypatch.setattr(settings, "entry_min_elapsed_seconds", 0)
    monkeypatch.setattr(settings, "entry_min_remaining_seconds", 5)
    monkeypatch.setattr(settings, "max_spread", 0.05)
    monkeypatch.setattr(settings, "settlement_basis_bps", 0.0)
    monkeypatch.setattr(settings, "ml_prediction_enabled", False)
    monkeypatch.setattr(settings, "brti_tick_interval_seconds", 1)


def _engine(level, up_ask, down_ask):
    now = time.time()
    start, end = now - 120, now + 180
    feed = FakeFeed(FakeMarket(start, end), _book("up", up_ask - 0.01, up_ask),
                    _book("down", down_ask - 0.01, down_ask))
    risk = RecordingRisk()
    engine = FiveMinuteLifecycleEngine(bankroll=1000.0, brti_engine=FakeBRTI(start, level),
                                       market_feed=feed, risk_manager=risk)
    return engine, feed, risk


def test_enters_up_when_index_is_well_above_strike_and_market_lags():
    engine, feed, risk = _engine(level=STRIKE * 1.003, up_ask=0.60, down_ask=0.41)
    engine.tick()  # discover + start + first run tick
    engine.tick()
    w = engine._market_windows["0xwin"]
    assert w.price_to_beat == pytest.approx(STRIKE)
    assert w.fair_prob_up > 0.9
    assert w.position_side == "YES"
    assert w.shares >= 5
    assert engine.bankroll == pytest.approx(1000.0 - w.cost_usd)
    shares = w.shares

    # Window ends, resolves Up -> each share pays $1.
    w.end_time = time.time() - 1
    engine.tick()
    assert w.phase == MarketPhase.RESOLVING
    feed.resolution = "UP"
    w.last_resolution_check = 0
    engine.tick()
    assert w.phase == MarketPhase.SETTLED
    assert w.won and w.pnl_usd == pytest.approx(shares - w.position_size_usd)
    assert engine.bankroll == pytest.approx(1000.0 + w.pnl_usd)
    assert risk.closed == [pytest.approx(w.pnl_usd)]


def test_no_trade_when_market_already_prices_the_fair_value():
    # Index at the strike -> fair ~0.5; asks at 0.51 cost ~0.527 after fees.
    engine, feed, risk = _engine(level=STRIKE, up_ask=0.51, down_ask=0.51)
    engine.tick()
    engine.tick()
    w = engine._market_windows["0xwin"]
    assert w.position_side == ""
    assert "net edge" in w.strategy_signal["reason"]


def test_window_without_strike_history_is_not_traded():
    engine, feed, risk = _engine(level=STRIKE * 1.003, up_ask=0.60, down_ask=0.41)
    engine.brti_engine.get_price_history = lambda seconds: []
    engine.tick()
    engine.tick()
    w = engine._market_windows["0xwin"]
    assert not w.tradeable and w.position_side == ""


def test_early_exit_when_bid_exceeds_model_value():
    engine, feed, risk = _engine(level=STRIKE * 1.003, up_ask=0.60, down_ask=0.41)
    engine.tick()
    w = engine._market_windows["0xwin"]
    assert w.position_side == "YES"
    # Index collapses back below the strike while the Up bid stays rich.
    engine.brti_engine.level = STRIKE * 0.997
    feed.books["up"] = _book("up", 0.70, 0.71)
    engine.tick()
    assert w.phase == MarketPhase.SETTLED
    assert w.shares == 0
    assert w.exit_price == pytest.approx(0.70)
