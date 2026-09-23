"""Tests for execution/executor.py — paper fill simulation and live guard rails."""
from types import SimpleNamespace

import pytest

from config.settings import LIVE_TRADING_ACK_PHRASE, settings
from execution.executor import LiveExecutor, PaperExecutor, simulate_taker

ASKS = [(0.50, 10.0), (0.51, 10.0), (0.60, 100.0)]
BIDS = [(0.49, 10.0), (0.48, 10.0), (0.30, 100.0)]
FEE = 0.07


def test_buy_walks_levels_up_to_bound_and_charges_fees():
    shares, notional, fees = simulate_taker("BUY", ASKS, 0.51, FEE, 1.0, usd_amount=1000)
    assert shares == pytest.approx(20.0)  # stops before the 0.60 level
    assert notional == pytest.approx(10 * 0.50 + 10 * 0.51)
    assert fees == pytest.approx(10 * FEE * 0.25 + 10 * FEE * 0.51 * 0.49)


def test_buy_budget_includes_fees():
    shares, notional, fees = simulate_taker("BUY", ASKS, 0.50, FEE, 1.0, usd_amount=2.5875)
    assert notional + fees == pytest.approx(2.5875)
    assert shares == pytest.approx(5.0)


def test_sell_respects_min_price():
    shares, notional, fees = simulate_taker("SELL", BIDS, 0.48, FEE, 1.0, shares=50)
    assert shares == pytest.approx(20.0)
    assert notional == pytest.approx(10 * 0.49 + 10 * 0.48)


def test_paper_executor_fill_shape():
    fill = PaperExecutor().buy("tok", 5.0, 0.50, ASKS, FEE, 1.0)
    assert fill.simulated and fill.shares > 0
    assert fill.cash_usd == pytest.approx(5.0)
    assert fill.avg_price == pytest.approx(0.50)
    assert PaperExecutor().buy("tok", 5.0, 0.40, ASKS, FEE, 1.0) is None  # nothing <= bound


class FakeConnector:
    def __init__(self, balance=100.0):
        self.balance = balance
        self.token_balance = 0.0
        self.orders = []

    def get_collateral_balance(self):
        return self.balance

    def get_token_balance(self, token_id):
        return self.token_balance

    def buy_fak(self, token_id, usd, max_price):
        self.orders.append(("BUY", token_id, usd, max_price))
        self.token_balance += usd / max_price
        return SimpleNamespace(ok=True, order_id="o1", making_amount=usd, taking_amount=usd / max_price)

    def sell_fak(self, token_id, shares, min_price):
        self.orders.append(("SELL", token_id, shares, min_price))
        return SimpleNamespace(ok=True, order_id="o2", making_amount=shares, taking_amount=shares * min_price)


@pytest.fixture
def live_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "trading_mode", "live")
    monkeypatch.setattr(settings, "live_trading_ack", LIVE_TRADING_ACK_PHRASE)
    monkeypatch.setattr(settings, "live_max_order_usd", 10.0)
    monkeypatch.setattr(settings, "live_max_open_exposure_usd", 25.0)
    monkeypatch.setattr(settings, "kill_switch_file", str(tmp_path / "KILL_SWITCH"))
    return tmp_path


def test_live_buy_happy_path_and_order_cap(live_settings):
    conn = FakeConnector()
    fill = LiveExecutor(conn).buy("tok", 50.0, 0.5, ASKS, FEE, 1.0)
    assert conn.orders == [("BUY", "tok", 10.0, 0.5)]  # capped to LIVE_MAX_ORDER_USD
    assert fill.shares == pytest.approx(20.0) and fill.cash_usd == pytest.approx(10.0)


def test_live_buy_blocked_without_ack(live_settings, monkeypatch):
    monkeypatch.setattr(settings, "live_trading_ack", "yes")
    conn = FakeConnector()
    assert LiveExecutor(conn).buy("tok", 5.0, 0.5, ASKS, FEE, 1.0) is None
    assert conn.orders == []


def test_live_buy_blocked_by_kill_switch(live_settings):
    (live_settings / "KILL_SWITCH").write_text("stop")
    conn = FakeConnector()
    assert LiveExecutor(conn).buy("tok", 5.0, 0.5, ASKS, FEE, 1.0) is None
    assert conn.orders == []


def test_live_buy_blocked_by_exposure_cap(live_settings):
    conn = FakeConnector()
    ex = LiveExecutor(conn, exposure_fn=lambda: 20.0)
    assert ex.buy("tok", 10.0, 0.5, ASKS, FEE, 1.0) is None


def test_live_buy_blocked_by_insufficient_balance(live_settings):
    conn = FakeConnector(balance=3.0)
    assert LiveExecutor(conn).buy("tok", 5.0, 0.5, ASKS, FEE, 1.0) is None
    assert conn.orders == []


def test_live_buy_rejected_order_returns_none(live_settings):
    conn = FakeConnector()
    conn.buy_fak = lambda *a: SimpleNamespace(ok=False, code="fak_not_filled", message="")
    assert LiveExecutor(conn).buy("tok", 5.0, 0.5, ASKS, FEE, 1.0) is None


def test_live_sell_never_exceeds_held_shares(live_settings):
    conn = FakeConnector()
    conn.token_balance = 4.0
    fill = LiveExecutor(conn).sell("tok", 10.0, 0.45, BIDS, FEE, 1.0)
    assert conn.orders == [("SELL", "tok", 4.0, 0.45)]
    assert fill.shares == pytest.approx(4.0)


def test_exits_allowed_even_with_kill_switch(live_settings):
    (live_settings / "KILL_SWITCH").write_text("stop")
    conn = FakeConnector()
    conn.token_balance = 4.0
    assert LiveExecutor(conn).sell("tok", 4.0, 0.45, BIDS, FEE, 1.0) is not None
