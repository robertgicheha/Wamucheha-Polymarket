"""Parsing tests for connectors/polymarket_connector.py (no network)."""
import json

import pytest

from connectors.polymarket_connector import PolymarketConnector, parse_book

# Shape of a real CLOB /book payload: bids ASCENDING, asks DESCENDING.
RAW_BOOK = {
    "asset_id": "tok",
    "timestamp": "1790193401880",
    "bids": [{"price": "0.01", "size": "1000"}, {"price": "0.47", "size": "20"}, {"price": "0.48", "size": "12"}],
    "asks": [{"price": "0.99", "size": "1000"}, {"price": "0.52", "size": "30"}, {"price": "0.50", "size": "8"}],
    "min_order_size": "5",
    "tick_size": "0.01",
}


def test_parse_book_takes_best_prices_not_first_elements():
    book = parse_book(RAW_BOOK)
    assert book.best_bid == 0.48
    assert book.bid_size == 12
    assert book.best_ask == 0.50
    assert book.ask_size == 8
    assert book.bids[0] == (0.48, 12.0) and book.asks[0] == (0.50, 8.0)
    assert book.spread == pytest.approx(0.02)
    assert book.mid == pytest.approx(0.49)
    assert book.min_order_size == 5


def test_parse_book_empty_side():
    book = parse_book({"asset_id": "t", "bids": [], "asks": [{"price": "0.6", "size": "1"}]})
    assert book.best_bid == 0.0 and book.best_ask == 0.6
    assert book.spread == 1.0


def _gamma_market(outcomes, prices, closed=False):
    return {
        "conditionId": "0xcond",
        "question": "Bitcoin Up or Down - test",
        "clobTokenIds": json.dumps(["tokA", "tokB"]),
        "outcomes": json.dumps(outcomes),
        "outcomePrices": json.dumps(prices),
        "eventStartTime": "2026-09-23T20:00:00Z",
        "endDate": "2026-09-23T20:05:00Z",
        "acceptingOrders": True,
        "orderMinSize": 5,
        "orderPriceMinTickSize": 0.01,
        "closed": closed,
    }


def test_updown_market_maps_tokens_by_label(monkeypatch):
    c = PolymarketConnector()
    # Outcomes deliberately reversed: the mapping must follow labels.
    monkeypatch.setattr(c, "_gamma_get", lambda path, params=None: [_gamma_market(["Down", "Up"], ["0.4", "0.6"])])
    m = c.get_updown_market("btc", 5, 1790193600)
    assert m.slug == "btc-updown-5m-1790193600"
    assert m.token_id_up == "tokB" and m.token_id_down == "tokA"
    assert m.end_time - m.start_time == 300


def test_resolution_requires_closed_and_decisive_prices(monkeypatch):
    c = PolymarketConnector()
    calls = {}

    def fake(path, params=None):
        calls.update(params)
        return [_gamma_market(["Up", "Down"], ["0", "1"], closed=True)]

    monkeypatch.setattr(c, "_gamma_get", fake)
    assert c.get_updown_resolution("btc-updown-5m-1") == "DOWN"
    assert calls["closed"] == "true"  # Gamma hides closed markets otherwise

    monkeypatch.setattr(c, "_gamma_get", lambda p, params=None: [_gamma_market(["Up", "Down"], ["0.6", "0.4"])])
    assert c.get_updown_resolution("x") is None
