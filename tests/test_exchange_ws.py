"""
Parser tests for the BRTI exchange feeds, using message shapes captured from
the live Kraken v2, Bitstamp and Gemini v2 WebSocket APIs, plus the BRTI
top-of-book validation.
"""
import json
import time

from connectors.exchange_ws_base import OrderbookLevel, OrderbookSnapshot
from data.brti.brti_engine import BRTIEngine
from data.brti.exchange_ws import BitstampWSS, GeminiWSS, KrakenWSS, get_all_exchange_ws


def test_symbols_and_subscriptions():
    ws = {w.exchange_name: w for w in get_all_exchange_ws("BTC-USD")}
    assert ws["kraken"].symbol == "BTC/USD"  # "XBT/USD" is rejected by Kraken v2
    assert json.loads(ws["bitstamp"]._build_subscribe_message())["data"]["channel"] == "order_book_btcusd"
    sub = json.loads(ws["gemini"]._build_subscribe_message())
    assert sub["subscriptions"] == [{"name": "l2", "symbols": ["BTCUSD"]}]
    assert ws["gemini"]._get_url() == "wss://api.gemini.com/v2/marketdata"


def test_kraken_v2_snapshot_and_update_with_list_payload():
    k = KrakenWSS(symbol="BTC/USD")
    snap = k._parse_message(json.dumps({
        "channel": "book", "type": "snapshot",
        "data": [{"symbol": "BTC/USD",
                  "bids": [{"price": 100.0, "qty": 1.0}, {"price": 99.0, "qty": 2.0}],
                  "asks": [{"price": 101.0, "qty": 1.5}],
                  "checksum": 1, "timestamp": "2026-09-23T20:21:04.336Z"}],
    }))
    assert snap.bids[0].price == 100.0 and snap.asks[0].price == 101.0
    upd = k._parse_message(json.dumps({
        "channel": "book", "type": "update",
        "data": [{"symbol": "BTC/USD", "bids": [{"price": 100.0, "qty": 0}], "asks": [], "checksum": 2}],
    }))
    assert upd.bids[0].price == 99.0


def test_bitstamp_data_event_is_parsed():
    b = BitstampWSS(symbol="BTC-USD")
    assert b._parse_message(json.dumps({"event": "bts:subscription_succeeded",
                                        "channel": "order_book_btcusd", "data": {}})) is None
    snap = b._parse_message(json.dumps({
        "event": "data", "channel": "order_book_btcusd",
        "data": {"timestamp": "1790194864", "bids": [["84338.17", "0.66"], ["84337.94", "0.12"]],
                 "asks": [["84340.00", "0.5"]]},
    }))
    assert snap.bids[0].price == 84338.17 and snap.asks[0].price == 84340.0


def test_gemini_l2_initial_book_then_changes():
    g = GeminiWSS(symbol="BTC-USD")
    snap = g._parse_message(json.dumps({
        "type": "l2_updates", "symbol": "BTCUSD",
        "changes": [["buy", "84341.41", "0.02"], ["buy", "84339.69", "0.03"], ["sell", "84345.00", "0.5"]],
    }))
    assert snap.bids[0].price == 84341.41 and snap.asks[0].price == 84345.0
    snap = g._parse_message(json.dumps({"type": "l2_updates", "symbol": "BTCUSD",
                                        "changes": [["buy", "84341.41", "0.0"]]}))
    assert snap.bids[0].price == 84339.69
    assert g._parse_message(json.dumps({"type": "heartbeat"})) is None


def _snapshot(exchange, bid, ask, size=1.0):
    return OrderbookSnapshot(
        exchange=exchange,
        bids=[OrderbookLevel(price=bid - i, size=size) for i in range(5)],
        asks=[OrderbookLevel(price=ask + i, size=size) for i in range(5)],
        timestamp=time.time(),
    )


def test_brti_validation_uses_top_of_book_not_depth_weighted_sides():
    engine = BRTIEngine(validation_enabled=True, max_divergence_bps=5.0)
    # Deep, evenly spread book: side-weighted averages sit far apart, but the
    # top of book is tight -> tick must be accepted.
    engine.update_orderbook(_snapshot("coinbase", 84000.0, 84000.5))
    tick = engine.calculate_tick()
    assert tick is not None and 83990 < tick.brti_price < 84010


def test_brti_validation_rejects_badly_crossed_feed():
    engine = BRTIEngine(validation_enabled=True, max_divergence_bps=5.0)
    engine.update_orderbook(_snapshot("coinbase", 84000.0, 84000.5))
    engine.update_orderbook(_snapshot("kraken", 84200.0, 84200.5))  # stale venue ~24 bps off
    assert engine.calculate_tick() is None
