"""
Exchange-specific WebSocket managers for BRTI constituent exchanges.

Each exchange has different:
  - WebSocket URLs
  - Subscription message formats
  - Message payload structures
  - Orderbook update mechanisms (snapshot vs diff)

All normalize to OrderbookSnapshot from connectors.exchange_ws_base.

Constituent exchanges (per CF Benchmarks):
  - Coinbase (BTC-USD)
  - Kraken (XBT/USD)
  - Bitstamp (btcusd)
  - Gemini (btcusd)
"""
import json
import logging
import time
from typing import Dict, List, Optional

from connectors.exchange_ws_base import (
    ExchangeWebSocketBase,
    OrderbookLevel,
    OrderbookSnapshot,
)

logger = logging.getLogger(__name__)


class CoinbaseWSS(ExchangeWebSocketBase):
    """
    Coinbase Advanced Trade WebSocket.
    Channel: level2_batch — incremental orderbook updates.
    URL: wss://ws-feed.exchange.coinbase.com
    """

    exchange_name = "coinbase"

    def _get_url(self) -> str:
        return "wss://ws-feed.exchange.coinbase.com"

    def _build_subscribe_message(self) -> str:
        return json.dumps({
            "type": "subscribe",
            "product_ids": [self.symbol],
            "channels": ["level2_batch"],
        })

    def _parse_message(self, raw: str) -> Optional[OrderbookSnapshot]:
        data = json.loads(raw)
        msg_type = data.get("type")

        if msg_type == "snapshot":
            return self._parse_snapshot(data)
        elif msg_type == "l2update":
            return self._apply_update(data)
        return None

    def _parse_snapshot(self, data: Dict) -> OrderbookSnapshot:
        bids = [
            OrderbookLevel(price=float(b[0]), size=float(b[1]))
            for b in data.get("bids", [])
            if float(b[1]) > 0
        ]
        asks = [
            OrderbookLevel(price=float(a[0]), size=float(a[1]))
            for a in data.get("asks", [])
            if float(a[1]) > 0
        ]
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)

        self._local_book = {
            "bids": {b.price: b.size for b in bids},
            "asks": {a.price: a.size for a in asks},
        }

        return OrderbookSnapshot(
            exchange=self.exchange_name,
            bids=bids,
            asks=asks,
            timestamp=time.time(),
        )

    def _apply_update(self, data: Dict) -> Optional[OrderbookSnapshot]:
        if not hasattr(self, "_local_book"):
            return None

        changes = data.get("changes", [])
        for change in changes:
            side = change[0]
            price = float(change[1])
            size = float(change[2])

            book = self._local_book["bids"] if side == "buy" else self._local_book["asks"]
            if size == 0:
                book.pop(price, None)
            else:
                book[price] = size

        bids = sorted(
            [OrderbookLevel(price=p, size=s) for p, s in self._local_book["bids"].items()],
            key=lambda x: x.price,
            reverse=True,
        )
        asks = sorted(
            [OrderbookLevel(price=p, size=s) for p, s in self._local_book["asks"].items()],
            key=lambda x: x.price,
        )

        return OrderbookSnapshot(
            exchange=self.exchange_name,
            bids=bids,
            asks=asks,
            timestamp=time.time(),
        )


class KrakenWSS(ExchangeWebSocketBase):
    """
    Kraken WebSocket v2.
    Channel: book — orderbook updates with depth.
    URL: wss://ws.kraken.com/v2

    Kraken uses a diff-based system. First message is a full snapshot,
    subsequent messages are diffs applied to the local book.
    """

    exchange_name = "kraken"

    def __init__(self, *args, depth: int = 25, **kwargs):
        super().__init__(*args, **kwargs)
        self._depth = depth
        self._local_book: Dict[str, Dict[float, float]] = {"bids": {}, "asks": {}}
        self._book_initialized = False

    def _get_url(self) -> str:
        return "wss://ws.kraken.com/v2"

    def _build_subscribe_message(self) -> str:
        return json.dumps({
            "method": "subscribe",
            "params": {
                "channel": "book",
                "symbol": [self.symbol],
                "depth": self._depth,
            },
        })

    def _parse_message(self, raw: str) -> Optional[OrderbookSnapshot]:
        data = json.loads(raw)

        if "method" in data and data.get("method") in ("subscribe", "heartbeat"):
            return None
        if "channel" not in data:
            return None

        channel = data.get("channel")
        if channel != "book":
            return None

        msg_type = data.get("type")
        book_data = data.get("data", {})

        if msg_type == "snapshot":
            return self._parse_snapshot(book_data)
        elif msg_type == "update":
            return self._apply_update(book_data)

        return None

    def _parse_snapshot(self, data: Dict) -> OrderbookSnapshot:
        self._local_book = {"bids": {}, "asks": {}}

        for bid in data.get("bids", []):
            price = float(bid["price"])
            qty = float(bid["qty"])
            if qty > 0:
                self._local_book["bids"][price] = qty

        for ask in data.get("asks", []):
            price = float(ask["price"])
            qty = float(ask["qty"])
            if qty > 0:
                self._local_book["asks"][price] = qty

        self._book_initialized = True
        return self._build_snapshot(data.get("timestamp", time.time()))

    def _apply_update(self, data: Dict) -> Optional[OrderbookSnapshot]:
        if not self._book_initialized:
            return None

        for bid in data.get("bids", []):
            price = float(bid["price"])
            qty = float(bid["qty"])
            if qty == 0:
                self._local_book["bids"].pop(price, None)
            else:
                self._local_book["bids"][price] = qty

        for ask in data.get("asks", []):
            price = float(ask["price"])
            qty = float(ask["qty"])
            if qty == 0:
                self._local_book["asks"].pop(price, None)
            else:
                self._local_book["asks"][price] = qty

        return self._build_snapshot(data.get("timestamp", time.time()))

    def _build_snapshot(self, timestamp) -> OrderbookSnapshot:
        bids = sorted(
            [OrderbookLevel(price=p, size=s) for p, s in self._local_book["bids"].items()],
            key=lambda x: x.price, reverse=True,
        )
        asks = sorted(
            [OrderbookLevel(price=p, size=s) for p, s in self._local_book["asks"].items()],
            key=lambda x: x.price,
        )

        if isinstance(timestamp, str):
            try:
                timestamp = float(timestamp)
            except (ValueError, TypeError):
                timestamp = time.time()

        return OrderbookSnapshot(
            exchange=self.exchange_name,
            bids=bids,
            asks=asks,
            timestamp=timestamp,
        )


class BitstampWSS(ExchangeWebSocketBase):
    """
    Bitstamp WebSocket.
    Channel: order_book — full orderbook snapshots (no diff).
    URL: wss://ws.bitstamp.net

    Bitstamp sends the full book on each update, so no local state tracking needed.
    """

    exchange_name = "bitstamp"

    def _get_url(self) -> str:
        return "wss://ws.bitstamp.net"

    def _build_subscribe_message(self) -> str:
        channel = f"order_book_{self.symbol.lower()}"
        return json.dumps({
            "event": "bts:subscribe",
            "data": {"channel": channel},
        })

    def _parse_message(self, raw: str) -> Optional[OrderbookSnapshot]:
        data = json.loads(raw)
        event = data.get("event", "")

        if "order_book" not in event:
            return None
        if event.endswith("data"):
            pass
        elif "data" in data:
            pass
        else:
            return None

        book_data = data.get("data", data)
        if "bids" not in book_data:
            return None

        bids = [
            OrderbookLevel(price=float(b[0]), size=float(b[1]))
            for b in book_data.get("bids", [])
            if float(b[1]) > 0
        ]
        asks = [
            OrderbookLevel(price=float(a[0]), size=float(a[1]))
            for a in book_data.get("asks", [])
            if float(a[1]) > 0
        ]

        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)

        ts = book_data.get("timestamp")
        if ts is None:
            ts = time.time()
        else:
            ts = float(ts)

        return OrderbookSnapshot(
            exchange=self.exchange_name,
            bids=bids,
            asks=asks,
            timestamp=ts,
        )


class GeminiWSS(ExchangeWebSocketBase):
    """
    Gemini WebSocket v2.
    Channel: l2 — level 2 orderbook updates.
    URL: wss://api.gemini.com/v2/marketdata/{symbol}

    Gemini sends full book as first message, then incremental updates.
    """

    exchange_name = "gemini"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._local_book: Dict[str, Dict[float, float]] = {"bids": {}, "asks": {}}
        self._book_initialized = False

    def _get_url(self) -> str:
        symbol = self.symbol.lower().replace("-", "")
        return f"wss://api.gemini.com/v2/marketdata/{symbol}"

    def _build_subscribe_message(self) -> str:
        # Gemini v2 auto-subscribes on connection, no explicit subscribe needed
        return ""

    def _parse_message(self, raw: str) -> Optional[OrderbookSnapshot]:
        data = json.loads(raw)

        if "heartbeat" in data:
            return None

        events = data.get("events", [])
        if not events:
            return None

        snapshot_event = None
        change_events = []

        for event in events:
            typ = event.get("type")
            if typ == "snapshot":
                snapshot_event = event
            elif typ == "change":
                change_events.append(event)

        if snapshot_event:
            return self._handle_snapshot(snapshot_event)
        elif change_events:
            return self._handle_changes(change_events, data.get("timestamp", time.time()))

        return None

    def _handle_snapshot(self, event: Dict) -> OrderbookSnapshot:
        self._local_book = {"bids": {}, "asks": {}}

        for item in event.get("bids", []):
            price = float(item["price"])
            qty = float(item["remaining"])
            if qty > 0:
                self._local_book["bids"][price] = qty

        for item in event.get("asks", []):
            price = float(item["price"])
            qty = float(item["remaining"])
            if qty > 0:
                self._local_book["asks"][price] = qty

        self._book_initialized = True
        return self._build_snapshot(time.time())

    def _handle_changes(self, changes: List[Dict], timestamp) -> Optional[OrderbookSnapshot]:
        if not self._book_initialized:
            return None

        for change in changes:
            side = change.get("side")
            price = float(change["price"])
            qty = float(change["remaining"])

            book = self._local_book["bids"] if side == "bid" else self._local_book["asks"]
            if qty == 0:
                book.pop(price, None)
            else:
                book[price] = qty

        return self._build_snapshot(timestamp)

    def _build_snapshot(self, timestamp) -> OrderbookSnapshot:
        bids = sorted(
            [OrderbookLevel(price=p, size=s) for p, s in self._local_book["bids"].items()],
            key=lambda x: x.price, reverse=True,
        )
        asks = sorted(
            [OrderbookLevel(price=p, size=s) for p, s in self._local_book["asks"].items()],
            key=lambda x: x.price,
        )
        return OrderbookSnapshot(
            exchange=self.exchange_name,
            bids=bids,
            asks=asks,
            timestamp=timestamp,
        )


# ── Factory ────────────────────────────────────────────────────────────

EXCHANGE_WS_MAP = {
    "coinbase": CoinbaseWSS,
    "kraken": KrakenWSS,
    "bitstamp": BitstampWSS,
    "gemini": GeminiWSS,
}


def create_exchange_ws(
    exchange: str,
    symbol: str,
    on_snapshot=None,
    **kwargs,
) -> ExchangeWebSocketBase:
    """Create an exchange WebSocket by name."""
    cls = EXCHANGE_WS_MAP.get(exchange.lower())
    if cls is None:
        raise ValueError(f"Unknown exchange: {exchange}. Available: {list(EXCHANGE_WS_MAP.keys())}")
    return cls(symbol=symbol, on_snapshot=on_snapshot, **kwargs)


def get_all_exchange_ws(
    symbol: str = "BTC-USD",
    on_snapshot=None,
    exchanges: Optional[List[str]] = None,
) -> List[ExchangeWebSocketBase]:
    """Create WebSocket instances for all BRTI constituent exchanges."""
    if exchanges is None:
        exchanges = ["coinbase", "kraken", "bitstamp", "gemini"]

    ws_instances = []
    for exchange in exchanges:
        try:
            # Kraken uses different symbol format
            kraken_symbol = "XBT/USD" if exchange == "kraken" else symbol
            ws = create_exchange_ws(
                exchange=exchange,
                symbol=kraken_symbol,
                on_snapshot=on_snapshot,
            )
            ws_instances.append(ws)
            logger.info("Created %s WebSocket for %s", exchange, symbol)
        except Exception as e:
            logger.error("Failed to create %s WebSocket: %s", exchange, e)

    return ws_instances
