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
import heapq
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

# Only the top of each venue's book matters for the index. Full-depth local
# books reach ~40k levels on Coinbase; re-sorting them on every message and
# merging them on every tick starved the whole process of CPU.
MAX_LEVELS_PER_SIDE = 100


def _top_levels(book: Dict[float, float], bids: bool) -> List["OrderbookLevel"]:
    """Best MAX_LEVELS_PER_SIDE levels of a {price: size} book, best first."""
    pick = heapq.nlargest if bids else heapq.nsmallest
    return [OrderbookLevel(price=p, size=book[p]) for p in pick(MAX_LEVELS_PER_SIDE, book)]



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
            bids=bids[:MAX_LEVELS_PER_SIDE],
            asks=asks[:MAX_LEVELS_PER_SIDE],
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

        bids = _top_levels(self._local_book["bids"], bids=True)
        asks = _top_levels(self._local_book["asks"], bids=False)

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
        # v2 sends "data": [ {symbol, bids, asks, checksum, timestamp} ]
        book_data = data.get("data") or {}
        if isinstance(book_data, list):
            book_data = book_data[0] if book_data else {}

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
        bids = _top_levels(self._local_book["bids"], bids=True)
        asks = _top_levels(self._local_book["asks"], bids=False)

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
        channel = f"order_book_{self.symbol.lower().replace('-', '').replace('/', '')}"
        return json.dumps({
            "event": "bts:subscribe",
            "data": {"channel": channel},
        })

    def _parse_message(self, raw: str) -> Optional[OrderbookSnapshot]:
        data = json.loads(raw)
        # Book updates arrive as {"event": "data", "channel": "order_book_btcusd", "data": {...}}
        if data.get("event") != "data" or not str(data.get("channel", "")).startswith("order_book"):
            return None

        book_data = data.get("data") or {}
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
        bids, asks = bids[:MAX_LEVELS_PER_SIDE], asks[:MAX_LEVELS_PER_SIDE]

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
    URL: wss://api.gemini.com/v2/marketdata (subscribe to l2 for e.g. BTCUSD)

    Gemini sends full book as first message, then incremental updates.
    """

    exchange_name = "gemini"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._local_book: Dict[str, Dict[float, float]] = {"bids": {}, "asks": {}}
        self._book_initialized = False

    def _get_url(self) -> str:
        return "wss://api.gemini.com/v2/marketdata"

    def _build_subscribe_message(self) -> str:
        symbol = self.symbol.upper().replace("-", "").replace("/", "")
        return json.dumps({"type": "subscribe", "subscriptions": [{"name": "l2", "symbols": [symbol]}]})

    def _parse_message(self, raw: str) -> Optional[OrderbookSnapshot]:
        """
        v2 l2 feed: the first "l2_updates" message carries the full book, later
        ones carry incremental changes; each change is [side, price, qty] with
        side "buy"/"sell" and qty 0 meaning the level was removed.
        """
        data = json.loads(raw)
        if data.get("type") != "l2_updates":
            return None
        changes = data.get("changes") or []
        if not self._book_initialized:
            self._local_book = {"bids": {}, "asks": {}}
            self._book_initialized = True
        for side, price, qty in changes:
            book = self._local_book["bids"] if side == "buy" else self._local_book["asks"]
            price, qty = float(price), float(qty)
            if qty == 0:
                book.pop(price, None)
            else:
                book[price] = qty
        return self._build_snapshot(time.time())

    def _build_snapshot(self, timestamp) -> OrderbookSnapshot:
        bids = _top_levels(self._local_book["bids"], bids=True)
        asks = _top_levels(self._local_book["asks"], bids=False)
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
            # Kraken's v2 API uses "BTC/USD" (the old "XBT/USD" is rejected)
            exchange_symbol = symbol.replace("-", "/") if exchange == "kraken" else symbol
            ws = create_exchange_ws(
                exchange=exchange,
                symbol=exchange_symbol,
                on_snapshot=on_snapshot,
            )
            ws_instances.append(ws)
            logger.info("Created %s WebSocket for %s", exchange, symbol)
        except Exception as e:
            logger.error("Failed to create %s WebSocket: %s", exchange, e)

    return ws_instances
