"""
Base WebSocket manager for exchange orderbook feeds.

Handles connection lifecycle, reconnection with exponential backoff,
heartbeat management, and provides a normalized orderbook snapshot interface.

Each exchange-specific WebSocket subclass normalizes its native format
into:
    {
        "bids": [{"price": float, "size": float}, ...],  # descending by price
        "asks": [{"price": float, "size": float}, ...],  # ascending by price
        "timestamp": float,  # unix timestamp of snapshot
        "exchange": str,
    }
"""
import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class OrderbookLevel:
    price: float
    size: float


@dataclass
class OrderbookSnapshot:
    exchange: str
    bids: List[OrderbookLevel]
    asks: List[OrderbookLevel]
    timestamp: float

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    @property
    def spread_bps(self) -> Optional[float]:
        mid = self.mid_price
        spread = self.spread
        if mid and mid > 0 and spread is not None:
            return (spread / mid) * 10000
        return None

    def bid_depth(self, levels: int = 5) -> float:
        return sum(l.size for l in self.bids[:levels])

    def ask_depth(self, levels: int = 5) -> float:
        return sum(l.size for l in self.asks[:levels])

    def to_dict(self) -> Dict:
        return {
            "exchange": self.exchange,
            "bids": [{"price": l.price, "size": l.size} for l in self.bids],
            "asks": [{"price": l.price, "size": l.size} for l in self.asks],
            "timestamp": self.timestamp,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "mid_price": self.mid_price,
            "spread": self.spread,
        }


class ExchangeWebSocketBase(ABC):
    """
    Base class for exchange WebSocket orderbook feeds.

    Subclasses must implement:
        - _build_subscribe_message() -> str
        - _parse_message(raw: str) -> Optional[OrderbookSnapshot]
        - _get_url() -> str
    """

    exchange_name: str = "unknown"

    def __init__(
        self,
        symbol: str,
        on_snapshot: Optional[Callable[[OrderbookSnapshot], None]] = None,
        max_reconnect_delay: float = 60.0,
        initial_reconnect_delay: float = 1.0,
    ):
        self.symbol = symbol
        self.on_snapshot = on_snapshot
        self.max_reconnect_delay = max_reconnect_delay
        self.initial_reconnect_delay = initial_reconnect_delay

        self._ws = None
        self._running = False
        self._reconnect_delay = initial_reconnect_delay
        self._last_snapshot: Optional[OrderbookSnapshot] = None
        self._snapshot_count = 0
        self._error_count = 0
        self._last_message_time: float = 0.0
        self._task: Optional[asyncio.Task] = None

    @abstractmethod
    def _get_url(self) -> str:
        """Return the WebSocket URL for this exchange."""
        raise NotImplementedError

    @abstractmethod
    def _build_subscribe_message(self) -> str:
        """Return the subscription message to send after connection."""
        raise NotImplementedError

    @abstractmethod
    def _parse_message(self, raw: str) -> Optional[OrderbookSnapshot]:
        """
        Parse a raw WebSocket message into an OrderbookSnapshot.
        Return None if the message is not an orderbook update (e.g., ping, ack).
        """
        raise NotImplementedError

    async def _connect(self):
        """Establish WebSocket connection."""
        import websockets

        url = self._get_url()
        logger.info("[%s] Connecting to %s for %s", self.exchange_name, url, self.symbol)

        self._ws = await websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
            max_size=10 * 1024 * 1024,  # 10MB — some exchanges send large snapshots
        )

        subscribe = self._build_subscribe_message()
        if subscribe:
            await self._ws.send(subscribe)
            logger.info("[%s] Subscribed: %s", self.exchange_name, subscribe[:200])

        self._reconnect_delay = self.initial_reconnect_delay
        logger.info("[%s] Connected to %s", self.exchange_name, self.symbol)

    async def _handle_messages(self):
        """Main message loop — receives, parses, and dispatches snapshots."""
        try:
            async for raw in self._ws:
                self._last_message_time = time.time()
                try:
                    snapshot = self._parse_message(raw)
                    if snapshot is not None:
                        self._last_snapshot = snapshot
                        self._snapshot_count += 1
                        if self.on_snapshot:
                            try:
                                self.on_snapshot(snapshot)
                            except Exception as e:
                                logger.error(
                                    "[%s] on_snapshot callback error: %s",
                                    self.exchange_name, e,
                                )
                except Exception as e:
                    logger.error("[%s] Parse error: %s", self.exchange_name, e)
                    self._error_count += 1
        except Exception as e:
            logger.warning("[%s] WebSocket closed: %s", self.exchange_name, e)
        finally:
            if self._ws:
                await self._ws.close()

    async def _run_loop(self):
        """Reconnection loop with exponential backoff."""
        while self._running:
            try:
                await self._connect()
                await self._handle_messages()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[%s] Connection error: %s", self.exchange_name, e)

            if not self._running:
                break

            logger.info(
                "[%s] Reconnecting in %.1fs...",
                self.exchange_name, self._reconnect_delay,
            )
            await asyncio.sleep(self._reconnect_delay)
            self._reconnect_delay = min(
                self._reconnect_delay * 2, self.max_reconnect_delay
            )

    def start(self):
        """Start the WebSocket in the current event loop (non-blocking)."""
        if self._running:
            return
        self._running = True
        try:
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(self._run_loop())
        except RuntimeError:
            self._task = asyncio.ensure_future(self._run_loop())

    async def start_async(self):
        """Start the WebSocket (awaitable)."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    def stop(self):
        """Stop the WebSocket."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    @property
    def last_snapshot(self) -> Optional[OrderbookSnapshot]:
        return self._last_snapshot

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and self._ws.open

    @property
    def stats(self) -> Dict:
        return {
            "exchange": self.exchange_name,
            "symbol": self.symbol,
            "connected": self.is_connected,
            "snapshots_received": self._snapshot_count,
            "errors": self._error_count,
            "last_message_age_s": (
                time.time() - self._last_message_time
                if self._last_message_time > 0
                else None
            ),
        }


class OrderbookConsolidator:
    """
    Merges orderbook snapshots from multiple exchanges into one
    consolidated orderbook, applying BRTI methodology rules:
    - Order size cap (default 100 BTC)
    - Removes potentially erroneous orderbooks
    """

    def __init__(self, order_size_cap: float = 100.0):
        self.order_size_cap = order_size_cap
        self._snapshots: Dict[str, OrderbookSnapshot] = {}
        self._exchange_blacklist: Dict[str, float] = {}

    def update_snapshot(self, snapshot: OrderbookSnapshot) -> None:
        """Store the latest snapshot from an exchange."""
        self._snapshots[snapshot.exchange] = snapshot

    def blacklist_exchange(self, exchange: str, until: float) -> None:
        """Temporarily exclude an exchange (e.g., anomalous data)."""
        self._exchange_blacklist[exchange] = until

    def _is_blacklisted(self, exchange: str) -> bool:
        until = self._exchange_blacklist.get(exchange, 0)
        if time.time() < until:
            return True
        self._exchange_blacklist.pop(exchange, None)
        return False

    def get_consolidated(self) -> Optional[Dict]:
        """
        Build consolidated orderbook from all active exchange snapshots.

        Returns dict with:
            bids: sorted descending by price, sizes summed across exchanges
            asks: sorted ascending by price, sizes summed across exchanges
            exchanges_used: list of exchanges contributing
            timestamp: latest snapshot timestamp
        """
        active_bids: Dict[float, float] = {}
        active_asks: Dict[float, float] = {}
        exchanges_used = []
        latest_ts = 0.0

        for exchange, snap in self._snapshots.items():
            if self._is_blacklisted(exchange):
                continue
            if time.time() - snap.timestamp > 5.0:
                continue

            exchanges_used.append(exchange)
            latest_ts = max(latest_ts, snap.timestamp)

            for level in snap.bids:
                capped_size = min(level.size, self.order_size_cap)
                price_key = round(level.price, 2)
                active_bids[price_key] = active_bids.get(price_key, 0) + capped_size

            for level in snap.asks:
                capped_size = min(level.size, self.order_size_cap)
                price_key = round(level.price, 2)
                active_asks[price_key] = active_asks.get(price_key, 0) + capped_size

        if not active_bids or not active_asks:
            return None

        sorted_bids = sorted(
            [{"price": p, "size": s} for p, s in active_bids.items()],
            key=lambda x: x["price"],
            reverse=True,
        )
        sorted_asks = sorted(
            [{"price": p, "size": s} for p, s in active_asks.items()],
            key=lambda x: x["price"],
        )

        return {
            "bids": sorted_bids,
            "asks": sorted_asks,
            "exchanges_used": exchanges_used,
            "timestamp": latest_ts,
        }
