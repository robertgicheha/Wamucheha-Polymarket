"""
Polymarket WebSocket connector for real-time orderbook and market data.

Provides millisecond-latency orderbook updates via WebSocket, which is
critical for 5-minute crypto markets where every second counts.

Based on the architecture from KaustubhPatange/polymarket-trade-engine:
- Subscribe to orderbook WebSocket for real-time bid/ask updates
- Subscribe to market WebSocket for price and trade updates
- Track multiple 5-minute market windows simultaneously
- Feed data into the strategy layer with zero polling overhead

Polymarket WebSocket endpoints:
- Orderbook: wss://ws-subscriptions-clob.polymarket.com/ws/market
- Price: via CLOB REST API (prices-history)

The hybrid CLOB design means:
- Off-chain matching (sub-second)
- On-chain settlement (Polygon)
- WebSocket gives us the off-chain view in real-time
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import websockets

from config.settings import settings

logger = logging.getLogger(__name__)

# Polymarket WebSocket URLs
POLYMARKET_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
POLYMARKET_WS_BOOK_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/book"


@dataclass
class OrderbookLevel:
    price: float
    size: float


@dataclass
class OrderbookSnapshot:
    """Real-time orderbook snapshot from Polymarket."""
    token_id: str
    bids: List[OrderbookLevel]
    asks: List[OrderbookLevel]
    timestamp: float
    market_id: str = ""

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
        return sum(l.price * l.size for l in self.bids[:levels])

    def ask_depth(self, levels: int = 5) -> float:
        return sum(l.price * l.size for l in self.asks[:levels])

    def bid_size_total(self, levels: int = 5) -> float:
        return sum(l.size for l in self.bids[:levels])

    def ask_size_total(self, levels: int = 5) -> float:
        return sum(l.size for l in self.asks[:levels])

    def imbalance(self) -> float:
        """Order book imbalance: positive = more bids, negative = more asks."""
        total_bid = self.bid_size_total(10)
        total_ask = self.ask_size_total(10)
        total = total_bid + total_ask
        if total == 0:
            return 0.0
        return (total_bid - total_ask) / total

    def to_dict(self) -> Dict:
        return {
            "token_id": self.token_id,
            "market_id": self.market_id,
            "bids": [{"price": l.price, "size": l.size} for l in self.bids],
            "asks": [{"price": l.price, "size": l.size} for l in self.asks],
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "mid_price": self.mid_price,
            "spread": self.spread,
            "spread_bps": self.spread_bps,
            "imbalance": self.imbalance(),
            "timestamp": self.timestamp,
        }


@dataclass
class TradeEvent:
    """A trade event from the Polymarket WebSocket."""
    token_id: str
    side: str  # "BUY" or "SELL"
    price: float
    size: float
    timestamp: float
    market_id: str = ""


@dataclass
class MarketState:
    """Tracks the state of a 5-minute market window."""
    condition_id: str
    token_id_yes: str
    token_id_no: str
    question: str
    price_to_beat: float
    start_time: float
    end_time: float
    current_yes_price: float = 0.5
    current_no_price: float = 0.5
    last_orderbook: Optional[OrderbookSnapshot] = None
    trades: List[TradeEvent] = field(default_factory=list)

    @property
    def time_remaining(self) -> float:
        return max(0.0, self.end_time - time.time())

    @property
    def time_elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def is_active(self) -> bool:
        now = time.time()
        return self.start_time <= now <= self.end_time

    @property
    def is_upcoming(self) -> bool:
        return time.time() < self.start_time

    @property
    def is_expired(self) -> bool:
        return time.time() > self.end_time


class PolymarketWebSocket:
    """
    Real-time WebSocket connection to Polymarket's CLOB.

    Subscribes to:
    1. Orderbook updates for specific token IDs (bids + asks)
    2. Trade events (when trades happen on the book)
    3. Market price updates

    Key design decisions from the article:
    - Minimize API calls, rely on WebSocket wherever possible
    - Order book is the source of truth for trading
    - Real-time updates give edge over REST polling
    - Track multiple market windows simultaneously
    """

    def __init__(
        self,
        on_orderbook: Optional[Callable[[OrderbookSnapshot], None]] = None,
        on_trade: Optional[Callable[[TradeEvent], None]] = None,
        on_price_update: Optional[Callable[[str, float, float], None]] = None,
    ):
        self.on_orderbook = on_orderbook
        self.on_trade = on_trade
        self.on_price_update = on_price_update

        self._ws = None
        self._running = False
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0
        self._subscriptions: List[str] = []
        self._market_states: Dict[str, MarketState] = {}
        self._last_snapshot: Optional[OrderbookSnapshot] = None
        self._snapshot_count = 0
        self._error_count = 0
        self._task: Optional[asyncio.Task] = None

    # ── Connection lifecycle ───────────────────────────────────────────

    async def _connect(self):
        """Establish WebSocket connection to Polymarket CLOB."""
        logger.info("Connecting to Polymarket WebSocket: %s", POLYMARKET_WS_URL)
        self._ws = await websockets.connect(
            POLYMARKET_WS_URL,
            ping_interval=30,
            ping_timeout=10,
            close_timeout=5,
            max_size=10 * 1024 * 1024,
        )
        self._reconnect_delay = 1.0
        logger.info("Connected to Polymarket WebSocket")

        # Re-subscribe to all tokens after reconnect
        for sub in self._subscriptions:
            await self._ws.send(json.dumps(sub))

    async def _handle_messages(self):
        """Main message loop — receives and dispatches orderbook updates."""
        try:
            async for raw in self._ws:
                try:
                    data = json.loads(raw)
                    self._dispatch_message(data)
                except json.JSONDecodeError:
                    logger.debug("Non-JSON message received")
                except Exception as e:
                    logger.error("Message handling error: %s", e)
                    self._error_count += 1
        except websockets.ConnectionClosed as e:
            logger.warning("Polymarket WebSocket closed: %s", e)
        except Exception as e:
            logger.warning("Polymarket WebSocket error: %s", e)
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
                logger.error("Polymarket WS connection error: %s", e)

            if not self._running:
                break

            logger.info("Reconnecting in %.1fs...", self._reconnect_delay)
            await asyncio.sleep(self._reconnect_delay)
            self._reconnect_delay = min(
                self._reconnect_delay * 2, self._max_reconnect_delay
            )

    def _dispatch_message(self, data: Dict):
        """Dispatch incoming WebSocket message to the appropriate handler."""
        msg_type = data.get("type", "")
        event_type = data.get("event", "")

        # Orderbook update
        if msg_type == "book" or event_type == "book":
            snapshot = self._parse_orderbook(data)
            if snapshot:
                self._last_snapshot = snapshot
                self._snapshot_count += 1
                if self.on_orderbook:
                    try:
                        self.on_orderbook(snapshot)
                    except Exception as e:
                        logger.error("on_orderbook callback error: %s", e)

        # Trade event
        elif msg_type == "trade" or event_type == "trade":
            trade = self._parse_trade(data)
            if trade and self.on_trade:
                try:
                    self.on_trade(trade)
                except Exception as e:
                    logger.error("on_trade callback error: %s", e)

        # Price update
        elif msg_type == "price" or event_type == "price":
            token_id = data.get("market", data.get("token_id", ""))
            yes_price = float(data.get("yes_price", data.get("price", 0.5)))
            no_price = float(data.get("no_price", 1.0 - yes_price))
            if self.on_price_update and token_id:
                try:
                    self.on_price_update(token_id, yes_price, no_price)
                except Exception as e:
                    logger.error("on_price_update callback error: %s", e)

    def _parse_orderbook(self, data: Dict) -> Optional[OrderbookSnapshot]:
        """Parse a raw WebSocket message into an OrderbookSnapshot."""
        try:
            asset_id = data.get("asset_id", data.get("token_id", ""))
            market_id = data.get("market", "")

            bids = []
            for b in data.get("bids", data.get("change", [])):
                if isinstance(b, dict):
                    price = float(b.get("price", b.get("p", 0)))
                    size = float(b.get("size", b.get("s", 0)))
                    if price > 0:
                        bids.append(OrderbookLevel(price=price, size=size))

            asks = []
            for a in data.get("asks", []):
                if isinstance(a, dict):
                    price = float(a.get("price", a.get("p", 0)))
                    size = float(a.get("size", a.get("s", 0)))
                    if price > 0:
                        asks.append(OrderbookLevel(price=price, size=size))

            # Sort: bids descending, asks ascending
            bids.sort(key=lambda x: x.price, reverse=True)
            asks.sort(key=lambda x: x.price)

            if not bids and not asks:
                return None

            return OrderbookSnapshot(
                token_id=asset_id,
                bids=bids,
                asks=asks,
                timestamp=time.time(),
                market_id=market_id,
            )
        except Exception as e:
            logger.error("Orderbook parse error: %s", e)
            return None

    def _parse_trade(self, data: Dict) -> Optional[TradeEvent]:
        """Parse a trade event from the WebSocket."""
        try:
            return TradeEvent(
                token_id=data.get("asset_id", data.get("token_id", "")),
                side=data.get("side", "BUY"),
                price=float(data.get("price", 0)),
                size=float(data.get("size", 0)),
                timestamp=time.time(),
                market_id=data.get("market", ""),
            )
        except Exception:
            return None

    # ── Public API ─────────────────────────────────────────────────────

    async def start_async(self):
        """Start the WebSocket connection."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    def stop(self):
        """Stop the WebSocket connection."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    async def subscribe_orderbook(self, token_ids: List[str]):
        """Subscribe to orderbook updates for specific token IDs."""
        if not self._ws or not self._ws.open:
            logger.warning("WebSocket not connected, queueing subscriptions")
            for tid in token_ids:
                self._subscriptions.append({
                    "type": "subscribe",
                    "channel": "book",
                    "assets_ids": [tid],
                })
            return

        msg = {
            "type": "subscribe",
            "channel": "book",
            "assets_ids": token_ids,
        }
        await self._ws.send(json.dumps(msg))
        logger.info("Subscribed to orderbook for %d tokens", len(token_ids))

    async def subscribe_trades(self, token_ids: List[str]):
        """Subscribe to trade events for specific token IDs."""
        if not self._ws or not self._ws.open:
            for tid in token_ids:
                self._subscriptions.append({
                    "type": "subscribe",
                    "channel": "trades",
                    "assets_ids": [tid],
                })
            return

        msg = {
            "type": "subscribe",
            "channel": "trades",
            "assets_ids": token_ids,
        }
        await self._ws.send(json.dumps(msg))
        logger.info("Subscribed to trades for %d tokens", len(token_ids))

    async def subscribe_prices(self, token_ids: List[str]):
        """Subscribe to price updates for specific token IDs."""
        if not self._ws or not self._ws.open:
            for tid in token_ids:
                self._subscriptions.append({
                    "type": "subscribe",
                    "channel": "price",
                    "assets_ids": [tid],
                })
            return

        msg = {
            "type": "subscribe",
            "channel": "price",
            "assets_ids": token_ids,
        }
        await self._ws.send(json.dumps(msg))

    async def unsubscribe(self, token_ids: List[str]):
        """Unsubscribe from updates for specific token IDs."""
        if self._ws and self._ws.open:
            msg = {
                "type": "unsubscribe",
                "assets_ids": token_ids,
            }
            await self._ws.send(json.dumps(msg))

    # ── Market state tracking ──────────────────────────────────────────

    def register_market(self, market: MarketState):
        """Register a 5-minute market window for tracking."""
        self._market_states[market.condition_id] = market

    def update_market_price(self, condition_id: str, yes_price: float, no_price: float):
        """Update current prices for a tracked market."""
        market = self._market_states.get(condition_id)
        if market:
            market.current_yes_price = yes_price
            market.current_no_price = no_price

    def get_active_markets(self) -> List[MarketState]:
        """Get all currently active (within 5-min window) markets."""
        return [m for m in self._market_states.values() if m.is_active]

    def get_upcoming_markets(self) -> List[MarketState]:
        """Get upcoming markets (not yet started)."""
        return [m for m in self._market_states.values() if m.is_upcoming]

    def get_expired_markets(self) -> List[MarketState]:
        """Get expired markets (past resolution)."""
        return [m for m in self._market_states.values() if m.is_expired]

    def cleanup_expired(self, max_age_seconds: float = 600):
        """Remove expired markets older than max_age_seconds."""
        now = time.time()
        to_remove = []
        for cid, market in self._market_states.items():
            if market.is_expired and (now - market.end_time) > max_age_seconds:
                to_remove.append(cid)
        for cid in to_remove:
            del self._market_states[cid]

    # ── Stats ──────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and self._ws.open

    @property
    def last_snapshot(self) -> Optional[OrderbookSnapshot]:
        return self._last_snapshot

    @property
    def stats(self) -> Dict:
        return {
            "connected": self.is_connected,
            "snapshots_received": self._snapshot_count,
            "errors": self._error_count,
            "active_markets": len(self.get_active_markets()),
            "tracked_markets": len(self._market_states),
            "subscriptions": len(self._subscriptions),
        }
