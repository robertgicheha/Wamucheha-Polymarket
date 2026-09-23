"""
Polymarket connector (CLOB V2, post 2026-04-28 exchange upgrade).

Public data (no auth) goes straight to the REST APIs:
  - Gamma API: market discovery / metadata / resolution status
  - CLOB API:  order books, fee parameters, price history

Authenticated trading uses the official `polymarket-client` SDK
(`pip install polymarket-client`), which supports every Polymarket wallet
type (Deposit Wallet, legacy Proxy / Safe, raw EOA), pUSD collateral, the V2
order struct, and gasless approvals / redemptions through the relayer. The
legacy `py-clob-client` does NOT work against production since the upgrade.

Docs: https://docs.polymarket.com/
"""
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import requests

from config.settings import settings

logger = logging.getLogger(__name__)

# Category keyword mapping for market filtering — crypto + gold only.
CATEGORY_KEYWORDS = {
    "crypto": [
        "bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "sol",
        "xrp", "ripple", "dogecoin", "doge", "price", "blockchain", "defi",
        "token", "coin", "mining", "halving", "gold", "xau",
        "above", "below", "up", "down", "beat",
    ],
}

# Defaults observed on crypto up/down markets; always refreshed per market
# from /clob-markets/{condition_id} before use.
DEFAULT_TAKER_FEE_RATE = 0.07
DEFAULT_TAKER_FEE_EXPONENT = 1.0


def _classify_market(question: str) -> str:
    """Classify a market question into a category based on keyword matching."""
    q_lower = question.lower()
    scores = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        scores[cat] = sum(1 for kw in keywords if kw in q_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "uncategorized"


def _json_list(value) -> list:
    """Gamma returns some list fields as JSON-encoded strings."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except ValueError:
            return []
    return []


def _iso_to_ts(value: str) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


@dataclass
class Market:
    condition_id: str
    question: str
    category: str
    yes_price: float
    no_price: float
    volume_24h: float
    end_date: str
    token_id_yes: str = ""
    token_id_no: str = ""
    event_id: str = ""
    description: str = ""
    active: bool = True
    closed: bool = False
    outcomes: str = ""
    outcome_prices: str = ""


@dataclass
class UpDownMarket:
    """One crypto "Up or Down" window, e.g. btc-updown-5m-1790193600."""
    condition_id: str
    slug: str
    question: str
    asset: str
    start_time: float
    end_time: float
    token_id_up: str
    token_id_down: str
    accepting_orders: bool
    min_order_size: float
    tick_size: float
    closed: bool = False

    def to_discovery_dict(self) -> Dict:
        """Shape expected by FiveMinuteLifecycleEngine.discover_market."""
        return {
            "condition_id": self.condition_id,
            "slug": self.slug,
            "question": self.question,
            "asset": self.asset,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "token_id_yes": self.token_id_up,
            "token_id_no": self.token_id_down,
            "min_order_size": self.min_order_size,
            "tick_size": self.tick_size,
        }


@dataclass
class BookTop:
    """Normalised order-book summary for one outcome token."""
    token_id: str
    best_bid: float = 0.0
    best_ask: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0
    # Levels sorted best-first: [(price, size), ...]
    bids: List[tuple] = field(default_factory=list)
    asks: List[tuple] = field(default_factory=list)
    min_order_size: float = 0.0
    tick_size: float = 0.01
    timestamp: float = 0.0

    @property
    def mid(self) -> float:
        if self.best_bid > 0 and self.best_ask > 0:
            return (self.best_bid + self.best_ask) / 2
        return self.best_bid or self.best_ask or 0.0

    @property
    def spread(self) -> float:
        if self.best_bid > 0 and self.best_ask > 0:
            return self.best_ask - self.best_bid
        return 1.0

    @property
    def imbalance(self) -> float:
        total = self.bid_size + self.ask_size
        return (self.bid_size - self.ask_size) / total if total > 0 else 0.0


@dataclass
class FeeParams:
    rate: float = DEFAULT_TAKER_FEE_RATE
    exponent: float = DEFAULT_TAKER_FEE_EXPONENT
    taker_only: bool = True


@dataclass
class PricePoint:
    timestamp: int
    price: float


def parse_book(raw: Dict) -> BookTop:
    """
    Normalise a raw CLOB /book payload. The CLOB returns bids ascending and
    asks descending, i.e. the BEST price is the LAST element on each side —
    so never take [0] as the top of book. We sort explicitly instead of
    relying on either order.
    """
    def levels(key: str) -> List[tuple]:
        out = []
        for lvl in raw.get(key) or []:
            try:
                price, size = float(lvl["price"]), float(lvl["size"])
            except (KeyError, TypeError, ValueError):
                continue
            if size > 0:
                out.append((price, size))
        return out

    bids = sorted(levels("bids"), key=lambda x: x[0], reverse=True)
    asks = sorted(levels("asks"), key=lambda x: x[0])
    try:
        ts = float(raw.get("timestamp") or 0) / 1000.0
    except (TypeError, ValueError):
        ts = 0.0
    return BookTop(
        token_id=str(raw.get("asset_id", "")),
        best_bid=bids[0][0] if bids else 0.0,
        best_ask=asks[0][0] if asks else 0.0,
        bid_size=bids[0][1] if bids else 0.0,
        ask_size=asks[0][1] if asks else 0.0,
        bids=bids,
        asks=asks,
        min_order_size=float(raw.get("min_order_size") or 0),
        tick_size=float(raw.get("tick_size") or 0.01),
        timestamp=ts or time.time(),
    )


class PolymarketConnector:
    def __init__(self):
        self.host = settings.polymarket_host.rstrip("/")
        self.gamma_host = settings.gamma_api_host.rstrip("/")
        self._secure_client = None
        self._client_lock = threading.Lock()
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})
        self._fee_cache: Dict[str, FeeParams] = {}

    # ── HTTP helpers ──────────────────────────────────────────────────

    def _get_json(self, url: str, params: Optional[Dict] = None, timeout: float = 10):
        resp = self._session.get(url, params=params or {}, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def _gamma_get(self, path: str, params: Optional[Dict] = None) -> list | dict:
        url = f"{self.gamma_host}{path}"
        try:
            return self._get_json(url, params, timeout=30)
        except requests.RequestException as e:
            logger.error("Gamma API request failed: %s %s — %s", url, params, e)
            return []

    # ── Gamma API (read-only, no auth) ────────────────────────────────

    def get_events(
        self,
        limit: int = 100,
        offset: int = 0,
        closed: bool = False,
        tag: Optional[str] = None,
    ) -> list:
        """Fetch events from the Gamma API."""
        params = {"limit": limit, "offset": offset, "closed": str(closed).lower()}
        if tag:
            params["tag"] = tag
        return self._gamma_get("/events", params)

    def get_markets(
        self,
        category: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        active: bool = True,
        closed: bool = False,
    ) -> List[Market]:
        """
        Fetch markets from the Gamma API, optionally filter by category.
        Uses keyword matching on the question text since Polymarket doesn't
        natively tag markets by category.
        """
        params = {"limit": limit, "offset": offset}
        if active:
            params["active"] = "true"
        if closed:
            params["closed"] = "true"
        raw_markets = self._gamma_get("/markets", params)
        if not isinstance(raw_markets, list):
            return []

        markets = []
        for m in raw_markets:
            market = self._to_market(m)
            if category and market.category != category:
                continue
            markets.append(market)
        return markets

    def _to_market(self, m: Dict) -> Market:
        question = m.get("question", "")
        prices = _json_list(m.get("outcomePrices"))
        token_ids = _json_list(m.get("clobTokenIds"))
        try:
            yes_price = float(prices[0]) if len(prices) >= 2 else 0.5
            no_price = float(prices[1]) if len(prices) >= 2 else 0.5
        except (TypeError, ValueError):
            yes_price, no_price = 0.5, 0.5
        return Market(
            condition_id=m.get("conditionId", m.get("condition_id", "")),
            question=question,
            category=_classify_market(question),
            yes_price=yes_price,
            no_price=no_price,
            volume_24h=float(m.get("volume24hr", 0) or 0),
            end_date=m.get("endDate", ""),
            token_id_yes=str(token_ids[0]) if len(token_ids) >= 2 else "",
            token_id_no=str(token_ids[1]) if len(token_ids) >= 2 else "",
            event_id=str(m.get("eventId", m.get("event_id", "")) or ""),
            description=m.get("description", ""),
            active=bool(m.get("active", True)),
            closed=bool(m.get("closed", False)),
            outcomes=json.dumps(_json_list(m.get("outcomes"))),
            outcome_prices=json.dumps(prices),
        )

    def get_market_by_condition(self, condition_id: str) -> Optional[Market]:
        """Fetch a single market by condition ID."""
        raw = self._gamma_get("/markets", {"condition_ids": condition_id})
        if isinstance(raw, list) and raw:
            return self._to_market(raw[0])
        return None

    def get_resolved_markets(
        self, category: Optional[str] = None, limit: int = 100
    ) -> List[Market]:
        """Fetch closed/resolved markets for backtesting."""
        return self.get_markets(category=category, limit=limit, active=False, closed=True)

    # ── Crypto up/down windows ────────────────────────────────────────

    def get_updown_market(
        self, asset: str, interval_minutes: int, window_start: int
    ) -> Optional[UpDownMarket]:
        """
        Look up one up/down window by its deterministic slug
        ({asset}-updown-{N}m-{window_start_unix}).
        """
        slug = f"{asset.lower()}-updown-{interval_minutes}m-{int(window_start)}"
        raw = self._gamma_get("/markets", {"slug": slug})
        if not isinstance(raw, list) or not raw:
            return None
        m = raw[0]
        token_ids = _json_list(m.get("clobTokenIds"))
        outcomes = [str(o).lower() for o in _json_list(m.get("outcomes"))]
        if len(token_ids) != 2 or len(outcomes) != 2:
            logger.warning("Up/down market %s has unexpected outcomes %s", slug, outcomes)
            return None
        # Map by label, not position, so an outcome reorder can never flip sides.
        try:
            up_idx = outcomes.index("up")
            down_idx = outcomes.index("down")
        except ValueError:
            logger.warning("Up/down market %s has no Up/Down outcomes: %s", slug, outcomes)
            return None
        start = _iso_to_ts(m.get("eventStartTime", "")) or float(window_start)
        end = _iso_to_ts(m.get("endDate", "")) or start + interval_minutes * 60
        return UpDownMarket(
            condition_id=m.get("conditionId", ""),
            slug=slug,
            question=m.get("question", ""),
            asset=asset.lower(),
            start_time=start,
            end_time=end,
            token_id_up=str(token_ids[up_idx]),
            token_id_down=str(token_ids[down_idx]),
            accepting_orders=bool(m.get("acceptingOrders", False)),
            min_order_size=float(m.get("orderMinSize") or 5),
            tick_size=float(m.get("orderPriceMinTickSize") or 0.01),
            closed=bool(m.get("closed", False)),
        )

    def get_updown_resolution(self, slug: str) -> Optional[str]:
        """
        Returns "UP" / "DOWN" once the market has resolved (one outcome price
        at 1), otherwise None. Gamma hides closed markets unless asked, and
        these windows typically resolve ~1 minute after they end.
        """
        raw = self._gamma_get("/markets", {"slug": slug, "closed": "true"})
        if not isinstance(raw, list) or not raw:
            return None
        m = raw[0]
        outcomes = [str(o).lower() for o in _json_list(m.get("outcomes"))]
        try:
            prices = [float(p) for p in _json_list(m.get("outcomePrices"))]
        except (TypeError, ValueError):
            return None
        if not m.get("closed") or len(prices) != 2 or len(outcomes) != 2:
            return None
        if max(prices) < 0.99:
            return None
        winner = outcomes[prices.index(max(prices))]
        return winner.upper() if winner in ("up", "down") else None

    # ── CLOB public data ──────────────────────────────────────────────

    def get_books(self, token_ids: List[str]) -> Dict[str, BookTop]:
        """Fetch several order books in one request."""
        if not token_ids:
            return {}
        try:
            resp = self._session.post(
                f"{self.host}/books",
                json=[{"token_id": t} for t in token_ids],
                timeout=5,
            )
            resp.raise_for_status()
            books = {}
            for raw in resp.json() or []:
                book = parse_book(raw)
                books[book.token_id] = book
            return books
        except (requests.RequestException, ValueError) as e:
            logger.warning("Order book fetch failed: %s", e)
            return {}

    def get_orderbook(self, token_id: str) -> Optional[BookTop]:
        """Fetch the order book for a single token ID."""
        try:
            return parse_book(self._get_json(f"{self.host}/book", {"token_id": token_id}, timeout=5))
        except (requests.RequestException, ValueError) as e:
            logger.warning("Order book fetch failed for %s: %s", token_id, e)
            return None

    def get_fee_params(self, condition_id: str) -> FeeParams:
        """Taker-fee parameters for a market, cached per condition ID."""
        cached = self._fee_cache.get(condition_id)
        if cached:
            return cached
        try:
            info = self._get_json(f"{self.host}/clob-markets/{condition_id}", timeout=5)
            fd = info.get("fd") or {}
            params = FeeParams(
                rate=float(fd.get("r", 0.0)),
                exponent=float(fd.get("e", 1.0) or 1.0),
                taker_only=bool(fd.get("to", True)),
            )
        except (requests.RequestException, ValueError, TypeError) as e:
            logger.warning(
                "Fee lookup failed for %s (%s) — assuming %.2f/%.1f",
                condition_id, e, DEFAULT_TAKER_FEE_RATE, DEFAULT_TAKER_FEE_EXPONENT,
            )
            return FeeParams()
        self._fee_cache[condition_id] = params
        return params

    def get_clob_version(self) -> Optional[int]:
        try:
            return int(self._get_json(f"{self.host}/version", timeout=5).get("version"))
        except (requests.RequestException, ValueError, TypeError):
            return None

    def get_price_history(
        self, token_id: str, interval: str = "max"
    ) -> List[PricePoint]:
        """
        Fetch price history for a token. For resolved markets, this often only
        returns 12h+ granularity. For live markets, finer granularity is available.
        """
        try:
            data = self._get_json(
                f"{self.host}/prices-history",
                {"market": token_id, "interval": interval},
                timeout=30,
            )
            return [
                PricePoint(timestamp=int(p.get("t", 0)), price=float(p.get("p", 0)))
                for p in data.get("history", [])
            ]
        except requests.RequestException as e:
            logger.error("Price history request failed for %s: %s", token_id, e)
            return []

    # ── Authenticated trading (polymarket-client SDK) ─────────────────

    def secure_client(self):
        """
        Lazily create the authenticated SDK client. API credentials are
        derived from the signing key when POLYMARKET_API_* are not set.
        """
        with self._client_lock:
            if self._secure_client is None:
                from polymarket import ApiKeyCreds, RelayerApiKey, SecureClient

                kwargs = {"private_key": _normalise_key(settings.polymarket_private_key)}
                if settings.polymarket_funder_address:
                    kwargs["wallet"] = settings.polymarket_funder_address
                if (settings.polymarket_api_key and settings.polymarket_api_secret
                        and settings.polymarket_api_passphrase):
                    kwargs["credentials"] = ApiKeyCreds(
                        key=settings.polymarket_api_key,
                        secret=settings.polymarket_api_secret,
                        passphrase=settings.polymarket_api_passphrase,
                    )
                if settings.polymarket_relayer_api_key:
                    kwargs["api_key"] = RelayerApiKey(
                        key=settings.polymarket_relayer_api_key,
                        address=settings.polymarket_relayer_api_key_address,
                    )
                self._secure_client = SecureClient.create(**kwargs)
                logger.info(
                    "Polymarket client ready: wallet=%s type=%s",
                    self._secure_client.wallet, self._secure_client.wallet_type,
                )
            return self._secure_client

    def get_collateral_balance(self) -> float:
        """pUSD available to the trading wallet, in dollars."""
        ba = self.secure_client().get_balance_allowance(asset_type="COLLATERAL")
        return ba.balance / 1e6

    def get_token_balance(self, token_id: str) -> float:
        """Outcome-token shares held by the trading wallet."""
        ba = self.secure_client().get_balance_allowance(asset_type="CONDITIONAL", token_id=token_id)
        return ba.balance / 1e6

    def buy_fak(self, token_id: str, usd_amount: float, max_price: float):
        """Taker BUY for up to `usd_amount`, never paying more than `max_price`."""
        return self.secure_client().place_market_order(
            token_id=token_id, side="BUY", amount=round(usd_amount, 2),
            max_price=max_price, order_type="FAK",
        )

    def sell_fak(self, token_id: str, shares: float, min_price: float):
        """Taker SELL of up to `shares`, never receiving less than `min_price`."""
        return self.secure_client().place_market_order(
            token_id=token_id, side="SELL", shares=shares,
            min_price=min_price, order_type="FAK",
        )

    def cancel_all(self) -> bool:
        try:
            self.secure_client().cancel_all()
            return True
        except Exception as e:
            logger.error("Cancel-all failed: %s", e)
            return False

    def trading_approvals_ready(self) -> bool:
        return bool(self.secure_client().get_trading_approvals_state().is_fully_approved)

    def list_redeemable_condition_ids(self) -> List[str]:
        client = self.secure_client()
        ids = []
        for pos in client.list_positions(status="REDEEMABLE"):
            if pos.redeemable and pos.condition_id not in ids:
                ids.append(pos.condition_id)
        return ids

    def redeem(self, condition_id: str) -> None:
        """Redeem resolved positions for a market back into pUSD."""
        handle = self.secure_client().redeem_positions(condition_id=condition_id)
        handle.wait()


def _normalise_key(key: str) -> str:
    key = (key or "").strip()
    return key if key.startswith("0x") else "0x" + key
