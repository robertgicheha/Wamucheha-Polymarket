"""
Polymarket CLOB connector. Wraps order placement, orderbook reads, market
metadata via the Gamma API, and CLOB trading via py-clob-client.

Gamma API (read-only, no auth): market discovery, event listing, price history.
CLOB API (auth required): order placement, orderbook, trade execution.

Install: pip install py-clob-client
Docs: https://docs.polymarket.com/
"""
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

from config.settings import settings

logger = logging.getLogger(__name__)

# Category keyword mapping for market filtering — crypto + gold only.
# Sports, politics, macro categories removed entirely.
CATEGORY_KEYWORDS = {
    "crypto": [
        "bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "sol",
        "xrp", "ripple", "dogecoin", "doge", "price", "blockchain", "defi",
        "token", "coin", "mining", "halving", "gold", "xau",
        "above", "below", "up", "down", "beat",
    ],
}


def _classify_market(question: str) -> str:
    """Classify a market question into a category based on keyword matching."""
    q_lower = question.lower()
    scores = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        scores[cat] = sum(1 for kw in keywords if kw in q_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "uncategorized"


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
class OrderResult:
    order_id: str
    status: str
    filled_size: float
    avg_price: float


@dataclass
class PricePoint:
    timestamp: int
    price: float


class PolymarketConnector:
    def __init__(self):
        self.host = settings.polymarket_host
        self.gamma_host = settings.gamma_api_host
        self._client = None
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    def _get_client(self):
        """Lazily initialize the CLOB client (requires auth)."""
        if self._client is None:
            from py_clob_client.client import ClobClient
            self._client = ClobClient(
                host=self.host,
                key=settings.polymarket_private_key,
                chain_id=137,
                creds={
                    "apiKey": settings.polymarket_api_key,
                    "secret": settings.polymarket_api_secret,
                    "passphrase": settings.polymarket_api_passphrase,
                },
            )
        return self._client

    # ── Gamma API (read-only, no auth) ────────────────────────────────

    def _gamma_get(self, path: str, params: Optional[Dict] = None) -> list | dict:
        url = f"{self.gamma_host}{path}"
        try:
            resp = self._session.get(url, params=params or {}, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error("Gamma API request failed: %s %s — %s", url, params, e)
            return [] if isinstance(params, dict) else {}

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
            question = m.get("question", "")
            market_category = _classify_market(question)
            if category and market_category != category:
                continue

            outcomes = m.get("outcomes", "")
            outcome_prices = m.get("outcomePrices", "")
            yes_price = 0.5
            no_price = 0.5
            if outcome_prices:
                try:
                    prices = [float(p) for p in outcome_prices.strip("[]").split(",")]
                    if len(prices) >= 2:
                        yes_price = prices[0]
                        no_price = prices[1]
                except (ValueError, IndexError):
                    pass

            token_ids = m.get("clobTokenIds", "")
            token_id_yes = ""
            token_id_no = ""
            if token_ids:
                try:
                    ids = [x.strip().strip('"') for x in token_ids.strip("[]").split(",")]
                    if len(ids) >= 2:
                        token_id_yes = ids[0]
                        token_id_no = ids[1]
                except (ValueError, IndexError):
                    pass

            markets.append(Market(
                condition_id=m.get("conditionId", m.get("condition_id", "")),
                question=question,
                category=market_category,
                yes_price=yes_price,
                no_price=no_price,
                volume_24h=float(m.get("volume24hr", 0) or 0),
                end_date=m.get("endDate", ""),
                token_id_yes=token_id_yes,
                token_id_no=token_id_no,
                event_id=m.get("eventId", m.get("event_id", "")),
                description=m.get("description", ""),
                active=m.get("active", True),
                closed=m.get("closed", False),
                outcomes=outcomes,
                outcome_prices=outcome_prices,
            ))
        return markets

    def get_market_by_condition(self, condition_id: str) -> Optional[Market]:
        """Fetch a single market by condition ID."""
        result = self._gamma_get(f"/markets/{condition_id}")
        if not result or not isinstance(result, dict):
            return None
        markets = self.get_markets(limit=1)
        # Gamma might not have a direct single-market endpoint, search instead
        params = {"conditionId": condition_id}
        raw = self._gamma_get("/markets", params)
        if isinstance(raw, list) and raw:
            m = raw[0]
            return Market(
                condition_id=condition_id,
                question=m.get("question", ""),
                category=_classify_market(m.get("question", "")),
                yes_price=0.5,
                no_price=0.5,
                volume_24h=float(m.get("volume24hr", 0) or 0),
                end_date=m.get("endDate", ""),
            )
        return None

    def get_resolved_markets(
        self, category: Optional[str] = None, limit: int = 100
    ) -> List[Market]:
        """Fetch closed/resolved markets for backtesting."""
        return self.get_markets(category=category, limit=limit, active=False, closed=True)

    # ── CLOB API (authenticated) ──────────────────────────────────────

    def get_orderbook(self, token_id: str) -> dict:
        """Fetch the orderbook for a given token ID."""
        try:
            client = self._get_client()
            book = client.get_order_book(token_id)
            return book
        except Exception as e:
            logger.error("Failed to fetch orderbook for %s: %s", token_id, e)
            return {"bids": [], "asks": []}

    def get_price_history(
        self, token_id: str, interval: str = "max"
    ) -> List[PricePoint]:
        """
        Fetch price history for a token. For resolved markets, this often only
        returns 12h+ granularity. For live markets, finer granularity is available.
        """
        try:
            resp = self._session.get(
                f"{self.host}/prices-history",
                params={"market": token_id, "interval": interval},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            history = data.get("history", [])
            return [
                PricePoint(timestamp=int(p.get("t", 0)), price=float(p.get("p", 0)))
                for p in history
            ]
        except requests.RequestException as e:
            logger.error("Price history request failed for %s: %s", token_id, e)
            return []

    def place_order(
        self, token_id: str, side: str, price: float, size: float
    ) -> OrderResult:
        """
        Build, sign, and submit an order via py-clob-client.
        In paper mode this should never be called.
        """
        if settings.trading_mode != "live":
            raise RuntimeError(
                "place_order called while TRADING_MODE != 'live' — use PaperBroker instead"
            )
        try:
            from py_clob_client.clob_types import OrderArgs, OrderType

            client = self._get_client()
            order_side = 1 if side.upper() == "BUY" else 2
            order_args = OrderArgs(
                price=price,
                size=size,
                side=order_side,
                token_id=token_id,
            )
            signed_order = client.create_order(order_args)
            resp = client.post_order(signed_order, OrderType.GTC)
            return OrderResult(
                order_id=resp.get("orderID", resp.get("id", "")),
                status=resp.get("status", "unknown"),
                filled_size=float(resp.get("size_matched", 0)),
                avg_price=float(resp.get("price", price)),
            )
        except Exception as e:
            logger.error("Order placement failed: %s", e)
            raise

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order by ID."""
        try:
            client = self._get_client()
            client.cancel(order_id)
            return True
        except Exception as e:
            logger.error("Cancel order failed for %s: %s", order_id, e)
            return False

    def get_positions(self) -> list:
        """Fetch current open positions for the wallet."""
        try:
            client = self._get_client()
            return client.get_positions()
        except Exception as e:
            logger.error("Failed to fetch positions: %s", e)
            return []

    def get_trades(self) -> list:
        """Fetch recent trade history for the wallet."""
        try:
            client = self._get_client()
            return client.get_trades()
        except Exception as e:
            logger.error("Failed to fetch trades: %s", e)
            return []
