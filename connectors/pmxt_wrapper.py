"""
PMXT Integration — unified API wrapper for Polymarket + Kalshi.

PMXT (github.com/pmxt-dev/pmxt) provides a single interface to:
  - Query markets across both platforms
  - Place orders on either platform
  - Read orderbooks
  - Track positions

This wrapper adds:
  - Price normalization across platforms
  - Cross-platform market matching
  - Unified order interface with slippage modeling
  - Paper trading mode

If PMXT is not installed, falls back to native connectors.
"""
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from config.settings import settings

logger = logging.getLogger(__name__)

# Try to import PMXT
PMXT_AVAILABLE = False
try:
    # PMXT is installed from github.com/pmxt-dev/pmxt
    # The exact import path depends on the package structure
    # This is a placeholder — actual import will be set up after PMXT install
    PMXT_AVAILABLE = False
    logger.info("PMXT not installed — using native connector fallback")
except ImportError:
    logger.info("PMXT not installed — using native connector fallback")


@dataclass
class UnifiedMarket:
    """Normalized market representation across platforms."""
    platform: str  # "polymarket" or "kalshi"
    market_id: str
    question: str
    category: str
    yes_price: float
    no_price: float
    volume_24h: float
    end_date: str
    active: bool
    liquidity: float = 0.0
    spread_bps: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "platform": self.platform,
            "market_id": self.market_id,
            "question": self.question,
            "yes_price": self.yes_price,
            "no_price": self.no_price,
            "volume_24h": self.volume_24h,
            "end_date": self.end_date,
            "active": self.active,
        }


@dataclass
class UnifiedOrder:
    """Unified order representation."""
    platform: str
    market_id: str
    side: str  # "YES" or "NO"
    price: float
    size_usd: float
    order_type: str = "limit"  # "limit" or "market"


@dataclass
class UnifiedFill:
    """Unified fill/execution result."""
    platform: str
    market_id: str
    side: str
    requested_price: float
    filled_price: float
    size_usd: float
    fee_usd: float
    order_id: str
    timestamp: float

    def to_dict(self) -> Dict:
        return {
            "platform": self.platform,
            "market_id": self.market_id,
            "side": self.side,
            "filled_price": self.filled_price,
            "size_usd": self.size_usd,
            "fee_usd": self.fee_usd,
            "order_id": self.order_id,
        }


class PMXTWrapper:
    """
    Unified wrapper for Polymarket + Kalshi via PMXT.

    Provides a single interface for:
    - Market discovery and querying
    - Price feeds
    - Order placement
    - Position management

    Falls back to native connectors if PMXT is unavailable.
    """

    def __init__(
        self,
        paper_mode: bool = True,
        slippage_pct: float = 0.5,
    ):
        self.paper_mode = paper_mode or (settings.trading_mode == "paper")
        self.slippage_pct = slippage_pct
        self._pmxt_client = None
        self._pm_connector = None
        self._kalshi_client = None

        self._init_connectors()

    def _init_connectors(self) -> None:
        """Initialize PMXT or fallback connectors."""
        if PMXT_AVAILABLE:
            try:
                self._init_pmxt()
                logger.info("PMXT client initialized")
            except Exception as e:
                logger.warning("PMXT init failed, falling back: %s", e)
                self._init_fallback()
        else:
            self._init_fallback()

    def _init_pmxt(self) -> None:
        """Initialize PMXT client."""
        # Placeholder — actual PMXT initialization
        # self._pmxt_client = pmxt.Client(
        #     pm_api_key=settings.polymarket_api_key,
        #     kalshi_api_key=settings.kalshi_api_key,
        # )
        pass

    def _init_fallback(self) -> None:
        """Initialize native connectors as fallback."""
        try:
            from connectors.polymarket_connector import PolymarketConnector
            self._pm_connector = PolymarketConnector()
            logger.info("Polymarket native connector initialized")
        except Exception as e:
            logger.warning("Polymarket connector init failed: %s", e)

    # ── Market Discovery ───────────────────────────────────────────────

    async def get_crypto_markets(self) -> List[UnifiedMarket]:
        """Get all active crypto prediction markets from both platforms."""
        markets = []

        # Polymarket
        pm_markets = await self._get_pm_markets()
        markets.extend(pm_markets)

        # Kalshi
        kalshi_markets = await self._get_kalshi_markets()
        markets.extend(kalshi_markets)

        return markets

    async def _get_pm_markets(self) -> List[UnifiedMarket]:
        """Get Polymarket crypto markets."""
        if self._pmxt_client:
            return await self._pmxt_get_pm_markets()
        elif self._pm_connector:
            return await self._native_pm_markets()
        return []

    async def _native_pm_markets(self) -> List[UnifiedMarket]:
        """Get PM markets using native connector."""
        try:
            # Use Gamma API for market discovery
            import requests
            resp = requests.get(
                f"{settings.gamma_api_host}/markets",
                params={"active": True, "closed": False, "limit": 100},
                timeout=10,
            )
            if resp.status_code != 200:
                return []

            markets = []
            for item in resp.json():
                question = item.get("question", "").lower()
                if not any(kw in question for kw in ["bitcoin", "btc", "crypto"]):
                    continue

                markets.append(UnifiedMarket(
                    platform="polymarket",
                    market_id=item.get("condition_id", ""),
                    question=item.get("question", ""),
                    category="crypto",
                    yes_price=float(item.get("outcome_prices", "[0.5,0.5]").strip("[]").split(",")[0]),
                    no_price=float(item.get("outcome_prices", "[0.5,0.5]").strip("[]").split(",")[1]),
                    volume_24h=float(item.get("volume_num_24hr", 0)),
                    end_date=item.get("end_date_iso", ""),
                    active=True,
                ))
            return markets
        except Exception as e:
            logger.error("Failed to get PM markets: %s", e)
            return []

    async def _get_kalshi_markets(self) -> List[UnifiedMarket]:
        """Get Kalshi crypto markets."""
        if self._pmxt_client:
            return await self._pmxt_get_kalshi_markets()
        return await self._native_kalshi_markets()

    async def _native_kalshi_markets(self) -> List[UnifiedMarket]:
        """Get Kalshi markets using native API."""
        try:
            import requests
            resp = requests.get(
                f"{settings.kalshi_api_base}/markets",
                params={"series_ticker": "BTC", "limit": 100},
                headers={"Authorization": f"Bearer {settings.kalshi_api_key}"},
                timeout=10,
            )
            if resp.status_code != 200:
                return []

            markets = []
            for item in resp.json().get("markets", []):
                markets.append(UnifiedMarket(
                    platform="kalshi",
                    market_id=item.get("ticker", ""),
                    question=item.get("title", ""),
                    category="crypto",
                    yes_price=float(item.get("yes_ask", 50)) / 100,
                    no_price=float(item.get("no_ask", 50)) / 100,
                    volume_24h=float(item.get("volume", 0)),
                    end_date=item.get("close_time", ""),
                    active=item.get("status", "") == "open",
                ))
            return markets
        except Exception as e:
            logger.error("Failed to get Kalshi markets: %s", e)
            return []

    async def _pmxt_get_pm_markets(self) -> List[UnifiedMarket]:
        """Get PM markets via PMXT."""
        # Placeholder for PMXT API call
        return []

    async def _pmxt_get_kalshi_markets(self) -> List[UnifiedMarket]:
        """Get Kalshi markets via PMXT."""
        # Placeholder for PMXT API call
        return []

    # ── Price Feeds ────────────────────────────────────────────────────

    async def get_price(self, platform: str, market_id: str) -> Optional[Tuple[float, float]]:
        """Get (yes_price, no_price) for a market."""
        if platform == "polymarket":
            return await self._get_pm_price(market_id)
        elif platform == "kalshi":
            return await self._get_kalshi_price(market_id)
        return None

    async def _get_pm_price(self, market_id: str) -> Optional[Tuple[float, float]]:
        """Get PM price from orderbook."""
        try:
            import requests
            resp = requests.get(
                f"{settings.polymarket_host}/book",
                params={"token_id": market_id},
                timeout=5,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            if bids and asks:
                best_bid = float(bids[0]["price"])
                best_ask = float(asks[0]["price"])
                mid = (best_bid + best_ask) / 2
                return (mid, 1 - mid)
        except Exception as e:
            logger.error("PM price fetch failed: %s", e)
        return None

    async def _get_kalshi_price(self, market_id: str) -> Optional[Tuple[float, float]]:
        """Get Kalshi price."""
        try:
            import requests
            resp = requests.get(
                f"{settings.kalshi_api_base}/markets/{market_id}",
                headers={"Authorization": f"Bearer {settings.kalshi_api_key}"},
                timeout=5,
            )
            if resp.status_code != 200:
                return None
            data = resp.json().get("market", {})
            yes_price = float(data.get("yes_bid", 50)) / 100
            no_price = float(data.get("no_bid", 50)) / 100
            return (yes_price, no_price)
        except Exception as e:
            logger.error("Kalshi price fetch failed: %s", e)
        return None

    # ── Order Execution ────────────────────────────────────────────────

    async def place_order(self, order: UnifiedOrder) -> Optional[UnifiedFill]:
        """Place an order on the specified platform."""
        if self.paper_mode:
            return self._simulate_fill(order)

        if order.platform == "polymarket":
            return await self._place_pm_order(order)
        elif order.platform == "kalshi":
            return await self._place_kalshi_order(order)
        return None

    def _simulate_fill(self, order: UnifiedOrder) -> UnifiedFill:
        """Simulate order fill with slippage."""
        if order.side == "YES":
            filled_price = order.price * (1 + self.slippage_pct / 100)
        else:
            filled_price = order.price * (1 - self.slippage_pct / 100)
        filled_price = max(0.01, min(0.99, filled_price))

        fee = order.size_usd * 0.001  # 0.1% simulated fee

        return UnifiedFill(
            platform=order.platform,
            market_id=order.market_id,
            side=order.side,
            requested_price=order.price,
            filled_price=filled_price,
            size_usd=order.size_usd,
            fee_usd=fee,
            order_id=f"sim_{int(time.time() * 1000)}",
            timestamp=time.time(),
        )

    async def _place_pm_order(self, order: UnifiedOrder) -> Optional[UnifiedFill]:
        """Place order on Polymarket."""
        try:
            if self._pmxt_client:
                return await self._pmxt_place_pm_order(order)
            elif self._pm_connector:
                result = self._pm_connector.place_order(
                    token_id=order.market_id,
                    side="BUY" if order.side == "YES" else "SELL",
                    price=order.price,
                    size=order.size_usd / order.price if order.price > 0 else 0,
                )
                return UnifiedFill(
                    platform="polymarket",
                    market_id=order.market_id,
                    side=order.side,
                    requested_price=order.price,
                    filled_price=result.get("filled_price", order.price),
                    size_usd=order.size_usd,
                    fee_usd=0.0,
                    order_id=result.get("order_id", ""),
                    timestamp=time.time(),
                )
        except Exception as e:
            logger.error("PM order failed: %s", e)
        return None

    async def _place_kalshi_order(self, order: UnifiedOrder) -> Optional[UnifiedFill]:
        """Place order on Kalshi."""
        try:
            if self._pmxt_client:
                return await self._pmxt_place_kalshi_order(order)
            # Native Kalshi order placement
            import requests
            payload = {
                "ticker": order.market_id,
                "action": "buy" if order.side == "YES" else "sell",
                "type": "limit" if order.order_type == "limit" else "market",
                "count": int(order.size_usd / order.price) if order.price > 0 else 0,
                "price": int(order.price * 100),  # Kalshi uses cents
            }
            resp = requests.post(
                f"{settings.kalshi_api_base}/portfolio/orders",
                json=payload,
                headers={"Authorization": f"Bearer {settings.kalshi_api_key}"},
                timeout=10,
            )
            if resp.status_code in (200, 201):
                data = resp.json().get("order", {})
                return UnifiedFill(
                    platform="kalshi",
                    market_id=order.market_id,
                    side=order.side,
                    requested_price=order.price,
                    filled_price=float(data.get("price", order.price)) / 100,
                    size_usd=order.size_usd,
                    fee_usd=0.0,
                    order_id=data.get("order_id", ""),
                    timestamp=time.time(),
                )
        except Exception as e:
            logger.error("Kalshi order failed: %s", e)
        return None

    async def _pmxt_place_pm_order(self, order: UnifiedOrder) -> Optional[UnifiedFill]:
        """Place PM order via PMXT."""
        # Placeholder
        return None

    async def _pmxt_place_kalshi_order(self, order: UnifiedOrder) -> Optional[UnifiedFill]:
        """Place Kalshi order via PMXT."""
        # Placeholder
        return None

    # ── Cross-Platform Matching ────────────────────────────────────────

    def find_arb_markets(self, pm_markets: List[UnifiedMarket], kalshi_markets: List[UnifiedMarket]) -> List[Dict]:
        """
        Match Polymarket and Kalshi markets for arbitrage.

        Returns list of matched market pairs:
        [{"pm_market": ..., "kalshi_market": ..., "similarity": float}]
        """
        matches = []
        for pm in pm_markets:
            pm_words = set(pm.question.lower().split())
            for kalshi in kalshi_markets:
                kalshi_words = set(kalshi.question.lower().split())
                overlap = len(pm_words & kalshi_words)
                total = len(pm_words | kalshi_words)
                if total > 0 and overlap / total > 0.3:
                    matches.append({
                        "pm_market": pm,
                        "kalshi_market": kalshi,
                        "similarity": overlap / total,
                    })
        return matches
