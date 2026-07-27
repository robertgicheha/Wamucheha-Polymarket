"""
Data aggregation layer. Provides a unified interface for fetching and
normalizing data from all sources: OHLCV, funding rates, news, and
on-chain data.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from config.settings import settings
from connectors.binance_connector import BinanceConnector
from connectors.okx_connector import OKXConnector

logger = logging.getLogger(__name__)

# Map Polymarket crypto market keywords to exchange symbols
# Includes Gold (XAU) as a crypto-adjacent commodity tracked on prediction markets
CRYPTO_SYMBOL_MAP = {
    "bitcoin": {"binance": "BTCUSDT", "okx": "BTC-USDT", "okx_swap": "BTC-USDT-SWAP"},
    "btc": {"binance": "BTCUSDT", "okx": "BTC-USDT", "okx_swap": "BTC-USDT-SWAP"},
    "ethereum": {"binance": "ETHUSDT", "okx": "ETH-USDT", "okx_swap": "ETH-USDT-SWAP"},
    "eth": {"binance": "ETHUSDT", "okx": "ETH-USDT", "okx_swap": "ETH-USDT-SWAP"},
    "solana": {"binance": "SOLUSDT", "okx": "SOL-USDT", "okx_swap": "SOL-USDT-SWAP"},
    "sol": {"binance": "SOLUSDT", "okx": "SOL-USDT", "okx_swap": "SOL-USDT-SWAP"},
    "xrp": {"binance": "XRPUSDT", "okx": "XRP-USDT", "okx_swap": "XRP-USDT-SWAP"},
    "ripple": {"binance": "XRPUSDT", "okx": "XRP-USDT", "okx_swap": "XRP-USDT-SWAP"},
    "dogecoin": {"binance": "DOGEUSDT", "okx": "DOGE-USDT", "okx_swap": "DOGE-USDT-SWAP"},
    "doge": {"binance": "DOGEUSDT", "okx": "DOGE-USDT", "okx_swap": "DOGE-USDT-SWAP"},
    "gold": {"binance": "PAXGUSDT", "okx": "PAXG-USDT", "okx_swap": "PAXG-USDT-SWAP"},
    "xau": {"binance": "PAXGUSDT", "okx": "PAXG-USDT", "okx_swap": "PAXG-USDT-SWAP"},
}


class DataAggregator:
    """Unified data fetching and normalization across all sources."""

    def __init__(self):
        self.binance = BinanceConnector()
        self.okx = OKXConnector()
        self._news_session = requests.Session()

    def detect_asset(self, question: str) -> Optional[str]:
        """Detect which crypto asset a market question refers to."""
        q_lower = question.lower()
        for keyword, symbols in CRYPTO_SYMBOL_MAP.items():
            if keyword in q_lower:
                return keyword
        return None

    # ── OHLCV Data ────────────────────────────────────────────────────

    def get_ohlcv(
        self,
        asset: str,
        interval: str = "1h",
        limit: int = 200,
        source: str = "binance",
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Binance or OKX, returned as a normalized DataFrame."""
        symbols = CRYPTO_SYMBOL_MAP.get(asset)
        if not symbols:
            logger.warning("Unknown asset: %s", asset)
            return pd.DataFrame()

        if source == "binance":
            candles = self.binance.get_ohlcv(symbols["binance"], interval, limit)
            if not candles:
                # Fallback to OKX
                candles = self.okx.get_ohlcv(symbols["okx"], interval, limit)
                source = "okx"
        else:
            candles = self.okx.get_ohlcv(symbols["okx"], interval, limit)
            if not candles:
                candles = self.binance.get_ohlcv(symbols["binance"], interval, limit)
                source = "binance"

        if not candles:
            return pd.DataFrame()

        df = pd.DataFrame(candles)

        # Normalize column names
        ts_col = "ts" if "ts" in df.columns else "open_time"
        df["timestamp"] = pd.to_datetime(df[ts_col], unit="ms")
        df = df.set_index("timestamp")
        df = df.sort_index()

        # Ensure standard columns
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns and col in df.columns:
                df[col] = df[col]

        return df

    def get_multi_timeframe_ohlcv(
        self, asset: str, timeframes: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV data across multiple timeframes."""
        if timeframes is None:
            timeframes = ["15m", "1h", "4h", "1d"]

        result = {}
        for tf in timeframes:
            result[tf] = self.get_ohlcv(asset, tf, limit=200)
        return result

    # ── Funding Rates ─────────────────────────────────────────────────

    def get_funding_rates(
        self, asset: str, limit: int = 100, source: str = "binance"
    ) -> pd.DataFrame:
        """Fetch funding rate history from Binance or OKX."""
        symbols = CRYPTO_SYMBOL_MAP.get(asset)
        if not symbols:
            return pd.DataFrame()

        if source == "binance":
            rates = self.binance.get_funding_rate_history(
                symbols["binance"], limit
            )
        else:
            rates = self.okx.get_funding_rate_history(
                symbols["okx_swap"], limit
            )

        if not rates:
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["timestamp"] = pd.to_datetime(df["funding_time"], unit="ms")
        df = df.set_index("timestamp").sort_index()
        return df

    def get_current_funding_rate(self, asset: str) -> Dict:
        """Get current funding rate from both exchanges for cross-validation."""
        symbols = CRYPTO_SYMBOL_MAP.get(asset)
        if not symbols:
            return {}

        binance_rate = self.binance.get_current_funding_rate(symbols["binance"])
        okx_rate = self.okx.get_current_funding_rate(symbols["okx_swap"])

        return {
            "binance": binance_rate,
            "okx": okx_rate,
            "cross_exchange_agreement": (
                self._rates_agree(binance_rate, okx_rate)
                if binance_rate and okx_rate
                else None
            ),
        }

    def _rates_agree(self, rate1: Optional[Dict], rate2: Optional[Dict]) -> bool:
        """Check if two funding rates agree in direction."""
        if not rate1 or not rate2:
            return False
        r1 = rate1.get("funding_rate", 0)
        r2 = rate2.get("funding_rate", 0)
        return (r1 > 0 and r2 > 0) or (r1 < 0 and r2 < 0) or (abs(r1) < 0.0001 and abs(r2) < 0.0001)

    # ── News / Sentiment ──────────────────────────────────────────────

    def get_news(
        self,
        query: str,
        days_back: int = 3,
        page_size: int = 20,
    ) -> List[Dict]:
        """Fetch news articles from NewsAPI."""
        if not settings.newsapi_key:
            return []

        from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        try:
            resp = self._news_session.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "from": from_date,
                    "sortBy": "relevancy",
                    "pageSize": page_size,
                    "apiKey": settings.newsapi_key,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                {
                    "title": a.get("title", ""),
                    "description": a.get("description", ""),
                    "content": a.get("content", ""),
                    "source": a.get("source", {}).get("name", ""),
                    "published_at": a.get("publishedAt", ""),
                    "url": a.get("url", ""),
                }
                for a in data.get("articles", [])
            ]
        except requests.RequestException as e:
            logger.error("NewsAPI request failed: %s", e)
            return []

    def get_headlines_for_market(self, question: str) -> List[str]:
        """Extract relevant headlines for a market question."""
        # Build search query from the market question
        keywords = [
            w for w in question.split()
            if len(w) > 3 and w.lower() not in {
                "will", "the", "this", "that", "have", "been", "from",
                "with", "your", "than", "what", "when", "where", "which",
                "there", "their", "about", "would", "could", "should",
            }
        ]
        query = " ".join(keywords[:5])
        articles = self.get_news(query, days_back=3, page_size=15)
        headlines = []
        for a in articles:
            title = a.get("title", "")
            desc = a.get("description", "")
            if title:
                headlines.append(title)
            if desc and desc != title:
                headlines.append(desc)
        return headlines

    # ── Cross-exchange ticker ──────────────────────────────────────────

    def get_ticker(self, asset: str) -> Dict:
        """Get current price and 24h stats from both exchanges."""
        symbols = CRYPTO_SYMBOL_MAP.get(asset)
        if not symbols:
            return {}

        binance_ticker = self.binance.get_ticker(symbols["binance"])
        okx_ticker = self.okx.get_ticker(symbols["okx"])

        return {
            "binance": binance_ticker,
            "okx": okx_ticker,
            "price_disagreement_bps": self._price_disagreement(
                binance_ticker, okx_ticker
            ),
        }

    def _price_disagreement(
        self, ticker1: Optional[Dict], ticker2: Optional[Dict]
    ) -> Optional[float]:
        """Calculate price disagreement between exchanges in basis points."""
        if not ticker1 or not ticker2:
            return None
        p1 = ticker1.get("last", 0)
        p2 = ticker2.get("last", 0)
        if p1 == 0 or p2 == 0:
            return None
        return abs(p1 - p2) / ((p1 + p2) / 2) * 10000

    # ── Sentiment from multiple sources ────────────────────────────────

    def get_aggregated_sentiment(self, question: str) -> Dict:
        """Get aggregated sentiment from news + market signals."""
        headlines = self.get_headlines_for_market(question)

        # Also check GDELT if enabled
        gdelt_articles = []
        if settings.gdelt_enabled:
            gdelt_articles = self._query_gdelt(question)

        all_texts = headlines + [a.get("title", "") for a in gdelt_articles]
        return {
            "headlines": headlines[:10],
            "gdelt_count": len(gdelt_articles),
            "total_sources": len(all_texts),
            "texts_for_sentiment": all_texts[:30],
        }

    def _query_gdelt(self, query: str) -> List[Dict]:
        """Query GDELT for recent articles."""
        try:
            resp = self._news_session.get(
                "https://api.gdelt.org/api/v2/doc/search",
                params={
                    "query": query,
                    "mode": "ArtList",
                    "maxrecords": 10,
                    "format": "json",
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("articles", [])
        except Exception:
            return []
