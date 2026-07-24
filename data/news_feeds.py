"""
News feed aggregation and processing. Handles NewsAPI, GDELT, and
RSS feeds. Provides structured text for sentiment analysis.
"""
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from html import unescape

import requests

from config.settings import settings

logger = logging.getLogger(__name__)

# Common RSS feeds for crypto/politics/macro news
RSS_FEEDS = {
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cointelegraph": "https://cointelegraph.com/rss",
    "decrypt": "https://decrypt.co/feed",
    "theblock": "https://www.theblock.co/rss.xml",
    "reuters_business": "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
    "bloomberg_markets": "https://feeds.bloomberg.com/markets/news.rss",
    "fed_calendar": "https://www.federalreserve.gov/feeds/press_all.xml",
}


def _clean_html(text: str) -> str:
    """Strip HTML tags and decode entities."""
    clean = unescape(text)
    clean = re.sub(r"<[^>]+>", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


class NewsFeedAggregator:
    """Aggregate and process news from multiple sources."""

    def __init__(self):
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "PolymarketBot/1.0"

    def get_newsapi_articles(
        self,
        query: str,
        days_back: int = 3,
        page_size: int = 20,
        language: str = "en",
    ) -> List[Dict]:
        """Fetch articles from NewsAPI."""
        if not settings.newsapi_key:
            return []

        from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        try:
            resp = self._session.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "from": from_date,
                    "sortBy": "relevancy",
                    "pageSize": page_size,
                    "language": language,
                    "apiKey": settings.newsapi_key,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                {
                    "title": _clean_html(a.get("title", "")),
                    "description": _clean_html(a.get("description", "")),
                    "content": _clean_html(a.get("content", "")),
                    "source": a.get("source", {}).get("name", ""),
                    "published_at": a.get("publishedAt", ""),
                    "url": a.get("url", ""),
                    "sentiment_score": 0.0,
                }
                for a in data.get("articles", [])
                if a.get("title")
            ]
        except requests.RequestException as e:
            logger.error("NewsAPI request failed: %s", e)
            return []

    def get_gdelt_articles(self, query: str, max_records: int = 20) -> List[Dict]:
        """Fetch articles from GDELT Event API."""
        if not settings.gdelt_enabled:
            return []

        try:
            resp = self._session.get(
                "https://api.gdelt.org/api/v2/doc/search",
                params={
                    "query": query,
                    "mode": "ArtList",
                    "maxrecords": max_records,
                    "format": "json",
                    "sort": "DateDesc",
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            articles = []
            for a in data.get("articles", []):
                articles.append({
                    "title": _clean_html(a.get("title", "")),
                    "description": _clean_html(a.get("socialimage", "")),
                    "content": "",
                    "source": a.get("domain", ""),
                    "published_at": a.get("seendate", ""),
                    "url": a.get("url", ""),
                    "language": a.get("language", ""),
                    "tone": float(a.get("tone", 0)),
                    "sentiment_score": float(a.get("tone", 0)) / 10,
                })
            return articles
        except Exception as e:
            logger.error("GDELT request failed: %s", e)
            return []

    def get_rss_feed(self, feed_name: str, max_items: int = 20) -> List[Dict]:
        """Fetch and parse an RSS feed."""
        url = RSS_FEEDS.get(feed_name)
        if not url:
            logger.warning("Unknown RSS feed: %s", feed_name)
            return []

        try:
            import xml.etree.ElementTree as ET

            resp = self._session.get(url, timeout=15)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)

            items = []
            for item in root.iter("item"):
                title = item.findtext("title", "")
                desc = item.findtext("description", "")
                pub_date = item.findtext("pubDate", "")
                link = item.findtext("link", "")

                if title:
                    items.append({
                        "title": _clean_html(title),
                        "description": _clean_html(desc),
                        "content": "",
                        "source": feed_name,
                        "published_at": pub_date,
                        "url": link,
                        "sentiment_score": 0.0,
                    })

                if len(items) >= max_items:
                    break

            return items
        except Exception as e:
            logger.error("RSS feed %s failed: %s", feed_name, e)
            return []

    def get_all_rss_feeds(self, max_per_feed: int = 10) -> List[Dict]:
        """Fetch from all configured RSS feeds."""
        all_articles = []
        for feed_name in RSS_FEEDS:
            articles = self.get_rss_feed(feed_name, max_per_feed)
            all_articles.extend(articles)
        return all_articles

    def get_market_relevant_news(
        self, question: str, category: str = "crypto"
    ) -> List[Dict]:
        """
        Get news articles relevant to a specific market question.
        Uses intelligent query construction based on category.
        """
        # Build a focused search query
        stop_words = {
            "will", "the", "this", "that", "have", "been", "from",
            "with", "your", "than", "what", "when", "where", "which",
            "there", "their", "about", "would", "could", "should",
            "above", "below", "between", "during", "before", "after",
        }
        keywords = [
            w for w in question.split()
            if len(w) > 2 and w.lower() not in stop_words
        ][:6]
        query = " ".join(keywords)

        articles = []

        # NewsAPI
        newsapi = self.get_newsapi_articles(query, days_back=3, page_size=15)
        articles.extend(newsapi)

        # GDELT
        gdelt = self.get_gdelt_articles(query, max_records=10)
        articles.extend(gdelt)

        # Category-specific RSS feeds
        if category == "crypto":
            for feed in ["coindesk", "cointelegraph", "decrypt", "theblock"]:
                rss = self.get_rss_feed(feed, max_items=5)
                articles.extend(rss)
        elif category in ("politics", "macro"):
            for feed in ["reuters_business", "bloomberg_markets", "fed_calendar"]:
                rss = self.get_rss_feed(feed, max_items=5)
                articles.extend(rss)

        # Deduplicate by title similarity
        seen_titles = set()
        unique = []
        for a in articles:
            title_key = a["title"][:50].lower()
            if title_key not in seen_titles and title_key:
                seen_titles.add(title_key)
                unique.append(a)

        return unique

    def get_crypto_news(self, symbol: str = "bitcoin") -> List[Dict]:
        """Get crypto-specific news for an asset."""
        queries = [symbol, f"{symbol} price", f"{symbol} market"]
        articles = []
        for q in queries:
            articles.extend(self.get_newsapi_articles(q, days_back=1, page_size=10))

        # Add crypto RSS
        for feed in ["coindesk", "cointelegraph", "decrypt"]:
            articles.extend(self.get_rss_feed(feed, max_items=5))

        # Deduplicate
        seen = set()
        unique = []
        for a in articles:
            key = a["title"][:50].lower()
            if key not in seen and key:
                seen.add(key)
                unique.append(a)
        return unique

    def get_politics_news(self, topic: str = "") -> List[Dict]:
        """Get politics/current events news."""
        query = topic or "politics election government"
        articles = self.get_newsapi_articles(query, days_back=2, page_size=20)
        for feed in ["reuters_business"]:
            articles.extend(self.get_rss_feed(feed, max_items=10))
        return articles

    def get_macro_news(self, topic: str = "") -> List[Dict]:
        """Get macro economics news."""
        query = topic or "federal reserve interest rate inflation economy"
        articles = self.get_newsapi_articles(query, days_back=2, page_size=20)
        for feed in ["reuters_business", "bloomberg_markets", "fed_calendar"]:
            articles.extend(self.get_rss_feed(feed, max_items=10))
        return articles
