"""
Market data feed for crypto "Up or Down" windows: discovers the current and
upcoming windows by slug and fetches both outcome books in one request.
Consumed by FiveMinuteLifecycleEngine.tick().
"""
import logging
import time
from typing import Dict, List, Optional

from connectors.polymarket_connector import BookTop, FeeParams, PolymarketConnector, UpDownMarket

logger = logging.getLogger(__name__)


class UpDownMarketFeed:
    def __init__(
        self,
        connector: Optional[PolymarketConnector] = None,
        assets: Optional[List[str]] = None,
        interval_minutes: int = 5,
        lookahead_windows: int = 1,
        discovery_interval_seconds: float = 30.0,
    ):
        self.connector = connector or PolymarketConnector()
        self.assets = [a.lower() for a in (assets or ["btc"])]
        self.interval_seconds = interval_minutes * 60
        self.interval_minutes = interval_minutes
        self.lookahead_windows = lookahead_windows
        self.discovery_interval_seconds = discovery_interval_seconds
        self._known: Dict[str, UpDownMarket] = {}  # slug -> market
        self._last_discovery = 0.0

    def discover(self, now: Optional[float] = None) -> List[UpDownMarket]:
        """
        Returns newly found windows (current + `lookahead_windows` ahead).
        Rate-limited to one sweep per `discovery_interval_seconds`.
        """
        now = now or time.time()
        if now - self._last_discovery < self.discovery_interval_seconds:
            return []
        self._last_discovery = now
        current_start = int(now) - int(now) % self.interval_seconds
        new = []
        for asset in self.assets:
            for k in range(self.lookahead_windows + 1):
                start = current_start + k * self.interval_seconds
                slug = f"{asset}-updown-{self.interval_minutes}m-{start}"
                if slug in self._known:
                    continue
                market = self.connector.get_updown_market(asset, self.interval_minutes, start)
                if market is None or not market.condition_id:
                    continue
                self._known[slug] = market
                new.append(market)
        # Forget windows that ended long ago.
        cutoff = now - 3 * self.interval_seconds
        self._known = {s: m for s, m in self._known.items() if m.end_time > cutoff}
        return new

    def get_books(self, token_ids: List[str]) -> Dict[str, BookTop]:
        return self.connector.get_books(token_ids)

    def get_fee_params(self, condition_id: str) -> FeeParams:
        return self.connector.get_fee_params(condition_id)

    def get_resolution(self, slug: str) -> Optional[str]:
        return self.connector.get_updown_resolution(slug)
