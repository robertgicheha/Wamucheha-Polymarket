"""
The Graph subgraph connector for Polymarket on-chain trade history.

IMPORTANT: Polymarket migrated to new CTF Exchange contracts on 2026-04-28.
The old subgraph indexer stopped returning complete data after that date.
This connector handles both endpoints:
  - Post-migration: GRAPH_SUBGRAPH_URL (current subgraph)
  - Pre-migration: GRAPH_SUBGRAPH_URL_LEGACY (historical data before 2026-04-28)

For data spanning the migration, both endpoints are queried and results merged.
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests

from config.settings import settings

logger = logging.getLogger(__name__)


# GraphQL queries
TRADES_QUERY_POST_MIGRATION = """
query GetTrades($first: Int!, $skip: Int!, $orderBy: String, $orderDirection: String,
                 $where: Trade_filter) {
  trades(first: $first, skip: $skip, orderBy: $orderBy, orderDirection: $orderDirection,
         where: $where) {
    id
    timestamp
    maker
    taker
    tokenId
    makerAmount
    takerAmount
    side
    transactionHash
    blockNumber
  }
}
"""

TRADES_QUERY_LEGACY = """
query GetTrades($first: Int!, $skip: Int!, $orderBy: String, $orderDirection: String,
                 $where: Trade_filter) {
  trades(first: $first, skip: $skip, orderBy: $orderBy, orderDirection: $orderDirection,
         where: $where) {
    id
    timestamp
    maker
    taker
    asset
    collateralAmount
    outcomeIndex
    makerAmount
    takerAmount
    side
    transactionHash
    blockNumber
  }
}
"""

MARKETS_QUERY = """
query GetMarkets($first: Int!, $skip: Int!, $where: Market_filter) {
  markets(first: $first, skip: $skip, where: $where) {
    id
    conditionId
    question
    endDate
    active
    closed
    volume
    liquidity
    outcomes
    outcomePrices
  }
}
"""

EVENTS_QUERY = """
query GetEvents($first: Int!, $skip: Int!, $where: Event_filter) {
  events(first: $first, skip: $skip, where: $where) {
    id
    title
    description
    startDate
    endDate
    active
    markets {
      id
      conditionId
      question
      volume
      liquidity
    }
  }
}
"""

POSITIONS_QUERY = """
query GetPositions($first: Int!, $skip: Int!, $where: Position_filter) {
  positions(first: $first, skip: $skip, where: $where) {
    id
    user
    conditionId
    size
    payout
    initialRecordedUpazila
  }
}
"""


class TheGraphConnector:
    def __init__(self):
        self.subgraph_url = settings.graph_subgraph_url
        self.subgraph_url_legacy = settings.graph_subgraph_url_legacy
        self.migration_date_str = settings.graph_migration_date
        self.migration_date = datetime.fromisoformat(self.migration_date_str)
        self._session = requests.Session()

    def _query(
        self,
        url: str,
        query: str,
        variables: Optional[Dict] = None,
        timeout: int = 30,
    ) -> Optional[Dict]:
        """Execute a GraphQL query against a subgraph endpoint."""
        if not url:
            return None
        try:
            resp = self._session.post(
                url,
                json={"query": query, "variables": variables or {}},
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            if "errors" in data:
                logger.error("GraphQL errors: %s", data["errors"])
                return None
            return data.get("data")
        except requests.RequestException as e:
            logger.error("GraphQL request failed to %s: %s", url, e)
            return None

    def _determine_endpoint(self, timestamp: Optional[int] = None) -> str:
        """Choose the correct subgraph endpoint based on the migration date."""
        if timestamp is not None:
            trade_date = datetime.utcfromtimestamp(timestamp)
            if trade_date < self.migration_date:
                return self.subgraph_url_legacy
        return self.subgraph_url

    # ── Trades ─────────────────────────────────────────────────────────

    def get_trades(
        self,
        maker: Optional[str] = None,
        condition_id: Optional[str] = None,
        token_id: Optional[str] = None,
        start_timestamp: Optional[int] = None,
        end_timestamp: Optional[int] = None,
        limit: int = 1000,
        use_migration_aware: bool = True,
    ) -> List[Dict]:
        """
        Fetch trades from the subgraph. If use_migration_aware is True and
        the time range spans the migration date, queries both endpoints.
        """
        if use_migration_aware and self._spans_migration(start_timestamp, end_timestamp):
            pre_trades = self._get_trades_from_endpoint(
                self.subgraph_url_legacy,
                TRADES_QUERY_LEGACY,
                maker, condition_id, token_id,
                start_timestamp,
                int(self.migration_date.timestamp()),
                limit,
            )
            post_trades = self._get_trades_from_endpoint(
                self.subgraph_url,
                TRADES_QUERY_POST_MIGRATION,
                maker, condition_id, token_id,
                int(self.migration_date.timestamp()),
                end_timestamp,
                limit,
            )
            all_trades = pre_trades + post_trades
            all_trades.sort(key=lambda t: t.get("timestamp", 0))
            return all_trades
        else:
            endpoint = self._determine_endpoint(start_timestamp)
            query = (
                TRADES_QUERY_LEGACY
                if endpoint == self.subgraph_url_legacy
                else TRADES_QUERY_POST_MIGRATION
            )
            return self._get_trades_from_endpoint(
                endpoint, query, maker, condition_id, token_id,
                start_timestamp, end_timestamp, limit,
            )

    def _get_trades_from_endpoint(
        self,
        url: str,
        query: str,
        maker: Optional[str],
        condition_id: Optional[str],
        token_id: Optional[str],
        start_timestamp: Optional[int],
        end_timestamp: Optional[int],
        limit: int,
    ) -> List[Dict]:
        """Paginated trade fetch from a single subgraph endpoint."""
        where_clauses = []
        if maker:
            where_clauses.append(f'maker: "{maker}"')
        if start_timestamp:
            where_clauses.append(f"timestamp_gte: {start_timestamp}")
        if end_timestamp:
            where_clauses.append(f"timestamp_lte: {end_timestamp}")
        if token_id:
            where_clauses.append(f'tokenId: "{token_id}"')

        where_str = ", ".join(where_clauses) if where_clauses else ""
        where_filter = f"{{{where_str}}}" if where_str else "null"

        all_trades = []
        skip = 0
        page_size = min(limit, 1000)

        while skip < limit:
            variables = {
                "first": page_size,
                "skip": skip,
                "orderBy": "timestamp",
                "orderDirection": "asc",
                "where": where_filter,
            }
            data = self._query(url, query, variables)
            if not data or "trades" not in data:
                break
            trades = data["trades"]
            if not trades:
                break
            all_trades.extend(trades)
            skip += page_size
            if len(trades) < page_size:
                break

        return all_trades[:limit]

    def _spans_migration(
        self, start: Optional[int], end: Optional[int]
    ) -> bool:
        """Check if a time range spans the migration date."""
        migration_ts = int(self.migration_date.timestamp())
        if start is None and end is None:
            return True  # unknown range, try both
        if start is None:
            start = 0
        if end is None:
            end = int(datetime.utcnow().timestamp())
        return start < migration_ts and end > migration_ts

    # ── Markets ────────────────────────────────────────────────────────

    def get_markets(
        self, active_only: bool = True, limit: int = 100
    ) -> List[Dict]:
        """Fetch markets from the subgraph."""
        where = "{active: true}" if active_only else None
        variables = {"first": limit, "skip": 0, "where": where}
        data = self._query(self.subgraph_url, MARKETS_QUERY, variables)
        if data and "markets" in data:
            return data["markets"]
        return []

    # ── Events ─────────────────────────────────────────────────────────

    def get_events(self, active_only: bool = True, limit: int = 100) -> List[Dict]:
        """Fetch events from the subgraph."""
        where = "{active: true}" if active_only else None
        variables = {"first": limit, "skip": 0, "where": where}
        data = self._query(self.subgraph_url, EVENTS_QUERY, variables)
        if data and "events" in data:
            return data["events"]
        return []

    # ── Positions ──────────────────────────────────────────────────────

    def get_positions(self, user_address: str, limit: int = 100) -> List[Dict]:
        """Fetch positions for a specific wallet address."""
        where = '{user: "%s"}' % user_address
        variables = {"first": limit, "skip": 0, "where": where}
        data = self._query(self.subgraph_url, POSITIONS_QUERY, variables)
        if data and "positions" in data:
            return data["positions"]
        return []

    # ── Aggregation helpers ────────────────────────────────────────────

    def get_trade_volume_by_day(
        self,
        condition_id: Optional[str] = None,
        days: int = 30,
    ) -> List[Dict]:
        """Aggregate trade volume by day for a market or overall."""
        now = int(datetime.utcnow().timestamp())
        start = now - (days * 86400)
        trades = self.get_trades(
            condition_id=condition_id,
            start_timestamp=start,
            end_timestamp=now,
            limit=10000,
        )

        daily = {}
        for t in trades:
            ts = t.get("timestamp", 0)
            day = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
            amount = float(t.get("makerAmount", 0)) + float(t.get("takerAmount", 0))
            if day not in daily:
                daily[day] = {"date": day, "volume": 0, "trade_count": 0}
            daily[day]["volume"] += amount
            daily[day]["trade_count"] += 1

        return sorted(daily.values(), key=lambda d: d["date"])

    def get_market_activity(
        self, condition_id: str, days: int = 7
    ) -> Dict:
        """Get activity summary for a specific market over N days."""
        now = int(datetime.utcnow().timestamp())
        start = now - (days * 86400)
        trades = self.get_trades(
            condition_id=condition_id,
            start_timestamp=start,
            end_timestamp=now,
            limit=5000,
        )
        if not trades:
            return {"condition_id": condition_id, "total_volume": 0, "trade_count": 0}

        total_volume = sum(
            float(t.get("makerAmount", 0)) + float(t.get("takerAmount", 0))
            for t in trades
        )
        unique_makers = len(set(t.get("maker", "") for t in trades))
        unique_takers = len(set(t.get("taker", "") for t in trades))

        return {
            "condition_id": condition_id,
            "total_volume": total_volume,
            "trade_count": len(trades),
            "unique_makers": unique_makers,
            "unique_takers": unique_takers,
            "avg_trade_size": total_volume / len(trades) if trades else 0,
        }
