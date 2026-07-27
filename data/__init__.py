import data.market_parser
from data.price_feeds import DataAggregator
from data.news_feeds import NewsFeedAggregator
from data.market_parser import (
    parse_crypto_market,
)

__all__ = [
    "DataAggregator",
    "NewsFeedAggregator",
    "parse_crypto_market"
]
