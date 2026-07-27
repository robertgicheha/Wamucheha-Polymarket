"""
Polymarket market parsing utilities. Extracts structured information
from market questions: asset, strike price, resolution date, etc.
"""
import re
from datetime import datetime
from typing import Dict, Optional, Tuple


def parse_crypto_market(question: str) -> Dict:
    """
    Parse a crypto price market question to extract:
    - asset: the cryptocurrency (btc, eth, sol, etc.)
    - direction: "above" or "below"
    - strike_price: the threshold price
    - resolution_date: when the market resolves
    """
    q = question.lower().strip()

    asset = "unknown"
    asset_map = {
        "bitcoin": "btc", "btc": "btc",
        "ethereum": "eth", "eth": "eth",
        "solana": "sol", "sol": "sol",
        "xrp": "xrp", "ripple": "xrp",
        "dogecoin": "doge", "doge": "doge",
        "cardano": "ada", "ada": "ada",
        "polkadot": "dot", "dot": "dot",
        "avalanche": "avax", "avax": "avax",
        "chainlink": "link", "link": "link",
        "litecoin": "ltc", "ltc": "ltc",
        "gold": "gold", "xau": "gold",
    }
    for keyword, symbol in asset_map.items():
        if keyword in q:
            asset = symbol
            break

    direction = "above"
    if "below" in q or "under" in q or "drop below" in q or "fall below" in q:
        direction = "below"
    elif "above" in q or "over" in q or "rise above" in q or "go above" in q:
        direction = "above"

    strike_price = 0.0
    price_patterns = [
        r"\$[\d,]+(?:\.\d+)?",           # $120,000 or $120.50
        r"[\d,]+(?:\.\d+)?\s*(?:dollars|usd)", # 120000 dollars
        r"(?:above|below|over|under|at)\s+\$?([\d,]+(?:\.\d+)?)",
    ]
    for pattern in price_patterns:
        match = re.search(pattern, q)
        if match:
            price_str = match.group(0) if match.lastindex is None else match.group(1)
            price_str = re.sub(r"[^\d.]", "", price_str)
            try:
                strike_price = float(price_str)
                break
            except ValueError:
                continue

    resolution_date = None
    date_patterns = [
        (r"(?:by|before|on|until)\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})", "%B %d %Y"),
        (r"(?:by|before|on|until)\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})", "%b %d %Y"),
        (r"(?:by|before|on|until)\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?", None),
        (r"(\d{4})-(\d{2})-(\d{2})", None),
        (r"(q[1-4]\s*\d{4})", None),
    ]
    for pattern, fmt in date_patterns:
        match = re.search(pattern, q)
        if match:
            date_str = match.group(0)
            try:
                if fmt:
                    # Remove ordinal suffixes
                    date_str = re.sub(r"(\d+)(?:st|nd|rd|th)", r"\1", date_str)
                    resolution_date = datetime.strptime(date_str, fmt)
                else:
                    # Try common formats
                    for try_fmt in ["%Y-%m-%d", "%B %d", "%b %d", "%Y-Q%q"]:
                        try:
                            resolution_date = datetime.strptime(date_str, try_fmt)
                            break
                        except ValueError:
                            continue
                if resolution_date:
                    if resolution_date.year == 1900:  # no year in match
                        resolution_date = resolution_date.replace(year=datetime.now().year)
                    break
            except ValueError:
                continue

    return {
        "asset": asset,
        "direction": direction,
        "strike_price": strike_price,
        "resolution_date": resolution_date,
        "original_question": question,
    }



