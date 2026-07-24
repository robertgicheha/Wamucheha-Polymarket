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


def parse_election_market(question: str) -> Dict:
    """Parse a politics/election market question."""
    q = question.lower().strip()

    candidates = []
    election_type = "unknown"
    position = "unknown"

    # Detect election type
    if any(w in q for w in ["president", "presidential", "election"]):
        election_type = "presidential"
    elif any(w in q for w in ["senate", "senatorial"]):
        election_type = "senate"
    elif any(w in q for w in ["governor"]):
        election_type = "governor"
    elif any(w in q for w in ["house", "congress", "representative"]):
        election_type = "house"

    # Try to extract candidate names (capitalized words after "will" or before "win")
    name_patterns = [
        r"will\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+win",
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+be",
    ]
    for pattern in name_patterns:
        matches = re.findall(pattern, question)
        candidates.extend(matches)

    # Resolution date
    resolution_date = None
    year_match = re.search(r"\b(20\d{2})\b", q)
    if year_match:
        resolution_date = datetime(int(year_match.group(1)), 11, 1)  # default to Nov of election year

    return {
        "election_type": election_type,
        "candidates": list(set(candidates)),
        "resolution_date": resolution_date,
        "original_question": question,
    }


def parse_sports_market(question: str) -> Dict:
    """Parse a sports market question."""
    q = question.lower().strip()

    sport = "unknown"
    sport_keywords = {
        "nba": ["nba", "basketball", "lakers", "celtics", "warriors"],
        "nfl": ["nfl", "football", "super bowl", "niners", "chiefs"],
        "mlb": ["mlb", "baseball", "yankees", "dodgers"],
        "nhl": ["nhl", "hockey", "stanley cup"],
        "soccer": ["soccer", "football", "premier league", "champions league", "world cup"],
        "tennis": ["tennis", "wimbledon", "us open", "roland garros"],
        "f1": ["f1", "formula", "grand prix"],
    }
    for s, keywords in sport_keywords.items():
        if any(kw in q for kw in keywords):
            sport = s
            break

    # Try to extract team names
    teams = []
    team_patterns = [
        r"(?:will|do)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:win|beat|defeat)",
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:vs\.?|versus)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
    ]
    for pattern in team_patterns:
        matches = re.findall(pattern, question)
        for m in matches:
            if isinstance(m, tuple):
                teams.extend(m)
            else:
                teams.append(m)

    return {
        "sport": sport,
        "teams": teams,
        "original_question": question,
    }


def parse_macro_market(question: str) -> Dict:
    """Parse a macro economics market question."""
    q = question.lower().strip()

    indicator = "unknown"
    indicator_keywords = {
        "fed_rate": ["fed", "interest rate", "fomc", "rate cut", "rate hike"],
        "cpi": ["cpi", "inflation", "consumer price"],
        "gdp": ["gdp", "economic growth", "recession"],
        "unemployment": ["unemployment", "jobs report", "nonfarm", "payroll"],
        "tariff": ["tariff", "trade war", "trade deal"],
    }
    for ind, keywords in indicator_keywords.items():
        if any(kw in q for kw in keywords):
            indicator = ind
            break

    # Extract date
    resolution_date = None
    month_map = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }
    for month_name, month_num in month_map.items():
        if month_name in q:
            year = datetime.now().year
            year_match = re.search(r"\b(20\d{2})\b", q)
            if year_match:
                year = int(year_match.group(1))
            resolution_date = datetime(year, month_num, 1)
            break

    return {
        "indicator": indicator,
        "resolution_date": resolution_date,
        "original_question": question,
    }
