"""
Paper trading broker. Trades against real, live Polymarket orderbook data but with
simulated money -- this is what TRADING_MODE=paper routes through instead of
connectors/polymarket_connector.py's place_order. Run this for 2-4 weeks minimum
before ever setting TRADING_MODE=live.

Deliberately models slippage and fees, since a strategy that only looks good
ignoring both is the single most common way retail trading bots fail in the wild.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List

from risk.risk_manager import RiskManager


@dataclass
class PaperFill:
    market_id: str
    side: str
    requested_price: float
    filled_price: float  # includes simulated slippage
    size_usd: float
    fee_usd: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class PaperBroker:
    """
    TODO: pull real orderbook depth from connectors/polymarket_connector.py
    (read-only, no auth needed for market data) to simulate realistic slippage
    based on actual book depth, rather than a flat assumed slippage %.
    """

    def __init__(self, risk_manager: RiskManager, assumed_slippage_pct: float = 0.5, fee_pct: float = 0.0):
        self.risk_manager = risk_manager
        self.assumed_slippage_pct = assumed_slippage_pct
        self.fee_pct = fee_pct  # Polymarket currently has no explicit trading fee,
                                 # but gas costs on Polygon apply -- factor those in
                                 # via a per-tx estimate once connectors are live
        self.fills: List[PaperFill] = []

    def simulate_order(self, market_id: str, category: str, side: str, price: float) -> PaperFill:
        size_usd = self.risk_manager.max_position_size(category)
        if size_usd <= 0:
            raise RuntimeError("No tradable capital available for this position")

        slippage_direction = 1 if side == "YES" else -1
        filled_price = price * (1 + slippage_direction * self.assumed_slippage_pct / 100)
        filled_price = max(0.01, min(0.99, filled_price))
        fee_usd = size_usd * (self.fee_pct / 100)

        fill = PaperFill(
            market_id=market_id,
            side=side,
            requested_price=price,
            filled_price=filled_price,
            size_usd=size_usd,
            fee_usd=fee_usd,
        )
        self.fills.append(fill)
        self.risk_manager.open_position(market_id, category, side, filled_price, size_usd)
        return fill
