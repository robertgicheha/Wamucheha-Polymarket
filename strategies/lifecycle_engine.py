"""
5-Minute Market Lifecycle Engine — the core trading engine for crypto prediction markets.

Inspired by KaustubhPatange/polymarket-trade-engine's "early-bird" architecture.
Each market is a lifecycle: start -> run -> end states.

The engine orchestrates these market lifecycles:
  1. DISCOVER: Find upcoming 5-minute markets (BTC, ETH, SOL, XRP, Gold)
  2. START: Create strategy instance, subscribe to orderbook, set price to beat
  3. RUN: Monitor orderbook, execute trades based on strategy signals
  4. END: Close positions, compute PnL, compound profits

Key insights from the article:
- Always start in a FUTURE market slot (at least one ahead of current)
- The order book is the source of truth for trading
- Minimize API calls, rely on WebSocket wherever possible
- Goal is NOT always to hold until resolution — sell when profit is appropriate
- No single strategy wins all the time — find windows where your edge exists
- Exit signals are more important than entry signals

Three-layer architecture:
  Layer 1 (Data): Polymarket WebSocket + Gamma API → real-time orderbook + market discovery
  Layer 2 (Strategy): Strategy engine processes data → trade decisions
  Layer 3 (Execution): Sign transactions, submit orders, track positions
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

from config.settings import settings

logger = logging.getLogger(__name__)


class MarketPhase(Enum):
    """Phases of a 5-minute market lifecycle."""
    DISCOVER = "discover"    # Finding upcoming markets
    START = "start"          # Setting up strategy, subscribing to feeds
    RUN = "run"              # Active trading within the 5-min window
    END = "end"              # Closing positions, computing PnL
    SETTLED = "settled"      # Market resolved, profits compounded


@dataclass
class MarketWindow:
    """Represents a single 5-minute market window."""
    condition_id: str
    question: str
    asset: str  # "btc", "eth", "sol", "xrp", "gold"
    price_to_beat: float
    token_id_yes: str
    token_id_no: str
    start_time: float  # unix timestamp
    end_time: float    # unix timestamp
    phase: MarketPhase = MarketPhase.DISCOVER

    # Current state
    current_yes_price: float = 0.5
    current_no_price: float = 0.5
    orderbook_snapshot: Optional[Dict] = None
    gap: float = 0.0  # current price - price_to_beat

    # Strategy
    strategy_name: str = ""
    strategy_signal: Optional[Dict] = None

    # Execution
    position_side: str = ""  # "YES" or "NO"
    position_size_usd: float = 0.0
    entry_price: float = 0.0
    entry_time: float = 0.0
    exit_price: float = 0.0
    exit_time: float = 0.0

    # PnL
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    won: bool = False

    @property
    def time_remaining(self) -> float:
        return max(0.0, self.end_time - time.time())

    @property
    def time_elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def progress_pct(self) -> float:
        total = self.end_time - self.start_time
        if total <= 0:
            return 1.0
        return min(1.0, self.time_elapsed / total)

    @property
    def is_active(self) -> bool:
        now = time.time()
        return self.start_time <= now <= self.end_time

    def to_dict(self) -> Dict:
        return {
            "condition_id": self.condition_id,
            "question": self.question,
            "asset": self.asset,
            "price_to_beat": self.price_to_beat,
            "current_yes_price": self.current_yes_price,
            "current_no_price": self.current_no_price,
            "gap": self.gap,
            "phase": self.phase.value,
            "time_remaining": round(self.time_remaining, 1),
            "progress_pct": round(self.progress_pct * 100, 1),
            "position_side": self.position_side,
            "position_size_usd": self.position_size_usd,
            "pnl_usd": round(self.pnl_usd, 4),
            "won": self.won,
        }


@dataclass
class EngineStats:
    """Aggregate engine statistics."""
    total_markets_observed: int = 0
    total_markets_traded: int = 0
    total_wins: int = 0
    total_losses: int = 0
    total_pnl_usd: float = 0.0
    total_volume_usd: float = 0.0
    avg_edge: float = 0.0
    win_rate: float = 0.0
    current_bankroll: float = 0.0
    peak_bankroll: float = 0.0
    uptime_seconds: float = 0.0
    markets_per_hour: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "total_markets_observed": self.total_markets_observed,
            "total_markets_traded": self.total_markets_traded,
            "total_wins": self.total_wins,
            "total_losses": self.total_losses,
            "total_pnl_usd": round(self.total_pnl_usd, 4),
            "total_volume_usd": round(self.total_volume_usd, 2),
            "win_rate": round(self.win_rate, 1),
            "current_bankroll": round(self.current_bankroll, 2),
            "uptime_hours": round(self.uptime_seconds / 3600, 1),
        }


class FiveMinuteLifecycleEngine:
    """
    The core engine that orchestrates 5-minute crypto prediction markets.

    Lifecycle for each market:
      DISCOVER → START → RUN → END → SETTLED

    The engine:
      1. Discovers upcoming 5-min markets via Gamma API
      2. Starts each market with a strategy
      3. Runs the strategy during the 5-min window using real-time orderbook
      4. Ends the market by closing positions and computing PnL
      5. Compounds profits into the bankroll

    Key design principles (from the article):
    - Always start in a FUTURE market slot
    - Order book is the source of truth
    - Minimize API calls, use WebSocket
    - Sell when profit is appropriate (don't always hold to resolution)
    - Track PnL per market for strategy evaluation
    """

    def __init__(
        self,
        bankroll: float = 100.0,
        strategy_name: str = "kelly",
        on_trade: Optional[Callable] = None,
        on_pnl: Optional[Callable] = None,
        on_notification: Optional[Callable] = None,
        arbitrage_engine=None,
        orderbook_arb=None,
        risk_manager=None,
        trade_logger=None,
    ):
        self.bankroll = bankroll
        self.initial_bankroll = bankroll
        self.strategy_name = strategy_name
        self.on_trade = on_trade
        self.on_pnl = on_pnl
        self.on_notification = on_notification
        self.arbitrage_engine = arbitrage_engine
        self.orderbook_arb = orderbook_arb
        self.risk_manager = risk_manager
        self.trade_logger = trade_logger

        self._market_windows: Dict[str, MarketWindow] = {}
        self._current_market: Optional[MarketWindow] = None
        self._next_market: Optional[MarketWindow] = None
        self._stats = EngineStats()
        self._stats.current_bankroll = bankroll
        self._stats.peak_bankroll = bankroll
        self._start_time = time.time()
        self._running = False
        self._trade_history: List[Dict] = []

        # Strategy instance
        self._strategy = None

        # Compounding settings
        self._compound_enabled = settings.reinvest_profits_only is False

    # ── Lifecycle management ───────────────────────────────────────────

    def discover_market(self, market_data: Dict) -> Optional[MarketWindow]:
        """
        Register a discovered 5-minute market from Gamma API data.
        Returns a MarketWindow if valid, None otherwise.
        """
        condition_id = market_data.get("condition_id", "")
        if not condition_id or condition_id in self._market_windows:
            return None

        # Extract asset from question
        question = market_data.get("question", "")
        asset = self._detect_asset(question)
        if asset == "unknown":
            return None

        # Parse price to beat
        price_to_beat = self._parse_price_to_beat(market_data)

        # Parse timing
        start_time = market_data.get("start_time", 0)
        end_time = market_data.get("end_time", 0)
        if start_time == 0 or end_time == 0:
            return None

        window = MarketWindow(
            condition_id=condition_id,
            question=question,
            asset=asset,
            price_to_beat=price_to_beat,
            token_id_yes=market_data.get("token_id_yes", ""),
            token_id_no=market_data.get("token_id_no", ""),
            start_time=start_time,
            end_time=end_time,
        )

        self._market_windows[condition_id] = window
        self._stats.total_markets_observed += 1

        logger.info(
            "Discovered market: %s | %s | Price to beat: $%.2f | Window: %s -> %s",
            asset.upper(), question[:60], price_to_beat,
            datetime.fromtimestamp(start_time, tz=timezone.utc).strftime("%H:%M:%S"),
            datetime.fromtimestamp(end_time, tz=timezone.utc).strftime("%H:%M:%S"),
        )

        return window

    def start_market(self, condition_id: str) -> bool:
        """
        Start a market — transition from DISCOVER to START phase.
        Creates strategy instance and prepares for trading.
        """
        window = self._market_windows.get(condition_id)
        if not window:
            return False

        if window.phase != MarketPhase.DISCOVER:
            logger.warning("Market %s already in phase %s", condition_id, window.phase)
            return False

        window.phase = MarketPhase.START
        window.strategy_name = self.strategy_name

        # Create strategy
        self._strategy = self._create_strategy(self.strategy_name)

        # Subscribe to orderbook
        if window.token_id_yes:
            logger.info(
                "Starting market %s: %s | Strategy: %s | Bankroll: $%.2f",
                condition_id[:8], window.asset.upper(),
                self.strategy_name, self.bankroll,
            )

        return True

    def run_market_tick(self, condition_id: str, orderbook: Optional[Dict] = None) -> bool:
        """
        Process a tick within an active market window.
        Called every second (or faster) with real-time orderbook data.
        """
        window = self._market_windows.get(condition_id)
        if not window or not window.is_active:
            return False

        # Transition to RUN if still in START
        if window.phase == MarketPhase.START:
            window.phase = MarketPhase.RUN

        if window.phase != MarketPhase.RUN:
            return False

        # Update orderbook
        if orderbook:
            window.orderbook_snapshot = orderbook
            window.current_yes_price = orderbook.get("mid_price", 0.5)
            window.current_no_price = 1.0 - window.current_yes_price

        # Calculate gap (current price vs price to beat)
        if window.orderbook_snapshot:
            mid = window.orderbook_snapshot.get("mid_price", 0.5)
            # Gap is the difference between current mid and 0.5 (neutral)
            window.gap = mid - 0.5

        # Get strategy signal
        if self._strategy and window.orderbook_snapshot:
            signal = self._get_strategy_signal(window)
            window.strategy_signal = signal

            # Execute trade if signal says to
            if signal and signal.get("should_trade", False):
                self._execute_trade(window, signal)

        # Check for early exit (take profit / stop loss)
        if window.position_side:
            self._check_exit_conditions(window)

        return True

    def end_market(self, condition_id: str, final_price: Optional[float] = None) -> bool:
        """
        End a market — transition from RUN to END phase.
        Close any open positions and compute PnL.
        """
        window = self._market_windows.get(condition_id)
        if not window:
            return False

        if window.phase not in (MarketPhase.RUN, MarketPhase.START):
            return False

        window.phase = MarketPhase.END

        # Close any open position at final price
        if window.position_side and window.entry_price > 0:
            if final_price is not None:
                window.exit_price = final_price
            else:
                window.exit_price = window.current_yes_price

            window.exit_time = time.time()

            # Calculate PnL
            if window.position_side == "YES":
                if final_price is not None and final_price > window.price_to_beat:
                    window.won = True
                    window.pnl_usd = window.position_size_usd * ((1.0 / window.entry_price) - 1)
                else:
                    window.won = False
                    window.pnl_usd = -window.position_size_usd
            else:  # NO side
                if final_price is not None and final_price < window.price_to_beat:
                    window.won = True
                    window.pnl_usd = window.position_size_usd * ((1.0 / (1 - window.entry_price)) - 1)
                else:
                    window.won = False
                    window.pnl_usd = -window.position_size_usd

            # Apply fees (crypto market taker fee: 7.2%)
            fee_rate = 0.072
            fee = abs(window.pnl_usd) * fee_rate if window.pnl_usd > 0 else window.position_size_usd * fee_rate * window.entry_price * (1 - window.entry_price)
            window.pnl_usd -= fee

            # Update bankroll with compounding
            self.bankroll += window.pnl_usd
            window.pnl_pct = (window.pnl_usd / window.position_size_usd * 100) if window.position_size_usd > 0 else 0

            # Log trade exit to database
            if self.trade_logger:
                try:
                    # Find the most recent trade_id for this condition_id
                    recent_trades = self.trade_logger.get_recent_trades(limit=50)
                    matching_trade = None
                    for t in recent_trades:
                        if t["condition_id"] == condition_id and t["exit_price"] is None:
                            matching_trade = t
                            break

                    if matching_trade:
                        exit_reason = "resolution"
                        if window.pnl_usd > 0 and window.time_remaining > 30:
                            exit_reason = "take_profit"
                        elif window.pnl_usd < 0 and window.time_remaining > 30:
                            exit_reason = "stop_loss"

                        self.trade_logger.log_exit(
                            trade_id=matching_trade["trade_id"],
                            exit_price=window.exit_price,
                            exit_reason=exit_reason,
                            pnl_usd=window.pnl_usd,
                            fees_usd=fee,
                            bankroll_after=self.bankroll,
                            metadata={
                                "asset": window.asset,
                                "entry_side": window.position_side,
                                "market_duration": window.end_time - window.start_time,
                            },
                        )
                except Exception as e:
                    logger.error("Trade log exit failed: %s", e)

            # Update stats
            self._stats.total_markets_traded += 1
            self._stats.total_pnl_usd += window.pnl_usd
            self._stats.total_volume_usd += window.position_size_usd
            if window.won:
                self._stats.total_wins += 1
            else:
                self._stats.total_losses += 1

            total_traded = self._stats.total_wins + self._stats.total_losses
            self._stats.win_rate = (self._stats.total_wins / total_traded * 100) if total_traded > 0 else 0

            # Track peak bankroll
            if self.bankroll > self._stats.peak_bankroll:
                self._stats.peak_bankroll = self.bankroll
            self._stats.current_bankroll = self.bankroll

            # Log trade
            trade_record = {
                "condition_id": condition_id,
                "asset": window.asset,
                "side": window.position_side,
                "entry_price": window.entry_price,
                "exit_price": window.exit_price,
                "size_usd": window.position_size_usd,
                "pnl_usd": window.pnl_usd,
                "won": window.won,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "strategy": window.strategy_name,
            }
            self._trade_history.append(trade_record)

            # Notify
            if self.on_pnl:
                self.on_pnl(trade_record)

            if self.on_notification:
                emoji = "W" if window.won else "L"
                self.on_notification(
                    f"[{emoji}] {window.asset.upper()} {window.position_side} | "
                    f"PnL: ${window.pnl_usd:+.4f} | "
                    f"Bankroll: ${self.bankroll:.2f} | "
                    f"Win Rate: {self._stats.win_rate:.1f}%"
                )

            logger.info(
                "Market END: %s %s %s | Entry: %.3f Exit: %.3f | "
                "PnL: $%.4f (%s) | Bankroll: $%.2f",
                condition_id[:8], window.asset.upper(), window.position_side,
                window.entry_price, window.exit_price,
                window.pnl_usd, "WIN" if window.won else "LOSS",
                self.bankroll,
            )

        window.phase = MarketPhase.SETTLED

        # Cleanup old markets
        self._cleanup_old_markets()

        return True

    # ── Strategy execution ─────────────────────────────────────────────

    def _create_strategy(self, name: str):
        """Create a strategy instance by name."""
        try:
            from strategies import get_strategy
            return get_strategy(name)
        except Exception as e:
            logger.error("Failed to create strategy %s: %s", name, e)
            return None

    def _get_strategy_signal(self, window: MarketWindow) -> Optional[Dict]:
        """
        Get a trading signal from the strategy for the current market.
        Uses orderbook data + market state to generate a signal.
        """
        if not self._strategy or not window.orderbook_snapshot:
            return None

        # Build extra context for the strategy
        extra = {
            "momentum": self._compute_momentum(window),
            "z_score": 0.0,
            "confidence": 0.5,
            "orderbook_imbalance": window.orderbook_snapshot.get("imbalance", 0),
            "spread_bps": window.orderbook_snapshot.get("spread_bps", 0),
            "time_remaining": window.time_remaining,
            "progress_pct": window.progress_pct,
        }

        # For 5-minute markets, the model probability is derived from
        # the orderbook mid price and the gap
        mid = window.current_yes_price
        model_prob = mid  # Use mid as base probability estimate

        # Adjust model_prob based on gap (price vs price_to_beat)
        if window.price_to_beat > 0:
            # If current price is above price_to_beat, YES is more likely
            gap_pct = (window.current_yes_price - 0.5) * 2  # normalize to [-1, 1]
            model_prob = max(0.01, min(0.99, 0.5 + gap_pct * 0.3))

        decision = self._strategy.should_trade(
            model_prob=model_prob,
            market_price=window.current_yes_price,
            bankroll=self.bankroll,
            tte_seconds=int(window.time_remaining),
            volatility=0.01,  # Will be replaced with real volatility
            extra=extra,
        )

        if decision.should_trade:
            return {
                "should_trade": True,
                "side": decision.side,
                "size_usd": decision.size_usd,
                "confidence": decision.confidence,
                "edge": decision.edge,
                "reason": decision.reason,
            }
        return {"should_trade": False}

    def _execute_trade(self, window: MarketWindow, signal: Dict):
        """Execute a trade based on strategy signal."""
        side = signal.get("side", "YES")
        size_usd = signal.get("size_usd", 0)

        if size_usd <= 0 or size_usd > self.bankroll:
            return

        # Cap position size
        max_position = self.bankroll * (settings.max_position_size_pct / 100)
        size_usd = min(size_usd, max_position)

        if side == "YES":
            entry_price = window.current_yes_price
        else:
            entry_price = window.current_no_price

        # Apply slippage
        slippage = 0.005  # 0.5% slippage
        if side == "YES":
            entry_price = min(0.99, entry_price * (1 + slippage))
        else:
            entry_price = max(0.01, entry_price * (1 - slippage))

        # Update window state
        window.position_side = side
        window.position_size_usd = size_usd
        window.entry_price = entry_price
        window.entry_time = time.time()

        # Deduct from bankroll
        self.bankroll -= size_usd

        # Log trade entry to database
        if self.trade_logger:
            try:
                self.trade_logger.log_entry(
                    condition_id=window.condition_id,
                    asset=window.asset,
                    side=side,
                    price=entry_price,
                    size_usd=size_usd,
                    strategy=self.strategy_name,
                    source="lifecycle",
                    market_question=window.question,
                    bankroll_after=self.bankroll,
                    metadata={
                        "price_to_beat": window.price_to_beat,
                        "signal_edge": signal.get("edge", 0),
                        "signal_reason": signal.get("reason", ""),
                    },
                )
            except Exception as e:
                logger.error("Trade log entry failed: %s", e)

        logger.info(
            "TRADE: %s %s @ %.3f ($%.2f) | Edge: %+.4f | %s",
            side, window.asset.upper(), entry_price, size_usd,
            signal.get("edge", 0), signal.get("reason", ""),
        )

        if self.on_trade:
            self.on_trade({
                "condition_id": window.condition_id,
                "asset": window.asset,
                "side": side,
                "entry_price": entry_price,
                "size_usd": size_usd,
                "edge": signal.get("edge", 0),
                "strategy": window.strategy_name,
            })

    def _check_exit_conditions(self, window: MarketWindow):
        """
        Check if we should exit early (take profit / stop loss).
        Key insight from the article: don't always hold to resolution.
        Sell when you have appropriate profit.
        """
        if not window.position_side or window.entry_price <= 0:
            return

        current_price = (
            window.current_yes_price if window.position_side == "YES"
            else window.current_no_price
        )

        if current_price <= 0:
            return

        # Calculate unrealized PnL %
        if window.position_side == "YES":
            unrealized_pct = (current_price - window.entry_price) / window.entry_price * 100
        else:
            unrealized_pct = (window.entry_price - current_price) / window.entry_price * 100

        # Take profit: exit if we have > 5% profit
        take_profit_pct = 5.0
        if unrealized_pct >= take_profit_pct:
            window.exit_price = current_price
            window.exit_time = time.time()
            pnl_usd = window.position_size_usd * (unrealized_pct / 100)
            window.pnl_usd = pnl_usd
            self.bankroll += window.position_size_usd + pnl_usd
            window.phase = MarketPhase.END
            logger.info("EARLY EXIT (TP): %s %s @ %.3f (+%.1f%%)", window.asset.upper(), window.position_side, current_price, unrealized_pct)
            return

        # Stop loss: exit if we have > 8% loss
        stop_loss_pct = -8.0
        if unrealized_pct <= stop_loss_pct:
            window.exit_price = current_price
            window.exit_time = time.time()
            pnl_usd = window.position_size_usd * (unrealized_pct / 100)
            window.pnl_usd = pnl_usd
            self.bankroll += window.position_size_usd + pnl_usd
            window.phase = MarketPhase.END
            logger.info("EARLY EXIT (SL): %s %s @ %.3f (%.1f%%)", window.asset.upper(), window.position_side, current_price, unrealized_pct)
            return

    # ── Helper methods ─────────────────────────────────────────────────

    def _detect_asset(self, question: str) -> str:
        """Detect which crypto asset a market question refers to."""
        q = question.lower()
        asset_keywords = {
            "btc": ["bitcoin", "btc"],
            "eth": ["ethereum", "eth"],
            "sol": ["solana", "sol"],
            "xrp": ["xrp", "ripple"],
            "gold": ["gold", "xau"],
        }
        for asset, keywords in asset_keywords.items():
            if any(kw in q for kw in keywords):
                return asset
        return "unknown"

    def _parse_price_to_beat(self, market_data: Dict) -> float:
        """Extract the price to beat from market data."""
        # Try various fields
        for field_name in ["price_to_beat", "strike_price", "target_price"]:
            val = market_data.get(field_name, 0)
            if val:
                return float(val)

        # Parse from question text
        import re
        question = market_data.get("question", "")
        match = re.search(r"\$[\d,]+(?:\.\d+)?", question)
        if match:
            price_str = match.group(0).replace("$", "").replace(",", "")
            try:
                return float(price_str)
            except ValueError:
                pass

        return 0.0

    def _compute_momentum(self, window: MarketWindow) -> float:
        """Compute price momentum from orderbook data."""
        if not window.orderbook_snapshot:
            return 0.0

        imbalance = window.orderbook_snapshot.get("imbalance", 0)
        # Positive imbalance = more bids = bullish momentum
        return imbalance * 0.1  # Scale down for strategy use

    def _cleanup_old_markets(self, max_age: float = 600):
        """Remove markets older than max_age seconds."""
        now = time.time()
        to_remove = []
        for cid, window in self._market_windows.items():
            if window.phase == MarketPhase.SETTLED:
                if (now - window.end_time) > max_age:
                    to_remove.append(cid)
        for cid in to_remove:
            del self._market_windows[cid]

    # ── Public API ─────────────────────────────────────────────────────

    @property
    def stats(self) -> EngineStats:
        self._stats.uptime_seconds = time.time() - self._start_time
        return self._stats

    @property
    def current_market(self) -> Optional[MarketWindow]:
        """Get the currently active market window."""
        for window in self._market_windows.values():
            if window.is_active and window.phase == MarketPhase.RUN:
                return window
        return None

    @property
    def next_market(self) -> Optional[MarketWindow]:
        """Get the next upcoming market window."""
        upcoming = [
            w for w in self._market_windows.values()
            if w.phase == MarketPhase.DISCOVER and w.start_time > time.time()
        ]
        if upcoming:
            return min(upcoming, key=lambda w: w.start_time)
        return None

    @property
    def trade_history(self) -> List[Dict]:
        return list(self._trade_history)

    def get_all_windows(self) -> List[Dict]:
        """Get all market windows as dicts."""
        return [w.to_dict() for w in self._market_windows.values()]

    def get_active_windows(self) -> List[MarketWindow]:
        """Get all currently active market windows."""
        return [
            w for w in self._market_windows.values()
            if w.is_active or w.phase in (MarketPhase.RUN, MarketPhase.START, MarketPhase.END)
        ]

    def get_total_pnl(self) -> float:
        """Get total PnL across all markets."""
        return self._stats.total_pnl_usd

    def tick(self):
        """Main tick function called from the orchestrator."""
        now = time.time()
        for window in self._market_windows.values():
            if window.phase == MarketPhase.DISCOVER and now >= window.start_time:
                self.start_market(window.condition_id)
            elif window.phase == MarketPhase.RUN and now >= window.end_time:
                self.end_market(window.condition_id)
            elif window.phase in (MarketPhase.START, MarketPhase.RUN):
                self.run_market_tick(window.condition_id)

    def get_performance_summary(self) -> Dict:
        """Get a summary of engine performance."""
        stats = self.stats
        return {
            "bankroll": round(self.bankroll, 2),
            "initial_bankroll": round(self.initial_bankroll, 2),
            "total_profit": round(self.bankroll - self.initial_bankroll, 2),
            "profit_pct": round((self.bankroll - self.initial_bankroll) / self.initial_bankroll * 100, 1) if self.initial_bankroll > 0 else 0,
            "total_trades": stats.total_markets_traded,
            "win_rate": round(stats.win_rate, 1),
            "total_pnl": round(stats.total_pnl_usd, 4),
            "peak_bankroll": round(stats.peak_bankroll, 2),
            "uptime_hours": round(stats.uptime_seconds / 3600, 1),
        }
