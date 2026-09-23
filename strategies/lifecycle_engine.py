"""
Up/Down Market Lifecycle Engine — the core trading engine for Polymarket's
crypto "Up or Down" windows (5-minute by default).

Each window is a lifecycle:
  DISCOVER  found via the market feed (slug {asset}-updown-5m-{start})
  START     window opened: capture the price to beat (60s index average
            ending at window start — mirrors the Chainlink TWAP strike)
  RUN       every tick: read both outcome books, price the window with the
            fair-value model, enter when the edge net of taker fees clears
            MIN_NET_EDGE, exit early when the bid pays more than the model
            says the position is worth
  RESOLVING window over, waiting for Polymarket's resolution (~1 min)
  SETTLED   PnL booked (and winning shares redeemed in live mode)

Pricing (strategies/fair_value.py): P(Up) from the BRTI index level vs the
strike, realized volatility, time left, and the TWAP averaging window; an
optional ML model (TTE orchestrator) tilts it. There is deliberately NO
fallback that echoes the market price — without a fair value we don't trade.

Execution goes through an executor (execution/executor.py): PaperExecutor
fills against the real book with real fees; LiveExecutor sends bounded FAK
orders. The engine code path is identical in both modes.
"""
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Dict, List, Optional

from scipy.stats import norm

from config.settings import settings
from data.btc_tick_buffer import build_feature_frame
from strategies.fair_value import (
    TWAP_WINDOW_SECONDS,
    all_in_buy_cost,
    binary_kelly_fraction,
    net_sell_proceeds,
    prob_up,
)

logger = logging.getLogger(__name__)

RESOLUTION_POLL_SECONDS = 10.0
RESOLUTION_GIVE_UP_SECONDS = 3 * 3600
ENTRY_RETRY_COOLDOWN_SECONDS = 5.0
MAX_ENTRY_ATTEMPTS = 3
BANKROLL_SYNC_SECONDS = 30.0
REDEEM_SWEEP_SECONDS = 300.0
MIN_STRIKE_SAMPLES = 20


class MarketPhase(Enum):
    """Phases of an up/down market lifecycle."""
    DISCOVER = "discover"
    START = "start"
    RUN = "run"
    END = "end"
    RESOLVING = "resolving"
    SETTLED = "settled"


@dataclass
class MarketWindow:
    """Represents a single up/down market window."""
    condition_id: str
    question: str
    asset: str  # "btc", "eth", "sol", "xrp"
    price_to_beat: float
    token_id_yes: str  # "Up" token
    token_id_no: str   # "Down" token
    start_time: float  # unix timestamp
    end_time: float    # unix timestamp
    phase: MarketPhase = MarketPhase.DISCOVER
    slug: str = ""
    min_order_size: float = 5.0
    fee_rate: float = 0.07
    fee_exponent: float = 1.0
    tradeable: bool = True
    skip_reason: str = ""

    # Current state
    current_yes_price: float = 0.5
    current_no_price: float = 0.5
    orderbook_snapshot: Optional[Dict] = None
    gap: float = 0.0
    fair_prob_up: Optional[float] = None

    # Strategy
    strategy_name: str = ""
    strategy_signal: Optional[Dict] = None
    entry_attempts: int = 0
    last_entry_attempt: float = 0.0

    # Position
    position_side: str = ""  # "YES" (Up) or "NO" (Down)
    position_token_id: str = ""
    shares: float = 0.0
    cost_usd: float = 0.0          # remaining cost basis incl. fees
    position_size_usd: float = 0.0  # original cost incl. fees
    fees_usd: float = 0.0
    realized_usd: float = 0.0      # proceeds from partial exits
    entry_price: float = 0.0
    entry_time: float = 0.0
    exit_price: float = 0.0
    exit_time: float = 0.0
    trade_id: str = ""
    outcome: str = ""
    last_resolution_check: float = 0.0

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

    @property
    def has_position(self) -> bool:
        return bool(self.position_side) and self.shares > 0

    def to_dict(self) -> Dict:
        return {
            "condition_id": self.condition_id,
            "question": self.question,
            "asset": self.asset,
            "price_to_beat": self.price_to_beat,
            "current_yes_price": self.current_yes_price,
            "current_no_price": self.current_no_price,
            "fair_prob_up": None if self.fair_prob_up is None else round(self.fair_prob_up, 4),
            "gap": self.gap,
            "phase": self.phase.value,
            "time_remaining": round(self.time_remaining, 1),
            "progress_pct": round(self.progress_pct * 100, 1),
            "position_side": self.position_side,
            "position_size_usd": round(self.position_size_usd, 4),
            "shares": round(self.shares, 4),
            "pnl_usd": round(self.pnl_usd, 4),
            "won": self.won,
            "tradeable": self.tradeable,
            "skip_reason": self.skip_reason,
        }


@dataclass
class EngineStats:
    """Aggregate engine statistics."""
    total_markets_observed: int = 0
    total_markets_traded: int = 0
    total_wins: int = 0
    total_losses: int = 0
    total_pnl_usd: float = 0.0
    total_fees_usd: float = 0.0
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
            "total_fees_usd": round(self.total_fees_usd, 4),
            "total_volume_usd": round(self.total_volume_usd, 2),
            "avg_edge": round(self.avg_edge, 4),
            "win_rate": round(self.win_rate, 1),
            "current_bankroll": round(self.current_bankroll, 2),
            "uptime_hours": round(self.uptime_seconds / 3600, 1),
        }


class FiveMinuteLifecycleEngine:
    """Orchestrates up/down windows from discovery to settlement."""

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
        tte_orchestrator=None,
        brti_engine=None,
        market_feed=None,
        executor=None,
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
        self.tte_orchestrator = tte_orchestrator
        self.brti_engine = brti_engine
        self.market_feed = market_feed
        if executor is None:
            from execution.executor import PaperExecutor
            executor = PaperExecutor()
        self.executor = executor

        self._market_windows: Dict[str, MarketWindow] = {}
        self._stats = EngineStats()
        self._stats.current_bankroll = bankroll
        self._stats.peak_bankroll = bankroll
        self._start_time = time.time()
        self._trade_history: List[Dict] = []
        self._edge_sum = 0.0
        self._last_bankroll_sync = 0.0
        self._last_redeem_sweep = 0.0

        # Kept for callers that inspect/override it; pricing no longer
        # depends on the generic strategy classes.
        self._strategy = None

    @property
    def mode_tag(self) -> str:
        return "LIVE" if getattr(self.executor, "live", False) else "PAPER"

    def _notify(self, message: str) -> None:
        if self.on_notification:
            try:
                self.on_notification(f"[{self.mode_tag}] {message}")
            except Exception as e:
                logger.warning("Notification callback failed: %s", e)

    # ── Lifecycle management ───────────────────────────────────────────

    def discover_market(self, market_data: Dict) -> Optional[MarketWindow]:
        """Register a discovered window. Returns the MarketWindow if new."""
        condition_id = market_data.get("condition_id", "")
        if not condition_id or condition_id in self._market_windows:
            return None

        question = market_data.get("question", "")
        asset = market_data.get("asset") or self._detect_asset(question)
        if asset == "unknown":
            return None

        start_time = market_data.get("start_time", 0)
        end_time = market_data.get("end_time", 0)
        if start_time == 0 or end_time == 0:
            return None

        window = MarketWindow(
            condition_id=condition_id,
            question=question,
            asset=asset,
            price_to_beat=self._parse_price_to_beat(market_data),
            token_id_yes=market_data.get("token_id_yes", ""),
            token_id_no=market_data.get("token_id_no", ""),
            start_time=start_time,
            end_time=end_time,
            slug=market_data.get("slug", ""),
            min_order_size=float(market_data.get("min_order_size", 5) or 5),
        )
        self._market_windows[condition_id] = window
        self._stats.total_markets_observed += 1
        logger.info(
            "Discovered market: %s | %s | Window: %s -> %s",
            asset.upper(), question[:60],
            datetime.fromtimestamp(start_time, tz=timezone.utc).strftime("%H:%M:%S"),
            datetime.fromtimestamp(end_time, tz=timezone.utc).strftime("%H:%M:%S"),
        )
        return window

    def start_market(self, condition_id: str) -> bool:
        """DISCOVER -> START: capture the strike and load fee parameters."""
        window = self._market_windows.get(condition_id)
        if not window:
            return False
        if window.phase != MarketPhase.DISCOVER:
            logger.warning("Market %s already in phase %s", condition_id, window.phase)
            return False

        window.phase = MarketPhase.START
        window.strategy_name = self.strategy_name
        self._strategy = self._create_strategy(self.strategy_name)

        if window.price_to_beat <= 0:
            strike = self._capture_strike(window)
            if strike:
                window.price_to_beat = strike
            else:
                window.tradeable = False
                window.skip_reason = "no index history covering window start"

        if self.market_feed is not None:
            try:
                fees = self.market_feed.get_fee_params(condition_id)
                window.fee_rate, window.fee_exponent = fees.rate, fees.exponent
            except Exception as e:
                logger.warning("Fee params unavailable for %s: %s", condition_id[:10], e)

        logger.info(
            "Starting market %s: %s | strike=%.2f | tradeable=%s %s",
            condition_id[:10], window.asset.upper(), window.price_to_beat,
            window.tradeable, window.skip_reason,
        )
        return True

    def run_market_tick(self, condition_id: str, orderbook: Optional[Dict] = None) -> bool:
        """Process one tick of an active window."""
        window = self._market_windows.get(condition_id)
        if not window or not window.is_active:
            return False
        if window.phase == MarketPhase.START:
            window.phase = MarketPhase.RUN
        if window.phase != MarketPhase.RUN:
            return False

        if orderbook:
            window.orderbook_snapshot = orderbook
            window.current_yes_price = orderbook.get("mid_price", window.current_yes_price)
            no_book = orderbook.get("no")
            window.current_no_price = no_book.mid if no_book is not None else 1.0 - window.current_yes_price
            window.gap = window.current_yes_price - 0.5

        # New entries only while the account isn't halted by the circuit breaker.
        risk_ok = self.risk_manager is None or not getattr(self.risk_manager, "halted", False)
        if window.orderbook_snapshot and risk_ok and not window.position_side:
            signal = self._get_strategy_signal(window)
            window.strategy_signal = signal
            if signal and signal.get("should_trade", False):
                self._execute_trade(window, signal)

        if window.has_position:
            btc_signal = self._compute_btc_signal(window) if window.asset == "btc" else None
            window.fair_prob_up = self._fair_probability(window, btc_signal)
            self._check_exit_conditions(window)
        return True

    def end_market(self, condition_id: str, final_price: Optional[float] = None) -> bool:
        """RUN -> RESOLVING (if holding) or SETTLED (if flat)."""
        window = self._market_windows.get(condition_id)
        if not window or window.phase not in (MarketPhase.RUN, MarketPhase.START):
            return False
        if window.has_position:
            window.phase = MarketPhase.RESOLVING
            logger.info("Window %s ended holding %s — awaiting resolution",
                        condition_id[:10], window.position_side)
        else:
            window.phase = MarketPhase.SETTLED
            self._cleanup_old_markets()
        return True

    def _check_resolution(self, window: MarketWindow, now: float) -> None:
        if now - window.last_resolution_check < RESOLUTION_POLL_SECONDS:
            return
        window.last_resolution_check = now

        outcome = None
        if self.market_feed is not None and window.slug:
            try:
                outcome = self.market_feed.get_resolution(window.slug)
            except Exception as e:
                logger.warning("Resolution lookup failed for %s: %s", window.slug, e)
        elif not getattr(self.executor, "live", False):
            outcome = self._estimate_outcome_from_index(window)

        if outcome is None:
            if now - window.end_time > RESOLUTION_GIVE_UP_SECONDS:
                logger.error("Window %s unresolved after %.0fh — leaving for redeem sweep",
                             window.slug or window.condition_id, RESOLUTION_GIVE_UP_SECONDS / 3600)
                self._notify(f"Window {window.slug} still unresolved; position left for redemption")
                window.phase = MarketPhase.SETTLED
            return

        window.outcome = outcome
        won = (outcome == "UP") == (window.position_side == "YES")
        payout = window.shares if won else 0.0
        self._close_position(window, payout, 1.0 if won else 0.0, "resolution")
        if won and getattr(self.executor, "live", False) and settings.auto_redeem_enabled:
            self.executor.redeem(window.condition_id)

    # ── Pricing ────────────────────────────────────────────────────────

    def _capture_strike(self, window: MarketWindow) -> Optional[float]:
        """Index average over the 60s ending at window start (the TWAP strike)."""
        if self.brti_engine is None or window.asset != "btc":
            return None
        lookback = int(time.time() - window.start_time + TWAP_WINDOW_SECONDS + 5)
        history = self.brti_engine.get_price_history(lookback)
        lo = window.start_time - TWAP_WINDOW_SECONDS
        prices = [p for ts, p in history if lo <= ts <= window.start_time and p > 0]
        covered = any(ts <= lo + 10 for ts, _ in history)
        if len(prices) < MIN_STRIKE_SAMPLES or not covered:
            return None
        return sum(prices) / len(prices)

    def _vol_per_second(self) -> Optional[float]:
        if self.brti_engine is None:
            return None
        vol = self.brti_engine.get_volatility(300) or self.brti_engine.get_volatility(60)
        if not vol:
            return None
        tick = max(float(settings.brti_tick_interval_seconds), 1e-3)
        return vol / math.sqrt(tick)

    def _fair_probability(self, window: MarketWindow, btc_signal: Optional[Dict] = None) -> Optional[float]:
        """P(window resolves Up), or None if we can't price it honestly."""
        if not window.tradeable or window.price_to_beat <= 0 or self.brti_engine is None:
            return None
        last_tick = self.brti_engine.last_tick
        if last_tick is None:
            return None
        sigma = self._vol_per_second()
        if sigma is None:
            return None
        tau = window.end_time - time.time()
        running = None
        if tau < TWAP_WINDOW_SECONDS:
            since = window.end_time - TWAP_WINDOW_SECONDS
            recent = [p for ts, p in self.brti_engine.get_price_history(int(TWAP_WINDOW_SECONDS) + 5)
                      if ts >= since and p > 0]
            running = sum(recent) / len(recent) if recent else None
        drift_z = 0.0
        if btc_signal and btc_signal.get("tte_prob") is not None:
            drift_z = (btc_signal["tte_prob"] - 0.5) * 2.0
        return prob_up(
            current_price=last_tick.brti_price,
            strike=window.price_to_beat,
            vol_per_second=sigma,
            tau_seconds=tau,
            running_twap=running,
            basis_bps=settings.settlement_basis_bps,
            drift_z=drift_z,
        )

    def _get_strategy_signal(self, window: MarketWindow) -> Optional[Dict]:
        """
        Decide whether to enter. Compares the model's probability for each
        side with the all-in cost of buying that side's best ask (price +
        taker fee), and sizes with fractional Kelly.
        """
        snapshot = window.orderbook_snapshot or {}
        up_book, down_book = snapshot.get("yes"), snapshot.get("no")
        btc_signal = self._compute_btc_signal(window) if window.asset == "btc" else None
        fair = self._fair_probability(window, btc_signal)
        window.fair_prob_up = fair

        if up_book is None or down_book is None:
            return {"should_trade": False, "reason": "no order book"}
        if fair is None:
            return {"should_trade": False, "reason": window.skip_reason or "no fair value"}

        elapsed = time.time() - window.start_time
        remaining = window.end_time - time.time()
        if elapsed < settings.entry_min_elapsed_seconds:
            return {"should_trade": False, "reason": "too early in window", "model_prob": fair}
        if remaining < settings.entry_min_remaining_seconds:
            return {"should_trade": False, "reason": "too late in window", "model_prob": fair}
        if window.entry_attempts >= MAX_ENTRY_ATTEMPTS:
            return {"should_trade": False, "reason": "entry attempts exhausted", "model_prob": fair}
        if time.time() - window.last_entry_attempt < ENTRY_RETRY_COOLDOWN_SECONDS:
            return {"should_trade": False, "reason": "entry cooldown", "model_prob": fair}

        best = None
        for side, book, prob in (("YES", up_book, fair), ("NO", down_book, 1.0 - fair)):
            ask = book.best_ask
            if ask <= 0 or book.spread > settings.max_spread:
                continue
            if not (settings.min_price_threshold <= ask <= settings.max_price_threshold):
                continue
            cost = all_in_buy_cost(ask, window.fee_rate, window.fee_exponent)
            edge = prob - cost
            if best is None or edge > best["edge"]:
                best = {"side": side, "book": book, "prob": prob, "ask": ask, "cost": cost, "edge": edge}

        if best is None:
            return {"should_trade": False, "reason": "no tradeable side (spread/price zone)", "model_prob": fair}
        if best["edge"] < settings.min_net_edge:
            return {"should_trade": False, "reason": f"net edge {best['edge']:+.4f} < {settings.min_net_edge}",
                    "model_prob": fair, "edge": best["edge"]}

        kelly = binary_kelly_fraction(best["prob"], best["cost"]) * settings.ml_kelly_fraction
        size = self.bankroll * kelly
        size = min(size, self.bankroll * settings.max_position_size_pct / 100)
        if self.risk_manager is not None and hasattr(self.risk_manager, "max_position_size"):
            size = min(size, self.risk_manager.max_position_size("crypto"))
        size = min(size, best["book"].ask_size * best["cost"])  # top level only
        if getattr(self.executor, "live", False):
            size = min(size, settings.live_max_order_usd)
        min_size = window.min_order_size * best["cost"]
        if size < min_size:
            return {"should_trade": False, "model_prob": fair, "edge": best["edge"],
                    "reason": f"size ${size:.2f} below minimum order ${min_size:.2f}"}

        return {
            "should_trade": True,
            "side": best["side"],
            "size_usd": round(size, 2),
            "price": best["ask"],
            "cost": best["cost"],
            "edge": best["edge"],
            "confidence": abs(fair - 0.5) * 2,
            "model_prob": fair,
            "reason": f"P(side)={best['prob']:.3f} vs cost {best['cost']:.3f}",
        }

    def _compute_btc_signal(self, window: MarketWindow) -> Optional[Dict]:
        """
        BTC momentum / z-score / volatility from BRTI history, plus — when
        ML_PREDICTION_ENABLED and the TTE model for this TTE is trained — the
        model's directional probability (tte_prob) and the implied P(Up).
        """
        if self.brti_engine is None:
            return None
        last_tick = self.brti_engine.last_tick
        if last_tick is None:
            return None

        history = self.brti_engine.get_price_history(120)
        if len(history) < 20:
            return None

        prices = [p for _, p in history]
        current_price = last_tick.brti_price
        mean = sum(prices) / len(prices)
        variance = sum((p - mean) ** 2 for p in prices) / len(prices)
        std = variance ** 0.5
        z_score = (current_price - mean) / std if std > 0 else 0.0
        momentum = (current_price - prices[0]) / prices[0] if prices[0] else 0.0
        volatility = self.brti_engine.get_volatility(60) or 0.0

        result = {
            "momentum": momentum,
            "z_score": z_score,
            "volatility": volatility,
            "confidence": 0.5,
            "model_prob": None,
            "tte_prob": None,
        }

        if not settings.ml_prediction_enabled or self.tte_orchestrator is None:
            return result

        tte = max(1, min(900, int(window.time_remaining)))
        model_set = self.tte_orchestrator.tte_models.get(tte)
        if model_set is None or not model_set.lr_baseline.fitted:
            logger.debug("TTE model for tte=%ds not trained yet", tte)
            return result

        features = build_feature_frame(self.brti_engine)
        if features is None:
            return result

        prediction = self.tte_orchestrator.predict(features, tte)
        tte_prob = prediction.get("probability", 0.5)
        result["tte_prob"] = tte_prob
        result["confidence"] = prediction.get("confidence", 0.5)
        result["model_prob"] = self._tte_prob_to_yes_probability(
            tte_prob, current_price, window.price_to_beat, volatility, tte,
        )
        return result

    @staticmethod
    def _tte_prob_to_yes_probability(
        tte_prob: float,
        current_price: float,
        price_to_beat: float,
        vol_per_second: float,
        tte_seconds: int,
    ) -> float:
        """
        Combine the TTE model's self-relative prediction ("higher than now in
        `tte_seconds`?") with the distance to price_to_beat into P(Up),
        using a lognormal / normal-CDF approximation.
        """
        if current_price <= 0 or price_to_beat <= 0:
            return 0.5
        sigma = max(vol_per_second, 1e-6) * math.sqrt(max(tte_seconds, 1))
        gap = math.log(current_price / price_to_beat)
        drift = (tte_prob - 0.5) * 2 * sigma
        d = (gap + drift) / sigma
        return float(max(0.01, min(0.99, norm.cdf(d))))

    # ── Execution ──────────────────────────────────────────────────────

    def _execute_trade(self, window: MarketWindow, signal: Dict):
        """Buy the chosen side through the executor."""
        side = signal.get("side", "YES")
        size_usd = float(signal.get("size_usd", 0))
        if size_usd <= 0 or size_usd > self.bankroll:
            return
        ok, reason = self.executor.entries_allowed()
        if not ok:
            logger.info("Entry skipped: %s", reason)
            return

        book = window.orderbook_snapshot["yes" if side == "YES" else "no"]
        token_id = window.token_id_yes if side == "YES" else window.token_id_no
        window.entry_attempts += 1
        window.last_entry_attempt = time.time()

        fill = self.executor.buy(
            token_id, size_usd, max_price=signal["price"], asks=book.asks,
            fee_rate=window.fee_rate, fee_exponent=window.fee_exponent,
        )
        if fill is None or fill.shares <= 0:
            logger.info("Entry not filled: %s %s $%.2f @<=%.2f", side, window.asset.upper(),
                        size_usd, signal["price"])
            return

        window.position_side = side
        window.position_token_id = token_id
        window.shares = fill.shares
        window.cost_usd = fill.cash_usd
        window.position_size_usd = fill.cash_usd
        window.fees_usd = fill.fees_usd
        window.entry_price = fill.avg_price
        window.entry_time = time.time()
        self.bankroll -= fill.cash_usd
        self._edge_sum += float(signal.get("edge", 0))

        if self.trade_logger:
            try:
                window.trade_id = self.trade_logger.log_entry(
                    condition_id=window.condition_id,
                    asset=window.asset,
                    side=side,
                    price=fill.avg_price,
                    size_usd=fill.cash_usd,
                    strategy="fair_value",
                    source="lifecycle_live" if self.executor.live else "lifecycle_paper",
                    market_question=window.question,
                    bankroll_after=self.bankroll,
                    metadata={
                        "slug": window.slug,
                        "price_to_beat": window.price_to_beat,
                        "shares": fill.shares,
                        "entry_fees": fill.fees_usd,
                        "signal_edge": signal.get("edge", 0),
                        "model_prob": signal.get("model_prob"),
                        "order_id": fill.order_id,
                    },
                )
            except Exception as e:
                logger.error("Trade log entry failed: %s", e)

        logger.info(
            "[%s] ENTRY %s %s: %.2f sh @ %.3f ($%.2f incl. $%.3f fees) | edge %+.4f | %s",
            self.mode_tag, "UP" if side == "YES" else "DOWN", window.asset.upper(),
            fill.shares, fill.avg_price, fill.cash_usd, fill.fees_usd,
            signal.get("edge", 0), signal.get("reason", ""),
        )
        self._notify(
            f"Entry {'UP' if side == 'YES' else 'DOWN'} {window.asset.upper()} "
            f"{fill.shares:.2f} sh @ {fill.avg_price:.3f} (${fill.cash_usd:.2f}) "
            f"| edge {signal.get('edge', 0):+.3f}\n{window.question}"
        )
        if self.on_trade:
            self.on_trade({
                "condition_id": window.condition_id,
                "asset": window.asset,
                "side": side,
                "entry_price": fill.avg_price,
                "size_usd": fill.cash_usd,
                "edge": signal.get("edge", 0),
                "strategy": window.strategy_name,
            })

    def _check_exit_conditions(self, window: MarketWindow):
        """
        Sell early only when the market pays more than the position is worth
        to us: net bid (after taker fee) - model value >= EXIT_EDGE. This
        covers both take-profit and stop-loss without fixed % thresholds that
        churn fees on normal 5-minute noise.
        """
        snapshot = window.orderbook_snapshot or {}
        book = snapshot.get("yes" if window.position_side == "YES" else "no")
        fair = window.fair_prob_up
        if book is None or fair is None or book.best_bid <= 0:
            return
        value = fair if window.position_side == "YES" else 1.0 - fair
        net_bid = net_sell_proceeds(book.best_bid, window.fee_rate, window.fee_exponent)
        if net_bid - value < settings.exit_edge:
            return

        fill = self.executor.sell(
            window.position_token_id, window.shares, min_price=book.best_bid, bids=book.bids,
            fee_rate=window.fee_rate, fee_exponent=window.fee_exponent,
        )
        if fill is None or fill.shares <= 0:
            return
        sold_fraction = min(1.0, fill.shares / window.shares)
        cost_released = window.cost_usd * sold_fraction
        window.fees_usd += fill.fees_usd
        self.bankroll += fill.cash_usd
        logger.info(
            "[%s] EARLY EXIT %s %s: sold %.2f sh @ %.3f for $%.2f (model value %.3f)",
            self.mode_tag, window.position_side, window.asset.upper(),
            fill.shares, fill.avg_price, fill.cash_usd, value,
        )
        if sold_fraction >= 0.999:
            window.shares = 0.0
            self._book_close(window, window.realized_usd + fill.cash_usd, fill.avg_price, "early_exit",
                             bankroll_credited=True)
        else:
            window.shares -= fill.shares
            window.cost_usd -= cost_released
            window.realized_usd += fill.cash_usd

    def _close_position(self, window: MarketWindow, payout: float, exit_price: float, reason: str):
        """Settle the remaining shares at resolution (payout $1 or $0 per share)."""
        self.bankroll += payout
        window.shares = 0.0
        self._book_close(window, window.realized_usd + payout, exit_price, reason, bankroll_credited=True)

    def _book_close(self, window: MarketWindow, total_proceeds: float, exit_price: float,
                    reason: str, bankroll_credited: bool):
        window.exit_price = exit_price
        window.exit_time = time.time()
        window.pnl_usd = total_proceeds - window.position_size_usd
        window.pnl_pct = (window.pnl_usd / window.position_size_usd * 100) if window.position_size_usd > 0 else 0
        window.won = window.pnl_usd > 0
        window.phase = MarketPhase.SETTLED

        self._stats.total_markets_traded += 1
        self._stats.total_pnl_usd += window.pnl_usd
        self._stats.total_fees_usd += window.fees_usd
        self._stats.total_volume_usd += window.position_size_usd
        if window.won:
            self._stats.total_wins += 1
        else:
            self._stats.total_losses += 1
        traded = self._stats.total_wins + self._stats.total_losses
        self._stats.win_rate = self._stats.total_wins / traded * 100 if traded else 0
        self._stats.avg_edge = self._edge_sum / traded if traded else 0
        self._stats.peak_bankroll = max(self._stats.peak_bankroll, self.bankroll)
        self._stats.current_bankroll = self.bankroll

        if self.trade_logger and window.trade_id:
            try:
                self.trade_logger.log_exit(
                    trade_id=window.trade_id,
                    exit_price=exit_price,
                    exit_reason=reason,
                    pnl_usd=window.pnl_usd,
                    fees_usd=window.fees_usd,
                    bankroll_after=self.bankroll,
                    metadata={"won": window.won, "outcome": window.outcome,
                              "entry_side": window.position_side},
                )
            except Exception as e:
                logger.error("Trade log exit failed: %s", e)

        if self.risk_manager is not None and hasattr(self.risk_manager, "record_closed_trade"):
            try:
                self.risk_manager.record_closed_trade(window.condition_id, "crypto", window.pnl_usd)
            except Exception as e:  # CircuitBreakerTripped: halt new entries, keep managing exits
                logger.critical("Circuit breaker tripped: %s", e)
                self._notify(f"CIRCUIT BREAKER — new entries halted: {e}")

        record = {
            "condition_id": window.condition_id,
            "slug": window.slug,
            "asset": window.asset,
            "side": window.position_side,
            "entry_price": window.entry_price,
            "exit_price": exit_price,
            "size_usd": window.position_size_usd,
            "pnl_usd": window.pnl_usd,
            "fees_usd": window.fees_usd,
            "won": window.won,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": window.strategy_name,
        }
        self._trade_history.append(record)
        if self.on_pnl:
            self.on_pnl(record)

        logger.info(
            "[%s] SETTLED %s %s (%s): %s pnl=$%+.2f fees=$%.3f bankroll=$%.2f",
            self.mode_tag, window.asset.upper(), window.position_side, reason,
            "WIN" if window.won else "LOSS", window.pnl_usd, window.fees_usd, self.bankroll,
        )
        self._notify(
            f"{'WIN' if window.won else 'LOSS'} {window.asset.upper()} "
            f"{'UP' if window.position_side == 'YES' else 'DOWN'} ({reason}) "
            f"pnl ${window.pnl_usd:+.2f} | fees ${window.fees_usd:.3f} | bankroll ${self.bankroll:.2f}"
        )
        self._cleanup_old_markets()

    def _estimate_outcome_from_index(self, window: MarketWindow) -> Optional[str]:
        """Paper-only fallback without a market feed: settle on our own index."""
        if self.brti_engine is None or window.price_to_beat <= 0:
            return None
        history = self.brti_engine.get_price_history(int(time.time() - window.end_time + 65))
        prices = [p for ts, p in history if window.end_time - TWAP_WINDOW_SECONDS <= ts <= window.end_time]
        if len(prices) < MIN_STRIKE_SAMPLES:
            return None
        return "UP" if sum(prices) / len(prices) >= window.price_to_beat else "DOWN"

    # ── Helper methods ─────────────────────────────────────────────────

    def _create_strategy(self, name: str):
        try:
            from strategies import get_strategy
            return get_strategy(name)
        except Exception as e:
            logger.error("Failed to create strategy %s: %s", name, e)
            return None

    def _detect_asset(self, question: str) -> str:
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
        """
        Explicit strike if the market data carries one. Up/down windows have
        none — their strike is captured from the index at window start.
        """
        for field_name in ["price_to_beat", "strike_price", "target_price"]:
            val = market_data.get(field_name, 0)
            if val:
                return float(val)
        return 0.0

    def _compute_momentum(self, window: MarketWindow) -> float:
        if not window.orderbook_snapshot:
            return 0.0
        return window.orderbook_snapshot.get("imbalance", 0) * 0.1

    def _cleanup_old_markets(self, max_age: float = 600):
        now = time.time()
        stale = [cid for cid, w in self._market_windows.items()
                 if w.phase == MarketPhase.SETTLED and (now - w.end_time) > max_age]
        for cid in stale:
            del self._market_windows[cid]

    def open_exposure(self) -> float:
        """Cost basis of all open positions (live exposure cap input)."""
        return sum(w.cost_usd for w in self._market_windows.values() if w.has_position)

    # ── Public API ─────────────────────────────────────────────────────

    @property
    def stats(self) -> EngineStats:
        self._stats.uptime_seconds = time.time() - self._start_time
        return self._stats

    @property
    def current_market(self) -> Optional[MarketWindow]:
        for window in self._market_windows.values():
            if window.is_active and window.phase == MarketPhase.RUN:
                return window
        return None

    @property
    def next_market(self) -> Optional[MarketWindow]:
        upcoming = [w for w in self._market_windows.values()
                    if w.phase == MarketPhase.DISCOVER and w.start_time > time.time()]
        return min(upcoming, key=lambda w: w.start_time) if upcoming else None

    @property
    def trade_history(self) -> List[Dict]:
        return list(self._trade_history)

    def get_all_windows(self) -> List[Dict]:
        return [w.to_dict() for w in self._market_windows.values()]

    def get_active_windows(self) -> List[MarketWindow]:
        return [
            w for w in self._market_windows.values()
            if w.is_active or w.phase in (MarketPhase.RUN, MarketPhase.START,
                                          MarketPhase.END, MarketPhase.RESOLVING)
        ]

    def get_total_pnl(self) -> float:
        return self._stats.total_pnl_usd

    def _book_snapshot(self, books: Dict, window: MarketWindow) -> Optional[Dict]:
        up, down = books.get(window.token_id_yes), books.get(window.token_id_no)
        if up is None or down is None:
            return None
        return {
            "yes": up,
            "no": down,
            "mid_price": up.mid,
            "imbalance": up.imbalance,
            "spread_bps": up.spread * 10000,
        }

    def tick(self):
        """Main tick, called every ORDERBOOK_UPDATE_INTERVAL by the fast loop."""
        now = time.time()

        if self.market_feed is not None:
            try:
                for market in self.market_feed.discover(now):
                    self.discover_market(market.to_discovery_dict())
            except Exception as e:
                logger.warning("Market discovery failed: %s", e)

        for window in list(self._market_windows.values()):
            if window.phase == MarketPhase.DISCOVER and now >= window.start_time:
                self.start_market(window.condition_id)
            if window.phase in (MarketPhase.START, MarketPhase.RUN) and now >= window.end_time:
                self.end_market(window.condition_id)

        running = [w for w in self._market_windows.values()
                   if w.phase in (MarketPhase.START, MarketPhase.RUN)]
        books = {}
        if running and self.market_feed is not None:
            tokens = [t for w in running for t in (w.token_id_yes, w.token_id_no) if t]
            books = self.market_feed.get_books(tokens)
        for window in running:
            self.run_market_tick(window.condition_id, self._book_snapshot(books, window))

        for window in list(self._market_windows.values()):
            if window.phase == MarketPhase.RESOLVING:
                self._check_resolution(window, now)

        if getattr(self.executor, "live", False):
            self._live_housekeeping(now)

    def _live_housekeeping(self, now: float) -> None:
        if now - self._last_bankroll_sync >= BANKROLL_SYNC_SECONDS:
            self._last_bankroll_sync = now
            balance = self.executor.collateral_balance()
            if balance is not None:
                self.bankroll = balance
                self._stats.current_bankroll = balance
        if settings.auto_redeem_enabled and now - self._last_redeem_sweep >= REDEEM_SWEEP_SECONDS:
            self._last_redeem_sweep = now
            self.executor.redeem_all()

    def get_performance_summary(self) -> Dict:
        stats = self.stats
        return {
            "mode": self.mode_tag,
            "bankroll": round(self.bankroll, 2),
            "initial_bankroll": round(self.initial_bankroll, 2),
            "total_profit": round(self.bankroll - self.initial_bankroll, 2),
            "profit_pct": round((self.bankroll - self.initial_bankroll) / self.initial_bankroll * 100, 1)
            if self.initial_bankroll > 0 else 0,
            "total_trades": stats.total_markets_traded,
            "win_rate": round(stats.win_rate, 1),
            "total_pnl": round(stats.total_pnl_usd, 4),
            "total_fees": round(stats.total_fees_usd, 4),
            "avg_edge": round(stats.avg_edge, 4),
            "peak_bankroll": round(stats.peak_bankroll, 2),
            "uptime_hours": round(stats.uptime_seconds / 3600, 1),
        }
