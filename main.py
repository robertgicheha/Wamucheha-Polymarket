"""
Main orchestrator — connects all modules into a unified trading bot.

Modules integrated:
  - BRTI Engine: real-time BTC price replication from exchange orderbooks
  - TTE Orchestrator: 900 ML models for time-to-expiry prediction
  - Strategy Engine: 6 strategies, selected by Sharpe/P&L
  - Arbitrage Engine: cross-platform PM + Limit Exchange / Opinion / Myriad
  - PMXT Wrapper: unified Polymarket API
  - Risk Manager: circuit breaker, stop-loss, compounding
  - WebSocket Engine: real-time Polymarket orderbook/trade feeds
  - Lifecycle Engine: 5-minute market lifecycle management

Cycle flow:
  1. BRTI engine ticks (1-sec) → feeds TTE models
  2. TTE models predict at each TTE → probability + confidence
  3. Strategy engine selects best strategy → trade decision
  4. Arbitrage engine scans for cross-platform spreads
  5. Orderbook arb scans 5-min markets for complement spreads
  6. Lifecycle engine manages 5-min market windows
  7. Risk checks → position sizing, circuit breaker
  8. Execute trades via paper broker
  9. Monitor positions, check exits
  10. Dashboard update, alerts
"""
import asyncio
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from alerts.notifier import Severity, notifier
from config.settings import settings

# Set up logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/bot.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def _init_brti_engine():
    """Initialize BRTI engine with exchange WebSockets."""
    from data.brti.brti_engine import BRTIEngine
    from data.brti.exchange_ws import get_all_exchange_ws

    engine = BRTIEngine()
    ws_instances = get_all_exchange_ws(
        symbol="BTC-USD",
        on_snapshot=engine.update_orderbook,
        exchanges=settings.brti_exchanges,
    )
    return engine, ws_instances


def _init_tte_orchestrator(brti_engine):
    """Initialize TTE training orchestrator, wired to train on live BRTI history."""
    from ml.tte_orchestrator import TTETrainingOrchestrator
    from data.btc_tick_buffer import raw_price_frame

    def _data_provider():
        return raw_price_frame(brti_engine)

    orchestrator = TTETrainingOrchestrator(data_provider=_data_provider)
    loaded = orchestrator.load_all()
    logger.info("TTE orchestrator: loaded %d model sets", loaded)
    return orchestrator


def _init_arbitrage_engine(bankroll: float):
    """Initialize cross-platform arbitrage engine."""
    from strategies.arbitrage import ArbitrageEngine
    return ArbitrageEngine(bankroll=bankroll)


def _init_orderbook_arb(bankroll: float):
    """Initialize orderbook-based intra-platform arbitrage for 5-min markets."""
    from strategies.arbitrage import IntraPlatformArbEngine
    return IntraPlatformArbEngine(bankroll=bankroll)


def _init_pmxt_wrapper():
    """Initialize PMXT unified API wrapper."""
    from connectors.pmxt_wrapper import PMXTWrapper
    return PMXTWrapper(paper_mode=(settings.trading_mode == "paper"))


def _init_strategies():
    """Initialize all trading strategies."""
    from strategies import get_all_strategies
    return get_all_strategies()


def _init_websocket_connector():
    """Initialize Polymarket WebSocket connector."""
    from connectors.polymarket_ws import PolymarketWebSocket
    ws = PolymarketWebSocket()
    return ws


def _init_lifecycle_engine(
    arb_engine, orderbook_arb, risk_manager, trade_logger=None,
    tte_orchestrator=None, brti_engine=None, pm_connector=None,
):
    """
    Initialize the up/down lifecycle engine with a live market feed and the
    executor for the current mode (real orders only when TRADING_MODE=live).
    """
    from connectors.updown_feed import UpDownMarketFeed
    from execution.executor import LiveExecutor, PaperExecutor
    from strategies.lifecycle_engine import FiveMinuteLifecycleEngine

    feed = UpDownMarketFeed(
        connector=pm_connector,
        assets=settings.updown_assets,
        interval_minutes=settings.updown_interval_minutes,
        discovery_interval_seconds=settings.market_discovery_interval,
    )
    live = settings.trading_mode == "live"
    executor = LiveExecutor(pm_connector) if live else PaperExecutor()
    engine = FiveMinuteLifecycleEngine(
        bankroll=risk_manager.bankroll,
        arbitrage_engine=arb_engine,
        orderbook_arb=orderbook_arb,
        risk_manager=risk_manager,
        trade_logger=trade_logger,
        tte_orchestrator=tte_orchestrator,
        brti_engine=brti_engine,
        market_feed=feed,
        executor=executor,
        on_notification=lambda msg: notifier.send(msg, Severity.INFO),
    )
    if live:
        executor.exposure_fn = engine.open_exposure
    return engine


def _live_startup(pm_connector):
    """
    Refuse to trade live unless every critical preflight check passes, then
    start from a clean slate: no stale resting orders, resolved winnings
    redeemed. Returns the initial pUSD balance.
    """
    from execution.preflight import critical_failures, format_report, run_preflight

    settings.validate_for_live_trading()
    checks = run_preflight(include_exchanges=settings.auto_funding_enabled)
    report = format_report(checks)
    logger.info("Preflight:\n%s", report)
    failures = critical_failures(checks)
    if failures:
        notifier.send(f"Live start ABORTED — preflight failed:\n{report}", Severity.CRITICAL)
        raise SystemExit(f"Preflight failed ({len(failures)} critical) — not trading. See log.")

    pm_connector.cancel_all()
    from execution.executor import LiveExecutor
    redeemed = LiveExecutor(pm_connector).redeem_all()
    balance = pm_connector.get_collateral_balance()
    logger.info("Live start: cancelled open orders, redeemed %d market(s), pUSD=$%.2f", redeemed, balance)
    return balance


# def _init_gnosis_relayer():
#     """Initialize Gnosis Safe relayer for gasless payments."""
#     if not settings.gnosis_safe_enabled:
#         return None
#     from connectors.gnosis_relayer import GnosisSafeRelayer, RelayerConfig
#     config = RelayerConfig(
#         safe_address=settings.gnosis_safe_address,
#         private_key=settings.gnosis_safe_owner_key,
#         rpc_url=settings.polygon_rpc_url or "https://polygon-rpc.com",
#         relayer_url="https://safe-relayer.gnosis.io",
#         chain_id=137,
#     )
#     return GnosisSafeRelayer(config)


def _fast_trading_loop(lifecycle_engine, stop_event: threading.Event) -> None:
    """
    Ticks the 5-minute lifecycle engine at settings.orderbook_update_interval
    (default 1s), independent of the main loop's slower
    signal_check_interval_seconds (default 300s — the length of an entire
    market window). Without this, a 5-min market got ~1 tick for its whole
    lifetime and take-profit/stop-loss/entries never got a real chance to
    fire regardless of what fed the model.
    """
    interval = settings.orderbook_update_interval
    logger.info("Fast trading loop started (interval=%.1fs)", interval)
    while not stop_event.is_set():
        tick_start = time.time()
        try:
            lifecycle_engine.tick()
        except Exception as e:
            logger.error("Fast trading loop tick failed: %s", e)
        elapsed = time.time() - tick_start
        stop_event.wait(max(0.0, interval - elapsed))
    logger.info("Fast trading loop stopped")


def _run_async(coro):
    """
    Run a one-shot async coroutine to completion in a throwaway event loop.
    Only safe for coroutines that fully finish their work before returning
    (e.g. a single HTTP call) — NOT for anything that schedules a background
    task meant to outlive the call (asyncio.create_task inside it), because
    this loop is closed immediately after, orphaning any such task before it
    ever runs a second iteration. Use `_start_background_loop` +
    `_run_on_background_loop` for those instead (BRTI engine, exchange/PM
    WebSockets, TTE retrain loop all fall in that category).
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _start_background_loop() -> asyncio.AbstractEventLoop:
    """
    Start one persistent asyncio event loop on a dedicated daemon thread,
    for the lifetime of the process. Long-running async components (BRTI
    engine tick loop, exchange WebSocket reconnect loops, Polymarket
    WebSocket, TTE retrain loop) get scheduled on this loop via
    `_run_on_background_loop` so they keep running for the bot's lifetime,
    instead of being silently orphaned by a throwaway loop that closes right
    after scheduling them.
    """
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True, name="asyncio-bg-loop")
    thread.start()
    return loop


def _run_on_background_loop(loop, coro, timeout: float = 30.0):
    """Schedule `coro` on the persistent background loop and wait for it to start/finish."""
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)


def _write_balance_state(risk_manager, arb_engine=None, lifecycle_engine=None, paper_broker=None):
    """Write current balance state to JSON for admin API reads."""
    summary = risk_manager.get_compounding_summary()
    state = {
        "principal": summary.get("principal", 0.0),
        "bankroll": summary.get("bankroll", 0.0),
        "total_profit": summary.get("total_profit", 0.0),
        "profit_pct": summary.get("profit_pct", 0.0),
        "total_withdrawn": summary.get("total_withdrawn", 0.0),
        "win_rate": summary.get("win_rate", 0.0),
        "total_trades": summary.get("total_trades", 0),
        "total_wins": summary.get("total_wins", 0),
        "total_losses": summary.get("total_losses", 0),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if arb_engine:
        arb_stats = arb_engine.stats
        state["arb_total_pnl"] = arb_stats.get("total_pnl", 0.0)
        state["arb_win_rate"] = arb_stats.get("win_rate", 0.0)
        state["arb_open_positions"] = arb_stats.get("open_positions", 0)
    if lifecycle_engine:
        state["lifecycle_active_markets"] = len(lifecycle_engine.get_active_windows())
        state["lifecycle_total_pnl"] = lifecycle_engine.get_total_pnl()
    if paper_broker:
        paper_summary = paper_broker.summary()
        state["paper_balance"] = paper_summary.get("balance", 0.0)
        state["paper_positions"] = paper_summary.get("open_positions", {})
        state["paper_trades"] = paper_summary.get("total_trades", 0)
        state["paper_fees"] = paper_summary.get("total_fees", 0.0)
    balance_path = Path(os.environ.get("BOT_DB_PATH", "data")) / "balance.json"
    balance_path.parent.mkdir(parents=True, exist_ok=True)
    balance_path.write_text(json.dumps(state, indent=2))


def _write_markets_state(lifecycle_engine):
    """Write active market windows to JSON for admin API reads."""
    windows = lifecycle_engine.get_active_windows()
    state = {
        "active_windows": [w.to_dict() for w in windows[:20]],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    markets_path = Path(os.environ.get("BOT_DB_PATH", "data")) / "markets_state.json"
    markets_path.parent.mkdir(parents=True, exist_ok=True)
    markets_path.write_text(json.dumps(state, indent=2))


def main():
    logger.info("=" * 60)
    logger.info("Polymarket Crypto Trading Bot starting...")
    logger.info("Mode: %s | Categories: %s", settings.trading_mode, settings.categories_enabled)
    logger.info("Lifecycle engine: %s", "enabled" if settings.lifecycle_engine_enabled else "disabled")
    # logger.info("Gnosis Safe relayer: %s", "enabled" if settings.gnosis_safe_enabled else "disabled")
    logger.info("Compounding: %s", "enabled" if settings.compound_enabled else "disabled")
    logger.info("=" * 60)

    from connectors.polymarket_connector import PolymarketConnector
    pm_connector = PolymarketConnector()
    starting_bankroll = settings.initial_stake_usd
    if settings.trading_mode == "live":
        starting_bankroll = _live_startup(pm_connector)

    # ── Initialize all modules ────────────────────────────────────────
    from risk.risk_manager import CircuitBreakerTripped, RiskManager
    from data.trade_logger import TradeLogger
    from data.scheduler import BotScheduler

    risk_manager = RiskManager(bankroll_usd=starting_bankroll)
    trade_logger = TradeLogger()

    brti_engine, brti_ws = _init_brti_engine()
    tte_orchestrator = _init_tte_orchestrator(brti_engine)
    arb_engine = _init_arbitrage_engine(bankroll=risk_manager.bankroll)
    orderbook_arb = _init_orderbook_arb(bankroll=risk_manager.bankroll)
    pmxt = _init_pmxt_wrapper()
    strategies = _init_strategies()
    ws_connector = None
    lifecycle_engine = None

    # ── Initialize lifecycle engine (5-min markets) ───────────────────
    if settings.lifecycle_engine_enabled:
        lifecycle_engine = _init_lifecycle_engine(
            arb_engine, orderbook_arb, risk_manager, trade_logger,
            tte_orchestrator=tte_orchestrator, brti_engine=brti_engine,
            pm_connector=pm_connector,
        )
        logger.info(
            "Lifecycle engine: %s | assets=%s %dm | ML tilt: %s | executor: %s",
            "enabled", settings.updown_assets, settings.updown_interval_minutes,
            "enabled" if settings.ml_prediction_enabled else "off (volatility fair value only)",
            "LIVE" if settings.trading_mode == "live" else "paper",
        )

    # ── Initialize WebSocket connector ────────────────────────────────
    if settings.lifecycle_engine_enabled:
        ws_connector = _init_websocket_connector()
        logger.info("Polymarket WebSocket connector initialized")

    # ── Paper broker (gated by TRAINING_MODE) ─────────────────────────
    paper_broker = None
    if settings.training_mode and settings.trading_mode == "paper":
        if settings.paper_backend == "pm_trader":
            try:
                from backtest.pm_trader_bridge import PMTraderBroker
                paper_broker = PMTraderBroker(
                    risk_manager,
                    data_dir=settings.pm_trader_data_dir,
                    starting_balance=settings.pm_trader_starting_balance,
                )
                logger.info("Paper broker: pm_trader backend (data_dir=%s)", settings.pm_trader_data_dir)
            except ImportError:
                logger.warning(
                    "polymarket-paper-trader not installed, falling back to local PaperLedger. "
                    "Install with: pip install polymarket-paper-trader"
                )
                from backtest.paper_broker import PaperBroker
                paper_broker = PaperBroker(
                    risk_manager,
                    state_file=settings.paper_ledger_state_file,
                    starting_balance=settings.initial_stake_usd,
                    fee_bps=settings.paper_fee_bps,
                )
        else:
            from backtest.paper_broker import PaperBroker
            paper_broker = PaperBroker(
                risk_manager,
                state_file=settings.paper_ledger_state_file,
                starting_balance=settings.initial_stake_usd,
                fee_bps=settings.paper_fee_bps,
            )
            logger.info("Paper broker: local PaperLedger backend")
    elif not settings.training_mode:
        logger.info("TRAINING_MODE=false — paper trading disabled, preparing for live trading")

    # ── Persistent background event loop ──────────────────────────────
    # BRTI's tick loop, the exchange WebSocket reconnect loops, the
    # Polymarket WebSocket, and the TTE retrain loop are all long-running
    # asyncio tasks meant to outlive the call that starts them. Scheduling
    # them via the old throwaway-loop `_run_async` would close the loop
    # (and orphan those tasks) the instant the start-up coroutine returned —
    # they need one event loop that stays alive for the process lifetime.
    bg_loop = _start_background_loop()

    # ── Start BRTI engine ─────────────────────────────────────────────
    async def _start_brti():
        for ws in brti_ws:
            await ws.start_async()
        await brti_engine.start()

    logger.info("Starting BRTI engine with %d exchange feeds...", len(brti_ws))
    _run_on_background_loop(bg_loop, _start_brti())

    # ── Start TTE retrain loop ─────────────────────────────────────────
    _run_on_background_loop(bg_loop, tte_orchestrator.start())

    # ── Start WebSocket feeds ─────────────────────────────────────────
    if ws_connector and settings.lifecycle_engine_enabled:
        async def _start_ws():
            await ws_connector.start_async()
        try:
            _run_on_background_loop(bg_loop, _start_ws())
            logger.info("Polymarket WebSocket feeds started")
        except Exception as e:
            logger.warning("WebSocket start failed: %s (will use REST fallback)", e)

    # ── Start dashboard ───────────────────────────────────────────────
    try:
        from dashboard.app import run as run_dashboard
        threading.Thread(target=run_dashboard, daemon=True).start()
    except Exception as e:
        logger.warning("Dashboard failed to start: %s", e)

    # ── Start fast trading loop (5-min markets) ───────────────────────
    fast_loop_stop = threading.Event()
    fast_loop_thread = None
    if lifecycle_engine:
        fast_loop_thread = threading.Thread(
            target=_fast_trading_loop,
            args=(lifecycle_engine, fast_loop_stop),
            daemon=True,
            name="fast-trading-loop",
        )
        fast_loop_thread.start()

    # ── Start scheduler (5-min pings + hourly reports) ────────────────
    scheduler = BotScheduler(
        trade_logger=trade_logger,
        notifier=notifier,
        status_interval_seconds=settings.status_ping_interval_seconds,
        report_interval_seconds=settings.hourly_report_interval_seconds,
        get_risk_summary=lambda: risk_manager.get_compounding_summary() if hasattr(risk_manager, 'get_compounding_summary') else {},
        get_arb_stats=lambda: arb_engine.stats if arb_engine else {},
        get_lifecycle_stats=lambda: {
            "active": len(lifecycle_engine.get_active_windows()) if lifecycle_engine else 0,
            "pnl": lifecycle_engine.get_total_pnl() if lifecycle_engine else 0,
            "win_rate": lifecycle_engine.stats.win_rate if lifecycle_engine else 0,
            "traded": lifecycle_engine.stats.total_markets_traded if lifecycle_engine else 0,
        } if lifecycle_engine else {},
    )
    scheduler.start()

    # ── Startup notification ──────────────────────────────────────────
    startup_msg = (
        f"Bot started in {settings.trading_mode.upper()} mode\n"
        f"Training Mode: {'ON (paper trading active)' if settings.training_mode else 'OFF (preparing for live)'}\n"
        f"Principal: ${risk_manager.principal:.2f}\n"
        f"BRTI exchanges: {settings.brti_exchanges}\n"
        f"Strategies: {list(strategies.keys())}\n"
        f"Arb min spread: {settings.arb_min_spread_cents}\u00a2\n"
        f"Compounding: {'ON' if settings.compound_enabled else 'OFF'}"
    )
    if settings.lifecycle_engine_enabled:
        startup_msg += "\n5-min lifecycle engine: ENABLED"
    # if settings.gnosis_safe_enabled:
    #     startup_msg += "\nGnosis Safe relayer: ENABLED"
    notifier.send(startup_msg, Severity.INFO)

    funding_manager = None
    if settings.trading_mode == "live" and settings.auto_funding_enabled:
        from connectors.binance_connector import BinanceConnector
        from connectors.okx_connector import OKXConnector
        from execution.funding import FundingManager
        funding_manager = FundingManager(
            okx=OKXConnector() if settings.okx_api_key else None,
            binance=BinanceConnector() if settings.binance_api_key else None,
        )

    # ── Main loop ─────────────────────────────────────────────────────
    cycle_count = 0
    last_report_cycle = 0

    try:
        while True:
            cycle_start = time.time()
            cycle_count += 1

            # ── 1. BRTI tick (auto via async, just read latest) ───────
            brti_price = brti_engine.last_tick
            if brti_price is None:
                logger.debug("No BRTI tick yet, waiting...")
                time.sleep(1)
                continue

            current_btc_price = brti_price.brti_price

            # ── 2. Update external price feeds for orderbook arb ──────
            orderbook_arb.update_external_price("BTC", current_btc_price)

            # ── 3-5. Trade decision for 5-min markets ──────────────────
            # Handled by the fast trading-loop thread (_fast_trading_loop),
            # which ticks lifecycle_engine at settings.orderbook_update_interval
            # instead of once per slow cycle here, using real BRTI-derived
            # features (strategies/lifecycle_engine.py::_compute_btc_signal)
            # rather than the placeholder random-noise features this dead
            # Kalshi-settlement-only block used to feed the model.

            # ── 6. Cross-platform arbitrage scan (simulation only) ────
            pm_markets = _run_async(pmxt.get_crypto_markets()) if settings.arb_scan_enabled else []
            for market in pm_markets:
                if market.platform == "polymarket":
                    arb_engine.update_pm_prices(
                        market.market_id,
                        market.yes_price,
                        market.no_price,
                        question=market.question,
                        volume_24h=market.volume_24h,
                    )
                elif market.platform == "limit_exchange":
                    arb_engine.update_limit_exchange_prices(
                        market.market_id,
                        market.yes_price,
                        market.no_price,
                        question=market.question,
                    )
                elif market.platform == "opinion":
                    arb_engine.update_opinion_prices(
                        market.market_id,
                        market.yes_price,
                        market.no_price,
                        question=market.question,
                    )
                elif market.platform == "myriad":
                    arb_engine.update_myriad_prices(
                        market.market_id,
                        market.yes_price,
                        market.no_price,
                        question=market.question,
                    )

            opportunities = arb_engine.scan_for_opportunities() if settings.arb_scan_enabled else []
            for opp in opportunities:
                if arb_engine.evaluate_opportunity(opp):
                    _run_async(arb_engine.execute_arb(opp))

            exits = arb_engine.check_exits()
            for pos_id in exits:
                arb_engine.close_position(pos_id)

            # ── 7. Orderbook-based intra-platform arb (5-min markets) ─
            if settings.lifecycle_engine_enabled and settings.arb_scan_enabled:
                ob_opps = orderbook_arb.scan_opportunities()
                for opp in ob_opps:
                    size_usd = orderbook_arb.size_position(opp)
                    if size_usd > 0:
                        logger.info(
                            "[NL-PAPER] \u26a0\ufe0f Orderbook arb\n"
                            "Market: %s | Direction: %s\n"
                            "Complement: %.4f | Net: %.4f\n"
                            "Size: $%.2f",
                            opp.asset, opp.direction,
                            opp.complement_spread, opp.net_profit_per_share, size_usd,
                        )
                        notifier.send(
                            f"Orderbook arb: {opp.direction} {opp.asset}\n"
                            f"Complement: ${opp.complement_spread:.4f} | "
                            f"Net profit: ${opp.net_profit_per_share:.4f}/share\n"
                            f"Size: ${size_usd:.2f}",
                            Severity.INFO,
                        )

            # ── 8. Lifecycle engine tick (5-min markets) ──────────────
            # Handled by the fast trading-loop thread (see
            # _fast_trading_loop) at settings.orderbook_update_interval,
            # not here — this slow loop's cadence (signal_check_interval_
            # seconds, default 300s) is the length of an entire market
            # window, so ticking here gave each window ~1 chance to trade.

            # ── 9. Stop-loss check ────────────────────────────────────
            # (handled by risk_manager in paper_broker)

            # ── 10. Circuit breaker ───────────────────────────────────
            risk_manager.check_circuit_breaker()

            # ── 10b. Auto-funding from OKX/Binance (opt-in, live only) ─
            if funding_manager is not None and lifecycle_engine is not None:
                wd_id = funding_manager.maybe_auto_fund(
                    lifecycle_engine.executor.collateral_balance(),
                    pm_connector.secure_client().wallet,
                )
                if wd_id:
                    notifier.send(
                        f"Auto-funding: withdrew ${settings.funding_topup_usd:.2f} USDC from "
                        f"{settings.funding_source} (id {wd_id})", Severity.WARNING,
                    )

            # ── 11. Write state files for admin API ───────────────────
            if cycle_count % 10 == 0:
                _write_balance_state(risk_manager, arb_engine, lifecycle_engine, paper_broker)
                if lifecycle_engine:
                    _write_markets_state(lifecycle_engine)

            # ── 12. Periodic database backup (every 6 hours) ──────────
            if cycle_count % 720 == 0:  # 720 cycles * 5min = 6 hours
                try:
                    trade_logger.backup()
                    logger.info("Periodic database backup completed")
                except Exception as e:
                    logger.error("Periodic backup failed: %s", e)

            # ── 12. Dashboard state update ────────────────────────────
            try:
                from dashboard.state import update_state, update_paper_state
                update_state(
                    risk_manager,
                    brti_engine=brti_engine,
                    arb_engine=arb_engine,
                    tte_orchestrator=tte_orchestrator,
                    lifecycle_engine=lifecycle_engine,
                    trade_logger=trade_logger,
                )
                if paper_broker:
                    update_paper_state(paper_broker)
            except Exception:
                pass

            # ── 13. Periodic reports ──────────────────────────────────
            if cycle_count - last_report_cycle >= settings.profit_report_interval_cycles:
                _send_profit_report(risk_manager, arb_engine, brti_engine, lifecycle_engine)
                last_report_cycle = cycle_count

            # ── Sleep ─────────────────────────────────────────────────
            elapsed = time.time() - cycle_start
            sleep_time = max(0, settings.signal_check_interval_seconds - elapsed)
            time.sleep(sleep_time)

    except CircuitBreakerTripped as e:
        logger.critical("Circuit breaker: %s", e)
        notifier.send(f"CIRCUIT BREAKER — bot halted: {e}", Severity.CRITICAL)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        _send_profit_report(risk_manager, arb_engine, brti_engine, lifecycle_engine)
        notifier.send("Bot stopped by user", Severity.INFO)
    except Exception as e:
        logger.critical("Bot crashed: %s", e, exc_info=True)
        notifier.send(f"Bot crashed: {e}", Severity.CRITICAL)
        raise
    finally:
        # Cleanup
        scheduler.stop()
        fast_loop_stop.set()
        if fast_loop_thread:
            fast_loop_thread.join(timeout=5)
        if settings.trading_mode == "live":
            # FAK orders never rest, but cancel anything left over regardless.
            pm_connector.cancel_all()
        _run_on_background_loop(bg_loop, brti_engine.stop())
        for ws in brti_ws:
            ws.stop()
        _run_on_background_loop(bg_loop, tte_orchestrator.stop())
        if ws_connector:
            try:
                ws_connector.stop()
            except Exception as e:
                logger.warning("WebSocket connector stop failed: %s", e)
        bg_loop.call_soon_threadsafe(bg_loop.stop)
        # Final database backup
        try:
            trade_logger.backup()
            logger.info("Final database backup completed")
        except Exception as e:
            logger.error("Final backup failed: %s", e)
        logger.info("Bot shutdown complete")


def _send_profit_report(risk_manager, arb_engine=None, brti_engine=None, lifecycle_engine=None):
    """Send comprehensive status report."""
    summary = risk_manager.get_compounding_summary()
    lines = [
        "=== STATUS REPORT ===",
        f"Principal: ${summary['principal']:.2f}",
        f"Bankroll: ${summary['bankroll']:.2f}",
        f"Profit: ${summary['total_profit']:.2f} ({summary['profit_pct']:.1f}%)",
        f"Withdrawn: ${summary['total_withdrawn']:.2f}",
        f"Drawdown: {summary['drawdown_pct']:.1f}%",
        f"Growth Rate: {summary['compound_growth_rate_annualized']:.1f}% (ann.)",
        f"Trades: {summary['total_trades']} (W:{summary['total_wins']} L:{summary['total_losses']})",
        f"Win Rate: {summary['win_rate']:.1f}%",
        f"Compounding: {'ON' if settings.compound_enabled else 'OFF'}",
    ]
    if arb_engine:
        arb_stats = arb_engine.stats
        lines.append(f"Arb P&L: ${arb_stats['total_pnl']:.4f} ({arb_stats['positions_closed']} trades)")
    if lifecycle_engine:
        active = lifecycle_engine.get_active_windows()
        lines.append(f"Active 5-min markets: {len(active)}")
        lines.append(f"Lifecycle P&L: ${lifecycle_engine.get_total_pnl():.4f}")
    if brti_engine and brti_engine.last_tick:
        lines.append(f"BRTI: ${brti_engine.last_tick.brti_price:,.2f}")
    notifier.send("\n".join(lines), Severity.INFO)


if __name__ == "__main__":
    main()
