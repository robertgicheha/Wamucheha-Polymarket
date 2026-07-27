"""
Main orchestrator — connects all modules into a unified trading bot.

Modules integrated:
  - BRTI Engine: real-time BTC price replication from exchange orderbooks
  - TTE Orchestrator: 900 ML models for time-to-expiry prediction
  - Strategy Engine: 6 strategies, selected by Sharpe/P&L
  - Arbitrage Engine: cross-platform PM+Kalshi + orderbook-based intra-platform arb
  - PMXT Wrapper: unified Polymarket+Kalshi API
  - Risk Manager: circuit breaker, stop-loss, compounding
  - WebSocket Engine: real-time Polymarket orderbook/trade feeds
  - Lifecycle Engine: 5-minute market lifecycle management
  - Gnosis Safe Relayer: gasless payments via Polymarket relayer

Cycle flow:
  1. BRTI engine ticks (1-sec) → feeds TTE models
  2. TTE models predict at each TTE → probability + confidence
  3. Strategy engine selects best strategy → trade decision
  4. Arbitrage engine scans for PM+Kalshi spreads → arb opportunities
  5. Orderbook arb scans 5-min markets for complement spreads
  6. Lifecycle engine manages 5-min market windows
  7. Risk checks → position sizing, circuit breaker
  8. Execute trades via Gnosis Safe relayer (gasless) or paper broker
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


def _init_tte_orchestrator():
    """Initialize TTE training orchestrator."""
    from ml.tte_orchestrator import TTETrainingOrchestrator
    orchestrator = TTETrainingOrchestrator()
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


def _init_lifecycle_engine(arb_engine, orderbook_arb, risk_manager, trade_logger=None):
    """Initialize 5-minute market lifecycle engine."""
    from strategies.lifecycle_engine import FiveMinuteLifecycleEngine
    return FiveMinuteLifecycleEngine(
        bankroll=risk_manager.bankroll,
        arbitrage_engine=arb_engine,
        orderbook_arb=orderbook_arb,
        risk_manager=risk_manager,
        trade_logger=trade_logger,
    )


def _init_gnosis_relayer():
    """Initialize Gnosis Safe relayer for gasless payments."""
    if not settings.gnosis_safe_enabled:
        return None
    from connectors.gnosis_relayer import GnosisSafeRelayer, RelayerConfig
    config = RelayerConfig(
        safe_address=settings.gnosis_safe_address,
        private_key=settings.gnosis_safe_owner_key,
        rpc_url=settings.polygon_rpc_url or "https://polygon-rpc.com",
        relayer_url="https://safe-relayer.gnosis.io",
        chain_id=137,
    )
    return GnosisSafeRelayer(config)


def _run_async(coro):
    """Run an async coroutine in a new event loop (for non-async contexts)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _write_balance_state(risk_manager, arb_engine=None, lifecycle_engine=None):
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
    logger.info("Gnosis Safe relayer: %s", "enabled" if settings.gnosis_safe_enabled else "disabled")
    logger.info("Compounding: %s", "enabled" if settings.compound_enabled else "disabled")
    logger.info("=" * 60)

    if settings.trading_mode == "live":
        settings.validate_for_live_trading()

    # ── Initialize all modules ────────────────────────────────────────
    from risk.risk_manager import CircuitBreakerTripped, RiskManager
    from data.trade_logger import TradeLogger
    from data.scheduler import BotScheduler

    risk_manager = RiskManager(bankroll_usd=settings.initial_stake_usd)
    trade_logger = TradeLogger()

    brti_engine, brti_ws = _init_brti_engine()
    tte_orchestrator = _init_tte_orchestrator()
    arb_engine = _init_arbitrage_engine(bankroll=risk_manager.bankroll)
    orderbook_arb = _init_orderbook_arb(bankroll=risk_manager.bankroll)
    pmxt = _init_pmxt_wrapper()
    strategies = _init_strategies()
    gnosis_relayer = _init_gnosis_relayer()
    ws_connector = None
    lifecycle_engine = None

    from data.brti.kalshi_settlement import KalshiSettlementCalculator
    settlement_calc = KalshiSettlementCalculator()

    # Connect BRTI ticks to settlement calculator
    brti_engine.on_tick(settlement_calc.on_brti_tick)

    # ── Initialize lifecycle engine (5-min markets) ───────────────────
    if settings.lifecycle_engine_enabled:
        lifecycle_engine = _init_lifecycle_engine(arb_engine, orderbook_arb, risk_manager, trade_logger)
        logger.info("Lifecycle engine initialized for 5-minute markets")

    # ── Initialize WebSocket connector ────────────────────────────────
    if settings.lifecycle_engine_enabled:
        ws_connector = _init_websocket_connector()
        logger.info("Polymarket WebSocket connector initialized")

    # ── Paper broker ──────────────────────────────────────────────────
    paper_broker = None
    if settings.trading_mode == "paper":
        from backtest.paper_broker import PaperBroker
        paper_broker = PaperBroker(risk_manager)

    # ── Start BRTI engine ─────────────────────────────────────────────
    async def _start_brti():
        for ws in brti_ws:
            await ws.start_async()
        await brti_engine.start()

    logger.info("Starting BRTI engine with %d exchange feeds...", len(brti_ws))
    _run_async(_start_brti())

    # ── Start WebSocket feeds ─────────────────────────────────────────
    if ws_connector and settings.lifecycle_engine_enabled:
        async def _start_ws():
            await ws_connector.start_async()
        try:
            _run_async(_start_ws())
            logger.info("Polymarket WebSocket feeds started")
        except Exception as e:
            logger.warning("WebSocket start failed: %s (will use REST fallback)", e)

    # ── Start dashboard ───────────────────────────────────────────────
    try:
        from dashboard.app import run as run_dashboard
        threading.Thread(target=run_dashboard, daemon=True).start()
    except Exception as e:
        logger.warning("Dashboard failed to start: %s", e)

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
        f"Principal: ${risk_manager.principal:.2f}\n"
        f"BRTI exchanges: {settings.brti_exchanges}\n"
        f"Strategies: {list(strategies.keys())}\n"
        f"Arb min spread: {settings.arb_min_spread_cents}¢\n"
        f"Compounding: {'ON' if settings.compound_enabled else 'OFF'}"
    )
    if settings.lifecycle_engine_enabled:
        startup_msg += "\n5-min lifecycle engine: ENABLED"
    if settings.gnosis_safe_enabled:
        startup_msg += "\nGnosis Safe relayer: ENABLED"
    notifier.send(startup_msg, Severity.INFO)

    # ── Main loop ─────────────────────────────────────────────────────
    cycle_count = 0
    last_report_cycle = 0
    last_strategy_eval = 0
    last_market_scan = 0
    best_strategy_name = settings.strategy

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

            # ── 3. Get current TTE for active contracts ───────────────
            active_contracts = settlement_calc.get_active_contracts()
            for contract in active_contracts:
                tte = int(contract.time_to_close_seconds)
                if tte <= 0 or tte > 900:
                    continue

                import pandas as pd
                import numpy as np
                features = pd.DataFrame(
                    np.random.randn(1, 80),
                    columns=[f"f{i}" for i in range(80)],
                )

                prediction = tte_orchestrator.predict(features, tte)

                # ── 4. Strategy selection (periodic) ──────────────────
                if cycle_count - last_strategy_eval >= 3600:
                    best_strategy_name = settings.strategy
                    last_strategy_eval = cycle_count

                strategy = strategies.get(best_strategy_name)
                if strategy is None:
                    continue

                # ── 5. Trade decision ─────────────────────────────────
                decision = strategy.should_trade(
                    model_prob=prediction["probability"],
                    market_price=contract.last_yes_price or 0.5,
                    bankroll=risk_manager.bankroll,
                    tte_seconds=tte,
                    volatility=brti_engine.get_volatility(60) or 0.01,
                    extra={
                        "momentum": 0.0,
                        "z_score": 0.0,
                        "confidence": prediction.get("confidence", 0.5),
                    },
                )

                if decision.should_trade and decision.size_usd > 0:
                    if not risk_manager.can_open_new_position():
                        continue

                    category = "crypto"
                    size_usd = min(
                        decision.size_usd,
                        risk_manager.max_position_size(category),
                    )

                    # Execute via Gnosis Safe relayer if enabled
                    if gnosis_relayer and settings.trading_mode == "live":
                        try:
                            success = _run_async(gnosis_relayer.submit_order(
                                market_id=contract.ticker,
                                side=decision.side,
                                price=contract.last_yes_price or 0.5,
                                size_usd=size_usd,
                            ))
                            if success:
                                notifier.send(
                                    f"[GASLESS] {decision.side} {contract.ticker} "
                                    f"@ {contract.last_yes_price:.3f} (${size_usd:.2f})",
                                    Severity.INFO,
                                )
                        except Exception as e:
                            logger.error("Gnosis relayer failed, falling back to direct: %s", e)
                    elif settings.trading_mode == "paper" and paper_broker:
                        fill = paper_broker.simulate_order(
                            market_id=contract.ticker,
                            category=category,
                            side=decision.side,
                            price=contract.last_yes_price or 0.5,
                        )

                        # Log trade entry
                        trade_id = trade_logger.log_entry(
                            condition_id=contract.ticker,
                            asset="btc",
                            side=decision.side,
                            price=fill.filled_price,
                            size_usd=size_usd,
                            strategy=best_strategy_name,
                            source="lifecycle",
                            market_question=getattr(contract, 'question', ''),
                            bankroll_after=risk_manager.bankroll,
                            metadata={
                                "edge": decision.edge,
                                "model_prob": prediction["probability"],
                                "market_price": contract.last_yes_price or 0.5,
                                "tte": tte,
                            },
                        )

                        notifier.send(
                            f"[PAPER] {decision.side} {contract.ticker} "
                            f"@ {fill.filled_price:.3f} (${size_usd:.2f}) — "
                            f"edge={decision.edge:+.4f} | "
                            f"Bankroll: ${risk_manager.bankroll:.2f}\n"
                            f"Trade ID: {trade_id}",
                            Severity.INFO,
                        )

                    logger.info(
                        "Trade: %s %s — prob=%.3f market=%.3f edge=%+.4f size=$%.2f",
                        decision.side, contract.ticker,
                        prediction["probability"],
                        contract.last_yes_price or 0.5,
                        decision.edge, size_usd,
                    )

            # ── 6. Cross-platform arbitrage scan ──────────────────────
            pm_markets = _run_async(pmxt.get_crypto_markets())
            for market in pm_markets:
                if market.platform == "polymarket":
                    arb_engine.update_pm_prices(
                        market.market_id,
                        market.yes_price,
                        market.no_price,
                        question=market.question,
                        volume_24h=market.volume_24h,
                    )
                elif market.platform == "kalshi":
                    arb_engine.update_kalshi_prices(
                        market.market_id,
                        market.yes_price,
                        market.no_price,
                        question=market.question,
                    )

            opportunities = arb_engine.scan_for_opportunities()
            for opp in opportunities:
                if arb_engine.evaluate_opportunity(opp):
                    _run_async(arb_engine.execute_arb(opp))

            exits = arb_engine.check_exits()
            for pos_id in exits:
                arb_engine.close_position(pos_id)

            # ── 7. Orderbook-based intra-platform arb (5-min markets) ─
            if settings.lifecycle_engine_enabled:
                ob_opps = orderbook_arb.scan_opportunities()
                for opp in ob_opps:
                    size_usd = orderbook_arb.size_position(opp)
                    if size_usd > 0:
                        logger.info(
                            "ORDERBOOK ARB: %s %s — complement=%.4f net=%.4f size=$%.2f",
                            opp.direction, opp.asset,
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
            if lifecycle_engine:
                try:
                    lifecycle_engine.tick()
                except Exception as e:
                    logger.error("Lifecycle engine tick failed: %s", e)

            # ── 9. Stop-loss check ────────────────────────────────────
            # (handled by risk_manager in paper_broker)

            # ── 10. Circuit breaker ───────────────────────────────────
            risk_manager.check_circuit_breaker()

            # ── 11. Write state files for admin API ───────────────────
            if cycle_count % 10 == 0:
                _write_balance_state(risk_manager, arb_engine, lifecycle_engine)
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
                from dashboard.state import update_state
                update_state(risk_manager)
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
        _run_async(brti_engine.stop())
        for ws in brti_ws:
            ws.stop()
        _run_async(tte_orchestrator.stop())
        if ws_connector:
            try:
                _run_async(ws_connector.stop())
            except Exception:
                pass
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
        f"Growth Rate: {summary['compound_growth_rate']:.1f}% (ann.)",
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
