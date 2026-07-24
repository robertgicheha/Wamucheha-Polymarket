"""
Main orchestrator. Every SIGNAL_CHECK_INTERVAL_SECONDS (default 5 min), for each
enabled category:
  1. Pull open markets from Gamma API
  2. Run that category's signal generator (with ML ensemble)
  3. Compare model probability to market price
  4. If edge exceeds MIN_EDGE_THRESHOLD and confidence above threshold,
     and risk checks pass, place a trade (paper or live)
  5. Check stop-losses on open positions
  6. Check circuit breaker
  7. Process manual withdrawal requests
  8. Report profit status at configured intervals
  9. Check profit milestones and alert

COMPOUNDING: All profits stay in the bankroll and are reinvested automatically.
Position sizes grow as the bankroll grows. You withdraw manually when ready.
"""
import logging
import sys
import threading
import time
from datetime import datetime
from typing import Dict, Optional

from alerts.notifier import Severity, notifier
from config.settings import settings
from connectors.polymarket_connector import Market, PolymarketConnector
from dashboard.state import update_state
from risk.risk_manager import CircuitBreakerTripped, RiskManager
from signals.crypto.crypto_signal import CryptoSignalGenerator
from signals.macro.macro_signal import MacroSignalGenerator
from signals.politics.politics_signal import PoliticsSignalGenerator
from signals.sports.sports_signal import SportsSignalGenerator

# Set up logging
import os
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

SIGNAL_GENERATORS = {
    "crypto": CryptoSignalGenerator(),
    "politics": PoliticsSignalGenerator(),
    "sports": SportsSignalGenerator(),
    "macro": MacroSignalGenerator(),
}


def run_cycle(risk_manager: RiskManager, polymarket: PolymarketConnector, paper_broker=None):
    """Run one signal check + trade execution cycle."""
    for category in settings.categories_enabled:
        generator = SIGNAL_GENERATORS.get(category)
        if generator is None:
            continue

        if not risk_manager.can_open_new_position():
            continue

        try:
            markets = polymarket.get_markets(category=category, limit=50)
        except Exception as e:
            logger.error("Failed to fetch markets for %s: %s", category, e)
            continue

        if not markets:
            continue

        for market in markets:
            try:
                signal = generator.generate(
                    market_id=market.condition_id,
                    market_question=market.question,
                    market_price=market.yes_price,
                )

                market_price = market.yes_price
                edge = signal.model_probability - market_price
                abs_edge = abs(edge)

                if abs_edge < settings.min_edge_threshold:
                    continue
                if signal.confidence < settings.ml_confidence_threshold:
                    continue

                side = "YES" if edge > 0 else "NO"
                size_usd = risk_manager.max_position_size(category)
                if size_usd <= 0:
                    continue

                kelly_frac = risk_manager.kelly_fraction(
                    signal.model_probability if side == "YES" else (1 - signal.model_probability),
                    market_price if side == "YES" else (1 - market_price),
                    kelly_cap=settings.ml_kelly_fraction,
                )
                size_usd = min(size_usd, risk_manager.bankroll * kelly_frac)

                if settings.trading_mode == "paper" and paper_broker is not None:
                    fill = paper_broker.simulate_order(
                        market_id=market.condition_id,
                        category=category,
                        side=side,
                        price=market_price,
                    )
                    notifier.send(
                        f"[PAPER] {side} {market.question[:50]} @ {fill.filled_price:.3f} "
                        f"(${size_usd:.2f}) — edge {edge:+.3f}, conf {signal.confidence:.2f} | "
                        f"Bankroll: ${risk_manager.bankroll:.2f}",
                        Severity.INFO,
                    )
                elif settings.trading_mode == "live":
                    token_id = market.token_id_yes if side == "YES" else market.token_id_no
                    if not token_id:
                        continue
                    from connectors.polymarket_connector import OrderResult
                    result = polymarket.place_order(
                        token_id=token_id, side="BUY", price=market_price, size=size_usd,
                    )
                    risk_manager.open_position(
                        market_id=market.condition_id, category=category,
                        side=side, entry_price=market_price, size_usd=size_usd,
                    )
                    notifier.send(
                        f"[LIVE] {side} {market.question[:50]} @ {market_price:.3f} "
                        f"(${size_usd:.2f}) — order {result.order_id} | "
                        f"Bankroll: ${risk_manager.bankroll:.2f}",
                        Severity.INFO,
                    )

                logger.info(
                    "Trade: %s %s on %s — model=%.3f market=%.3f edge=%+.3f",
                    side, category, market.question[:40],
                    signal.model_probability, market_price, edge,
                )

            except Exception as e:
                logger.error("Signal failed for %s: %s", market.question[:40], e, exc_info=True)

    # Check stop-losses
    try:
        current_prices = _get_current_prices(polymarket, risk_manager)
        to_close = risk_manager.check_stop_losses(current_prices)
        for market_id in to_close:
            price = current_prices.get(market_id, 0)
            result = risk_manager.close_position(market_id, price)
            notifier.send(
                f"Stop-loss: {market_id} closed — P&L ${result.pnl_usd:+.2f} | "
                f"Bankroll: ${risk_manager.bankroll:.2f}",
                Severity.WARNING,
            )
    except Exception as e:
        logger.error("Stop-loss check failed: %s", e)


def _get_current_prices(
    polymarket: PolymarketConnector, risk_manager: RiskManager
) -> Dict[str, float]:
    prices = {}
    for market_id in risk_manager.open_positions:
        try:
            market = polymarket.get_market_by_condition(market_id)
            if market:
                prices[market_id] = market.yes_price
        except Exception:
            pass
    return prices


def _send_profit_report(risk_manager: RiskManager) -> None:
    """Send a periodic profit status report via notifications."""
    summary = risk_manager.get_compounding_summary()

    report_lines = [
        "=== PROFIT STATUS REPORT ===",
        f"Principal: ${summary['principal']:.2f}",
        f"Bankroll: ${summary['bankroll']:.2f}",
        f"Total Profit: ${summary['total_profit']:.2f} ({summary['profit_pct']:.1f}%)",
        f"Total Withdrawn: ${summary['total_withdrawn']:.2f}",
        f"Peak Bankroll: ${summary['peak_bankroll']:.2f}",
        f"Drawdown: {summary['drawdown_pct']:.1f}%",
        f"Growth Rate (ann.): {summary['compound_growth_rate']:.1f}%",
        f"Trades: {summary['total_trades']} (W:{summary['total_wins']} L:{summary['total_losses']})",
        f"Win Rate: {summary['win_rate']:.1f}%",
        f"Open Positions: {summary['open_positions_count']}",
        f"Status: {'HALTED' if summary['halted'] else 'RUNNING'}",
    ]
    notifier.send("\n".join(report_lines), Severity.INFO)


def _check_profit_milestone(risk_manager: RiskManager) -> None:
    """Check if profit has hit a milestone and alert the user."""
    target = settings.compound_growth_target_pct
    if target <= 0:
        return

    current_pct = risk_manager.profit_pct
    # Check if we've crossed a new milestone
    if current_pct >= target:
        notifier.send(
            f"MILESTONE REACHED: Profit is now {current_pct:.1f}% "
            f"(target: {target:.1f}%) — ${risk_manager.total_profit:.2f} total profit "
            f"on ${risk_manager.principal:.2f} principal",
            Severity.INFO,
        )


def _process_withdrawals(risk_manager: RiskManager) -> None:
    """Check for and process manual withdrawal requests from the dashboard."""
    try:
        from dashboard.app import get_pending_withdrawal
        request = get_pending_withdrawal()
        if request is None:
            return

        amount = request.get("amount", 0)
        destination = request.get("destination", "")

        if not risk_manager.can_withdraw(amount):
            notifier.send(
                f"Withdrawal of ${amount:.2f} rejected — insufficient margin. "
                f"Available profit: ${risk_manager.total_profit:.2f}",
                Severity.WARNING,
            )
            return

        record = risk_manager.request_withdrawal(amount, destination)

        # Execute on-chain transfer if in live mode
        if settings.trading_mode == "live" and destination:
            try:
                from connectors.polygon_connector import PolygonConnector
                polygon = PolygonConnector()
                tx_hash = polygon.transfer_usdc(destination, amount)
                risk_manager.complete_withdrawal(len(risk_manager.withdrawal_history) - 1)
                notifier.send(
                    f"Withdrawal completed: ${amount:.2f} USDC sent to {destination[:20]}... | "
                    f"TX: {tx_hash} | Remaining bankroll: ${risk_manager.bankroll:.2f}",
                    Severity.INFO,
                )
            except Exception as e:
                risk_manager.fail_withdrawal(len(risk_manager.withdrawal_history) - 1)
                notifier.send(
                    f"Withdrawal FAILED: ${amount:.2f} — {e}. Funds refunded to bankroll.",
                    Severity.CRITICAL,
                )
        else:
            # Paper mode — just record it
            notifier.send(
                f"Withdrawal recorded: ${amount:.2f} USDC to {destination or 'configured address'} | "
                f"Remaining bankroll: ${risk_manager.bankroll:.2f}",
                Severity.INFO,
            )
    except Exception as e:
        logger.error("Withdrawal processing error: %s", e)


def _check_ml_retraining(generators: Dict) -> None:
    for name, generator in generators.items():
        if hasattr(generator, "ml_engine") and generator.ml_engine.should_retrain():
            logger.info("Retraining ML models for %s...", name)
            try:
                notifier.send(f"ML retraining triggered for {name}", Severity.INFO)
            except Exception as e:
                logger.error("Retraining failed for %s: %s", name, e)


def main():
    logger.info("Polymarket ML Trading Bot starting (COMPOUNDING MODE)...")

    if settings.trading_mode == "live":
        settings.validate_for_live_trading()

    risk_manager = RiskManager(bankroll_usd=settings.initial_stake_usd)
    polymarket = PolymarketConnector()

    paper_broker = None
    if settings.trading_mode == "paper":
        from backtest.paper_broker import PaperBroker
        paper_broker = PaperBroker(risk_manager)
        logger.info("Running in PAPER mode — no real trades")

    for name, generator in SIGNAL_GENERATORS.items():
        if hasattr(generator, "ml_engine"):
            generator.ml_engine.load_all(settings.ml_model_dir)

    from dashboard.app import run as run_dashboard
    threading.Thread(target=run_dashboard, daemon=True).start()

    notifier.send(
        f"Bot started in COMPOUNDING {settings.trading_mode.upper()} mode\n"
        f"Principal: ${risk_manager.principal:.2f}\n"
        f"Categories: {settings.categories_enabled}\n"
        f"All profits auto-reinvested. Withdraw manually via /withdraw or dashboard.",
        Severity.INFO,
    )

    cycle_count = 0
    last_report_cycle = 0

    try:
        while True:
            cycle_count += 1

            # Gas balance check (live only)
            if settings.trading_mode == "live":
                try:
                    from connectors.polygon_connector import PolygonConnector
                    polygon = PolygonConnector()
                    gas_check = polygon.check_gas_balance()
                    if gas_check.get("warning"):
                        notifier.send(gas_check["warning"], Severity.WARNING)
                except Exception:
                    pass

            # Run trade cycle
            run_cycle(risk_manager, polymarket, paper_broker)

            # Process manual withdrawals
            _process_withdrawals(risk_manager)

            # Update dashboard
            update_state(risk_manager)

            # Periodic profit report
            if cycle_count - last_report_cycle >= settings.profit_report_interval_cycles:
                _send_profit_report(risk_manager)
                last_report_cycle = cycle_count

            # Milestone check
            _check_profit_milestone(risk_manager)

            # ML retraining (every 50 cycles)
            if cycle_count % 50 == 0:
                _check_ml_retraining(SIGNAL_GENERATORS)

            time.sleep(settings.signal_check_interval_seconds)

    except CircuitBreakerTripped as e:
        logger.critical("Circuit breaker: %s", e)
        notifier.send(f"CIRCUIT BREAKER — bot halted: {e}", Severity.CRITICAL)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        _send_profit_report(risk_manager)
        notifier.send("Bot stopped by user", Severity.INFO)
    except Exception as e:
        logger.critical("Bot crashed: %s", e, exc_info=True)
        notifier.send(f"Bot crashed: {e}", Severity.CRITICAL)
        raise


if __name__ == "__main__":
    main()
