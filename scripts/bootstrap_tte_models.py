"""
One-time bootstrap: pretrain the 900-bin TTE model bank
(ml/tte_orchestrator.py) on historical BTC klines, so the bot isn't cold on
day one — the live BRTI tick buffer starts empty and needs real uptime to
accumulate enough history to train on otherwise.

Klines only give OHLCV (no orderbook depth), so bootstrap-trained models
rely mainly on the momentum/technical/time-based feature groups in
ml/btc_features.py — that's fine, and matches the module's own "LR baseline
is gold standard, L1b must beat it on Brier score to be included" design.
Orderbook/microstructure features enrich the models further once the bot
has been running live and accumulating real BRTI tick history.

Usage:
    python scripts/bootstrap_tte_models.py
    python scripts/bootstrap_tte_models.py --lookback-days 90 --interval 1m

Run this once before setting ML_PREDICTION_ENABLED=true. Training all 900
TTE bins (each with an LR baseline + XGBoost + LSTM + GRU + a logistic
blender + a neural meta-learner) takes a while — expect anywhere from
minutes to a few hours depending on hardware and --lookback-days. It's safe
to re-run; results overwrite the previous model bank in settings.ml_model_dir.
"""
import argparse
import asyncio
import logging
import sys
import time

import pandas as pd

sys.path.insert(0, ".")

from config.settings import settings  # noqa: E402
from connectors.binance_connector import BinanceConnector  # noqa: E402
from ml.tte_orchestrator import TTETrainingOrchestrator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BINANCE_KLINE_LIMIT = 1000  # Binance's max candles per request


def fetch_historical_klines(symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
    """
    Paginate Binance klines backward from now to cover `lookback_days`,
    since a single request is capped at BINANCE_KLINE_LIMIT candles.
    """
    connector = BinanceConnector()
    interval_ms = {
        "1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000,
    }.get(interval)
    if interval_ms is None:
        raise ValueError(f"Unsupported interval for bootstrap: {interval}")

    end_time = int(time.time() * 1000)
    start_time = end_time - lookback_days * 86_400_000
    all_candles = []

    cursor = start_time
    while cursor < end_time:
        candles = connector.get_ohlcv(
            symbol=symbol, interval=interval, limit=BINANCE_KLINE_LIMIT,
            start_time=cursor, end_time=end_time,
        )
        if not candles:
            break
        all_candles.extend(candles)
        last_ts = candles[-1]["ts"]
        if last_ts <= cursor:
            break
        cursor = last_ts + interval_ms
        logger.info(
            "Fetched %d candles (up to %s), %d total so far",
            len(candles), pd.to_datetime(last_ts, unit="ms"), len(all_candles),
        )
        time.sleep(0.2)  # be polite to Binance's public rate limits

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles).drop_duplicates(subset="ts").sort_values("ts")
    return pd.DataFrame({
        "timestamp": df["ts"].values / 1000.0,  # seconds, matching BRTITick.timestamp
        "price": df["close"].values,
    })


async def main_async(args) -> None:
    logger.info(
        "Fetching %d days of %s %s klines for bootstrap training...",
        args.lookback_days, args.symbol, args.interval,
    )
    data = fetch_historical_klines(args.symbol, args.interval, args.lookback_days)
    if data.empty:
        logger.error("No historical data fetched — aborting bootstrap.")
        return

    logger.info("Fetched %d candles spanning %.1f days", len(data), args.lookback_days)

    orchestrator = TTETrainingOrchestrator(model_dir=args.model_dir)
    result = await orchestrator.train_full(data)

    logger.info("Bootstrap training complete: %s", result)
    if result.get("mean_brier") is not None and result["mean_brier"] >= 0.25:
        logger.warning(
            "Mean Brier score (%.4f) is no better than a coin flip (0.25) — "
            "check the label construction / feature coverage before trusting "
            "these models in ML_PREDICTION_ENABLED mode.",
            result["mean_brier"],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1m", choices=["1m", "5m", "15m", "1h"])
    parser.add_argument(
        "--lookback-days", type=int, default=min(settings.ml_lookback_days, 14),
        help="Default is capped at 14 days to keep a first bootstrap run's "
             "wall-clock time reasonable; pass --lookback-days 90 (or "
             "settings.ml_lookback_days) for a fuller sweep once you've "
             "confirmed the pipeline works end-to-end.",
    )
    parser.add_argument("--model-dir", default=settings.ml_model_dir)
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
