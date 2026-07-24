# Polymarket Multi-Category Trading Bot

Automated trading bot for Polymarket, funded via OKX -> Polygon, trading across
Crypto, Politics, Sports, and Macro Economics markets.

## Read this before you touch live capital

**There is no realistic version of this bot that hits 90%+ accuracy, and it is not
implemented here.** Polymarket prices are aggregated market consensus. A system that
reliably beat that consensus by 20-40+ points would either be exploiting a genuine
market inefficiency (rare, usually on illiquid markets you can't size into) or would
get arbitraged away almost immediately. Realistic, sustainable directional accuracy
on well-chosen subsets is in the 55-65% range. The risk management layer in this repo
(stop-loss, circuit breaker, position sizing) exists specifically because the edge is
real but modest, and a few consecutive losses can otherwise wipe out weeks of gains.

Treat any backtest or paper-trading result above ~70% accuracy as a sign of overfitting
or a data leak, not a sign the model is unusually good.

## Status

This is a scaffold, not a finished system. Risk management and connector interfaces
are implemented. Each category's signal-generation pipeline (`signals/<category>/`)
is stubbed with a clear interface and TODOs — these are genuinely different projects
per category (price-series modeling for crypto vs. news/NLP for politics/macro vs.
stats-based modeling for sports) and need to be built and backtested independently
before going live.

## Architecture

```
connectors/       Polymarket CLOB, Polygon (web3.py), OKX (funding only)
signals/          Per-category signal generation -> calibrated probability
risk/             Stop-loss, circuit breaker, position sizing (Kelly-fraction)
ml/               Shared model utilities (ensemble, calibration, retraining)
data/             News feeds, price feeds, historical resolution data
dashboard/        Monitoring dashboard
alerts/           Telegram, Discord, email notifications
backtest/         Backtest harness against historical Polymarket data
config/           .env-based configuration
```

## Build order (recommended)

1. Get `connectors/` working end-to-end in **paper mode** (real odds, fake money)
2. Build and backtest the **crypto** signal pipeline first — it has the most
   tractable ground truth (verifiable price feed at resolution time)
3. Only after crypto backtests show a real, non-overfit edge, extend to
   politics/sports/macro — each needs its own data pipeline and backtest
4. Run paper mode for at least 2-4 weeks across all categories before flipping
   `TRADING_MODE=live` in `.env`

## Setup

```bash
cp config/.env.example config/.env
# fill in your API keys/wallet details
pip install -r requirements.txt
python backtest/run_backtest.py --category crypto   # start here
```
