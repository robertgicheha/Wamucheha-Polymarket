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

## Going live on Polymarket (CLOB V2 / pUSD)

Polymarket replaced its exchange on 2026-04-28 (CTF Exchange V2, pUSD collateral,
new order format). The bot uses the official `polymarket-client` SDK (Python ≥ 3.11);
the old `py-clob-client` no longer works against production.

**How money flows**

```
OKX / Binance ──USDC on Polygon──▶ your Polymarket bridge deposit address
                                     └─ auto-wrapped to pUSD ─▶ your Polymarket wallet (the "funder")
                                                                   ▲ orders signed by POLYMARKET_PRIVATE_KEY
```

OKX and Binance are *funding sources*. They are never the trading wallet or the
CLOB "funder". The funder is your Polymarket account wallet (profile menu on
polymarket.com).

**Checklist**

1. Create a fresh wallet and Polymarket account. Put its key in `POLYMARKET_PRIVATE_KEY`,
   the account wallet in `POLYMARKET_FUNDER_ADDRESS`, and a Relayer API key in
   `POLYMARKET_RELAYER_API_KEY(_ADDRESS)`.
2. `python scripts/preflight.py --setup-approvals --yes` (one-time trading approvals).
3. Set `FUNDING_DEPOSIT_ADDRESS` to your Polymarket bridge address and whitelist it on
   OKX (IP-whitelist the bot host) and/or Binance (withdrawals need an IP-restricted key).
   Fund with `python scripts/fund.py --source okx --amount 50 --yes`.
4. Paper-trade (`TRADING_MODE=paper`). Paper fills walk the real order book and pay
   the real taker fee, so paper PnL is a fair preview.
5. `python scripts/preflight.py` must print `READY`. Then set `TRADING_MODE=live` and
   `LIVE_TRADING_ACK=I understand this trades real money`. `main.py` re-runs the
   preflight and refuses to start on any critical failure.

**Strategy.** For each BTC up/down window, the strike is the 60s index average at
window start (the market settles on Chainlink's 60s TWAP). A volatility model prices
P(Up), and the bot buys a side only when that probability beats the ask *plus the
taker fee* (0.07·p·(1−p) per share) by `MIN_NET_EDGE`. It sells early only when the
bid, net of fees, beats the model value by `EXIT_EDGE`; otherwise it holds to
resolution (about a minute after the window ends) and redeems automatically.

**Guard rails.** `LIVE_MAX_ORDER_USD`, `LIVE_MAX_OPEN_EXPOSURE_USD`, the circuit
breaker, and a kill switch: `touch KILL_SWITCH` stops new entries immediately while
exits and redemptions continue.

## BTC 5-minute prediction models

The 5-minute crypto lifecycle engine (`strategies/lifecycle_engine.py`) can
trade on real ML predictions from the TTE model bank (`ml/tte_orchestrator.py`)
instead of its market-price-echo fallback heuristic. Since the live BRTI tick
buffer starts empty, pretrain on historical data first:

```bash
python scripts/bootstrap_tte_models.py   # one-time, ~14 days of 1m BTC klines by default
```

Check the printed Brier scores (below 0.25 = better than a coin flip) before
setting `ML_PREDICTION_ENABLED=true` in `.env`. Until that flag is set, or for
any TTE bin that isn't trained yet, the engine falls back to the previous
heuristic rather than trading on an untrained model.
