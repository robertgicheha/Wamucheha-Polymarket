# Data Sources for Crypto Category Backtesting

Concrete, verified sources to wire up first, since crypto is the recommended
starting category.

## 1. Polymarket market discovery — Gamma API

Read-only, free, no API key needed.

```
GET https://gamma-api.polymarket.com/events?limit=100&closed=true&tag=crypto
GET https://gamma-api.polymarket.com/markets?limit=100
```

Use this to find resolved crypto markets and pull each market's `clobTokenIds`
(you need the CLOB *token_id*, not the `condition_id`, for the next step —
mixing these up is the #1 reason people get empty responses from CLOB).

## 2. Polymarket price history — CLOB API

Also unauthenticated and free.

```
GET https://clob.polymarket.com/prices-history?market={token_id}&interval=max
```

**Known limitation to design around:** for resolved/closed markets, this endpoint
often only returns data at 12-hour+ granularity, even for markets that had much
finer-grained price movement while live. Practical implication: for markets you
want *minute-level* backtest resolution on, capture the price history in
near-real-time (while the market is still open) and store it yourself, rather
than relying on pulling fine-grained history after resolution. For markets
resolved before you started capturing, you'll only get coarse granularity — fine
for validating longer-horizon strategies, not for anything intraday.

Also has a live orderbook + trades WebSocket (`wss://ws-subscriptions-clob.polymarket.com/ws/market`)
worth hooking into now so you're building forward history from day one, in
parallel with backtesting on what's retroactively available.

## 3. Polymarket on-chain trade history — The Graph subgraph

Note: **Polymarket migrated to new CTF Exchange contracts on 2026-04-28** and the
old subgraph indexer stopped returning complete data after that point. If you
need trade-level history spanning the migration, you'll need to either (a) use
the current subgraph for post-migration data and a preserved historical dataset
for pre-migration data, or (b) read on-chain events directly via an indexer like
HyperSync/Envio, which several open-source Polymarket data tools switched to
after the migration. Worth checking `docs.polymarket.com` and the subgraph repo
for the current endpoint before building against this — it's the piece most
likely to have moved again since this was written.

## 4. Paid/managed options (worth it once you're past the prototype stage)

If self-hosting the above becomes a bottleneck: there are a few third-party
vendors now selling structured historical Polymarket data (minute-level price
history, L2 orderbook snapshots, bulk exports) as an API or Parquet download,
generally covering data from around August 2025 onward. Not necessary to start,
but worth evaluating once the strategy is validated and you want faster
iteration than hand-rolling pagination against the free endpoints.

## 5. Crypto OHLCV (the other half of the crypto signal)

Free, well-documented, no surprises:
- Binance: `GET https://api.binance.com/api/v3/klines` — OHLCV candles, any interval
- OKX: `GET https://www.okx.com/api/v5/market/candles` — same idea, useful since
  you're already integrating OKX for funding
- Funding rate data (for the order-book-imbalance signal): Binance/OKX both expose
  a funding-rate history endpoint for perpetual futures on the same asset

## Build order for the crypto backtest specifically

1. Pull ~50-100 resolved BTC/ETH price-threshold markets via Gamma
2. For each, pull whatever CLOB price history is available (accept the 12h
   granularity limit on already-resolved markets for now)
3. Pull matching OHLCV from Binance/OKX for the same time window
4. Fit the GARCH model on OHLCV only first, backtest against the resolved
   outcomes, check calibration — this alone is a legitimate first milestone
   before the LSTM/sentiment layers are added
5. Only add complexity (LSTM, order-book imbalance, sentiment) if it
   demonstrably improves calibration over the GARCH-only baseline — each layer
   you add is also a layer that can overfit
