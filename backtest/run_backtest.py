"""
Backtest harness. Run this against historical resolved Polymarket markets BEFORE
any paper or live trading. If a strategy doesn't show a real edge here -- after
accounting for slippage and fees -- it has no business going to paper mode.

Usage:
    python backtest/run_backtest.py --category crypto --start 2025-01-01 --end 2025-12-31

TODO once signal generators are implemented:
  1. Pull historical resolved markets for the category (Polymarket has a
     historical data API / subgraph for this)
  2. For each market, run the signal generator using ONLY data available before
     the market resolved (strict point-in-time correctness -- this is the #1
     source of backtest overfitting: accidentally leaking future data)
  3. Compare model probability vs. market price at that point in time
  4. Simulate trade decisions through PaperBroker + RiskManager
  5. Report: win rate, Brier score (calibration), Sharpe-like risk-adjusted
     return, max drawdown, and performance BY CATEGORY and BY MONTH (a strategy
     that only worked in one regime is not a strategy)

A backtest Brier score close to what you'd get from a coin flip, or a win rate
that only looks good on paper before slippage, means the strategy isn't ready --
go back to the signal layer, not straight to paper trading.
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, choices=["crypto", "politics", "sports", "macro"])
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    raise NotImplementedError(
        f"Backtest for '{args.category}' not yet implemented -- build the "
        f"signal generator for this category first (see signals/{args.category}/), "
        f"then wire historical data loading here."
    )


if __name__ == "__main__":
    main()
