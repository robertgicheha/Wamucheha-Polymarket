"""
Move USDC from OKX or Binance to your Polymarket account (USDC on Polygon →
your Polymarket bridge deposit address → auto-wrapped to pUSD).

    python scripts/fund.py --status
    python scripts/fund.py --source okx --amount 50 --yes

Requires TRADING_MODE=live, FUNDING_DEPOSIT_ADDRESS set to the bridge address
Polymarket issues for your trading wallet (verified before sending), and that
address whitelisted on the exchange. Caps: FUNDING_MAX_WITHDRAWAL_USD per
withdrawal, FUNDING_MAX_DAILY_USD per rolling 24h.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402
from connectors.binance_connector import BinanceConnector  # noqa: E402
from connectors.okx_connector import OKXConnector  # noqa: E402
from connectors.polymarket_connector import PolymarketConnector  # noqa: E402
from execution.funding import FundingManager  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["okx", "binance"])
    parser.add_argument("--amount", type=float)
    parser.add_argument("--status", action="store_true", help="show balances and caps only")
    parser.add_argument("--yes", action="store_true", help="confirm the withdrawal")
    args = parser.parse_args()

    manager = FundingManager(
        okx=OKXConnector() if settings.okx_api_key else None,
        binance=BinanceConnector() if settings.binance_api_key else None,
    )
    for name, client in manager.sources.items():
        try:
            print(f"{name}: ${client.get_usdc_balance():.2f} USDC available")
        except Exception as e:
            print(f"{name}: unavailable — {e}")
    print(f"sent in last 24h: ${manager.withdrawn_last_24h():.2f} / cap ${settings.funding_max_daily_usd:.2f}")
    print(f"destination: {settings.funding_deposit_address or '(FUNDING_DEPOSIT_ADDRESS not set)'}")
    if args.status:
        return 0
    if not (args.source and args.amount):
        parser.error("--source and --amount are required (or use --status)")
    if not args.yes:
        print(f"Would withdraw ${args.amount:.2f} USDC from {args.source}. Re-run with --yes to send.")
        return 2

    wallet = PolymarketConnector().secure_client().wallet
    wd_id = manager.fund(args.source, args.amount, wallet)
    print(f"Withdrawal submitted: {wd_id}. pUSD usually arrives within a few minutes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
