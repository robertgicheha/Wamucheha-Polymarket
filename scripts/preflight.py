"""
Production readiness check — run before (and after any config change to)
live trading. Read-only unless --setup-approvals is given.

    python scripts/preflight.py                       # full report
    python scripts/preflight.py --no-exchanges        # skip OKX/Binance
    python scripts/preflight.py --setup-approvals --yes
        # submit the standard Polymarket trading approvals for your wallet
        # (gasless via the relayer for Deposit/Proxy/Safe wallets; an EOA
        # pays gas in POL)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution.preflight import critical_failures, format_report, run_preflight  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-exchanges", action="store_true", help="skip OKX/Binance checks")
    parser.add_argument("--setup-approvals", action="store_true",
                        help="submit missing Polymarket trading approvals")
    parser.add_argument("--yes", action="store_true", help="confirm on-chain actions")
    args = parser.parse_args()

    if args.setup_approvals:
        if not args.yes:
            print("--setup-approvals submits transactions; re-run with --yes to confirm.")
            return 2
        from connectors.polymarket_connector import PolymarketConnector
        client = PolymarketConnector().secure_client()
        print(f"Setting up trading approvals for {client.wallet} ({client.wallet_type})...")
        client.setup_trading_approvals().wait()
        print("Approvals:", "complete" if client.get_trading_approvals_state().is_fully_approved
              else "STILL MISSING")

    checks = run_preflight(include_exchanges=not args.no_exchanges)
    print(format_report(checks))
    failures = critical_failures(checks)
    print()
    if failures:
        print(f"NOT READY: {len(failures)} critical check(s) failed.")
        return 1
    print("READY for live trading.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
