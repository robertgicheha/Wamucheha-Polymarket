"""
Central config loader. Everything else in the codebase should import `settings`
from here rather than calling os.environ directly, so there's one source of truth
and one place to validate required values before the bot starts trading.
"""
import os
from dataclasses import dataclass, field
from typing import List

from dotenv import dotenv_values, load_dotenv

_ENV_FILE = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(_ENV_FILE)

# `KEY=    # comment` (blank value + inline comment) is read by python-dotenv
# as the comment text itself; treat such values as empty so e.g. an unset
# address never becomes the string "# where to send...".
for _key, _val in dotenv_values(_ENV_FILE).items():
    if _val is not None and _val.strip().startswith("#") and os.environ.get(_key) == _val:
        os.environ[_key] = ""


def _bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _float(key: str, default: float) -> float:
    val = os.getenv(key)
    return float(val) if val not in (None, "") else default


def _int(key: str, default: int) -> int:
    val = os.getenv(key)
    return int(val) if val not in (None, "") else default


def _list_float(key: str, default: List[float]) -> List[float]:
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return default
    return [float(x.strip()) for x in val.split(",") if x.strip()]


LIVE_TRADING_ACK_PHRASE = "I understand this trades real money"


def _looks_like_address(value: str) -> bool:
    v = value.strip()
    return len(v) == 42 and v.startswith("0x") and all(c in "0123456789abcdefABCDEF" for c in v[2:])


def _list_str(key: str, default: List[str]) -> List[str]:
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return default
    return [x.strip() for x in val.split(",") if x.strip()]


@dataclass
class Settings:
    trading_mode: str = os.getenv("TRADING_MODE", "paper")
    training_mode: bool = _bool("TRAINING_MODE", True)
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Polymarket
    polymarket_private_key: str = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    polymarket_api_key: str = os.getenv("POLYMARKET_API_KEY", "")
    polymarket_api_secret: str = os.getenv("POLYMARKET_API_SECRET", "")
    polymarket_api_passphrase: str = os.getenv("POLYMARKET_API_PASSPHRASE", "")
    polymarket_host: str = os.getenv("POLYMARKET_HOST", "https://clob.polymarket.com")
    gamma_api_host: str = os.getenv("GAMMA_API_HOST", "https://gamma-api.polymarket.com")
    # The Polymarket account wallet that HOLDS the pUSD and positions (the
    # "funder" in CLOB terms). For accounts made on polymarket.com this is the
    # address shown in the profile menu (Deposit / Proxy / Safe wallet), NOT
    # your signer EOA and NOT an OKX/Binance deposit address. Leave empty to
    # use the signer key's own Deposit Wallet.
    polymarket_funder_address: str = os.getenv("POLYMARKET_FUNDER_ADDRESS", "")
    # Relayer API key (polymarket.com → Settings → API Keys → Relayer API Keys)
    # enables gasless approvals and redemptions for Deposit/Proxy/Safe wallets.
    polymarket_relayer_api_key: str = os.getenv("POLYMARKET_RELAYER_API_KEY", "")
    polymarket_relayer_api_key_address: str = os.getenv("POLYMARKET_RELAYER_API_KEY_ADDRESS", "")

    # # Kalshi (commented out)
    # kalshi_api_key: str = os.getenv("KALSHI_API_KEY", "")
    # kalshi_private_key: str = os.getenv("KALSHI_PRIVATE_KEY", "")
    # kalshi_api_base: str = os.getenv("KALSHI_API_BASE", "https://trading-api.kalshi.com/trade-api/v2")
    # kalshi_ws_base: str = os.getenv("KALSHI_WS_BASE", "wss://trading-api.kalshi.com/ws/v1")

    # Arbitrage platforms
    limit_exchange_enabled: bool = _bool("LIMIT_EXCHANGE_ENABLED", True)
    opinion_enabled: bool = _bool("OPINION_ENABLED", True)
    myriad_enabled: bool = _bool("MYRIAD_ENABLED", True)

    # PMXT (unified prediction market API)
    pmxt_api_key: str = os.getenv("PMXT_API_KEY", "")
    pmxt_wallet_address: str = os.getenv("PMXT_WALLET_ADDRESS", "")
    pmxt_private_key: str = os.getenv("PMXT_PRIVATE_KEY", "")

    # Polygon
    polygon_rpc_url: str = os.getenv("POLYGON_RPC_URL", "")
    # Comma-separated backup RPCs, tried in order when the primary fails.
    polygon_rpc_fallback_urls: List[str] = field(
        default_factory=lambda: _list_str("POLYGON_RPC_FALLBACK_URLS", ["https://polygon.drpc.org"])
    )
    polygon_wallet_address: str = os.getenv("POLYGON_WALLET_ADDRESS", "")
    polygon_rpc_timeout_seconds: float = _float("POLYGON_RPC_TIMEOUT_SECONDS", 10.0)

    # The Graph subgraph
    graph_subgraph_url: str = os.getenv("GRAPH_SUBGRAPH_URL", "")
    graph_subgraph_url_legacy: str = os.getenv("GRAPH_SUBGRAPH_URL_LEGACY", "")
    graph_migration_date: str = os.getenv("GRAPH_MIGRATION_DATE", "2026-04-28")

    # Binance (OHLCV + funding rates)
    binance_api_key: str = os.getenv("BINANCE_API_KEY", "")
    binance_api_secret: str = os.getenv("BINANCE_API_SECRET", "")

    # OKX
    okx_api_key: str = os.getenv("OKX_API_KEY", "")
    okx_api_secret: str = os.getenv("OKX_API_SECRET", "")
    okx_api_passphrase: str = os.getenv("OKX_API_PASSPHRASE", "")

    # Funding (OKX / Binance → Polymarket). The exchanges are funding SOURCES:
    # the bot withdraws USDC on Polygon from them to FUNDING_DEPOSIT_ADDRESS,
    # which must be your Polymarket deposit address (polymarket.com → Deposit,
    # Polygon USDC) and must be whitelisted on the exchange. Withdrawals are
    # only ever sent to that single address.
    funding_deposit_address: str = os.getenv("FUNDING_DEPOSIT_ADDRESS", "")
    auto_funding_enabled: bool = _bool("AUTO_FUNDING_ENABLED", False)
    funding_source: str = os.getenv("FUNDING_SOURCE", "okx")  # okx | binance
    funding_min_balance_usd: float = _float("FUNDING_MIN_BALANCE_USD", 20.0)
    funding_topup_usd: float = _float("FUNDING_TOPUP_USD", 50.0)
    funding_max_withdrawal_usd: float = _float("FUNDING_MAX_WITHDRAWAL_USD", 100.0)
    funding_max_daily_usd: float = _float("FUNDING_MAX_DAILY_USD", 200.0)

    # News / data
    newsapi_key: str = os.getenv("NEWSAPI_KEY", "")
    gdelt_enabled: bool = _bool("GDELT_ENABLED", True)

    # Stake & compounding
    initial_stake_usd: float = _float("INITIAL_STAKE_USD", 100.0)
    compound_enabled: bool = _bool("COMPOUND_ENABLED", True)
    reinvest_profits_only: bool = _bool("REINVEST_PROFITS_ONLY", False)
    compound_growth_target_pct: float = _float("COMPOUND_GROWTH_TARGET_PCT", 0.0)
    profit_report_interval_cycles: int = _int("PROFIT_REPORT_INTERVAL_CYCLES", 72)
    withdrawal_enabled: bool = _bool("WITHDRAWAL_ENABLED", True)
    withdrawal_destination_address: str = os.getenv("WITHDRAWAL_DESTINATION_ADDRESS", "")

    # Categories
    categories_enabled: List[str] = field(
        default_factory=lambda: _list_str("CATEGORIES_ENABLED", ["crypto"])
    )
    signal_check_interval_seconds: int = _int("SIGNAL_CHECK_INTERVAL_SECONDS", 300)
    min_edge_threshold: float = _float("MIN_EDGE_THRESHOLD", 0.05)

    # Risk management
    stop_loss_pct: float = _float("STOP_LOSS_PCT", 8.0)
    circuit_breaker_daily_drawdown_pct: float = _float(
        "CIRCUIT_BREAKER_DAILY_DRAWDOWN_PCT", 5.0
    )
    circuit_breaker_max_consecutive_losses: int = _int(
        "CIRCUIT_BREAKER_MAX_CONSECUTIVE_LOSSES", 3
    )
    max_position_size_pct: float = _float("MAX_POSITION_SIZE_PCT", 5.0)
    max_concurrent_positions: int = _int("MAX_CONCURRENT_POSITIONS", 10)
    max_exposure_per_category_pct: float = _float(
        "MAX_EXPOSURE_PER_CATEGORY_PCT", 40.0
    )

    # Price zone filtering (based on empirical calibration research)
    min_price_threshold: float = _float("MIN_PRICE_THRESHOLD", 0.08)
    max_price_threshold: float = _float("MAX_PRICE_THRESHOLD", 0.85)

    # Trading strategy
    strategy: str = os.getenv("STRATEGY", "kelly")

    # Paper trading (PaperLedger)
    paper_backend: str = os.getenv("PAPER_BACKEND", "pm_trader")  # "pm_trader" | "local"
    paper_ledger_state_file: str = os.getenv("PAPER_LEDGER_STATE_FILE", "data/paper_ledger_state.json")
    paper_fee_bps: float = _float("PAPER_FEE_BPS", 200.0)
    pm_trader_data_dir: str = os.getenv("PM_TRADER_DATA_DIR", "data/pm_trader")
    pm_trader_starting_balance: float = _float("PM_TRADER_STARTING_BALANCE", 10000.0)

    # ML / AI
    ml_model_dir: str = os.getenv("ML_MODEL_DIR", "models")
    ml_retrain_interval_hours: int = _int("ML_RETRAIN_INTERVAL_HOURS", 168)
    ml_ensemble_weights: List[float] = field(
        default_factory=lambda: _list_float("ML_ENSEMBLE_WEIGHTS", [0.35, 0.25, 0.20, 0.20])
    )
    ml_sentiment_model: str = os.getenv(
        "ML_SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    ml_lookback_days: int = _int("ML_LOOKBACK_DAYS", 90)
    ml_min_samples_for_training: int = _int("ML_MIN_SAMPLES_FOR_TRAINING", 50)
    ml_kelly_fraction: float = _float("ML_KELLY_FRACTION", 0.25)
    ml_confidence_threshold: float = _float("ML_CONFIDENCE_THRESHOLD", 0.3)
    ml_prediction_enabled: bool = _bool("ML_PREDICTION_ENABLED", False)
    # Allow live trading on the volatility fair-value model alone (no ML).
    # Only enable after paper results show a positive net edge.
    allow_baseline_live: bool = _bool("ALLOW_BASELINE_LIVE", False)

    # BRTI Engine
    brti_exchanges: List[str] = field(
        default_factory=lambda: _list_str(
            "BRTI_EXCHANGES", ["coinbase", "kraken", "bitstamp", "gemini"]
        )
    )
    brti_order_size_cap: float = _float("BRTI_ORDER_SIZE_CAP", 100.0)
    brti_max_volume: float = _float("BRTI_MAX_VOLUME", 5000.0)
    brti_deviation_threshold: float = _float("BRTI_DEVIATION_THRESHOLD", 0.005)
    brti_validation_enabled: bool = _bool("BRTI_VALIDATION_ENABLED", True)
    brti_max_divergence_bps: float = _float("BRTI_MAX_DIVERGENCE_BPS", 5.0)
    brti_tick_interval_seconds: int = _int("BRTI_TICK_INTERVAL_SECONDS", 1)
    # brti_kalshi_avg_window: int = _int("BRTI_KALSHI_AVG_WINDOW", 60)

    # ML BTC prediction (Layer 1)
    ml_btc_train_parallel: bool = _bool("ML_BTC_TRAIN_PARALLEL", True)
    ml_btc_max_workers: int = _int("ML_BTC_MAX_WORKERS", 8)
    ml_retrain_hour: int = _int("ML_RETRAIN_HOUR", 3)
    ml_lstm_epochs: int = _int("ML_LSTM_EPOCHS", 50)
    ml_sequence_length: int = _int("ML_SEQUENCE_LENGTH", 60)

    # Arbitrage
    arb_min_spread_cents: float = _float("ARB_MIN_SPREAD_CENTS", 5.0)
    arb_max_hold_minutes: int = _int("ARB_MAX_HOLD_MINUTES", 30)

    # Alerts
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    discord_webhook_url: str = os.getenv("DISCORD_WEBHOOK_URL", "")
    alert_email_smtp_host: str = os.getenv("ALERT_EMAIL_SMTP_HOST", "smtp.gmail.com")
    alert_email_smtp_port: int = _int("ALERT_EMAIL_SMTP_PORT", 587)
    alert_email_from: str = os.getenv("ALERT_EMAIL_FROM", "")
    alert_email_to: str = os.getenv("ALERT_EMAIL_TO", "")
    alert_email_password: str = os.getenv("ALERT_EMAIL_PASSWORD", "")

    dashboard_port: int = _int("DASHBOARD_PORT", 8080)
    dashboard_host: str = os.getenv("DASHBOARD_HOST", "0.0.0.0")
    dashboard_username: str = os.getenv("DASHBOARD_USERNAME", "")
    dashboard_password: str = os.getenv("DASHBOARD_PASSWORD", "")

    # # Gnosis Safe / Relayer (gasless payments) — commented out
    # gnosis_safe_enabled: bool = _bool("GNOSIS_SAFE_ENABLED", False)
    # gnosis_safe_address: str = os.getenv("GNOSIS_SAFE_ADDRESS", "")
    # gnosis_safe_owner_key: str = os.getenv("GNOSIS_SAFE_OWNER_KEY", "")
    # relayer_enabled: bool = _bool("RELAYER_ENABLED", True)
    # relayer_gas_limit: int = _int("RELAYER_GAS_LIMIT", 500000)
    # batch_transactions: bool = _bool("BATCH_TRANSACTIONS", True)

    # 5-minute market engine
    lifecycle_engine_enabled: bool = _bool("LIFECYCLE_ENGINE_ENABLED", True)
    market_discovery_interval: int = _int("MARKET_DISCOVERY_INTERVAL", 30)
    early_exit_take_profit_pct: float = _float("EARLY_EXIT_TAKE_PROFIT_PCT", 5.0)
    early_exit_stop_loss_pct: float = _float("EARLY_EXIT_STOP_LOSS_PCT", 8.0)
    orderbook_update_interval: float = _float("ORDERBOOK_UPDATE_INTERVAL", 1.0)
    # Which up/down markets to trade: assets and window length in minutes.
    updown_assets: List[str] = field(
        default_factory=lambda: _list_str("UPDOWN_ASSETS", ["btc"])
    )
    updown_interval_minutes: int = _int("UPDOWN_INTERVAL_MINUTES", 5)

    # Execution & live-trading guard rails
    # Net edge (model prob − all-in cost incl. taker fee) required to enter.
    min_net_edge: float = _float("MIN_NET_EDGE", 0.03)
    # Sell early only if the bid (after fees) beats model value by this much.
    exit_edge: float = _float("EXIT_EDGE", 0.02)
    max_spread: float = _float("MAX_SPREAD", 0.04)
    # No entries in the first/last N seconds of a window.
    entry_min_elapsed_seconds: int = _int("ENTRY_MIN_ELAPSED_SECONDS", 20)
    entry_min_remaining_seconds: int = _int("ENTRY_MIN_REMAINING_SECONDS", 20)
    # Reference-price (Chainlink vs our exchange index) basis risk, in bps,
    # added to the volatility used by the fair-value model.
    settlement_basis_bps: float = _float("SETTLEMENT_BASIS_BPS", 2.0)
    live_max_order_usd: float = _float("LIVE_MAX_ORDER_USD", 10.0)
    live_max_open_exposure_usd: float = _float("LIVE_MAX_OPEN_EXPOSURE_USD", 30.0)
    # Must be set to exactly this phrase to allow live orders.
    live_trading_ack: str = os.getenv("LIVE_TRADING_ACK", "")
    kill_switch_file: str = os.getenv("KILL_SWITCH_FILE", "KILL_SWITCH")
    auto_redeem_enabled: bool = _bool("AUTO_REDEEM_ENABLED", True)
    # Cross-platform arb is simulation-only; it never places real orders.
    arb_scan_enabled: bool = _bool("ARB_SCAN_ENABLED", False)

    # Scheduler (5-min pings + hourly reports)
    status_ping_interval_seconds: int = _int("STATUS_PING_INTERVAL_SECONDS", 300)
    hourly_report_interval_seconds: int = _int("HOURLY_REPORT_INTERVAL_SECONDS", 3600)

    # Database & backup
    trade_log_db_path: str = os.getenv("TRADE_LOG_DB_PATH", "data/trade_log.db")
    backup_interval_hours: int = _int("BACKUP_INTERVAL_HOURS", 6)
    backup_retain_days: int = _int("BACKUP_RETAIN_DAYS", 30)

    def validate_for_live_trading(self) -> None:
        """Call this before allowing TRADING_MODE=live. Fails loud, not silent."""
        problems = []
        for field_name in ("polymarket_private_key", "polygon_rpc_url"):
            if not getattr(self, field_name):
                problems.append(f"missing {field_name.upper()}")
        if self.live_trading_ack != LIVE_TRADING_ACK_PHRASE:
            problems.append(
                f'LIVE_TRADING_ACK must be set to "{LIVE_TRADING_ACK_PHRASE}"'
            )
        if not self.ml_prediction_enabled and not self.allow_baseline_live:
            problems.append(
                "neither ML_PREDICTION_ENABLED nor ALLOW_BASELINE_LIVE is set — "
                "refusing to trade live without a validated model"
            )
        if self.live_max_order_usd <= 0 or self.live_max_open_exposure_usd <= 0:
            problems.append("LIVE_MAX_ORDER_USD and LIVE_MAX_OPEN_EXPOSURE_USD must be > 0")
        for addr_field in ("polymarket_funder_address", "funding_deposit_address",
                           "withdrawal_destination_address"):
            value = getattr(self, addr_field)
            if value and not _looks_like_address(value):
                problems.append(f"{addr_field.upper()} is not a valid 0x address")
        if self.auto_funding_enabled and not self.funding_deposit_address:
            problems.append("AUTO_FUNDING_ENABLED requires FUNDING_DEPOSIT_ADDRESS")
        if problems:
            raise RuntimeError(
                "Cannot start live trading:\n  - " + "\n  - ".join(problems)
            )


settings = Settings()
