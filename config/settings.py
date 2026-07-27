"""
Central config loader. Everything else in the codebase should import `settings`
from here rather than calling os.environ directly, so there's one source of truth
and one place to validate required values before the bot starts trading.
"""
import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


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


def _list_str(key: str, default: List[str]) -> List[str]:
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return default
    return [x.strip() for x in val.split(",") if x.strip()]


@dataclass
class Settings:
    trading_mode: str = os.getenv("TRADING_MODE", "paper")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Polymarket
    polymarket_private_key: str = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    polymarket_api_key: str = os.getenv("POLYMARKET_API_KEY", "")
    polymarket_api_secret: str = os.getenv("POLYMARKET_API_SECRET", "")
    polymarket_api_passphrase: str = os.getenv("POLYMARKET_API_PASSPHRASE", "")
    polymarket_host: str = os.getenv("POLYMARKET_HOST", "https://clob.polymarket.com")
    gamma_api_host: str = os.getenv("GAMMA_API_HOST", "https://gamma-api.polymarket.com")

    # Kalshi
    kalshi_api_key: str = os.getenv("KALSHI_API_KEY", "")
    kalshi_private_key: str = os.getenv("KALSHI_PRIVATE_KEY", "")
    kalshi_api_base: str = os.getenv("KALSHI_API_BASE", "https://trading-api.kalshi.com/trade-api/v2")
    kalshi_ws_base: str = os.getenv("KALSHI_WS_BASE", "wss://trading-api.kalshi.com/ws/v1")

    # PMXT (unified prediction market API)
    pmxt_api_key: str = os.getenv("PMXT_API_KEY", "")
    pmxt_wallet_address: str = os.getenv("PMXT_WALLET_ADDRESS", "")
    pmxt_private_key: str = os.getenv("PMXT_PRIVATE_KEY", "")

    # Polygon
    polygon_rpc_url: str = os.getenv("POLYGON_RPC_URL", "")
    polygon_wallet_address: str = os.getenv("POLYGON_WALLET_ADDRESS", "")

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
    brti_max_divergence_bps: float = _float("BRTI_MAX_DIVERGENCE_BPS", 0.5)
    brti_tick_interval_seconds: int = _int("BRTI_TICK_INTERVAL_SECONDS", 1)
    brti_kalshi_avg_window: int = _int("BRTI_KALSHI_AVG_WINDOW", 60)

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

    # Gnosis Safe / Relayer (gasless payments)
    gnosis_safe_enabled: bool = _bool("GNOSIS_SAFE_ENABLED", False)
    gnosis_safe_address: str = os.getenv("GNOSIS_SAFE_ADDRESS", "")
    gnosis_safe_owner_key: str = os.getenv("GNOSIS_SAFE_OWNER_KEY", "")
    relayer_enabled: bool = _bool("RELAYER_ENABLED", True)
    relayer_gas_limit: int = _int("RELAYER_GAS_LIMIT", 500000)
    batch_transactions: bool = _bool("BATCH_TRANSACTIONS", True)

    # 5-minute market engine
    lifecycle_engine_enabled: bool = _bool("LIFECYCLE_ENGINE_ENABLED", True)
    market_discovery_interval: int = _int("MARKET_DISCOVERY_INTERVAL", 30)
    early_exit_take_profit_pct: float = _float("EARLY_EXIT_TAKE_PROFIT_PCT", 5.0)
    early_exit_stop_loss_pct: float = _float("EARLY_EXIT_STOP_LOSS_PCT", 8.0)
    orderbook_update_interval: float = _float("ORDERBOOK_UPDATE_INTERVAL", 1.0)

    # Scheduler (5-min pings + hourly reports)
    status_ping_interval_seconds: int = _int("STATUS_PING_INTERVAL_SECONDS", 300)
    hourly_report_interval_seconds: int = _int("HOURLY_REPORT_INTERVAL_SECONDS", 3600)

    # Database & backup
    trade_log_db_path: str = os.getenv("TRADE_LOG_DB_PATH", "data/trade_log.db")
    backup_interval_hours: int = _int("BACKUP_INTERVAL_HOURS", 6)
    backup_retain_days: int = _int("BACKUP_RETAIN_DAYS", 30)

    def validate_for_live_trading(self) -> None:
        """Call this before allowing TRADING_MODE=live. Fails loud, not silent."""
        missing = []
        required = [
            "polymarket_private_key",
            "polymarket_api_key",
            "polygon_rpc_url",
            "polygon_wallet_address",
        ]
        for field_name in required:
            if not getattr(self, field_name):
                missing.append(field_name.upper())
        if missing:
            raise RuntimeError(
                f"Cannot start live trading, missing required config: {missing}"
            )
        if self.trading_mode == "live":
            print(
                "!! TRADING_MODE=live -- this bot will place real orders with real "
                "capital. Confirm you have run paper mode for at least 2-4 weeks "
                "and reviewed backtest results before proceeding."
            )


settings = Settings()
