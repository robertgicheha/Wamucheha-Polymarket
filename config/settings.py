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
    reinvest_profits_only: bool = _bool("REINVEST_PROFITS_ONLY", False)
    compound_growth_target_pct: float = _float("COMPOUND_GROWTH_TARGET_PCT", 0.0)
    profit_report_interval_cycles: int = _int("PROFIT_REPORT_INTERVAL_CYCLES", 72)
    withdrawal_enabled: bool = _bool("WITHDRAWAL_ENABLED", True)
    withdrawal_destination_address: str = os.getenv("WITHDRAWAL_DESTINATION_ADDRESS", "")

    # Categories
    categories_enabled: List[str] = field(
        default_factory=lambda: [
            c.strip()
            for c in os.getenv("CATEGORIES_ENABLED", "crypto").split(",")
            if c.strip()
        ]
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
