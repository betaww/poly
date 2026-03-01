"""
Configuration for Polymarket trading system.
All tunable parameters in one place.
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WalletConfig:
    """Polygon wallet & API auth."""
    private_key: str = ""
    funder_address: str = ""  # proxy wallet address (if using Magic/email wallet)
    signature_type: int = 0   # 0=EOA, 1=Magic/email, 2=Browser proxy
    chain_id: int = 137       # Polygon mainnet

    @classmethod
    def from_env(cls) -> "WalletConfig":
        return cls(
            private_key=os.environ.get("POLYMARKET_PRIVATE_KEY", ""),
            funder_address=os.environ.get("POLYMARKET_FUNDER_ADDRESS", ""),
            signature_type=int(os.environ.get("POLYMARKET_SIGNATURE_TYPE", "0")),
        )


@dataclass
class APIConfig:
    """API endpoints."""
    clob_host: str = "https://clob.polymarket.com"
    gamma_host: str = "https://gamma-api.polymarket.com"
    data_host: str = "https://data-api.polymarket.com"

    # API credentials (derived from private key via py-clob-client)
    api_key: str = ""
    api_secret: str = ""
    api_passphrase: str = ""

    @classmethod
    def from_env(cls) -> "APIConfig":
        return cls(
            api_key=os.environ.get("POLYMARKET_API_KEY", ""),
            api_secret=os.environ.get("POLYMARKET_API_SECRET", ""),
            api_passphrase=os.environ.get("POLYMARKET_API_PASSPHRASE", ""),
        )


@dataclass
class MarketConfig:
    """Market scanning parameters."""
    # Assets to trade
    assets: list = field(default_factory=lambda: ["btc", "eth"])  # E7: multi-asset
    timeframes: list = field(default_factory=lambda: ["5m", "15m", "1h", "4h"])
    timeframe: str = "5m"
    # Slug patterns per timeframe
    SLUG_PATTERNS: dict = field(default_factory=lambda: {
        "5m":  "{asset}-updown-5m-{ts}",
        "15m": "{asset}-updown-15m-{ts}",
        "1h":  "{asset}-updown-1h-{ts}",
        "4h":  "{asset}-updown-4h-{ts}",
    })
    # Timeframe durations in seconds
    DURATIONS: dict = field(default_factory=lambda: {
        "5m": 300,
        "15m": 900,
        "1h": 3600,
        "4h": 14400,
    })


@dataclass
class StrategyConfig:
    """Strategy tuning knobs."""
    # --- Market Making (legacy CryptoMM, kept for reference) ---
    base_spread: float = 0.03          # 3 cents base spread
    min_spread: float = 0.02           # minimum spread
    max_spread: float = 0.08           # max spread in high volatility
    order_size_usd: float = 10.0       # per-side order size
    max_position_usd: float = 50.0     # max exposure per round
    refresh_interval_ms: int = 1000    # quote refresh rate

    # --- DirectionalSniper (v6) ---
    sniper_base_price: float = 0.60    # maker bid price (break-even = win_rate)
    sniper_min_confidence: float = 0.55  # minimum confidence to fire (lowered from 0.65 — oracle 88% accurate)
    sniper_window_start: int = 10      # T-10s start committing
    sniper_window_end: int = 5         # T-5s stop (maker needs fill time)
    # E2: Kelly sizing parameters
    sniper_bankroll: float = 100.0     # starting bankroll for Kelly sizing
    sniper_min_size: float = 5.0       # minimum order size USD
    sniper_max_kelly_fraction: float = 0.15  # max fraction of bankroll per trade

    # --- Oracle Arbitrage (legacy, kept for reference) ---
    oracle_confidence_threshold: float = 0.95  # min confidence to trade
    oracle_min_edge: float = 0.05              # 5% minimum edge
    commitment_window_start: int = 10          # T-10s start committing (GTC maker)
    commitment_window_end: int = 5              # T-5s stop (need time for maker fill)
    kelly_fraction: float = 0.25               # 1/4 Kelly sizing

    # --- Risk Management ---
    daily_loss_limit_usd: float = 100.0
    consecutive_loss_pause: int = 3            # pause after N consecutive losses
    pause_duration_minutes: int = 15
    max_rounds_per_hour: int = 12              # max 12 x 5min rounds


@dataclass
class PaperSimConfig:
    """Paper trading simulator parameters."""
    # Latency model
    latency_mean_ms: float = 80.0       # VPS in same region as CLOB
    latency_stddev_ms: float = 30.0     # jitter
    latency_min_ms: float = 20.0
    latency_max_ms: float = 500.0
    # Order book model (5-min crypto markets have thin books)
    book_depth_usd: float = 2_000.0     # $2K per side (real: $500-5K)
    book_near_concentration: float = 0.3  # 30% depth within 1 tick
    # Rate limiting
    rate_limit_per_min: int = 60
    # Default volatility for GTC fill probability
    default_volatility: float = 0.05    # 5% for crypto 5-min


@dataclass
class OracleConfig:
    """Synthetic oracle configuration."""
    # Exchange/oracle weights for price aggregation
    # Pyth: highest weight — pull-based like Chainlink, >0.999 correlation
    # Chainlink: 0.0 weight — used only for drift calibration, not pricing
    weights: dict = field(default_factory=lambda: {
        "pyth": 0.40,       # Pull-based oracle, closest to Chainlink Data Streams
        "binance": 0.30,    # Highest volume CEX
        "coinbase": 0.15,   # US regulated, reliable
        "okx": 0.15,        # Asian market coverage
        "chainlink": 0.0,   # Calibration only — not used in pricing
    })
    # CEX WebSocket endpoints
    binance_ws: str = "wss://stream.binance.com:9443/ws/btcusdt@ticker"
    binance_ws_eth: str = "wss://stream.binance.com:9443/ws/ethusdt@ticker"  # E7
    coinbase_ws: str = "wss://ws-feed.exchange.coinbase.com"
    # Outlier rejection threshold
    outlier_threshold_pct: float = 0.002  # 0.2%
    # Learning rate for online weight adjustment
    learning_rate: float = 0.005


@dataclass
class RedisConfig:
    """Redis pub/sub for inter-node communication."""
    host: str = "localhost"
    port: int = 6379
    password: str = ""
    db: int = 0

    # Channel names
    ch_prediction: str = "poly:prediction"    # Alienware → VPS
    ch_telemetry: str = "poly:telemetry"      # VPS → Mac
    ch_control: str = "poly:control"          # Mac → VPS

    @classmethod
    def from_env(cls) -> "RedisConfig":
        return cls(
            host=os.environ.get("POLY_REDIS_HOST", "localhost"),
            port=int(os.environ.get("POLY_REDIS_PORT", "6379")),
            password=os.environ.get("POLY_REDIS_PASSWORD", ""),
        )


@dataclass
class AlertConfig:
    """E8: Telegram alerting configuration."""
    enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    # Alert thresholds
    brain_offline_seconds: int = 30
    consecutive_coinflip_alert: int = 3
    clob_disconnect_seconds: int = 60
    fill_rate_min: float = 0.10  # alert if fill rate drops below 10%
    fill_rate_max: float = 0.80  # alert if fill rate exceeds 80% (unrealistic)

    @classmethod
    def from_env(cls) -> "AlertConfig":
        token = os.environ.get("POLY_TELEGRAM_TOKEN", "")
        chat_id = os.environ.get("POLY_TELEGRAM_CHAT_ID", "")
        return cls(
            enabled=bool(token and chat_id),
            telegram_bot_token=token,
            telegram_chat_id=chat_id,
        )


@dataclass
class Config:
    """Master configuration."""
    wallet: WalletConfig = field(default_factory=WalletConfig)
    api: APIConfig = field(default_factory=APIConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    paper_sim: PaperSimConfig = field(default_factory=PaperSimConfig)
    oracle: OracleConfig = field(default_factory=OracleConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)  # E8

    # Operating mode
    mode: str = "paper"  # "paper" or "live"
    # Node role
    node: str = "standalone"  # "standalone", "vps", "brain", "dashboard"
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            wallet=WalletConfig.from_env(),
            api=APIConfig.from_env(),
            redis=RedisConfig.from_env(),
            alert=AlertConfig.from_env(),  # E8
            mode=os.environ.get("POLYMARKET_MODE", "paper"),
            node=os.environ.get("POLY_NODE", "standalone"),
        )
