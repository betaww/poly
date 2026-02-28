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
    assets: list[str] = field(default_factory=lambda: ["btc", "eth"])
    # Timeframe: 5m, 15m, 1h, 4h
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
    # --- Market Making ---
    base_spread: float = 0.03          # 3 cents base spread
    min_spread: float = 0.02           # minimum spread
    max_spread: float = 0.08           # max spread in high volatility
    order_size_usd: float = 10.0       # per-side order size
    max_position_usd: float = 50.0     # max exposure per round
    refresh_interval_ms: int = 1000    # quote refresh rate

    # --- Oracle Arbitrage ---
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
    # Order book model
    book_depth_usd: float = 10_000.0    # $10K per side
    book_near_concentration: float = 0.3  # 30% depth within 1 tick
    # Rate limiting
    rate_limit_per_min: int = 60
    # Default volatility for GTC fill probability
    default_volatility: float = 0.05    # 5% for crypto 5-min


@dataclass
class OracleConfig:
    """Synthetic oracle configuration."""
    # Exchange weights for price aggregation
    weights: dict = field(default_factory=lambda: {
        # Only include exchanges we actually have feeds for
        # Kraken removed: no feed → permanently missing → confidence penalty
        "binance": 0.50,     # was 0.45, absorbed kraken share
        "coinbase": 0.30,
        "okx": 0.20,         # was 0.10, absorbed kraken share
    })
    # CEX WebSocket endpoints
    binance_ws: str = "wss://stream.binance.com:9443/ws/btcusdt@ticker"
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
class Config:
    """Master configuration."""
    wallet: WalletConfig = field(default_factory=WalletConfig)
    api: APIConfig = field(default_factory=APIConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    paper_sim: PaperSimConfig = field(default_factory=PaperSimConfig)
    oracle: OracleConfig = field(default_factory=OracleConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)

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
            mode=os.environ.get("POLYMARKET_MODE", "paper"),
            node=os.environ.get("POLY_NODE", "standalone"),
        )
