"""
Central, immutable configuration for the crypto paper-trading simulation engine.

Every tunable lives here. Secrets and deployment-specific overrides come from the
environment (optionally via a local ``.env`` file); the tradable universe comes from
``config/assets.yaml``.

Import contract
---------------
    from config.settings import SETTINGS
    SETTINGS.risk.max_risk_per_trade
    SETTINGS.assets                  # tuple[AssetConfig, ...]

``load_settings()`` is memoised, so configuration is parsed and validated exactly once
per process and is safe to import from scheduler threads.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from dotenv import load_dotenv

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

BASE_DIR: Path = Path(__file__).resolve().parents[1]
CONFIG_DIR: Path = BASE_DIR / "config"
DATA_DIR: Path = BASE_DIR / "data"
LOG_DIR: Path = BASE_DIR / "logs"
REPORT_DIR: Path = BASE_DIR / "reports"
ASSETS_FILE: Path = CONFIG_DIR / "assets.yaml"

for _directory in (DATA_DIR, LOG_DIR, REPORT_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# ``.env`` never overrides a real environment variable (container/CI wins).
load_dotenv(BASE_DIR / ".env", override=False)


class ConfigError(RuntimeError):
    """Raised when the configuration is internally inconsistent or unsafe."""


# --------------------------------------------------------------------------------------
# Environment helpers
# --------------------------------------------------------------------------------------

_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off"}


def _env_str(key: str, default: str = "") -> str:
    raw = os.environ.get(key)
    return default if raw is None else raw.strip()


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ConfigError(f"{key}={raw!r} is not a valid boolean")


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"{key}={raw!r} is not a valid float") from exc


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        return int(float(raw))
    except ValueError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"{key}={raw!r} is not a valid int") from exc


# --------------------------------------------------------------------------------------
# Settings sections
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExchangeSettings:
    """CCXT client construction and sandbox behaviour."""

    exchange_id: str = "bybit"
    api_key: str = ""
    api_secret: str = ""
    use_sandbox: bool = True
    market_type: str = "swap"              # "swap" (linear perps) or "spot"
    quote_currency: str = "USDT"
    place_testnet_orders: bool = False     # mirror simulated fills onto the testnet book
    enable_rate_limit: bool = True
    timeout_ms: int = 20_000
    max_retries: int = 5
    retry_base_delay: float = 1.5
    retry_max_delay: float = 45.0
    ohlcv_limit: int = 400                 # must exceed the slowest EMA period

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key and self.api_secret)


@dataclass(frozen=True, slots=True)
class RiskSettings:
    """Hard risk envelope. The executor refuses to breach any of these."""

    initial_corpus: float = 10_000.0
    max_risk_per_trade: float = 0.05       # 5% of current equity
    position_sizing_mode: str = "notional"  # "notional" | "risk"
    max_portfolio_exposure: float = 0.60   # sum of open notionals / equity
    max_open_positions: int = 3
    max_positions_per_asset: int = 1
    allow_shorts: bool = True
    min_order_notional: float = 10.0
    equity_stop: float = 0.70              # halt new entries below 70% of start corpus
    require_stop_loss: bool = True         # every order must carry SL *and* TP


@dataclass(frozen=True, slots=True)
class SimulationSettings:
    """Simulated microstructure costs."""

    fee_rate: float = 0.001                # 0.1% taker fee, charged per side
    slippage_rate: float = 0.001           # 0.1% adverse slippage on entry and exit
    conservative_fills: bool = True        # if SL and TP both touched in a bar, SL wins


@dataclass(frozen=True, slots=True)
class StrategySettings:
    """Indicator periods and decision thresholds."""

    timeframe: str = "4h"
    rsi_period: int = 14
    ema_fast_period: int = 50
    ema_slow_period: int = 200
    atr_period: int = 14

    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    rsi_buy_veto: float = 80.0             # never chase into extreme strength
    rsi_sell_veto: float = 20.0

    weight_trend: float = 0.45
    weight_momentum: float = 0.25
    weight_sentiment: float = 0.30

    buy_threshold: float = 0.30
    sell_threshold: float = -0.30
    min_confidence: float = 0.25

    atr_sl_multiple: float = 2.0
    atr_tp_multiple: float = 3.0
    min_stop_pct: float = 0.008            # floor so costs cannot dominate the stop
    max_stop_pct: float = 0.10
    max_atr_pct: float = 0.12              # volatility circuit breaker -> HOLD
    trend_flip_exit: bool = True           # close on a confirmed regime flip


@dataclass(frozen=True, slots=True)
class SentimentSettings:
    """News / Fear & Greed ingestion."""

    enabled: bool = True
    fear_greed_url: str = "https://api.alternative.me/fng/"
    cryptopanic_url: str = "https://cryptopanic.com/api/v1/posts/"
    cryptopanic_token: str = ""
    http_timeout: float = 12.0
    max_retries: int = 3
    retry_base_delay: float = 1.0
    cache_ttl_seconds: int = 900            # one fetch per 15 minutes, max
    failure_cooldown_seconds: int = 600     # negative cache: one retry ladder per 10min
    weight_fear_greed: float = 0.45
    weight_news: float = 0.55
    neutral_on_failure: bool = True         # degrade to 0.0 rather than crash a cycle


@dataclass(frozen=True, slots=True)
class ScheduleSettings:
    """Cadence and the hard 30-day experiment boundary."""

    cycle_hours: float = 4.0
    experiment_days: int = 30
    deadline_check_minutes: int = 15
    misfire_grace_seconds: int = 600
    run_immediately: bool = True
    timezone: str = "UTC"

    @property
    def cycle_seconds(self) -> int:
        return int(self.cycle_hours * 3600)

    @property
    def experiment_seconds(self) -> int:
        return int(self.experiment_days * 86_400)


@dataclass(frozen=True, slots=True)
class StorageSettings:
    """SQLite durability knobs (WAL + busy timeout handle concurrent writers)."""

    db_path: Path = DATA_DIR / "simulation.sqlite3"
    busy_timeout_ms: int = 10_000
    journal_mode: str = "WAL"
    synchronous: str = "NORMAL"
    write_retries: int = 6
    write_retry_delay: float = 0.35
    report_dir: Path = REPORT_DIR


@dataclass(frozen=True, slots=True)
class AssetConfig:
    """One tradable instrument."""

    symbol: str                            # canonical/display symbol, e.g. BTC/USDT
    exchange_symbol: str                   # symbol handed to CCXT (swaps get :USDT)
    base: str
    quote: str
    enabled: bool = True
    timeframe: str = "4h"
    weight: float = 1.0                    # scales the per-trade allocation
    min_notional: float = 10.0
    sentiment_currency: str | None = None   # CryptoPanic currency filter, e.g. BTC

    @property
    def sentiment_key(self) -> str:
        return self.sentiment_currency or self.base


@dataclass(frozen=True, slots=True)
class Settings:
    """Fully-resolved configuration tree."""

    exchange: ExchangeSettings
    risk: RiskSettings
    simulation: SimulationSettings
    strategy: StrategySettings
    sentiment: SentimentSettings
    schedule: ScheduleSettings
    storage: StorageSettings
    assets: tuple[AssetConfig, ...]
    log_level: str = "INFO"
    log_file: Path = LOG_DIR / "engine.log"

    # -- derived helpers ---------------------------------------------------------------

    @property
    def enabled_assets(self) -> tuple[AssetConfig, ...]:
        return tuple(a for a in self.assets if a.enabled)

    def asset(self, symbol: str) -> AssetConfig:
        for candidate in self.assets:
            if symbol in (candidate.symbol, candidate.exchange_symbol):
                return candidate
        raise KeyError(f"unknown asset: {symbol}")

    # -- validation --------------------------------------------------------------------

    def validate(self) -> "Settings":
        risk, sim, strat = self.risk, self.simulation, self.strategy

        if risk.initial_corpus <= 0:
            raise ConfigError("INITIAL_CORPUS must be > 0")
        if not 0 < risk.max_risk_per_trade <= 0.5:
            raise ConfigError("MAX_RISK_PER_TRADE must be in (0, 0.5]")
        if risk.position_sizing_mode not in {"notional", "risk"}:
            raise ConfigError("POSITION_SIZING_MODE must be 'notional' or 'risk'")
        if not 0 < risk.max_portfolio_exposure <= 1.0:
            raise ConfigError("MAX_PORTFOLIO_EXPOSURE must be in (0, 1]")
        if risk.max_open_positions < 1:
            raise ConfigError("MAX_OPEN_POSITIONS must be >= 1")
        if not risk.require_stop_loss:
            raise ConfigError("stop-loss/take-profit attachment cannot be disabled")

        if not 0 <= sim.fee_rate < 0.01:
            raise ConfigError("FEE_RATE must be in [0, 0.01)")
        if not 0 <= sim.slippage_rate < 0.05:
            raise ConfigError("SLIPPAGE_RATE must be in [0, 0.05)")

        if strat.ema_fast_period >= strat.ema_slow_period:
            raise ConfigError("ema_fast_period must be < ema_slow_period")
        if strat.rsi_period < 2 or strat.atr_period < 2:
            raise ConfigError("indicator periods must be >= 2")
        if strat.buy_threshold <= 0 or strat.sell_threshold >= 0:
            raise ConfigError("buy_threshold must be > 0 and sell_threshold < 0")
        if strat.min_stop_pct >= strat.max_stop_pct:
            raise ConfigError("min_stop_pct must be < max_stop_pct")
        if strat.atr_tp_multiple <= strat.atr_sl_multiple:
            raise ConfigError("take-profit multiple must exceed the stop-loss multiple")
        # The stop must clear round-trip costs or the edge is structurally negative.
        round_trip_cost = 2 * (sim.fee_rate + sim.slippage_rate)
        if strat.min_stop_pct <= round_trip_cost:
            raise ConfigError(
                f"min_stop_pct ({strat.min_stop_pct:.4f}) must exceed round-trip "
                f"cost ({round_trip_cost:.4f})"
            )

        warm_up = strat.ema_slow_period + strat.rsi_period + 5
        if self.exchange.ohlcv_limit < warm_up:
            raise ConfigError(
                f"ohlcv_limit ({self.exchange.ohlcv_limit}) is below the indicator "
                f"warm-up requirement ({warm_up})"
            )

        if self.exchange.market_type not in {"swap", "spot"}:
            raise ConfigError("MARKET_TYPE must be 'swap' or 'spot'")
        if risk.allow_shorts and self.exchange.market_type == "spot":
            raise ConfigError(
                "ALLOW_SHORTS=true requires MARKET_TYPE=swap (spot cannot go short)"
            )

        if self.schedule.cycle_hours <= 0:
            raise ConfigError("CYCLE_HOURS must be > 0")
        if self.schedule.experiment_days < 1:
            raise ConfigError("EXPERIMENT_DAYS must be >= 1")

        sentiment = self.sentiment
        if sentiment.http_timeout <= 0:
            raise ConfigError("SENTIMENT_HTTP_TIMEOUT must be > 0")
        if sentiment.max_retries < 1:
            raise ConfigError("SENTIMENT_MAX_RETRIES must be >= 1")
        if sentiment.failure_cooldown_seconds < 0:
            raise ConfigError("SENTIMENT_FAILURE_COOLDOWN must be >= 0")

        if not self.enabled_assets:
            raise ConfigError(f"no enabled assets found in {ASSETS_FILE}")

        seen: set[str] = set()
        for asset in self.assets:
            if asset.symbol in seen:
                raise ConfigError(f"duplicate asset symbol: {asset.symbol}")
            seen.add(asset.symbol)

        return self


# --------------------------------------------------------------------------------------
# assets.yaml loading
# --------------------------------------------------------------------------------------


def _to_exchange_symbol(symbol: str, market_type: str, quote: str) -> str:
    """Bybit linear perpetuals are addressed as ``BASE/QUOTE:QUOTE`` in CCXT."""
    if market_type != "swap" or ":" in symbol:
        return symbol
    return f"{symbol}:{quote}"


def load_assets(
    path: Path = ASSETS_FILE,
    *,
    market_type: str = "swap",
    default_timeframe: str = "4h",
) -> tuple[AssetConfig, ...]:
    if not path.exists():
        raise ConfigError(f"asset universe file not found: {path}")

    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse {path}: {exc}") from exc

    if not isinstance(raw, Mapping):
        raise ConfigError(f"{path} must contain a mapping at the top level")

    defaults: Mapping[str, Any] = raw.get("defaults") or {}
    universe: Sequence[Any] = raw.get("universe") or []
    if not isinstance(universe, Sequence) or isinstance(universe, (str, bytes)):
        raise ConfigError(f"{path}: 'universe' must be a list")

    assets: list[AssetConfig] = []
    for index, entry in enumerate(universe):
        if isinstance(entry, str):
            entry = {"symbol": entry}
        if not isinstance(entry, Mapping):
            raise ConfigError(f"{path}: universe[{index}] must be a string or mapping")

        symbol = str(entry.get("symbol", "")).strip().upper()
        if "/" not in symbol:
            raise ConfigError(
                f"{path}: universe[{index}] symbol {symbol!r} must look like BASE/QUOTE"
            )
        base, quote = symbol.split("/", 1)
        quote = quote.split(":", 1)[0]

        assets.append(
            AssetConfig(
                symbol=symbol,
                exchange_symbol=_to_exchange_symbol(symbol, market_type, quote),
                base=base,
                quote=quote,
                enabled=bool(entry.get("enabled", defaults.get("enabled", True))),
                timeframe=str(
                    entry.get("timeframe", defaults.get("timeframe", default_timeframe))
                ),
                weight=float(entry.get("weight", defaults.get("weight", 1.0))),
                min_notional=float(
                    entry.get("min_notional", defaults.get("min_notional", 10.0))
                ),
                sentiment_currency=(
                    str(entry["sentiment_currency"]).upper()
                    if entry.get("sentiment_currency")
                    else base
                ),
            )
        )

    return tuple(assets)


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    """Build, validate and memoise the settings tree."""

    exchange = ExchangeSettings(
        exchange_id=_env_str("EXCHANGE_ID", "bybit").lower(),
        api_key=_env_str("BYBIT_API_KEY"),
        api_secret=_env_str("BYBIT_API_SECRET"),
        use_sandbox=_env_bool("USE_SANDBOX", True),
        market_type=_env_str("MARKET_TYPE", "swap").lower(),
        quote_currency=_env_str("QUOTE_CURRENCY", "USDT").upper(),
        place_testnet_orders=_env_bool("PLACE_TESTNET_ORDERS", False),
        timeout_ms=_env_int("EXCHANGE_TIMEOUT_MS", 20_000),
        max_retries=_env_int("EXCHANGE_MAX_RETRIES", 5),
        ohlcv_limit=_env_int("OHLCV_LIMIT", 400),
    )

    risk = RiskSettings(
        initial_corpus=_env_float("INITIAL_CORPUS", 10_000.0),
        max_risk_per_trade=_env_float("MAX_RISK_PER_TRADE", 0.05),
        position_sizing_mode=_env_str("POSITION_SIZING_MODE", "notional").lower(),
        max_portfolio_exposure=_env_float("MAX_PORTFOLIO_EXPOSURE", 0.60),
        max_open_positions=_env_int("MAX_OPEN_POSITIONS", 3),
        allow_shorts=_env_bool("ALLOW_SHORTS", True),
        min_order_notional=_env_float("MIN_ORDER_NOTIONAL", 10.0),
        equity_stop=_env_float("EQUITY_STOP", 0.70),
    )

    simulation = SimulationSettings(
        fee_rate=_env_float("FEE_RATE", 0.001),
        slippage_rate=_env_float("SLIPPAGE_RATE", 0.001),
    )

    strategy = StrategySettings(
        timeframe=_env_str("TIMEFRAME", "4h"),
        rsi_period=_env_int("RSI_PERIOD", 14),
        ema_fast_period=_env_int("EMA_FAST_PERIOD", 50),
        ema_slow_period=_env_int("EMA_SLOW_PERIOD", 200),
    )

    sentiment = SentimentSettings(
        enabled=_env_bool("SENTIMENT_ENABLED", True),
        fear_greed_url=_env_str("FEAR_GREED_URL", "https://api.alternative.me/fng/"),
        cryptopanic_url=_env_str(
            "CRYPTOPANIC_URL", "https://cryptopanic.com/api/v1/posts/"
        ),
        cryptopanic_token=_env_str("CRYPTOPANIC_TOKEN"),
        http_timeout=_env_float("SENTIMENT_HTTP_TIMEOUT", 12.0),
        max_retries=_env_int("SENTIMENT_MAX_RETRIES", 3),
        cache_ttl_seconds=_env_int("SENTIMENT_CACHE_TTL", 900),
        failure_cooldown_seconds=_env_int("SENTIMENT_FAILURE_COOLDOWN", 600),
    )

    schedule = ScheduleSettings(
        cycle_hours=_env_float("CYCLE_HOURS", 4.0),
        experiment_days=_env_int("EXPERIMENT_DAYS", 30),
        run_immediately=_env_bool("RUN_IMMEDIATELY", True),
    )

    storage = StorageSettings(
        db_path=Path(_env_str("DB_PATH", str(DATA_DIR / "simulation.sqlite3"))),
    )

    assets = load_assets(
        ASSETS_FILE,
        market_type=exchange.market_type,
        default_timeframe=strategy.timeframe,
    )

    return Settings(
        exchange=exchange,
        risk=risk,
        simulation=simulation,
        strategy=strategy,
        sentiment=sentiment,
        schedule=schedule,
        storage=storage,
        assets=assets,
        log_level=_env_str("LOG_LEVEL", "INFO").upper(),
    ).validate()


SETTINGS: Settings = load_settings()

__all__ = [
    "AssetConfig",
    "BASE_DIR",
    "ConfigError",
    "ExchangeSettings",
    "RiskSettings",
    "ScheduleSettings",
    "SentimentSettings",
    "Settings",
    "SimulationSettings",
    "StorageSettings",
    "StrategySettings",
    "SETTINGS",
    "load_assets",
    "load_settings",
]
