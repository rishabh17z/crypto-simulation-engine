"""
Market-data ingestion: OHLCV via CCXT plus the technical indicator stack.

Responsibilities
----------------
* Build and own the CCXT client (sandbox/testnet aware) shared with the executor.
* Pull OHLCV candles with retries for timeouts, rate limits and DDoS protection.
* Compute RSI(14) with Wilder smoothing, EMA(50)/EMA(200) and ATR(14), and package the
  result as an immutable :class:`MarketSnapshot`.

The last candle returned by an exchange is the *in-progress* one. Indicators are
therefore computed on closed candles only (the forming bar is dropped), which is what
keeps a live run consistent with any backtest of the same rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Sequence

import ccxt
import numpy as np
import pandas as pd

from config.settings import AssetConfig, Settings
from src.common import (
    get_logger,
    is_finite,
    ms_to_datetime,
    retry_on,
    safe_float,
    utc_now,
)

LOGGER = get_logger(__name__)

# Transient, worth retrying: socket errors, exchange 5xx, throttling, timeouts.
TRANSIENT_EXCHANGE_ERRORS: tuple[type[BaseException], ...] = (
    ccxt.NetworkError,          # superclass of RequestTimeout / ExchangeNotAvailable
    ccxt.RequestTimeout,
    ccxt.ExchangeNotAvailable,
    ccxt.DDoSProtection,
    ccxt.RateLimitExceeded,
    ccxt.OnMaintenance,
)

OHLCV_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


class MarketDataError(RuntimeError):
    """Raised when usable market data cannot be produced for an asset."""


class InsufficientDataError(MarketDataError):
    """Not enough closed candles to warm up the slowest indicator."""


# --------------------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------------------


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average (``adjust=False`` => standard recursive EMA)."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder's RSI - the definition every charting package uses.

    A simple average of the first ``period`` gains/losses seeds the recursion, after
    which gains and losses are smoothed with alpha = 1/period. A zero average loss
    yields RSI 100 rather than a division error.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 -> pure uptrend -> RSI 100; avg_gain == 0 -> RSI 0.
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    rsi = rsi.where(~((avg_gain == 0.0) & (avg_loss > 0.0)), 0.0)
    return rsi


def atr_wilder(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Average True Range with Wilder smoothing; drives stop/target distances."""
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


# --------------------------------------------------------------------------------------
# Snapshot
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """Indicator state for one asset at one closed candle."""

    symbol: str
    exchange_symbol: str
    timeframe: str
    candle_time: datetime
    fetched_at: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    prev_close: float
    rsi: float
    ema_fast: float
    ema_slow: float
    atr: float
    ema_fast_period: int
    ema_slow_period: int
    candles: int
    last_price: float = 0.0          # live ticker price when available, else close

    # -- derived -----------------------------------------------------------------------

    @property
    def price(self) -> float:
        """Price used for sizing and fill simulation."""
        return self.last_price or self.close

    @property
    def atr_pct(self) -> float:
        return self.atr / self.close if self.close else 0.0

    @property
    def ema_spread_pct(self) -> float:
        """(fast - slow) / slow: the signed strength of the trend regime."""
        return (self.ema_fast - self.ema_slow) / self.ema_slow if self.ema_slow else 0.0

    @property
    def regime(self) -> str:
        spread = self.ema_spread_pct
        if spread > 0.002:
            return "bull"
        if spread < -0.002:
            return "bear"
        return "neutral"

    @property
    def price_vs_ema_fast_pct(self) -> float:
        return (self.close - self.ema_fast) / self.ema_fast if self.ema_fast else 0.0

    @property
    def bar_return_pct(self) -> float:
        return (self.close - self.prev_close) / self.prev_close if self.prev_close else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "candle_time": self.candle_time.isoformat(),
            "close": round(self.close, 8),
            "last_price": round(self.price, 8),
            "rsi": round(self.rsi, 2),
            "ema_fast": round(self.ema_fast, 8),
            "ema_slow": round(self.ema_slow, 8),
            "atr": round(self.atr, 8),
            "atr_pct": round(self.atr_pct, 5),
            "ema_spread_pct": round(self.ema_spread_pct, 5),
            "regime": self.regime,
            "bar_return_pct": round(self.bar_return_pct, 5),
            "candles": self.candles,
        }


# --------------------------------------------------------------------------------------
# CCXT client
# --------------------------------------------------------------------------------------


def build_exchange(settings: Settings) -> ccxt.Exchange:
    """
    Instantiate the CCXT client in sandbox (testnet) mode.

    ``set_sandbox_mode(True)`` swaps Bybit's REST hosts for ``api-testnet.bybit.com``.
    Credentials are optional for public OHLCV but required for private endpoints, so a
    missing key is a warning, not a hard failure - the simulation still runs.
    """
    exchange_id = settings.exchange.exchange_id
    if not hasattr(ccxt, exchange_id):
        raise MarketDataError(f"ccxt has no exchange named {exchange_id!r}")

    klass = getattr(ccxt, exchange_id)
    exchange: ccxt.Exchange = klass(
        {
            "apiKey": settings.exchange.api_key or None,
            "secret": settings.exchange.api_secret or None,
            "enableRateLimit": settings.exchange.enable_rate_limit,
            "timeout": settings.exchange.timeout_ms,
            "options": {
                "defaultType": settings.exchange.market_type,
                "recvWindow": 10_000,
            },
        }
    )

    if settings.exchange.use_sandbox:
        try:
            exchange.set_sandbox_mode(True)
            LOGGER.info("%s sandbox/testnet mode ENABLED", exchange_id)
        except (ccxt.NotSupported, AttributeError) as exc:
            raise MarketDataError(
                f"{exchange_id} does not expose a CCXT sandbox mode: {exc}"
            ) from exc
    else:
        LOGGER.warning(
            "USE_SANDBOX=false - %s client is pointed at PRODUCTION endpoints",
            exchange_id,
        )

    if not settings.exchange.has_credentials:
        LOGGER.warning(
            "no API credentials configured: public market data only, "
            "testnet order placement disabled"
        )
    return exchange


class PriceFetcher:
    """Fetches candles/tickers and turns them into :class:`MarketSnapshot` objects."""

    def __init__(self, settings: Settings, exchange: ccxt.Exchange | None = None):
        self.settings = settings
        self.exchange = exchange or build_exchange(settings)
        self._markets_loaded = False

    # -- exchange plumbing -------------------------------------------------------------

    @property
    def _retry_kwargs(self) -> dict[str, Any]:
        return {
            "attempts": self.settings.exchange.max_retries,
            "base_delay": self.settings.exchange.retry_base_delay,
            "max_delay": self.settings.exchange.retry_max_delay,
        }

    def load_markets(self, reload: bool = False) -> None:
        """Load (and cache) the market map; needed for symbol validation + precision."""
        if self._markets_loaded and not reload:
            return

        @retry_on(
            TRANSIENT_EXCHANGE_ERRORS,
            attempts=self.settings.exchange.max_retries,
            base_delay=self.settings.exchange.retry_base_delay,
            max_delay=self.settings.exchange.retry_max_delay,
            logger=LOGGER,
        )
        def _load() -> None:
            self.exchange.load_markets(reload)

        _load()
        self._markets_loaded = True
        LOGGER.info("loaded %d markets from %s", len(self.exchange.markets or {}),
                    self.settings.exchange.exchange_id)

    def resolve_symbol(self, asset: AssetConfig) -> str:
        """Return a symbol the exchange actually lists, falling back to the spot form."""
        self.load_markets()
        markets = self.exchange.markets or {}
        for candidate in (asset.exchange_symbol, asset.symbol):
            if candidate in markets:
                return candidate
        raise MarketDataError(
            f"{asset.symbol} is not listed on "
            f"{self.settings.exchange.exchange_id} "
            f"({self.settings.exchange.market_type} markets)"
        )

    # -- raw data ----------------------------------------------------------------------

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int | None = None
    ) -> pd.DataFrame:
        """OHLCV as a UTC-indexed DataFrame with the in-progress candle removed."""
        bars = limit or self.settings.exchange.ohlcv_limit

        @retry_on(
            TRANSIENT_EXCHANGE_ERRORS,
            attempts=self.settings.exchange.max_retries,
            base_delay=self.settings.exchange.retry_base_delay,
            max_delay=self.settings.exchange.retry_max_delay,
            logger=LOGGER,
        )
        def _fetch(params: dict[str, Any] | None = None) -> list[list[float]]:
            return self.exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, limit=bars, params=params or {}
            )

        try:
            raw = _fetch()
            if raw and len(raw) < bars:
                # Some venues cap candles per request (KuCoin futures: 200), which is
                # too few to warm up EMA200; let CCXT stitch several pages together.
                calls = 1 + -(-bars // len(raw))
                paged = _fetch({"paginate": True, "paginationCalls": calls})
                if len(paged) > len(raw):
                    raw = paged[-bars:]
        except ccxt.BadSymbol as exc:
            raise MarketDataError(f"bad symbol {symbol!r}: {exc}") from exc
        except ccxt.ExchangeError as exc:  # auth, banned, malformed request...
            raise MarketDataError(f"exchange rejected OHLCV for {symbol}: {exc}") from exc

        if not raw:
            raise InsufficientDataError(f"{symbol}: exchange returned no candles")

        frame = pd.DataFrame(raw, columns=list(OHLCV_COLUMNS))
        frame = frame.dropna(subset=["close"])
        frame["datetime"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        frame = frame.drop_duplicates(subset="timestamp", keep="last")
        frame = frame.sort_values("timestamp").set_index("datetime")

        # Drop the forming candle: its close keeps moving until the bar completes.
        timeframe_ms = self.exchange.parse_timeframe(timeframe) * 1000
        now_ms = self.exchange.milliseconds()
        if float(frame["timestamp"].iloc[-1]) + timeframe_ms > now_ms:
            frame = frame.iloc[:-1]

        if frame.empty:
            raise InsufficientDataError(f"{symbol}: no closed candles available")
        return frame

    def fetch_last_price(self, symbol: str) -> float:
        """Live mark price; ``0.0`` (caller falls back to candle close) on failure."""

        @retry_on(
            TRANSIENT_EXCHANGE_ERRORS,
            attempts=max(2, self.settings.exchange.max_retries - 2),
            base_delay=self.settings.exchange.retry_base_delay,
            max_delay=self.settings.exchange.retry_max_delay,
            logger=LOGGER,
        )
        def _fetch() -> dict[str, Any]:
            return self.exchange.fetch_ticker(symbol)

        try:
            ticker = _fetch()
        except Exception as exc:  # noqa: BLE001 - ticker is a best-effort refinement
            LOGGER.warning("ticker fetch failed for %s (%s); using candle close",
                           symbol, exc)
            return 0.0

        for key in ("last", "close", "mark", "bid", "ask"):
            price = safe_float(ticker.get(key), 0.0)
            if price > 0:
                return price
        return 0.0

    # -- snapshots ---------------------------------------------------------------------

    def snapshot(self, asset: AssetConfig, *, with_ticker: bool = True) -> MarketSnapshot:
        """Full indicator snapshot for one asset. Raises :class:`MarketDataError`."""
        strategy = self.settings.strategy
        symbol = self.resolve_symbol(asset)
        timeframe = asset.timeframe or strategy.timeframe
        frame = self.fetch_ohlcv(symbol, timeframe)

        warm_up = strategy.ema_slow_period + 2
        if len(frame) < warm_up:
            raise InsufficientDataError(
                f"{asset.symbol}: {len(frame)} closed candles < {warm_up} required for "
                f"EMA{strategy.ema_slow_period}"
            )

        close = frame["close"].astype(float)
        high = frame["high"].astype(float)
        low = frame["low"].astype(float)

        rsi_series = rsi_wilder(close, strategy.rsi_period)
        ema_fast_series = ema(close, strategy.ema_fast_period)
        ema_slow_series = ema(close, strategy.ema_slow_period)
        atr_series = atr_wilder(high, low, close, strategy.atr_period)

        last = frame.iloc[-1]
        values = {
            "rsi": safe_float(rsi_series.iloc[-1], float("nan")),
            "ema_fast": safe_float(ema_fast_series.iloc[-1], float("nan")),
            "ema_slow": safe_float(ema_slow_series.iloc[-1], float("nan")),
            "atr": safe_float(atr_series.iloc[-1], float("nan")),
        }
        if not is_finite(*values.values()):
            raise InsufficientDataError(
                f"{asset.symbol}: indicators still warming up {values}"
            )

        last_price = self.fetch_last_price(symbol) if with_ticker else 0.0

        snapshot = MarketSnapshot(
            symbol=asset.symbol,
            exchange_symbol=symbol,
            timeframe=timeframe,
            candle_time=ms_to_datetime(float(last["timestamp"])),
            fetched_at=utc_now(),
            open=safe_float(last["open"]),
            high=safe_float(last["high"]),
            low=safe_float(last["low"]),
            close=safe_float(last["close"]),
            volume=safe_float(last["volume"]),
            prev_close=safe_float(frame["close"].iloc[-2]),
            rsi=values["rsi"],
            ema_fast=values["ema_fast"],
            ema_slow=values["ema_slow"],
            atr=max(values["atr"], 0.0),
            ema_fast_period=strategy.ema_fast_period,
            ema_slow_period=strategy.ema_slow_period,
            candles=len(frame),
            last_price=last_price,
        )
        LOGGER.info(
            "%s %s | px=%.6f rsi=%.1f ema%d=%.6f ema%d=%.6f atr=%.6f (%s)",
            snapshot.symbol, timeframe, snapshot.price, snapshot.rsi,
            strategy.ema_fast_period, snapshot.ema_fast,
            strategy.ema_slow_period, snapshot.ema_slow, snapshot.atr,
            snapshot.regime,
        )
        return snapshot

    def snapshots(
        self, assets: Iterable[AssetConfig]
    ) -> tuple[dict[str, MarketSnapshot], dict[str, str]]:
        """
        Snapshot many assets.

        One bad symbol must never abort a cycle, so failures are collected and returned
        alongside the successes as ``(snapshots, errors)``.
        """
        snapshots: dict[str, MarketSnapshot] = {}
        errors: dict[str, str] = {}
        for asset in assets:
            try:
                snapshots[asset.symbol] = self.snapshot(asset)
            except MarketDataError as exc:
                errors[asset.symbol] = str(exc)
                LOGGER.error("market data unavailable for %s: %s", asset.symbol, exc)
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                errors[asset.symbol] = f"{type(exc).__name__}: {exc}"
                LOGGER.exception("unexpected failure snapshotting %s", asset.symbol)
        return snapshots, errors

    def mark_prices(self, symbols: Sequence[str]) -> dict[str, float]:
        """Current prices for open-position marking (used between full snapshots)."""
        prices: dict[str, float] = {}
        for symbol in symbols:
            try:
                asset = self.settings.asset(symbol)
                resolved = self.resolve_symbol(asset)
            except (KeyError, MarketDataError):
                resolved = symbol
            price = self.fetch_last_price(resolved)
            if price > 0:
                prices[symbol] = price
        return prices

    def close(self) -> None:
        """Release the underlying HTTP session if the CCXT build exposes one."""
        closer = getattr(self.exchange, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:  # noqa: BLE001 - teardown must never raise
                LOGGER.debug("exchange close() raised during teardown", exc_info=True)


__all__ = [
    "InsufficientDataError",
    "MarketDataError",
    "MarketSnapshot",
    "PriceFetcher",
    "TRANSIENT_EXCHANGE_ERRORS",
    "atr_wilder",
    "build_exchange",
    "ema",
    "rsi_wilder",
]
