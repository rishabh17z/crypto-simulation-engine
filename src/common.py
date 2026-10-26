"""
Cross-cutting utilities shared by the ingestion, engine and storage layers.

Deliberately dependency-light so it can be imported from anywhere without creating
import cycles: logging setup, a jittered exponential-backoff retry decorator
(network timeouts / rate limits / locked SQLite), UTC time helpers and small numeric
guards.
"""

from __future__ import annotations

import functools
import logging
import logging.handlers
import math
import random
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

T = TypeVar("T")

ISO_FORMAT = "%Y-%m-%dT%H:%M:%S%z"

_LOGGING_CONFIGURED = False


# --------------------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------------------


def utc_now() -> datetime:
    """Timezone-aware current UTC time (never use naive datetimes in this codebase)."""
    return datetime.now(timezone.utc)


def to_iso(moment: datetime) -> str:
    """Serialise to a sortable ISO-8601 UTC string for SQLite TEXT columns."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


def from_iso(value: str | None) -> datetime | None:
    """Parse an ISO-8601 string written by :func:`to_iso`; ``None`` passes through."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def ms_to_datetime(milliseconds: float) -> datetime:
    """CCXT timestamps are epoch milliseconds."""
    return datetime.fromtimestamp(milliseconds / 1000.0, tz=timezone.utc)


def humanize_timedelta(delta: timedelta) -> str:
    total = int(max(delta.total_seconds(), 0))
    days, rem = divmod(total, 86_400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def new_uid(prefix: str = "") -> str:
    """Short, collision-resistant identifier used for trade/cycle correlation."""
    token = uuid.uuid4().hex[:12]
    return f"{prefix}-{token}" if prefix else token


# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------


def configure_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    *,
    max_bytes: int = 5_000_000,
    backups: int = 3,
) -> None:
    """Idempotently configure the root logger for console + rotating file output."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(root.handlers):  # replace anything a library installed
        root.removeHandler(handler)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fmt.converter = time.gmtime  # log in UTC to match stored timestamps

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        rotating = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backups, encoding="utf-8"
        )
        rotating.setFormatter(fmt)
        root.addHandler(rotating)

    # These libraries are chatty at INFO and drown the engine's own narrative.
    for noisy in ("urllib3", "ccxt", "apscheduler.scheduler", "apscheduler.executors"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# --------------------------------------------------------------------------------------
# Retry
# --------------------------------------------------------------------------------------


class RetryExhausted(RuntimeError):
    """Raised when every retry attempt for a transient failure has been consumed."""

    def __init__(self, func_name: str, attempts: int, last_error: BaseException):
        super().__init__(
            f"{func_name} failed after {attempts} attempt(s): "
            f"{type(last_error).__name__}: {last_error}"
        )
        self.func_name = func_name
        self.attempts = attempts
        self.last_error = last_error


def backoff_delay(
    attempt: int, base: float, cap: float, jitter: float = 0.3
) -> float:
    """Exponential backoff with full-ish jitter; ``attempt`` is 1-based."""
    raw = min(cap, base * (2 ** max(attempt - 1, 0)))
    spread = raw * jitter
    return max(0.0, raw + random.uniform(-spread, spread))


def retry_on(
    exceptions: tuple[type[BaseException], ...],
    *,
    attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    logger: logging.Logger | None = None,
    reraise_as: type[BaseException] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Retry ``exceptions`` with jittered exponential backoff.

    Used for CCXT ``NetworkError``/``RateLimitExceeded``/``RequestTimeout``, HTTP
    failures in the sentiment scraper, and ``sqlite3.OperationalError`` ("database is
    locked"). Non-listed exceptions propagate immediately - a bad symbol or an
    authentication failure is not something retrying can fix.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        log = logger or logging.getLogger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_error: BaseException | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_error = exc
                    if attempt == attempts:
                        break
                    delay = backoff_delay(attempt, base_delay, max_delay)
                    log.warning(
                        "%s transient failure (%s/%s): %s: %s - retrying in %.2fs",
                        func.__qualname__,
                        attempt,
                        attempts,
                        type(exc).__name__,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
            assert last_error is not None
            error = RetryExhausted(func.__qualname__, attempts, last_error)
            if reraise_as is not None:
                raise reraise_as(str(error)) from last_error
            raise error from last_error

        return wrapper

    return decorator


# --------------------------------------------------------------------------------------
# Numeric guards
# --------------------------------------------------------------------------------------


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce exchange/API payload values; NaN, None and junk collapse to ``default``."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(out) or math.isinf(out) else out


def is_finite(*values: Any) -> bool:
    for value in values:
        if value is None:
            return False
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        if math.isnan(number) or math.isinf(number):
            return False
    return True


def percent(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def mean(values: Iterable[float]) -> float:
    data = list(values)
    return sum(data) / len(data) if data else 0.0


def stdev(values: Iterable[float]) -> float:
    data = list(values)
    if len(data) < 2:
        return 0.0
    mu = mean(data)
    variance = sum((x - mu) ** 2 for x in data) / (len(data) - 1)
    return math.sqrt(variance)


__all__ = [
    "RetryExhausted",
    "backoff_delay",
    "clamp",
    "configure_logging",
    "from_iso",
    "get_logger",
    "humanize_timedelta",
    "is_finite",
    "mean",
    "ms_to_datetime",
    "new_uid",
    "percent",
    "retry_on",
    "safe_float",
    "stdev",
    "to_iso",
    "utc_now",
]
