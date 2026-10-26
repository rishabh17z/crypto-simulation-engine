"""
Sentiment ingestion: Alternative.me Fear & Greed + CryptoPanic news flow.

Output contract
---------------
Every public method returns a :class:`SentimentSnapshot` whose ``score`` is normalised
to ``[-1.0, +1.0]``:

    -1.0  maximal fear / overwhelmingly negative news flow
     0.0  neutral (also the value used when a source is unreachable)
    +1.0  maximal greed / overwhelmingly positive news flow

Composition:
    score = w_fg * fear_greed_norm + w_news * news_score          (weights renormalised
                                                                   over available sources)

Operational posture
-------------------
Sentiment is an *input*, never a dependency. Rate limits, outages, missing tokens and
schema drift all degrade to a neutral score with ``degraded=True`` so the trading cycle
still runs; the decision layer simply loses the sentiment term.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

import requests

from config.settings import Settings
from src.common import clamp, get_logger, retry_on, safe_float, utc_now

LOGGER = get_logger(__name__)

TRANSIENT_HTTP_ERRORS: tuple[type[BaseException], ...] = (
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
)

USER_AGENT = "crypto-simulation-engine/1.0 (paper-trading research)"

# Fallback lexicon for CryptoPanic plans that do not expose vote counts.
POSITIVE_TERMS: frozenset[str] = frozenset(
    """
    surge surges rally rallies soar soars jump jumps gain gains gained bullish bull
    breakout breaks adoption approval approved partnership upgrade upgraded inflow
    inflows accumulate accumulation record high highs etf institutional buy buys
    buying launch launches integration listing listed support rebound recovery
    optimistic greenlight milestone
    """.split()
)
NEGATIVE_TERMS: frozenset[str] = frozenset(
    """
    crash crashes plunge plunges plummet slump slumps tumble tumbles drop drops fall
    falls fell bearish bear selloff sell-off liquidation liquidations hack hacked
    exploit exploited breach stolen scam fraud lawsuit sue sued sec ban banned
    banning crackdown delist delisted outflow outflows fear fud warning warns risk
    bankruptcy insolvency halt halted investigation downgrade rug
    """.split()
)


class SentimentError(RuntimeError):
    """Raised only when the caller explicitly demands a hard failure."""


# --------------------------------------------------------------------------------------
# Snapshot
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SentimentSnapshot:
    """Normalised sentiment for one asset (or the whole market)."""

    symbol: str
    score: float                        # [-1.0, 1.0]
    fetched_at: datetime
    fear_greed_value: int | None = None     # raw 0..100
    fear_greed_label: str | None = None
    fear_greed_norm: float | None = None    # [-1, 1]
    news_score: float | None = None         # [-1, 1]
    news_count: int = 0
    sources: tuple[str, ...] = ()
    degraded: bool = False
    notes: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        if self.score >= 0.5:
            return "very bullish"
        if self.score >= 0.15:
            return "bullish"
        if self.score <= -0.5:
            return "very bearish"
        if self.score <= -0.15:
            return "bearish"
        return "neutral"

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "score": round(self.score, 4),
            "label": self.label,
            "fear_greed_value": self.fear_greed_value,
            "fear_greed_label": self.fear_greed_label,
            "news_score": None if self.news_score is None else round(self.news_score, 4),
            "news_count": self.news_count,
            "sources": list(self.sources),
            "degraded": self.degraded,
            "notes": list(self.notes),
        }

    @classmethod
    def neutral(cls, symbol: str, note: str) -> "SentimentSnapshot":
        return cls(
            symbol=symbol,
            score=0.0,
            fetched_at=utc_now(),
            degraded=True,
            notes=(note,),
        )


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float


class _Failed:
    """Sentinel cached after a failed fetch (a cache *miss* is ``None``)."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<failed>"


FAILED = _Failed()


# --------------------------------------------------------------------------------------
# Scraper
# --------------------------------------------------------------------------------------


class SentimentScraper:
    """Thread-safe, TTL-cached sentiment aggregator."""

    def __init__(self, settings: Settings, session: requests.Session | None = None):
        self.settings = settings
        self.cfg = settings.sentiment
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        self._cache: dict[str, _CacheEntry] = {}
        self._lock = threading.RLock()
        # Flipped off permanently if CryptoPanic rejects the token (401/403).
        self._news_enabled = bool(self.cfg.cryptopanic_token)
        if self.cfg.enabled and not self._news_enabled:
            LOGGER.warning(
                "CRYPTOPANIC_TOKEN not set - sentiment will use Fear & Greed only"
            )

    # -- cache -------------------------------------------------------------------------

    def _cache_get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if entry.expires_at < time.monotonic():
                self._cache.pop(key, None)
                return None
            return entry.value

    def _cache_put(self, key: str, value: Any, ttl: int | None = None) -> None:
        with self._lock:
            # `ttl if ttl is not None` and not `ttl or ...`: a caller-supplied 0 is a
            # meaningful "expire immediately", not a request for the default.
            effective_ttl = ttl if ttl is not None else self.cfg.cache_ttl_seconds
            self._cache[key] = _CacheEntry(
                value=value,
                expires_at=time.monotonic() + effective_ttl,
            )

    def _cache_failure(self, key: str) -> None:
        """
        Negative-cache a dead source for ``failure_cooldown_seconds``.

        Without this, a source that is firewalled or down costs the full retry ladder
        once *per asset*: three assets x three attempts x a 12s connect timeout is
        nearly two minutes of dead time every cycle, and the answer is the same every
        time. One attempt per cooldown window is enough to detect recovery.
        """
        self._cache_put(key, FAILED, ttl=self.cfg.failure_cooldown_seconds)

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    # -- HTTP --------------------------------------------------------------------------

    def _request(self, url: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """GET JSON with retries on timeouts/connection errors/429/5xx."""

        @retry_on(
            TRANSIENT_HTTP_ERRORS + (SentimentError,),
            attempts=self.cfg.max_retries,
            base_delay=self.cfg.retry_base_delay,
            max_delay=15.0,
            logger=LOGGER,
        )
        def _get() -> dict[str, Any]:
            response = self.session.get(
                url, params=dict(params or {}), timeout=self.cfg.http_timeout
            )
            if response.status_code in (401, 403):
                # Credential problem - retrying cannot help.
                raise PermissionError(
                    f"{url} returned {response.status_code}: check the API token"
                )
            if response.status_code == 429:
                wait = safe_float(response.headers.get("Retry-After"), 0.0)
                if wait > 0:
                    LOGGER.warning("%s rate limited; honouring Retry-After=%.0fs", url, wait)
                    time.sleep(min(wait, 30.0))
                raise SentimentError(f"{url} rate limited (429)")
            if response.status_code >= 500:
                raise SentimentError(f"{url} server error ({response.status_code})")
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise SentimentError(f"{url} returned a non-object JSON payload")
            return payload

        return _get()

    # -- Fear & Greed ------------------------------------------------------------------

    def fetch_fear_greed(self) -> tuple[int, str] | None:
        """``(value 0..100, label)`` for the market-wide index, or ``None``."""
        cached = self._cache_get("fear_greed")
        if cached is FAILED:
            LOGGER.debug("Fear & Greed in failure cooldown; skipping fetch")
            return None
        if cached is not None:
            return cached

        try:
            payload = self._request(self.cfg.fear_greed_url, {"limit": 1, "format": "json"})
            entries = payload.get("data") or []
            if not entries:
                raise SentimentError("empty 'data' array")
            first = entries[0]
            value = int(round(safe_float(first.get("value"), 50.0)))
            value = int(clamp(value, 0, 100))
            label = str(first.get("value_classification") or "Unknown")
        except (PermissionError, SentimentError, ValueError, TypeError, KeyError) as exc:
            LOGGER.warning(
                "Fear & Greed unavailable (%s); backing off for %ds",
                exc, self.cfg.failure_cooldown_seconds,
            )
            self._cache_failure("fear_greed")
            return None
        except Exception as exc:  # noqa: BLE001 - retries already exhausted
            LOGGER.warning(
                "Fear & Greed fetch failed (%s); backing off for %ds",
                exc, self.cfg.failure_cooldown_seconds,
            )
            self._cache_failure("fear_greed")
            return None

        self._cache_put("fear_greed", (value, label))
        LOGGER.info("Fear & Greed index: %d (%s)", value, label)
        return value, label

    @staticmethod
    def normalise_fear_greed(value: int) -> float:
        """0..100 (extreme fear..extreme greed) -> -1..+1."""
        return clamp((value - 50.0) / 50.0, -1.0, 1.0)

    # -- CryptoPanic -------------------------------------------------------------------

    def fetch_news_score(self, currency: str) -> tuple[float, int] | None:
        """
        ``(score in [-1, 1], article_count)`` for one currency, or ``None``.

        Each post contributes a bounded vote-based polarity, weighted by recency
        (24-hour half-life). Posts without vote data fall back to a title lexicon so a
        free-tier token still produces signal.
        """
        if not (self.cfg.enabled and self._news_enabled):
            return None

        cache_key = f"news:{currency}"
        cached = self._cache_get(cache_key)
        if cached is FAILED:
            LOGGER.debug("CryptoPanic %s in failure cooldown; skipping fetch", currency)
            return None
        if cached is not None:
            return cached

        params = {
            "auth_token": self.cfg.cryptopanic_token,
            "currencies": currency,
            "public": "true",
            "kind": "news",
        }
        try:
            payload = self._request(self.cfg.cryptopanic_url, params)
        except PermissionError as exc:
            LOGGER.error("CryptoPanic rejected the token (%s) - disabling news source", exc)
            self._news_enabled = False
            return None
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "CryptoPanic unavailable for %s (%s); backing off for %ds",
                currency, exc, self.cfg.failure_cooldown_seconds,
            )
            self._cache_failure(cache_key)
            return None

        posts = payload.get("results") or []
        if not isinstance(posts, list) or not posts:
            self._cache_put(cache_key, (0.0, 0))
            return 0.0, 0

        now = utc_now()
        weighted_sum = 0.0
        weight_total = 0.0
        counted = 0

        for post in posts[:50]:
            if not isinstance(post, Mapping):
                continue
            polarity = self._post_polarity(post)
            if polarity is None:
                continue
            age_hours = self._post_age_hours(post, now)
            recency_weight = 0.5 ** (age_hours / 24.0)      # 24h half-life
            importance = 1.0
            votes = post.get("votes")
            if isinstance(votes, Mapping):
                importance += min(safe_float(votes.get("important"), 0.0), 10.0) / 10.0
            weight = recency_weight * importance
            weighted_sum += polarity * weight
            weight_total += weight
            counted += 1

        score = clamp(weighted_sum / weight_total, -1.0, 1.0) if weight_total else 0.0
        result = (score, counted)
        self._cache_put(cache_key, result)
        LOGGER.info(
            "CryptoPanic %s: score=%+.3f from %d post(s)", currency, score, counted
        )
        return result

    @staticmethod
    def _post_polarity(post: Mapping[str, Any]) -> float | None:
        """Vote-derived polarity in [-1, 1]; falls back to a title lexicon."""
        votes = post.get("votes")
        if isinstance(votes, Mapping):
            positive = safe_float(votes.get("positive"), 0.0) + safe_float(
                votes.get("liked"), 0.0
            )
            negative = safe_float(votes.get("negative"), 0.0) + safe_float(
                votes.get("disliked"), 0.0
            ) + safe_float(votes.get("toxic"), 0.0)
            total = positive + negative
            if total > 0:
                # tanh damps a single brigaded post from dominating the average.
                return clamp(math.tanh((positive - negative) / max(total, 3.0)), -1.0, 1.0)

        title = str(post.get("title") or "").lower()
        if not title:
            return None
        tokens = {token.strip(".,:;!?()[]\"'").lower() for token in title.split()}
        hits_pos = len(tokens & POSITIVE_TERMS)
        hits_neg = len(tokens & NEGATIVE_TERMS)
        if hits_pos == hits_neg == 0:
            return 0.0
        return clamp((hits_pos - hits_neg) / float(hits_pos + hits_neg), -1.0, 1.0)

    @staticmethod
    def _post_age_hours(post: Mapping[str, Any], now: datetime) -> float:
        raw = post.get("published_at") or post.get("created_at")
        if not raw:
            return 12.0
        try:
            published = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            return 12.0
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        return max((now - published).total_seconds() / 3600.0, 0.0)

    # -- composition -------------------------------------------------------------------

    def get_sentiment(self, symbol: str, currency: str | None = None) -> SentimentSnapshot:
        """Composite sentiment for one asset. Never raises."""
        if not self.cfg.enabled:
            return SentimentSnapshot.neutral(symbol, "sentiment disabled by config")

        currency = (currency or symbol.split("/")[0]).upper()

        fear_greed = self.fetch_fear_greed()
        news = self.fetch_news_score(currency)

        components: list[tuple[float, float]] = []   # (value, weight)
        sources: list[str] = []
        notes: list[str] = []

        fg_norm: float | None = None
        if fear_greed is not None:
            fg_norm = self.normalise_fear_greed(fear_greed[0])
            components.append((fg_norm, self.cfg.weight_fear_greed))
            sources.append("fear_greed")
        else:
            notes.append("fear_greed unavailable")

        news_score: float | None = None
        if news is not None:
            news_score, news_count = news
            if news_count == 0:
                notes.append("no recent news articles")
            components.append((news_score, self.cfg.weight_news))
            sources.append("cryptopanic")
        else:
            news_count = 0
            notes.append("cryptopanic unavailable")

        if not components:
            if not self.cfg.neutral_on_failure:
                raise SentimentError(f"no sentiment source available for {symbol}")
            return SentimentSnapshot.neutral(symbol, "; ".join(notes) or "no sources")

        # Renormalise over the sources that actually answered.
        weight_total = sum(weight for _, weight in components) or 1.0
        score = clamp(
            sum(value * weight for value, weight in components) / weight_total, -1.0, 1.0
        )

        snapshot = SentimentSnapshot(
            symbol=symbol,
            score=score,
            fetched_at=utc_now(),
            fear_greed_value=fear_greed[0] if fear_greed else None,
            fear_greed_label=fear_greed[1] if fear_greed else None,
            fear_greed_norm=fg_norm,
            news_score=news_score,
            news_count=news_count,
            sources=tuple(sources),
            degraded=len(components) < 2,
            notes=tuple(notes),
        )
        LOGGER.info(
            "sentiment %s: %+.3f (%s) sources=%s%s",
            symbol, snapshot.score, snapshot.label, ",".join(sources) or "none",
            " [degraded]" if snapshot.degraded else "",
        )
        return snapshot

    def bulk_sentiment(self, assets: Any) -> dict[str, SentimentSnapshot]:
        """Sentiment for every asset in the universe, keyed by display symbol."""
        out: dict[str, SentimentSnapshot] = {}
        for asset in assets:
            try:
                out[asset.symbol] = self.get_sentiment(asset.symbol, asset.sentiment_key)
            except Exception as exc:  # noqa: BLE001 - never break a cycle
                LOGGER.exception("sentiment failed for %s", asset.symbol)
                out[asset.symbol] = SentimentSnapshot.neutral(
                    asset.symbol, f"{type(exc).__name__}: {exc}"
                )
        return out

    def close(self) -> None:
        try:
            self.session.close()
        except Exception:  # noqa: BLE001
            LOGGER.debug("session close raised during teardown", exc_info=True)


__all__ = ["SentimentError", "SentimentScraper", "SentimentSnapshot"]
