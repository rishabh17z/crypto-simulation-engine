"""
Decision layer: indicators + sentiment -> exactly one of BUY / SELL / HOLD, with a
mandatory take-profit and stop-loss attached to every actionable signal.

Two interchangeable paths, same output type
-------------------------------------------
1. :class:`DecisionEngine` - the default, deterministic heuristic. Reproducible,
   testable, no external dependency, and auditable after the fact because every
   component of the score is recorded.
2. :func:`build_decision_prompt` / :func:`parse_decision_response` - formats the same
   feature set into a strict prompt and parses a model's reply back into a
   :class:`TradeDecision`. Provider-agnostic: hand the prompt to whatever LLM client you
   like and feed the raw text back. Anything that is not a well-formed BUY/SELL/HOLD is
   rejected and the caller falls back to the heuristic, so a hallucinating model can
   never place a trade.

Scoring model
-------------
    score = w_trend * trend + w_momentum * momentum + w_sentiment * sentiment   in [-1, 1]

    trend     0.70 * tanh(ema_spread / 2%)  +  0.30 * tanh(price_vs_ema_fast / 3%)
    momentum  (RSI - 50) / 25, clipped  -> trend-following, not mean-reverting
    sentiment the normalised [-1, 1] composite from the sentiment scraper

    score >= buy_threshold   -> BUY
    score <= sell_threshold  -> SELL
    otherwise                -> HOLD

Vetoes (applied after scoring; any single veto forces HOLD): volatility circuit breaker
(ATR% too high), RSI exhaustion (never chase a vertical move), confidence floor, and
shorting disabled.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from config.settings import Settings
from src.common import clamp, get_logger
from src.data_ingestion.price_fetcher import MarketSnapshot
from src.data_ingestion.sentiment_scraper import SentimentSnapshot

LOGGER = get_logger(__name__)


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

    @classmethod
    def parse(cls, raw: Any) -> "Action":
        text = str(raw or "").strip().upper()
        if text in cls.__members__:
            return cls[text]
        raise ValueError(f"not a valid action: {raw!r}")


class ExitReason(str, Enum):
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    SIGNAL_FLIP = "SIGNAL_FLIP"
    END_OF_EXPERIMENT = "END_OF_EXPERIMENT"
    MANUAL = "MANUAL"


@dataclass(frozen=True, slots=True)
class TradeDecision:
    """A fully-specified, risk-bounded trading instruction."""

    symbol: str
    action: Action
    confidence: float                    # [0, 1]
    score: float                         # [-1, 1]
    reference_price: float
    stop_loss: float | None
    take_profit: float | None
    rationale: str
    components: Mapping[str, float] = field(default_factory=dict)
    vetoes: tuple[str, ...] = ()
    source: str = "heuristic"

    @property
    def is_actionable(self) -> bool:
        return self.action is not Action.HOLD

    @property
    def side(self) -> str:
        return "long" if self.action is Action.BUY else "short"

    @property
    def stop_distance_pct(self) -> float:
        if not (self.stop_loss and self.reference_price):
            return 0.0
        return abs(self.reference_price - self.stop_loss) / self.reference_price

    @property
    def target_distance_pct(self) -> float:
        if not (self.take_profit and self.reference_price):
            return 0.0
        return abs(self.take_profit - self.reference_price) / self.reference_price

    @property
    def risk_reward(self) -> float:
        stop = self.stop_distance_pct
        return (self.target_distance_pct / stop) if stop else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action.value,
            "confidence": round(self.confidence, 4),
            "score": round(self.score, 4),
            "reference_price": round(self.reference_price, 8),
            "stop_loss": None if self.stop_loss is None else round(self.stop_loss, 8),
            "take_profit": None if self.take_profit is None else round(self.take_profit, 8),
            "risk_reward": round(self.risk_reward, 3),
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "vetoes": list(self.vetoes),
            "source": self.source,
            "rationale": self.rationale,
        }

    @classmethod
    def hold(
        cls,
        symbol: str,
        price: float,
        reason: str,
        *,
        score: float = 0.0,
        confidence: float = 0.0,
        components: Mapping[str, float] | None = None,
        vetoes: tuple[str, ...] = (),
        source: str = "heuristic",
    ) -> "TradeDecision":
        return cls(
            symbol=symbol,
            action=Action.HOLD,
            confidence=confidence,
            score=score,
            reference_price=price,
            stop_loss=None,
            take_profit=None,
            rationale=reason,
            components=dict(components or {}),
            vetoes=vetoes,
            source=source,
        )


@dataclass(frozen=True, slots=True)
class ExitEvaluation:
    """Result of re-examining an open position against fresh data."""

    should_close: bool
    reason: ExitReason | None = None
    detail: str = ""


class DecisionEngine:
    """Deterministic rule engine. Stateless: safe to share across threads."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.strategy = settings.strategy
        self.risk = settings.risk

        total = (
            self.strategy.weight_trend
            + self.strategy.weight_momentum
            + self.strategy.weight_sentiment
        )
        if total <= 0:
            raise ValueError("strategy weights must sum to a positive number")
        self._w_trend = self.strategy.weight_trend / total
        self._w_momentum = self.strategy.weight_momentum / total
        self._w_sentiment = self.strategy.weight_sentiment / total

    # -- feature extraction ------------------------------------------------------------

    def trend_component(self, snapshot: MarketSnapshot) -> float:
        """EMA regime (70%) blended with price location vs the fast EMA (30%)."""
        regime = math.tanh(snapshot.ema_spread_pct / 0.02)
        location = math.tanh(snapshot.price_vs_ema_fast_pct / 0.03)
        return clamp(0.70 * regime + 0.30 * location, -1.0, 1.0)

    def momentum_component(self, snapshot: MarketSnapshot) -> float:
        """RSI as trend confirmation: 75 -> +1, 50 -> 0, 25 -> -1."""
        return clamp((snapshot.rsi - 50.0) / 25.0, -1.0, 1.0)

    def sentiment_component(self, sentiment: SentimentSnapshot | None) -> float:
        return 0.0 if sentiment is None else clamp(sentiment.score, -1.0, 1.0)

    @staticmethod
    def sentiment_available(sentiment: SentimentSnapshot | None) -> bool:
        """
        True only when a source actually answered.

        A score of 0.0 is ambiguous on its own: it means "the market is balanced" when
        a source answered, and "we know nothing" when none did. Those must not be
        treated alike, so availability is read from ``sources``, not from the score.
        """
        return sentiment is not None and bool(sentiment.sources)

    def composite_score(
        self, snapshot: MarketSnapshot, sentiment: SentimentSnapshot | None
    ) -> tuple[float, dict[str, float], bool]:
        """
        ``(score, components, sentiment_available)``.

        When no sentiment source answered, its weight is redistributed across trend and
        momentum rather than multiplied by a fabricated 0.0. Counting absent data as
        neutral would shrink the attainable score to +/-0.70 against unchanged
        thresholds - a silent 30% handicap that makes the engine look inert during an
        outage instead of trading the signals it still has.
        """
        trend = self.trend_component(snapshot)
        momentum = self.momentum_component(snapshot)
        available = self.sentiment_available(sentiment)

        if available:
            sentiment_value = self.sentiment_component(sentiment)
            w_trend, w_momentum, w_sentiment = (
                self._w_trend, self._w_momentum, self._w_sentiment,
            )
        else:
            sentiment_value = 0.0
            technical_weight = self._w_trend + self._w_momentum
            w_trend = self._w_trend / technical_weight
            w_momentum = self._w_momentum / technical_weight
            w_sentiment = 0.0

        score = clamp(
            w_trend * trend + w_momentum * momentum + w_sentiment * sentiment_value,
            -1.0,
            1.0,
        )
        components = {
            "trend": trend,
            "momentum": momentum,
            "sentiment": sentiment_value,
            "sentiment_weight": w_sentiment,
            "rsi": snapshot.rsi,
            "atr_pct": snapshot.atr_pct,
            "ema_spread_pct": snapshot.ema_spread_pct,
        }
        return score, components, available

    # -- levels ------------------------------------------------------------------------

    def compute_levels(
        self, action: Action, entry: float, atr: float
    ) -> tuple[float, float]:
        """
        ATR-derived stop-loss and take-profit, clamped to the configured band.

        The stop distance is floored at ``min_stop_pct`` (validated to exceed round-trip
        fees + slippage) and capped at ``max_stop_pct``. When the raw ATR stop is
        clamped, the target is widened proportionally so the configured reward:risk
        ratio survives.
        """
        if entry <= 0:
            raise ValueError("entry price must be positive")

        atr = max(atr, 0.0)
        raw_stop = atr * self.strategy.atr_sl_multiple
        stop_distance = clamp(
            raw_stop if raw_stop > 0 else entry * self.strategy.min_stop_pct,
            entry * self.strategy.min_stop_pct,
            entry * self.strategy.max_stop_pct,
        )
        rr = self.strategy.atr_tp_multiple / self.strategy.atr_sl_multiple
        target_distance = max(atr * self.strategy.atr_tp_multiple, stop_distance * rr)

        if action is Action.BUY:
            return entry - stop_distance, entry + target_distance
        if action is Action.SELL:
            return entry + stop_distance, entry - target_distance
        raise ValueError("HOLD has no stop/target levels")

    # -- confidence --------------------------------------------------------------------

    def _confidence(
        self,
        score: float,
        components: Mapping[str, float],
        sentiment: SentimentSnapshot | None,
    ) -> float:
        confidence = clamp(abs(score), 0.0, 1.0)
        signs = {
            math.copysign(1.0, value)
            for key, value in components.items()
            if abs(value) > 0.05 and key in ("trend", "momentum", "sentiment")
        }
        if len(signs) == 1:                       # every factor agrees
            confidence = clamp(confidence + 0.10, 0.0, 1.0)
        if sentiment is not None and sentiment.degraded:
            confidence *= 0.85                    # partial information, less conviction
        return round(confidence, 4)

    # -- main entry point --------------------------------------------------------------

    def decide(
        self,
        snapshot: MarketSnapshot,
        sentiment: SentimentSnapshot | None = None,
        *,
        open_side: str | None = None,
    ) -> TradeDecision:
        """
        Produce a decision for one asset.

        ``open_side`` ("long"/"short"/None) lets the engine avoid re-entering an
        existing position; exits are handled by :meth:`evaluate_exit`.
        """
        price = snapshot.price
        if price <= 0:
            return TradeDecision.hold(snapshot.symbol, price, "invalid reference price")

        score, components, sentiment_ok = self.composite_score(snapshot, sentiment)
        confidence = self._confidence(score, components, sentiment)
        if not sentiment_ok:
            LOGGER.debug(
                "%s: no sentiment source answered; scoring on technicals alone "
                "(weights renormalised to trend %.2f / momentum %.2f)",
                snapshot.symbol,
                self._w_trend / (self._w_trend + self._w_momentum),
                self._w_momentum / (self._w_trend + self._w_momentum),
            )

        # --- raw signal ---
        if score >= self.strategy.buy_threshold:
            action = Action.BUY
        elif score <= self.strategy.sell_threshold:
            action = Action.SELL
        else:
            return TradeDecision.hold(
                snapshot.symbol,
                price,
                f"score {score:+.3f} inside the neutral band "
                f"[{self.strategy.sell_threshold:+.2f}, "
                f"{self.strategy.buy_threshold:+.2f}]",
                score=score,
                confidence=confidence,
                components=components,
            )

        # --- vetoes ---
        vetoes: list[str] = []
        if snapshot.atr_pct > self.strategy.max_atr_pct:
            vetoes.append(
                f"volatility circuit breaker: ATR {snapshot.atr_pct:.2%} > "
                f"{self.strategy.max_atr_pct:.2%}"
            )
        if action is Action.BUY and snapshot.rsi >= self.strategy.rsi_buy_veto:
            vetoes.append(
                f"RSI {snapshot.rsi:.1f} >= buy veto {self.strategy.rsi_buy_veto:.0f} "
                "(exhausted upside)"
            )
        if action is Action.SELL and snapshot.rsi <= self.strategy.rsi_sell_veto:
            vetoes.append(
                f"RSI {snapshot.rsi:.1f} <= sell veto {self.strategy.rsi_sell_veto:.0f} "
                "(exhausted downside)"
            )
        if confidence < self.strategy.min_confidence:
            vetoes.append(
                f"confidence {confidence:.2f} < floor {self.strategy.min_confidence:.2f}"
            )
        if action is Action.SELL and open_side is None and not self.risk.allow_shorts:
            vetoes.append("short entries disabled (ALLOW_SHORTS=false)")
        if open_side == "long" and action is Action.BUY:
            vetoes.append("already long this asset; no pyramiding")
        if open_side == "short" and action is Action.SELL:
            vetoes.append("already short this asset; no pyramiding")

        if vetoes:
            return TradeDecision.hold(
                snapshot.symbol,
                price,
                "; ".join(vetoes),
                score=score,
                confidence=confidence,
                components=components,
                vetoes=tuple(vetoes),
            )

        stop_loss, take_profit = self.compute_levels(action, price, snapshot.atr)
        rationale = self._explain(action, snapshot, sentiment, score, components)

        decision = TradeDecision(
            symbol=snapshot.symbol,
            action=action,
            confidence=confidence,
            score=score,
            reference_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            rationale=rationale,
            components=components,
            source="heuristic",
        )
        LOGGER.info(
            "%s -> %s (score %+.3f, conf %.2f) SL=%.6f TP=%.6f R:R=%.2f",
            decision.symbol, decision.action.value, decision.score,
            decision.confidence, stop_loss, take_profit, decision.risk_reward,
        )
        return decision

    def _explain(
        self,
        action: Action,
        snapshot: MarketSnapshot,
        sentiment: SentimentSnapshot | None,
        score: float,
        components: Mapping[str, float],
    ) -> str:
        if self.sentiment_available(sentiment):
            assert sentiment is not None
            sentiment_text = f"sentiment {sentiment.score:+.2f} ({sentiment.label})"
        else:
            sentiment_text = "sentiment unavailable (weights renormalised)"
        return (
            f"{action.value}: {snapshot.regime} regime "
            f"(EMA{snapshot.ema_fast_period}/EMA{snapshot.ema_slow_period} spread "
            f"{snapshot.ema_spread_pct:+.2%}), RSI {snapshot.rsi:.1f}, "
            f"ATR {snapshot.atr_pct:.2%}, {sentiment_text}; composite score "
            f"{score:+.3f} (trend {components['trend']:+.2f} / momentum "
            f"{components['momentum']:+.2f} / sentiment {components['sentiment']:+.2f})"
        )

    # -- exits -------------------------------------------------------------------------

    def evaluate_exit(
        self,
        side: str,
        snapshot: MarketSnapshot,
        sentiment: SentimentSnapshot | None = None,
    ) -> ExitEvaluation:
        """
        Signal-based exit check for an open position.

        Price-based exits (stop-loss / take-profit) are enforced by the executor against
        candle highs and lows; this covers the softer case where the regime that
        justified the trade has flipped before either level is touched.
        """
        if not self.strategy.trend_flip_exit:
            return ExitEvaluation(False)

        score, _components, _available = self.composite_score(snapshot, sentiment)

        if side == "long" and score <= self.strategy.sell_threshold:
            return ExitEvaluation(
                True,
                ExitReason.SIGNAL_FLIP,
                f"long position with bearish composite score {score:+.3f} "
                f"(regime {snapshot.regime}, RSI {snapshot.rsi:.1f})",
            )
        if side == "short" and score >= self.strategy.buy_threshold:
            return ExitEvaluation(
                True,
                ExitReason.SIGNAL_FLIP,
                f"short position with bullish composite score {score:+.3f} "
                f"(regime {snapshot.regime}, RSI {snapshot.rsi:.1f})",
            )
        return ExitEvaluation(False)


# --------------------------------------------------------------------------------------
# Optional LLM path (provider-agnostic)
# --------------------------------------------------------------------------------------

DECISION_SCHEMA_HINT = (
    '{"action": "BUY|SELL|HOLD", "confidence": 0.0-1.0, '
    '"stop_loss": <number|null>, "take_profit": <number|null>, "rationale": "<= 2 sentences"}'
)


def build_decision_prompt(
    snapshot: MarketSnapshot,
    sentiment: SentimentSnapshot | None,
    settings: Settings,
    *,
    open_side: str | None = None,
) -> str:
    """Render the same feature set the heuristic sees as a strict, parseable prompt."""
    strategy = settings.strategy
    features = {
        "symbol": snapshot.symbol,
        "timeframe": snapshot.timeframe,
        "candle_close_utc": snapshot.candle_time.isoformat(),
        "price": round(snapshot.price, 8),
        "rsi_14": round(snapshot.rsi, 2),
        f"ema_{strategy.ema_fast_period}": round(snapshot.ema_fast, 8),
        f"ema_{strategy.ema_slow_period}": round(snapshot.ema_slow, 8),
        "ema_spread_pct": round(snapshot.ema_spread_pct * 100, 3),
        "price_vs_ema_fast_pct": round(snapshot.price_vs_ema_fast_pct * 100, 3),
        "atr_14": round(snapshot.atr, 8),
        "atr_pct": round(snapshot.atr_pct * 100, 3),
        "regime": snapshot.regime,
        "sentiment_score": None if sentiment is None else round(sentiment.score, 4),
        "sentiment_label": None if sentiment is None else sentiment.label,
        "fear_greed_index": None if sentiment is None else sentiment.fear_greed_value,
        "news_articles": 0 if sentiment is None else sentiment.news_count,
        "existing_position": open_side or "none",
    }
    constraints = {
        "max_risk_per_trade_pct": round(settings.risk.max_risk_per_trade * 100, 2),
        "stop_distance_band_pct": [
            round(strategy.min_stop_pct * 100, 2),
            round(strategy.max_stop_pct * 100, 2),
        ],
        "min_reward_to_risk": round(
            strategy.atr_tp_multiple / strategy.atr_sl_multiple, 2
        ),
        "shorts_allowed": settings.risk.allow_shorts,
    }
    return (
        "You are a disciplined systematic crypto trader operating a paper-trading "
        "account. Decide the action for ONE instrument from the features below.\n\n"
        "RULES\n"
        "1. Reply with a single JSON object and nothing else.\n"
        f"2. Schema: {DECISION_SCHEMA_HINT}\n"
        "3. BUY or SELL REQUIRE both stop_loss and take_profit as absolute prices.\n"
        "4. stop_loss must sit below price for BUY and above price for SELL; "
        "take_profit on the opposite side.\n"
        "5. The stop distance must stay inside stop_distance_band_pct and the "
        "reward:risk ratio must be at least min_reward_to_risk.\n"
        "6. If the evidence is mixed or volatility is extreme, answer HOLD with null "
        "levels. HOLD is always acceptable.\n\n"
        f"FEATURES\n{json.dumps(features, indent=2)}\n\n"
        f"CONSTRAINTS\n{json.dumps(constraints, indent=2)}\n"
    )


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
_ACTION_WORD = re.compile(r"\b(BUY|SELL|HOLD)\b", re.IGNORECASE)


def parse_decision_response(
    raw: str,
    snapshot: MarketSnapshot,
    settings: Settings,
) -> TradeDecision:
    """
    Parse a model reply into a validated :class:`TradeDecision`.

    Raises :class:`ValueError` on anything malformed or risk-inconsistent so the caller
    can fall back to the heuristic. Levels that violate the configured band are
    *rejected*, not silently repaired - a model is not allowed to widen the risk
    envelope.
    """
    if not raw or not raw.strip():
        raise ValueError("empty model response")

    payload: dict[str, Any] | None = None
    match = _JSON_BLOCK.search(raw)
    if match:
        try:
            candidate = json.loads(match.group(0))
            if isinstance(candidate, dict):
                payload = candidate
        except json.JSONDecodeError:
            payload = None

    if payload is None:
        word = _ACTION_WORD.search(raw)
        if not word:
            raise ValueError("response contained neither JSON nor a BUY/SELL/HOLD token")
        payload = {"action": word.group(1).upper()}

    action = Action.parse(payload.get("action"))
    price = snapshot.price
    confidence = clamp(float(payload.get("confidence") or 0.0), 0.0, 1.0)
    rationale = str(payload.get("rationale") or "model decision")[:500]

    if action is Action.HOLD:
        return TradeDecision.hold(
            snapshot.symbol, price, rationale, confidence=confidence, source="llm"
        )

    if action is Action.SELL and not settings.risk.allow_shorts:
        raise ValueError("model returned SELL but shorting is disabled")

    try:
        stop_loss = float(payload["stop_loss"])
        take_profit = float(payload["take_profit"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{action.value} requires numeric stop_loss and take_profit") from exc

    strategy = settings.strategy
    direction = 1.0 if action is Action.BUY else -1.0
    if (stop_loss - price) * direction >= 0:
        raise ValueError("stop_loss is on the wrong side of the entry price")
    if (take_profit - price) * direction <= 0:
        raise ValueError("take_profit is on the wrong side of the entry price")

    stop_pct = abs(price - stop_loss) / price
    if not strategy.min_stop_pct <= stop_pct <= strategy.max_stop_pct:
        raise ValueError(
            f"stop distance {stop_pct:.2%} outside the permitted band "
            f"[{strategy.min_stop_pct:.2%}, {strategy.max_stop_pct:.2%}]"
        )
    min_rr = strategy.atr_tp_multiple / strategy.atr_sl_multiple
    reward_risk = (abs(take_profit - price) / price) / stop_pct
    if reward_risk < min_rr - 1e-9:
        raise ValueError(f"reward:risk {reward_risk:.2f} below minimum {min_rr:.2f}")

    return TradeDecision(
        symbol=snapshot.symbol,
        action=action,
        confidence=confidence,
        score=direction * confidence,
        reference_price=price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        rationale=rationale,
        components={"model_confidence": confidence},
        source="llm",
    )


__all__ = [
    "Action",
    "DECISION_SCHEMA_HINT",
    "DecisionEngine",
    "ExitEvaluation",
    "ExitReason",
    "TradeDecision",
    "build_decision_prompt",
    "parse_decision_response",
]
