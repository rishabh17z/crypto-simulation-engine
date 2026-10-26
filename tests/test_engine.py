#!/usr/bin/env python3
"""
Offline regression tests for the simulation engine.

No network and no exchange: synthetic :class:`MarketSnapshot` objects drive the decision
engine, and the executor runs against a temporary SQLite file with the CCXT client set
to ``None``. Covers the paths that a single live cycle cannot reach on demand -
stop-outs, take-profits, short entries, risk rejections, restart recovery and the final
report.

Run with pytest, or standalone:

    python tests/test_engine.py
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SETTINGS, Settings  # noqa: E402
from src.common import utc_now  # noqa: E402
from src.data_ingestion.price_fetcher import (  # noqa: E402
    MarketSnapshot,
    atr_wilder,
    ema,
    rsi_wilder,
)
from src.data_ingestion.sentiment_scraper import SentimentSnapshot  # noqa: E402
from src.engine.decision_logic import (  # noqa: E402
    Action,
    DecisionEngine,
    ExitReason,
    build_decision_prompt,
    parse_decision_response,
)
from src.engine.testnet_executor import TestnetExecutor  # noqa: E402
from src.storage.db_manager import Database  # noqa: E402

TOLERANCE = 1e-6


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


def make_snapshot(
    *,
    symbol: str = "BTC/USDT",
    close: float = 100.0,
    rsi: float = 55.0,
    ema_fast: float = 100.0,
    ema_slow: float = 95.0,
    atr: float = 2.0,
    high: float | None = None,
    low: float | None = None,
    candle_age_hours: float = 1.0,
) -> MarketSnapshot:
    return MarketSnapshot(
        symbol=symbol,
        exchange_symbol=f"{symbol}:USDT",
        timeframe="4h",
        candle_time=utc_now() - timedelta(hours=candle_age_hours),
        fetched_at=utc_now(),
        open=close,
        high=high if high is not None else close * 1.01,
        low=low if low is not None else close * 0.99,
        close=close,
        volume=1_000.0,
        prev_close=close * 0.999,
        rsi=rsi,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        atr=atr,
        ema_fast_period=50,
        ema_slow_period=200,
        candles=400,
        last_price=close,
    )


def sentiment(score: float) -> SentimentSnapshot:
    return SentimentSnapshot(
        symbol="BTC/USDT",
        score=score,
        fetched_at=utc_now(),
        fear_greed_value=int(50 + score * 50),
        sources=("fear_greed", "cryptopanic"),
    )


def unavailable_sentiment() -> SentimentSnapshot:
    """What the scraper returns when no source answered at all."""
    return SentimentSnapshot.neutral("BTC/USDT", "all sources unreachable")


class FakeSession:
    """Counts HTTP attempts and always fails, to exercise the negative cache."""

    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.attempts = 0

    def get(self, *_args, **_kwargs):
        import requests

        self.attempts += 1
        raise requests.exceptions.ConnectionError("simulated network block")

    def close(self) -> None:
        pass


def temp_setup(**risk_overrides) -> tuple[Settings, Database, TestnetExecutor]:
    tmp = Path(tempfile.mkdtemp(prefix="sim-test-")) / "test.sqlite3"
    settings = SETTINGS
    if risk_overrides:
        settings = replace(settings, risk=replace(settings.risk, **risk_overrides))
    database = Database(tmp).connect()
    executor = TestnetExecutor(settings, database, price_fetcher=None)
    executor.bootstrap()
    return settings, database, executor


# --------------------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------------------


def test_rsi_bounds_and_extremes() -> None:
    import pandas as pd

    rising = pd.Series([float(i) for i in range(1, 60)])
    falling = pd.Series([float(i) for i in range(60, 1, -1)])
    assert abs(rsi_wilder(rising, 14).iloc[-1] - 100.0) < TOLERANCE, "monotonic up -> 100"
    assert abs(rsi_wilder(falling, 14).iloc[-1] - 0.0) < TOLERANCE, "monotonic down -> 0"

    noisy = pd.Series([100 + (3 if i % 2 else -3) for i in range(80)], dtype=float)
    value = rsi_wilder(noisy, 14).iloc[-1]
    assert 0.0 <= value <= 100.0 and 30.0 < value < 70.0, f"oscillator mid-range, got {value}"


def test_ema_and_atr_are_finite_after_warmup() -> None:
    import pandas as pd

    close = pd.Series([100.0 + i * 0.5 for i in range(260)])
    high = close * 1.01
    low = close * 0.99
    assert pd.isna(ema(close, 200).iloc[198]), "EMA must not emit before warm-up"
    assert ema(close, 200).iloc[-1] > 0
    assert atr_wilder(high, low, close, 14).iloc[-1] > 0


# --------------------------------------------------------------------------------------
# Decision logic
# --------------------------------------------------------------------------------------


def test_bull_regime_with_positive_sentiment_buys() -> None:
    engine = DecisionEngine(SETTINGS)
    decision = engine.decide(
        make_snapshot(close=100, rsi=62, ema_fast=102, ema_slow=96), sentiment(0.6)
    )
    assert decision.action is Action.BUY, decision.rationale
    assert decision.stop_loss < decision.reference_price < decision.take_profit
    assert decision.risk_reward >= 1.49, decision.risk_reward


def test_bear_regime_with_negative_sentiment_sells() -> None:
    engine = DecisionEngine(SETTINGS)
    decision = engine.decide(
        make_snapshot(close=100, rsi=38, ema_fast=96, ema_slow=104), sentiment(-0.6)
    )
    assert decision.action is Action.SELL, decision.rationale
    assert decision.stop_loss > decision.reference_price > decision.take_profit


def test_neutral_band_holds() -> None:
    engine = DecisionEngine(SETTINGS)
    decision = engine.decide(
        make_snapshot(close=100, rsi=50, ema_fast=100, ema_slow=100), sentiment(0.0)
    )
    assert decision.action is Action.HOLD
    assert decision.stop_loss is None and decision.take_profit is None


def test_volatility_circuit_breaker_vetoes() -> None:
    engine = DecisionEngine(SETTINGS)
    decision = engine.decide(
        make_snapshot(close=100, rsi=65, ema_fast=105, ema_slow=95, atr=25.0),
        sentiment(0.9),
    )
    assert decision.action is Action.HOLD
    assert any("circuit breaker" in veto for veto in decision.vetoes), decision.vetoes


def test_rsi_exhaustion_vetoes_a_buy() -> None:
    engine = DecisionEngine(SETTINGS)
    decision = engine.decide(
        make_snapshot(close=100, rsi=88, ema_fast=105, ema_slow=95), sentiment(0.9)
    )
    assert decision.action is Action.HOLD
    assert any("buy veto" in veto for veto in decision.vetoes), decision.vetoes


def test_stop_distance_is_clamped_into_the_configured_band() -> None:
    engine = DecisionEngine(SETTINGS)
    strategy = SETTINGS.strategy
    for atr in (0.0, 0.01, 5.0, 500.0):
        stop, target = engine.compute_levels(Action.BUY, 100.0, atr)
        stop_pct = (100.0 - stop) / 100.0
        assert strategy.min_stop_pct - TOLERANCE <= stop_pct <= strategy.max_stop_pct + TOLERANCE, (
            f"atr={atr} produced a {stop_pct:.4%} stop"
        )
        rr = (target - 100.0) / (100.0 - stop)
        assert rr >= strategy.atr_tp_multiple / strategy.atr_sl_multiple - TOLERANCE


def test_min_stop_clears_round_trip_costs() -> None:
    cost = 2 * (SETTINGS.simulation.fee_rate + SETTINGS.simulation.slippage_rate)
    assert SETTINGS.strategy.min_stop_pct > cost


def test_exit_evaluation_flips_on_regime_change() -> None:
    engine = DecisionEngine(SETTINGS)
    bearish = make_snapshot(close=100, rsi=32, ema_fast=94, ema_slow=106)
    evaluation = engine.evaluate_exit("long", bearish, sentiment(-0.7))
    assert evaluation.should_close and evaluation.reason is ExitReason.SIGNAL_FLIP
    assert not engine.evaluate_exit("short", bearish, sentiment(-0.7)).should_close


def test_absent_sentiment_renormalises_instead_of_counting_as_neutral() -> None:
    """Missing data must not be scored as a neutral reading."""
    engine = DecisionEngine(SETTINGS)
    snapshot = make_snapshot(close=100.0, rsi=62, ema_fast=102, ema_slow=96)

    neutral_but_present = replace(sentiment(0.0), sources=("fear_greed",))
    score_present, _, available_present = engine.composite_score(
        snapshot, neutral_but_present
    )
    score_absent, components, available_absent = engine.composite_score(
        snapshot, unavailable_sentiment()
    )

    assert available_present and not available_absent
    assert components["sentiment_weight"] == 0.0, "absent sentiment must carry no weight"
    assert score_absent > score_present, (
        f"renormalised score {score_absent:.4f} should exceed the damped "
        f"score {score_present:.4f}"
    )
    # Technical-only weights must still be a proper distribution.
    assert abs(score_absent - (0.45 * components["trend"] + 0.25 * components["momentum"])
               / 0.70) < 1e-9


def test_outage_does_not_suppress_an_otherwise_valid_signal() -> None:
    """
    The regression this guards: during a sentiment outage the old code capped the
    score at +/-0.70 against an unchanged +0.30 threshold, turning tradable setups
    into HOLD and making the engine look inert.
    """
    engine = DecisionEngine(SETTINGS)
    # Calibrated so the renormalised score clears +0.30 but 0.70x of it does not.
    snapshot = make_snapshot(close=101.346, rsi=56, ema_fast=101.346, ema_slow=100.0)

    absent = engine.decide(snapshot, unavailable_sentiment())
    present = engine.decide(snapshot, replace(sentiment(0.0), sources=("fear_greed",)))

    assert absent.action is Action.BUY, f"{absent.score:+.4f}: {absent.rationale}"
    assert present.action is Action.HOLD, f"{present.score:+.4f}"
    assert "renormalised" in absent.rationale
    assert absent.stop_loss is not None and absent.take_profit is not None


def test_a_genuinely_neutral_reading_still_damps_the_score() -> None:
    """The flip side: a source that answered 0.0 must keep its weight."""
    engine = DecisionEngine(SETTINGS)
    snapshot = make_snapshot(close=100.0, rsi=62, ema_fast=102, ema_slow=96)
    answered = replace(sentiment(0.0), sources=("fear_greed", "cryptopanic"))
    _score, components, available = engine.composite_score(snapshot, answered)
    assert available
    assert abs(components["sentiment_weight"] - 0.30) < 1e-9


def test_failed_sentiment_source_is_attempted_once_per_cooldown() -> None:
    """A firewalled source must not cost the retry ladder once per asset."""
    from dataclasses import replace as dc_replace

    from src.data_ingestion.sentiment_scraper import SentimentScraper

    settings = replace(
        SETTINGS,
        sentiment=dc_replace(
            SETTINGS.sentiment,
            max_retries=3,
            retry_base_delay=0.01,
            failure_cooldown_seconds=600,
            cryptopanic_token="",       # Fear & Greed is the only source under test
        ),
    )
    session = FakeSession()
    scraper = SentimentScraper(settings, session=session)
    try:
        results = scraper.bulk_sentiment(settings.enabled_assets)
        assert len(results) == len(settings.enabled_assets)
        assert all(s.score == 0.0 and s.degraded for s in results.values())
        assert all(not s.sources for s in results.values())
        # One retry ladder in total, not one per asset.
        assert session.attempts == settings.sentiment.max_retries, (
            f"expected {settings.sentiment.max_retries} attempts, "
            f"got {session.attempts}"
        )
    finally:
        scraper.close()


def test_cooldown_expiry_allows_a_fresh_attempt() -> None:
    from dataclasses import replace as dc_replace

    from src.data_ingestion.sentiment_scraper import SentimentScraper

    settings = replace(
        SETTINGS,
        sentiment=dc_replace(
            SETTINGS.sentiment,
            max_retries=1,
            retry_base_delay=0.01,
            failure_cooldown_seconds=0,   # expires immediately
            cryptopanic_token="",
        ),
    )
    session = FakeSession()
    scraper = SentimentScraper(settings, session=session)
    try:
        scraper.get_sentiment("BTC/USDT", "BTC")
        first = session.attempts
        scraper.get_sentiment("ETH/USDT", "ETH")
        assert session.attempts > first, "an expired cooldown must permit a retry"
    finally:
        scraper.close()


def test_disabled_sentiment_makes_no_http_calls() -> None:
    from dataclasses import replace as dc_replace

    from src.data_ingestion.sentiment_scraper import SentimentScraper

    settings = replace(
        SETTINGS, sentiment=dc_replace(SETTINGS.sentiment, enabled=False)
    )
    session = FakeSession()
    scraper = SentimentScraper(settings, session=session)
    try:
        snapshot = scraper.get_sentiment("BTC/USDT", "BTC")
        assert session.attempts == 0
        assert snapshot.score == 0.0 and not snapshot.sources
    finally:
        scraper.close()


# --------------------------------------------------------------------------------------
# LLM adapter
# --------------------------------------------------------------------------------------


def test_prompt_contains_every_feature_and_constraint() -> None:
    prompt = build_decision_prompt(make_snapshot(), sentiment(0.3), SETTINGS)
    for token in ("rsi_14", "ema_50", "ema_200", "atr_pct", "sentiment_score",
                  "max_risk_per_trade_pct", "min_reward_to_risk", "BUY|SELL|HOLD"):
        assert token in prompt, f"prompt is missing {token}"


def test_parser_accepts_a_valid_response() -> None:
    snapshot = make_snapshot(close=100.0)
    raw = (
        'Sure, here is my call:\n'
        '{"action": "BUY", "confidence": 0.7, "stop_loss": 97.0, '
        '"take_profit": 104.6, "rationale": "trend intact"}'
    )
    decision = parse_decision_response(raw, snapshot, SETTINGS)
    assert decision.action is Action.BUY and decision.source == "llm"
    assert decision.risk_reward >= 1.5 - 1e-9


def test_parser_rejects_risk_violations_and_garbage() -> None:
    snapshot = make_snapshot(close=100.0)
    bad_inputs = [
        '{"action": "BUY", "stop_loss": 103.0, "take_profit": 110.0}',   # wrong side
        '{"action": "BUY", "stop_loss": 99.9, "take_profit": 101.0}',    # stop too tight
        '{"action": "BUY", "stop_loss": 50.0, "take_profit": 160.0}',    # stop too wide
        '{"action": "BUY", "stop_loss": 97.0, "take_profit": 101.0}',    # R:R too low
        '{"action": "BUY"}',                                             # no levels
        '{"action": "YOLO", "stop_loss": 97.0, "take_profit": 105.0}',   # bad action
        "I think we should probably wait and see.",                      # no token
        "",                                                              # empty
    ]
    for raw in bad_inputs:
        try:
            parse_decision_response(raw, snapshot, SETTINGS)
        except ValueError:
            continue
        raise AssertionError(f"parser accepted an invalid response: {raw!r}")


def test_parser_passes_hold_through() -> None:
    decision = parse_decision_response(
        '{"action": "HOLD", "confidence": 0.1, "rationale": "mixed"}',
        make_snapshot(),
        SETTINGS,
    )
    assert decision.action is Action.HOLD and not decision.is_actionable


# --------------------------------------------------------------------------------------
# Execution & accounting
# --------------------------------------------------------------------------------------


def test_fees_and_slippage_are_charged_on_both_sides() -> None:
    settings, database, executor = temp_setup()
    try:
        engine = DecisionEngine(settings)
        snapshot = make_snapshot(close=100.0, rsi=62, ema_fast=102, ema_slow=96)
        decision = engine.decide(snapshot, sentiment(0.6))
        position, reason = executor.open_position(
            decision, snapshot, equity=settings.risk.initial_corpus
        )
        assert position is not None, reason

        # Entry fills above the reference by exactly the slippage rate.
        expected_entry = 100.0 * (1 + settings.simulation.slippage_rate)
        assert abs(position.entry_price - expected_entry) < TOLERANCE
        assert abs(position.entry_fee - position.notional * settings.simulation.fee_rate) < TOLERANCE

        # A flat round trip must cost exactly two fees plus two slippage crossings.
        result = executor.close_position(position, 100.0, ExitReason.MANUAL)
        assert result.realized_pnl < 0, "a flat round trip cannot be profitable"
        assert abs(result.fees - (position.entry_fee + result.exit_price * position.size
                                  * settings.simulation.fee_rate)) < TOLERANCE
        expected_equity = settings.risk.initial_corpus + result.realized_pnl
        assert abs(executor.equity({}) - expected_equity) < 1e-4, (
            f"equity {executor.equity({})} != {expected_equity}"
        )
        assert not executor.positions
    finally:
        database.close()


def test_notional_sizing_respects_the_risk_cap() -> None:
    settings, database, executor = temp_setup()
    try:
        engine = DecisionEngine(settings)
        snapshot = make_snapshot(close=100.0, rsi=62, ema_fast=102, ema_slow=96)
        decision = engine.decide(snapshot, sentiment(0.6))
        position, reason = executor.open_position(
            decision, snapshot, equity=settings.risk.initial_corpus
        )
        assert position is not None, reason
        cap = settings.risk.initial_corpus * settings.risk.max_risk_per_trade
        assert position.notional <= cap + TOLERANCE, (
            f"notional {position.notional} exceeded the {cap} cap"
        )
        assert position.notional > cap * 0.9, "sizing should use most of the budget"
    finally:
        database.close()


def test_stop_loss_is_triggered_by_the_candle_low() -> None:
    settings, database, executor = temp_setup()
    try:
        engine = DecisionEngine(settings)
        entry = make_snapshot(close=100.0, rsi=62, ema_fast=102, ema_slow=96)
        decision = engine.decide(entry, sentiment(0.6))
        position, _ = executor.open_position(
            decision, entry, equity=settings.risk.initial_corpus
        )
        assert position is not None

        # A later bar that wicked straight through the stop.
        breached = make_snapshot(
            close=position.stop_loss * 0.999,
            low=position.stop_loss * 0.98,
            high=position.entry_price,
            rsi=40, ema_fast=100, ema_slow=100,
            candle_age_hours=-1.0,       # candle closed after the entry
        )
        results = executor.update_positions({"BTC/USDT": breached},
                                            {"BTC/USDT": breached.close})
        assert len(results) == 1 and results[0].exit_reason == "STOP_LOSS"
        assert results[0].realized_pnl < 0
        # Loss stays within the configured per-trade envelope.
        assert abs(results[0].realized_pnl) <= (
            settings.risk.initial_corpus * settings.risk.max_risk_per_trade
        )
    finally:
        database.close()


def test_take_profit_is_triggered_by_the_candle_high() -> None:
    settings, database, executor = temp_setup()
    try:
        engine = DecisionEngine(settings)
        entry = make_snapshot(close=100.0, rsi=62, ema_fast=102, ema_slow=96)
        decision = engine.decide(entry, sentiment(0.6))
        position, _ = executor.open_position(
            decision, entry, equity=settings.risk.initial_corpus
        )
        assert position is not None

        winner = make_snapshot(
            close=position.take_profit * 1.001,
            high=position.take_profit * 1.02,
            low=position.entry_price,
            rsi=68, ema_fast=105, ema_slow=96,
            candle_age_hours=-1.0,
        )
        results = executor.update_positions({"BTC/USDT": winner},
                                            {"BTC/USDT": winner.close})
        assert len(results) == 1 and results[0].exit_reason == "TAKE_PROFIT"
        assert results[0].realized_pnl > 0
    finally:
        database.close()


def test_a_bar_that_touches_both_levels_resolves_conservatively() -> None:
    settings, database, executor = temp_setup()
    try:
        engine = DecisionEngine(settings)
        entry = make_snapshot(close=100.0, rsi=62, ema_fast=102, ema_slow=96)
        decision = engine.decide(entry, sentiment(0.6))
        position, _ = executor.open_position(
            decision, entry, equity=settings.risk.initial_corpus
        )
        assert position is not None

        both = make_snapshot(
            close=100.0,
            high=position.take_profit * 1.01,
            low=position.stop_loss * 0.99,
            rsi=55, ema_fast=100, ema_slow=100,
            candle_age_hours=-1.0,
        )
        results = executor.update_positions({"BTC/USDT": both}, {"BTC/USDT": 100.0})
        assert len(results) == 1
        assert results[0].exit_reason == "STOP_LOSS", "conservative_fills must favour the stop"
    finally:
        database.close()


def test_a_stale_candle_cannot_close_a_fresh_position() -> None:
    settings, database, executor = temp_setup()
    try:
        engine = DecisionEngine(settings)
        entry = make_snapshot(close=100.0, rsi=62, ema_fast=102, ema_slow=96)
        decision = engine.decide(entry, sentiment(0.6))
        position, _ = executor.open_position(
            decision, entry, equity=settings.risk.initial_corpus
        )
        assert position is not None

        # A bar that closed BEFORE the entry, with a range covering the stop.
        stale = make_snapshot(
            close=100.0,
            high=position.take_profit * 1.05,
            low=position.stop_loss * 0.95,
            candle_age_hours=48.0,
        )
        results = executor.update_positions({"BTC/USDT": stale}, {"BTC/USDT": 100.0})
        assert not results, "a pre-entry candle must not trigger protective exits"
        assert "BTC/USDT" in executor.positions
    finally:
        database.close()


def test_short_position_pnl_is_inverted() -> None:
    settings, database, executor = temp_setup()
    try:
        engine = DecisionEngine(settings)
        snapshot = make_snapshot(close=100.0, rsi=38, ema_fast=96, ema_slow=104)
        decision = engine.decide(snapshot, sentiment(-0.6))
        assert decision.action is Action.SELL
        position, reason = executor.open_position(
            decision, snapshot, equity=settings.risk.initial_corpus
        )
        assert position is not None, reason
        assert position.side == "short"
        assert position.entry_price < 100.0, "a short sells below the reference"

        result = executor.close_position(position, 90.0, ExitReason.TAKE_PROFIT)
        assert result.realized_pnl > 0, "a short must profit from a falling price"
        assert executor.cash > settings.risk.initial_corpus
    finally:
        database.close()


def test_risk_gates_reject_bad_orders() -> None:
    settings, database, executor = temp_setup(max_open_positions=1)
    try:
        engine = DecisionEngine(settings)
        snapshot = make_snapshot(close=100.0, rsi=62, ema_fast=102, ema_slow=96)
        decision = engine.decide(snapshot, sentiment(0.6))

        # Equity circuit breaker, asserted on an empty book so the capacity gates
        # downstream cannot mask it.
        floor = settings.risk.initial_corpus * settings.risk.equity_stop
        reason = executor.pre_trade_check(decision, floor - 1)
        assert reason and "circuit breaker" in reason, reason

        position, _ = executor.open_position(
            decision, snapshot, equity=settings.risk.initial_corpus
        )
        assert position is not None

        # Same asset again -> no pyramiding.
        _, reason = executor.open_position(
            decision, snapshot, equity=settings.risk.initial_corpus
        )
        assert reason and "already open" in reason, reason

        # A different asset -> max_open_positions.
        other = make_snapshot(symbol="ETH/USDT", close=100.0, rsi=62,
                              ema_fast=102, ema_slow=96)
        other_decision = engine.decide(other, sentiment(0.6))
        _, reason = executor.open_position(
            other_decision, other, equity=settings.risk.initial_corpus
        )
        assert reason and "max open positions" in reason, reason

        # Missing protective levels -> hard reject.
        from dataclasses import replace as dc_replace

        naked = dc_replace(other_decision, stop_loss=None, take_profit=None)
        reason = executor.pre_trade_check(naked, settings.risk.initial_corpus)
        assert reason and "mandatory" in reason, reason
    finally:
        database.close()


def test_exposure_cap_limits_the_book() -> None:
    settings, database, executor = temp_setup(
        max_open_positions=10, max_risk_per_trade=0.30, max_portfolio_exposure=0.50
    )
    try:
        engine = DecisionEngine(settings)
        for symbol in ("BTC/USDT", "ETH/USDT", "SOL/USDT"):
            snapshot = make_snapshot(symbol=symbol, close=100.0, rsi=62,
                                     ema_fast=102, ema_slow=96)
            decision = engine.decide(snapshot, sentiment(0.6))
            executor.open_position(decision, snapshot,
                                   equity=settings.risk.initial_corpus)
        exposure = sum(p.notional for p in executor.positions.values())
        cap = settings.risk.initial_corpus * settings.risk.max_portfolio_exposure
        assert exposure <= cap + 1.0, f"exposure {exposure} exceeded cap {cap}"
    finally:
        database.close()


def test_liquidation_flattens_everything() -> None:
    settings, database, executor = temp_setup()
    try:
        engine = DecisionEngine(settings)
        for symbol in ("BTC/USDT", "ETH/USDT"):
            snapshot = make_snapshot(symbol=symbol, close=100.0, rsi=62,
                                     ema_fast=102, ema_slow=96)
            decision = engine.decide(snapshot, sentiment(0.6))
            executor.open_position(decision, snapshot,
                                   equity=settings.risk.initial_corpus)
        assert len(executor.positions) == 2

        results = executor.liquidate_all({"BTC/USDT": 110.0, "ETH/USDT": 90.0})
        assert len(results) == 2 and not executor.positions
        assert all(r.exit_reason == "END_OF_EXPERIMENT" for r in results)
        assert not database.open_trades()
        assert len(database.closed_trades()) == 2
        # Cash is the whole account again once the book is flat.
        assert abs(executor.equity({}) - executor.cash) < TOLERANCE
    finally:
        database.close()


def test_state_survives_a_restart() -> None:
    settings, database, executor = temp_setup()
    try:
        engine = DecisionEngine(settings)
        snapshot = make_snapshot(close=100.0, rsi=62, ema_fast=102, ema_slow=96)
        decision = engine.decide(snapshot, sentiment(0.6))
        position, _ = executor.open_position(
            decision, snapshot, equity=settings.risk.initial_corpus
        )
        assert position is not None
        cash_before = executor.cash

        # Simulate a process restart against the same database file.
        revived = TestnetExecutor(settings, database, price_fetcher=None)
        revived.bootstrap()
        assert abs(revived.cash - cash_before) < TOLERANCE
        assert "BTC/USDT" in revived.positions
        restored = revived.positions["BTC/USDT"]
        assert restored.trade_uid == position.trade_uid
        assert abs(restored.entry_price - position.entry_price) < TOLERANCE
        assert abs(restored.stop_loss - position.stop_loss) < TOLERANCE
        assert abs(restored.size - position.size) < TOLERANCE
    finally:
        database.close()


# --------------------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------------------


def test_concurrent_writers_do_not_corrupt_the_database() -> None:
    import threading

    tmp = Path(tempfile.mkdtemp(prefix="sim-test-")) / "concurrent.sqlite3"
    database = Database(tmp).connect()
    errors: list[BaseException] = []
    writes_per_thread = 25

    def writer(worker: int) -> None:
        try:
            for index in range(writes_per_thread):
                database.log_decision(
                    {
                        "cycle_uid": f"w{worker}",
                        "timestamp": "2026-01-01T00:00:00+00:00",
                        "asset": "BTC/USDT",
                        "action": "HOLD",
                        "confidence": 0.1,
                        "score": 0.0,
                        "price": 100.0 + index,
                        "executed": 0,
                    }
                )
                database.record_equity(
                    equity=10_000.0, cash=10_000.0, unrealized_pnl=0.0,
                    realized_pnl_cum=0.0, fees_cum=0.0, open_positions=0,
                    exposure_pct=0.0, cycle_uid=f"w{worker}",
                )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    try:
        assert not errors, f"concurrent writes raised: {errors[:3]}"
        assert len(database.decisions(limit=1000)) == 8 * writes_per_thread
        assert len(database.equity_curve()) == 8 * writes_per_thread
        assert database._execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        database.close()


def test_closing_a_trade_twice_is_a_no_op() -> None:
    settings, database, executor = temp_setup()
    try:
        engine = DecisionEngine(settings)
        snapshot = make_snapshot(close=100.0, rsi=62, ema_fast=102, ema_slow=96)
        decision = engine.decide(snapshot, sentiment(0.6))
        position, _ = executor.open_position(
            decision, snapshot, equity=settings.risk.initial_corpus
        )
        assert position is not None
        executor.close_position(position, 105.0, ExitReason.TAKE_PROFIT)
        assert database.close_trade(
            position.trade_uid, exit_price=105.0, fees=0.0, slippage_cost=0.0,
            realized_pnl=999.0, gross_pnl=999.0, return_pct=9.99,
            exit_reason="MANUAL",
        ) is False, "a CLOSED trade must not be re-closed"
    finally:
        database.close()


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def test_report_metrics_are_consistent() -> None:
    settings, database, executor = temp_setup()
    try:
        engine = DecisionEngine(settings)
        # One winner, one loser.
        for symbol, exit_price in (("BTC/USDT", 112.0), ("ETH/USDT", 94.0)):
            snapshot = make_snapshot(symbol=symbol, close=100.0, rsi=62,
                                     ema_fast=102, ema_slow=96)
            decision = engine.decide(snapshot, sentiment(0.6))
            position, _ = executor.open_position(
                decision, snapshot, equity=executor.equity({})
            )
            assert position is not None
            executor.close_position(position, exit_price, ExitReason.MANUAL)

        stats = database.trade_stats()
        assert stats["trades"] == 2 and stats["wins"] == 1 and stats["losses"] == 1
        assert stats["fees"] > 0
        realized, fees = database.realized_totals()
        assert abs(realized - stats["net_pnl"]) < TOLERANCE
        assert abs(fees - stats["fees"]) < TOLERANCE
        # Net = gross - fees, to the cent.
        assert abs(stats["net_pnl"] - (stats["gross_pnl"] - stats["fees"])) < 1e-4
    finally:
        database.close()


# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------


def _collect() -> list:
    return [value for name, value in sorted(globals().items())
            if name.startswith("test_") and callable(value)]


def main() -> int:
    tests = _collect()
    failures: list[tuple[str, BaseException]] = []
    for test in tests:
        name = test.__name__
        try:
            test()
        except BaseException as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"pass  {name}")

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        print("\nfailures:")
        for name, exc in failures:
            print(f"  - {name}: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    import logging

    logging.disable(logging.CRITICAL)   # keep the test output readable
    raise SystemExit(main())
