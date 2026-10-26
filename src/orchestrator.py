"""
Orchestration: the 4-hour trading loop, the hard 30-day boundary, and the final report.

Lifecycle
---------
1. ``bootstrap``   - open the database, stamp the experiment window (persisted, so a
                     restart does **not** extend the deadline), restore cash + positions.
2. ``run_cycle``   - every ``CYCLE_HOURS``:
                       a. refresh market data for the universe
                       b. sweep open positions for stop-loss / take-profit hits
                       c. refresh sentiment
                       d. per asset: exit check if in a position, otherwise decide and
                          size a new entry
                       e. snapshot equity
3. ``deadline_watch`` - every 15 minutes, independent of the trading cadence, so day 30
                     is honoured within minutes rather than at the next 4-hour tick.
4. ``finalize``    - liquidate everything at market, mark the experiment COMPLETED and
                     write the P&L report to ``reports/``.

A cycle is fully defensive: a single asset's data failure, a sentiment outage or an
exchange timeout degrades that asset only. Unhandled errors are recorded against the
cycle row and the loop continues - a 30-day experiment cannot be allowed to die on one
bad HTTP response.
"""

from __future__ import annotations

import json
import signal
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from config.settings import SETTINGS, Settings
from src.common import (
    configure_logging,
    from_iso,
    get_logger,
    humanize_timedelta,
    mean,
    new_uid,
    percent,
    safe_float,
    stdev,
    to_iso,
    utc_now,
)
from src.data_ingestion.price_fetcher import MarketSnapshot, PriceFetcher
from src.data_ingestion.sentiment_scraper import SentimentScraper, SentimentSnapshot
from src.engine.decision_logic import Action, DecisionEngine, ExitReason, TradeDecision
from src.engine.testnet_executor import TestnetExecutor, TradeResult
from src.storage.db_manager import (
    KEY_CYCLE_COUNT,
    KEY_END_AT,
    KEY_LAST_CYCLE,
    KEY_START_AT,
    KEY_STATUS,
    Database,
    open_database,
)

LOGGER = get_logger(__name__)

STATUS_RUNNING = "RUNNING"
STATUS_COMPLETED = "COMPLETED"

# Below this many equity observations a Sharpe ratio is not reported at all.
MIN_SHARPE_OBSERVATIONS = 8


def _fmt(value: Any) -> str:
    """Render a metric that may legitimately be undefined."""
    if value is None:
        return "n/a (insufficient data)"
    return f"{value}"


@dataclass(slots=True)
class CycleReport:
    """What one trading cycle did."""

    cycle_uid: str
    day_index: int
    started_at: datetime
    assets_evaluated: int = 0
    data_errors: int = 0
    opened: int = 0
    closed: int = 0
    equity: float = 0.0
    actions: dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.actions is None:
            self.actions = {"BUY": 0, "SELL": 0, "HOLD": 0}


class Orchestrator:
    """Owns every component and the experiment clock."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or SETTINGS
        configure_logging(self.settings.log_level, self.settings.log_file)

        self.db: Database = open_database(self.settings)
        self.fetcher = PriceFetcher(self.settings)
        self.scraper = SentimentScraper(self.settings)
        self.engine = DecisionEngine(self.settings)
        self.executor = TestnetExecutor(self.settings, self.db, self.fetcher)

        self.scheduler: BlockingScheduler | None = None
        self._lock = threading.RLock()       # one cycle at a time, ever
        self._finalized = False
        self._shutting_down = False

    # ---------------------------------------------------------------------------------
    # Experiment clock
    # ---------------------------------------------------------------------------------

    def bootstrap(self) -> None:
        """Idempotently establish (or resume) the experiment window."""
        start_raw = self.db.get_state(KEY_START_AT)
        if start_raw is None:
            start = utc_now()
            end = start + timedelta(days=self.settings.schedule.experiment_days)
            self.db.set_state(KEY_START_AT, to_iso(start))
            self.db.set_state(KEY_END_AT, to_iso(end))
            self.db.set_state(KEY_STATUS, STATUS_RUNNING)
            self.db.set_state(KEY_CYCLE_COUNT, "0")
            LOGGER.info(
                "experiment STARTED %s -> %s (%d days, %.1fh cadence, corpus %.2f %s)",
                to_iso(start), to_iso(end), self.settings.schedule.experiment_days,
                self.settings.schedule.cycle_hours, self.settings.risk.initial_corpus,
                self.settings.exchange.quote_currency,
            )
        else:
            LOGGER.info(
                "resuming experiment started %s (ends %s, status %s)",
                start_raw, self.db.get_state(KEY_END_AT), self.db.get_state(KEY_STATUS),
            )

        self.executor.bootstrap()
        try:
            self.fetcher.load_markets()
        except Exception as exc:  # noqa: BLE001 - retried later, per cycle
            LOGGER.error("could not preload markets at startup: %s", exc)

    @property
    def start_at(self) -> datetime:
        return from_iso(self.db.get_state(KEY_START_AT)) or utc_now()

    @property
    def end_at(self) -> datetime:
        stored = from_iso(self.db.get_state(KEY_END_AT))
        if stored:
            return stored
        return self.start_at + timedelta(days=self.settings.schedule.experiment_days)

    @property
    def status(self) -> str:
        return str(self.db.get_state(KEY_STATUS, STATUS_RUNNING))

    @property
    def day_index(self) -> int:
        """1-based experiment day."""
        elapsed = utc_now() - self.start_at
        return min(
            int(elapsed.total_seconds() // 86_400) + 1,
            self.settings.schedule.experiment_days,
        )

    def deadline_reached(self) -> bool:
        return utc_now() >= self.end_at

    def time_remaining(self) -> timedelta:
        return max(self.end_at - utc_now(), timedelta(0))

    # ---------------------------------------------------------------------------------
    # Trading cycle
    # ---------------------------------------------------------------------------------

    def run_cycle(self) -> CycleReport | None:
        """Execute one full decision cycle. Never raises."""
        with self._lock:
            if self.status == STATUS_COMPLETED:
                LOGGER.info("experiment already completed; skipping cycle")
                return None
            if self.deadline_reached():
                LOGGER.warning("30-day deadline reached at cycle start - finalising")
                self.finalize()
                return None

            report = CycleReport(
                cycle_uid=new_uid("cycle"),
                day_index=self.day_index,
                started_at=utc_now(),
            )
            self.db.start_cycle(report.cycle_uid, report.day_index)
            LOGGER.info(
                "=== cycle %s | day %d/%d | %s remaining ===",
                report.cycle_uid, report.day_index,
                self.settings.schedule.experiment_days,
                humanize_timedelta(self.time_remaining()),
            )

            error: str | None = None
            try:
                self._execute_cycle(report)
            except Exception as exc:  # noqa: BLE001 - the loop must survive
                error = f"{type(exc).__name__}: {exc}"
                LOGGER.exception("cycle %s failed", report.cycle_uid)

            self.db.finish_cycle(
                report.cycle_uid,
                assets_evaluated=report.assets_evaluated,
                orders_opened=report.opened,
                orders_closed=report.closed,
                status="ERROR" if error else "OK",
                error=error,
            )
            self.db.set_state(KEY_LAST_CYCLE, to_iso(utc_now()))
            self.db.set_state(
                KEY_CYCLE_COUNT, str(self.db.get_state_int(KEY_CYCLE_COUNT, 0) + 1)
            )
            LOGGER.info(
                "=== cycle %s done | opened %d, closed %d, equity %.2f %s "
                "(%s vs corpus) ===",
                report.cycle_uid, report.opened, report.closed, report.equity,
                self.settings.exchange.quote_currency,
                percent(report.equity / self.settings.risk.initial_corpus - 1.0),
            )
            return report

    def _execute_cycle(self, report: CycleReport) -> None:
        assets = self.settings.enabled_assets

        # 1. Market data -----------------------------------------------------------
        snapshots, data_errors = self.fetcher.snapshots(assets)
        report.data_errors = len(data_errors)
        prices: dict[str, float] = {s.symbol: s.price for s in snapshots.values()}

        # Open positions whose asset failed to refresh still need a mark.
        for symbol in self.executor.positions:
            if symbol not in prices:
                position = self.executor.positions[symbol]
                live = self.fetcher.fetch_last_price(position.exchange_symbol)
                if live > 0:
                    prices[symbol] = live

        if not snapshots:
            LOGGER.error(
                "no market data for any asset this cycle (%d error(s)); nothing to do",
                len(data_errors),
            )
            report.equity = self.executor.equity(prices)
            self._record_equity(report, prices)
            return

        # 2. Protective levels on existing positions -------------------------------
        for result in self.executor.update_positions(snapshots, prices):
            report.closed += 1
            LOGGER.info("protective exit: %s", result.summary())

        # 3. Sentiment -------------------------------------------------------------
        sentiments: dict[str, SentimentSnapshot] = self.scraper.bulk_sentiment(assets)

        # 4. Per-asset decisions ---------------------------------------------------
        # Exits run first so freed slots are available; entries then execute
        # strongest-first, so when positions/exposure are capped the slots go to the
        # highest-confidence signals rather than to whichever asset is listed first.
        entries: list[tuple[str, MarketSnapshot, SentimentSnapshot | None, TradeDecision]] = []
        for asset in assets:
            snapshot = snapshots.get(asset.symbol)
            if snapshot is None:
                continue
            report.assets_evaluated += 1
            sentiment = sentiments.get(asset.symbol)
            try:
                if asset.symbol in self.executor.positions:
                    self._handle_exit(asset.symbol, snapshot, sentiment, report)
                else:
                    decision = self.engine.decide(snapshot, sentiment, open_side=None)
                    entries.append((asset.symbol, snapshot, sentiment, decision))
            except Exception:  # noqa: BLE001 - isolate per-asset failures
                LOGGER.exception("decision/execution failed for %s", asset.symbol)

        entries.sort(
            key=lambda e: (e[3].is_actionable, e[3].confidence, abs(e[3].score)),
            reverse=True,
        )
        for symbol, snapshot, sentiment, decision in entries:
            try:
                self._handle_entry(symbol, snapshot, sentiment, decision, report)
            except Exception:  # noqa: BLE001 - isolate per-asset failures
                LOGGER.exception("decision/execution failed for %s", symbol)

        # 5. Equity mark -----------------------------------------------------------
        prices.update({s.symbol: s.price for s in snapshots.values()})
        report.equity = self.executor.equity(prices)
        self._record_equity(report, prices)

    def _handle_exit(
        self,
        symbol: str,
        snapshot: MarketSnapshot,
        sentiment: SentimentSnapshot | None,
        report: CycleReport,
    ) -> None:
        """In a position: look for a signal-based exit."""
        position = self.executor.positions[symbol]
        evaluation = self.engine.evaluate_exit(position.side, snapshot, sentiment)
        if evaluation.should_close:
            result = self.executor.close_by_signal(
                symbol, snapshot.price, evaluation.detail
            )
            if result is not None:
                report.closed += 1
        else:
            LOGGER.info(
                "%s: holding open %s position (mark %.6f, unrealised %+.2f%%)",
                symbol, position.side, snapshot.price,
                position.unrealized_pct(snapshot.price) * 100,
            )

    def _handle_entry(
        self,
        symbol: str,
        snapshot: MarketSnapshot,
        sentiment: SentimentSnapshot | None,
        decision: TradeDecision,
        report: CycleReport,
    ) -> None:
        """Flat: execute (or log the skip of) a precomputed entry decision."""
        sentiment_score = None if sentiment is None else sentiment.score
        report.actions[decision.action.value] = (
            report.actions.get(decision.action.value, 0) + 1
        )

        executed = False
        skip_reason: str | None = None
        if decision.is_actionable:
            equity = self.executor.equity({symbol: snapshot.price})
            opened, skip_reason = self.executor.open_position(
                decision,
                snapshot,
                equity=equity,
                cycle_uid=report.cycle_uid,
                sentiment_score=sentiment_score,
            )
            executed = opened is not None
            if executed:
                report.opened += 1
        else:
            skip_reason = decision.rationale

        self.db.log_decision(
            {
                "cycle_uid": report.cycle_uid,
                "timestamp": to_iso(utc_now()),
                "asset": symbol,
                "action": decision.action.value,
                "confidence": decision.confidence,
                "score": decision.score,
                "price": decision.reference_price,
                "rsi": snapshot.rsi,
                "ema_fast": snapshot.ema_fast,
                "ema_slow": snapshot.ema_slow,
                "atr": snapshot.atr,
                "sentiment_score": sentiment_score,
                "take_profit": decision.take_profit,
                "stop_loss": decision.stop_loss,
                "executed": 1 if executed else 0,
                "skip_reason": None if executed else skip_reason,
                "rationale": decision.rationale,
            }
        )

    def _record_equity(self, report: CycleReport, prices: Mapping[str, float]) -> None:
        state = self.executor.portfolio(prices)
        realized, fees = self.db.realized_totals()
        self.db.record_equity(
            equity=state.equity,
            cash=state.cash,
            unrealized_pnl=state.unrealized_pnl,
            realized_pnl_cum=realized,
            fees_cum=fees,
            open_positions=state.open_positions,
            exposure_pct=state.exposure_pct,
            cycle_uid=report.cycle_uid,
        )
        report.equity = state.equity

    # ---------------------------------------------------------------------------------
    # Deadline + finalisation
    # ---------------------------------------------------------------------------------

    def deadline_watch(self) -> None:
        """Lightweight job: liquidate as soon as the 30-day window closes."""
        if self.status == STATUS_COMPLETED or self._finalized:
            self._stop_scheduler()
            return
        if self.deadline_reached():
            LOGGER.warning("deadline watch tripped - finalising the experiment")
            self.finalize()
        else:
            LOGGER.debug("deadline watch: %s remaining",
                         humanize_timedelta(self.time_remaining()))

    def finalize(self, reason: str | ExitReason = ExitReason.END_OF_EXPERIMENT) -> dict[str, Any]:
        """Liquidate, mark the experiment complete, write the report, stop the loop."""
        with self._lock:
            if self._finalized:
                return self.build_report()

            LOGGER.warning("FINALISING: liquidating all open positions at market")
            prices = self.fetcher.mark_prices(list(self.executor.positions.keys()))
            results = self.executor.liquidate_all(prices, reason)
            for result in results:
                LOGGER.info("liquidated: %s", result.summary())

            state = self.executor.portfolio({})
            realized, fees = self.db.realized_totals()
            self.db.record_equity(
                equity=state.equity,
                cash=state.cash,
                unrealized_pnl=0.0,
                realized_pnl_cum=realized,
                fees_cum=fees,
                open_positions=0,
                exposure_pct=0.0,
                cycle_uid="final",
            )
            self.db.set_state(KEY_STATUS, STATUS_COMPLETED)
            self._finalized = True

            report = self.build_report()
            self.write_report(report)
            LOGGER.warning("experiment COMPLETE - final equity %.2f %s (%s)",
                           report["summary"]["final_equity"],
                           self.settings.exchange.quote_currency,
                           report["summary"]["total_return_pct"])
            self._stop_scheduler()
            return report

    # ---------------------------------------------------------------------------------
    # Reporting
    # ---------------------------------------------------------------------------------

    def build_report(self) -> dict[str, Any]:
        """Aggregate the full P&L report from the database."""
        stats = self.db.trade_stats()
        curve = self.db.equity_curve()
        initial = self.settings.risk.initial_corpus

        equity_values = [safe_float(row["equity"]) for row in curve] or [initial]
        final_equity = equity_values[-1]
        peak = initial
        max_drawdown = 0.0
        for value in [initial, *equity_values]:
            peak = max(peak, value)
            if peak > 0:
                max_drawdown = max(max_drawdown, (peak - value) / peak)

        returns: list[float] = []
        previous = initial
        for value in equity_values:
            if previous > 0:
                returns.append(value / previous - 1.0)
            previous = value
        # A Sharpe ratio from a handful of observations is noise dressed as a metric,
        # so it is reported as null until the sample is large enough to mean anything.
        volatility = stdev(returns)
        periods_per_year = (365 * 24) / max(self.settings.schedule.cycle_hours, 1e-9)
        sharpe: float | None = None
        if len(returns) >= MIN_SHARPE_OBSERVATIONS and volatility > 0:
            sharpe = mean(returns) / volatility * (periods_per_year ** 0.5)

        trades = int(stats.get("trades", 0) or 0)
        wins = int(stats.get("wins", 0) or 0)
        losses = int(stats.get("losses", 0) or 0)
        gross_profit = safe_float(stats.get("gross_profit"))
        gross_loss = safe_float(stats.get("gross_loss"))
        net_pnl = safe_float(stats.get("net_pnl"))
        win_rate = wins / trades if trades else 0.0
        avg_win = gross_profit / wins if wins else 0.0
        avg_loss = gross_loss / losses if losses else 0.0

        started = self.start_at
        ends = self.end_at
        elapsed_days = max((utc_now() - started).total_seconds() / 86_400, 0.0)

        return {
            "generated_at": to_iso(utc_now()),
            "experiment": {
                "status": self.status,
                "started_at": to_iso(started),
                "ends_at": to_iso(ends),
                "planned_days": self.settings.schedule.experiment_days,
                "elapsed_days": round(elapsed_days, 2),
                "cycles_completed": self.db.get_state_int(KEY_CYCLE_COUNT, 0),
                "cycle_hours": self.settings.schedule.cycle_hours,
                "exchange": self.settings.exchange.exchange_id,
                "sandbox": self.settings.exchange.use_sandbox,
                "market_type": self.settings.exchange.market_type,
                "universe": [a.symbol for a in self.settings.enabled_assets],
            },
            "risk_config": {
                "initial_corpus": initial,
                "max_risk_per_trade_pct": self.settings.risk.max_risk_per_trade * 100,
                "position_sizing_mode": self.settings.risk.position_sizing_mode,
                "max_portfolio_exposure_pct": self.settings.risk.max_portfolio_exposure * 100,
                "max_open_positions": self.settings.risk.max_open_positions,
                "fee_rate_pct": self.settings.simulation.fee_rate * 100,
                "slippage_rate_pct": self.settings.simulation.slippage_rate * 100,
                "allow_shorts": self.settings.risk.allow_shorts,
            },
            "summary": {
                "initial_corpus": round(initial, 2),
                "final_equity": round(final_equity, 2),
                "net_pnl": round(final_equity - initial, 2),
                "total_return_pct": percent(final_equity / initial - 1.0 if initial else 0.0),
                "realized_pnl": round(net_pnl, 2),
                "gross_pnl": round(safe_float(stats.get("gross_pnl")), 2),
                "total_fees": round(safe_float(stats.get("fees")), 2),
                "total_slippage": round(safe_float(stats.get("slippage")), 2),
                "open_positions_remaining": len(self.executor.positions),
                "cash": round(self.executor.cash, 2),
            },
            "performance": {
                "closed_trades": trades,
                "wins": wins,
                "losses": losses,
                "win_rate": percent(win_rate),
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "expectancy": round(win_rate * avg_win - (1 - win_rate) * avg_loss, 2),
                # null rather than inf when there are no losses: inf is not valid
                # strict JSON and would break any downstream consumer.
                "profit_factor": (
                    round(gross_profit / gross_loss, 3) if gross_loss > 0 else None
                ),
                "best_trade": round(safe_float(stats.get("best")), 2),
                "worst_trade": round(safe_float(stats.get("worst")), 2),
                "max_drawdown": percent(max_drawdown),
                "sharpe_annualised": None if sharpe is None else round(sharpe, 3),
                "equity_observations": len(equity_values),
            },
            "per_asset": [
                {
                    "asset": row["asset"],
                    "trades": row["trades"],
                    "wins": row["wins"],
                    "net_pnl": round(safe_float(row["net_pnl"]), 2),
                    "fees": round(safe_float(row["fees"]), 2),
                }
                for row in self.db.per_asset_stats()
            ],
            "exit_reasons": [
                {
                    "reason": row["exit_reason"],
                    "count": row["n"],
                    "net_pnl": round(safe_float(row["net_pnl"]), 2),
                }
                for row in self.db.exit_reason_breakdown()
            ],
            "decision_mix": {
                row["action"]: row["n"] for row in self.db.action_breakdown()
            },
            "open_positions": self.executor.open_positions_table({}),
        }

    def render_report(self, report: Mapping[str, Any]) -> str:
        """Human-readable Markdown version of the report."""
        experiment = report["experiment"]
        summary = report["summary"]
        performance = report["performance"]
        quote = self.settings.exchange.quote_currency

        lines = [
            "# 30-Day Crypto Paper-Trading Report",
            "",
            f"*Generated {report['generated_at']}*",
            "",
            "## Experiment",
            "",
            f"- Status: **{experiment['status']}**",
            f"- Window: {experiment['started_at']} -> {experiment['ends_at']} "
            f"({experiment['planned_days']} days planned, "
            f"{experiment['elapsed_days']} elapsed)",
            f"- Cycles completed: {experiment['cycles_completed']} "
            f"(every {experiment['cycle_hours']}h)",
            f"- Venue: {experiment['exchange']} "
            f"{'TESTNET' if experiment['sandbox'] else 'LIVE'} "
            f"({experiment['market_type']})",
            f"- Universe: {', '.join(experiment['universe'])}",
            "",
            "## Headline P&L",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Initial corpus | {summary['initial_corpus']:.2f} {quote} |",
            f"| Final equity | {summary['final_equity']:.2f} {quote} |",
            f"| Net P&L | {summary['net_pnl']:+.2f} {quote} |",
            f"| Total return | {summary['total_return_pct']} |",
            f"| Gross P&L (pre-cost) | {summary['gross_pnl']:+.2f} {quote} |",
            f"| Fees paid | {summary['total_fees']:.2f} {quote} |",
            f"| Slippage cost | {summary['total_slippage']:.2f} {quote} |",
            f"| Cash | {summary['cash']:.2f} {quote} |",
            f"| Open positions remaining | {summary['open_positions_remaining']} |",
            "",
            "## Performance",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Closed trades | {performance['closed_trades']} |",
            f"| Wins / losses | {performance['wins']} / {performance['losses']} |",
            f"| Win rate | {performance['win_rate']} |",
            f"| Average win | {performance['avg_win']:+.2f} {quote} |",
            f"| Average loss | {performance['avg_loss']:.2f} {quote} |",
            f"| Expectancy / trade | {performance['expectancy']:+.2f} {quote} |",
            f"| Profit factor | {_fmt(performance['profit_factor'])} |",
            f"| Best trade | {performance['best_trade']:+.2f} {quote} |",
            f"| Worst trade | {performance['worst_trade']:+.2f} {quote} |",
            f"| Max drawdown | {performance['max_drawdown']} |",
            f"| Sharpe (annualised) | {_fmt(performance['sharpe_annualised'])} |",
            "",
        ]

        if report["per_asset"]:
            lines += ["## Per asset", "", "| Asset | Trades | Wins | Net P&L | Fees |",
                      "| --- | --- | --- | --- | --- |"]
            lines += [
                f"| {row['asset']} | {row['trades']} | {row['wins']} | "
                f"{row['net_pnl']:+.2f} | {row['fees']:.2f} |"
                for row in report["per_asset"]
            ]
            lines.append("")

        if report["exit_reasons"]:
            lines += ["## Exit reasons", "", "| Reason | Count | Net P&L |",
                      "| --- | --- | --- |"]
            lines += [
                f"| {row['reason']} | {row['count']} | {row['net_pnl']:+.2f} |"
                for row in report["exit_reasons"]
            ]
            lines.append("")

        if report["decision_mix"]:
            mix = ", ".join(f"{k}: {v}" for k, v in sorted(report["decision_mix"].items()))
            lines += ["## Decision mix", "", mix, ""]

        lines += [
            "## Risk configuration",
            "",
            "```json",
            json.dumps(report["risk_config"], indent=2),
            "```",
            "",
            "---",
            "",
            "*Simulated results from testnet/paper trading. Fees and slippage are "
            "modelled, not actual. Not investment advice.*",
        ]
        return "\n".join(lines)

    def write_report(self, report: Mapping[str, Any]) -> tuple[Path, Path]:
        """Persist the report as JSON + Markdown under ``reports/``."""
        directory = self.settings.storage.report_dir
        directory.mkdir(parents=True, exist_ok=True)
        stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")

        json_path = directory / f"final_report_{stamp}.json"
        json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

        md_path = directory / f"final_report_{stamp}.md"
        md_path.write_text(self.render_report(report), encoding="utf-8")

        latest = directory / "latest_report.md"
        latest.write_text(self.render_report(report), encoding="utf-8")

        LOGGER.info("report written: %s and %s", json_path.name, md_path.name)
        return json_path, md_path

    def status_lines(self) -> list[str]:
        """Compact live status for the CLI."""
        prices = self.fetcher.mark_prices(list(self.executor.positions.keys()))
        state = self.executor.portfolio(prices)
        realized, fees = self.db.realized_totals()
        initial = self.settings.risk.initial_corpus
        lines = [
            f"status            : {self.status}",
            f"started           : {to_iso(self.start_at)}",
            f"ends              : {to_iso(self.end_at)} "
            f"({humanize_timedelta(self.time_remaining())} remaining)",
            f"day               : {self.day_index}/{self.settings.schedule.experiment_days}",
            f"cycles completed  : {self.db.get_state_int(KEY_CYCLE_COUNT, 0)}",
            f"equity            : {state.equity:.2f} "
            f"({percent(state.equity / initial - 1.0)} vs corpus)",
            f"cash              : {state.cash:.2f}",
            f"unrealised P&L    : {state.unrealized_pnl:+.2f}",
            f"realised P&L      : {realized:+.2f}",
            f"fees paid         : {fees:.2f}",
            f"exposure          : {percent(state.exposure_pct)} "
            f"(cap {percent(self.settings.risk.max_portfolio_exposure)})",
            f"open positions    : {state.open_positions}/"
            f"{self.settings.risk.max_open_positions}",
        ]
        for row in self.executor.open_positions_table(prices):
            lines.append(
                f"  - {row['symbol']} {row['side']} {row['size']:.8f} @ "
                f"{row['entry']:.6f} | mark {row['mark']:.6f} | "
                f"SL {row['stop_loss']:.6f} TP {row['take_profit']:.6f} | "
                f"{row['unrealized_pct']:+.2%}"
            )
        return lines

    # ---------------------------------------------------------------------------------
    # Scheduler
    # ---------------------------------------------------------------------------------

    def run(self) -> None:
        """Start the blocking scheduler and run until day 30 (or Ctrl-C)."""
        self.bootstrap()

        if self.status == STATUS_COMPLETED:
            LOGGER.warning(
                "this experiment is already COMPLETED - reset the database "
                "(python main.py reset) to start a new 30-day run"
            )
            print("\n".join(self.status_lines()))
            return

        if self.deadline_reached():
            LOGGER.warning("deadline already passed on startup - finalising immediately")
            self.finalize()
            return

        schedule = self.settings.schedule
        self.scheduler = BlockingScheduler(timezone=schedule.timezone)
        self.scheduler.add_job(
            self.run_cycle,
            trigger=IntervalTrigger(hours=schedule.cycle_hours),
            id="trading_cycle",
            name=f"{schedule.cycle_hours}h trading cycle",
            max_instances=1,            # never overlap cycles
            coalesce=True,              # collapse missed runs into one
            misfire_grace_time=schedule.misfire_grace_seconds,
            next_run_time=utc_now() if schedule.run_immediately else None,
        )
        self.scheduler.add_job(
            self.deadline_watch,
            trigger=IntervalTrigger(minutes=schedule.deadline_check_minutes),
            id="deadline_watch",
            name="30-day deadline watch",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=schedule.misfire_grace_seconds,
        )

        self._install_signal_handlers()
        LOGGER.info(
            "scheduler starting: trading every %.1fh, deadline checked every %dmin, "
            "%s until day %d",
            schedule.cycle_hours, schedule.deadline_check_minutes,
            humanize_timedelta(self.time_remaining()), schedule.experiment_days,
        )
        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            LOGGER.warning("interrupted - shutting down without liquidating")
        finally:
            self.shutdown()

    def _install_signal_handlers(self) -> None:
        def _handler(signum: int, _frame: Any) -> None:
            LOGGER.warning("signal %s received - stopping the scheduler", signum)
            self._shutting_down = True
            self._stop_scheduler()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                # Not on the main thread (e.g. embedded) - nothing to install.
                LOGGER.debug("could not install handler for signal %s", sig)

    def _stop_scheduler(self) -> None:
        if self.scheduler is not None and self.scheduler.running:
            try:
                self.scheduler.shutdown(wait=False)
            except Exception:  # noqa: BLE001
                LOGGER.debug("scheduler shutdown raised", exc_info=True)

    def shutdown(self) -> None:
        """Release every external resource. Safe to call repeatedly."""
        LOGGER.info("shutting down: closing exchange, HTTP session and database")
        for closer in (self.fetcher.close, self.scraper.close, self.db.close):
            try:
                closer()
            except Exception:  # noqa: BLE001 - teardown must not raise
                LOGGER.debug("teardown step failed", exc_info=True)


__all__ = ["CycleReport", "Orchestrator", "STATUS_COMPLETED", "STATUS_RUNNING"]
