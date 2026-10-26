#!/usr/bin/env python3
"""
Entry point for the 30-day crypto paper-trading simulation engine.

Usage
-----
    python main.py run          # start the 30-day experiment (blocking scheduler)
    python main.py once         # execute exactly one 4-hour cycle, then exit
    python main.py status       # print the live portfolio and experiment clock
    python main.py report       # rebuild the P&L report from the database
    python main.py liquidate    # flatten the book and finalise early
    python main.py doctor       # verify config, exchange, sentiment and database
    python main.py reset        # DESTRUCTIVE: wipe the database and start over

All state lives in SQLite (``data/simulation.sqlite3``), so ``run`` can be stopped and
restarted freely: the experiment window, cash balance and open positions are restored
and the 30-day deadline is *not* extended by a restart.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the project importable regardless of the caller's working directory.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SETTINGS, ConfigError  # noqa: E402
from src.common import configure_logging, get_logger, humanize_timedelta, percent  # noqa: E402

LOGGER = get_logger("main")

BANNER = r"""
  crypto-simulation-engine - 30-day paper-trading experiment
  ----------------------------------------------------------
"""


def _print_config() -> None:
    exchange, risk, sim, schedule = (
        SETTINGS.exchange,
        SETTINGS.risk,
        SETTINGS.simulation,
        SETTINGS.schedule,
    )
    print(BANNER)
    print(f"  venue          : {exchange.exchange_id} "
          f"{'TESTNET/SANDBOX' if exchange.use_sandbox else '!! PRODUCTION !!'} "
          f"({exchange.market_type})")
    creds = "present" if exchange.has_credentials else "absent (public data only)"
    mirror = (
        "ON (orders mirrored to the testnet book)"
        if exchange.place_testnet_orders
        else "OFF (pure simulation)"
    )
    print(f"  credentials    : {creds}")
    print(f"  order mirror   : {mirror}")
    print(f"  corpus         : {risk.initial_corpus:,.2f} {exchange.quote_currency}")
    print(f"  risk / trade   : {percent(risk.max_risk_per_trade)} "
          f"({risk.position_sizing_mode} sizing)")
    print(f"  exposure cap   : {percent(risk.max_portfolio_exposure)}, "
          f"max {risk.max_open_positions} positions, shorts "
          f"{'allowed' if risk.allow_shorts else 'disabled'}")
    print(f"  costs          : fee {percent(sim.fee_rate)}/side, "
          f"slippage {percent(sim.slippage_rate)}/side")
    print(f"  cadence        : every {schedule.cycle_hours}h for "
          f"{schedule.experiment_days} days")
    print(f"  universe       : {', '.join(a.symbol for a in SETTINGS.enabled_assets)}")
    print(f"  database       : {SETTINGS.storage.db_path}")
    print()


def cmd_run(_args: argparse.Namespace) -> int:
    from src.orchestrator import Orchestrator

    _print_config()
    # Without credentials or order mirroring a production client can only read public
    # market data, so there is nothing to confirm (and no TTY under launchd/nohup).
    exchange = SETTINGS.exchange
    can_trade = exchange.has_credentials or exchange.place_testnet_orders
    if not exchange.use_sandbox and can_trade:
        confirm = input(
            "USE_SANDBOX is false: this will hit PRODUCTION endpoints. Type 'I ACCEPT' "
            "to continue: "
        )
        if confirm.strip() != "I ACCEPT":
            print("aborted.")
            return 1

    orchestrator = Orchestrator(SETTINGS)
    orchestrator.run()
    return 0


def cmd_once(_args: argparse.Namespace) -> int:
    from src.orchestrator import Orchestrator

    _print_config()
    orchestrator = Orchestrator(SETTINGS)
    try:
        orchestrator.bootstrap()
        report = orchestrator.run_cycle()
        if report is None:
            print("no cycle executed (experiment completed or deadline reached).")
        else:
            print(
                f"\ncycle {report.cycle_uid} | day {report.day_index} | "
                f"evaluated {report.assets_evaluated} asset(s), "
                f"opened {report.opened}, closed {report.closed}, "
                f"data errors {report.data_errors}\n"
                f"actions: {report.actions}\n"
                f"equity : {report.equity:.2f} {SETTINGS.exchange.quote_currency}"
            )
        print()
        print("\n".join(orchestrator.status_lines()))
    finally:
        orchestrator.shutdown()
    return 0


def cmd_status(_args: argparse.Namespace) -> int:
    from src.orchestrator import Orchestrator

    orchestrator = Orchestrator(SETTINGS)
    try:
        print(BANNER)
        print("\n".join(orchestrator.status_lines()))
    finally:
        orchestrator.shutdown()
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from src.orchestrator import Orchestrator

    orchestrator = Orchestrator(SETTINGS)
    try:
        report = orchestrator.build_report()
        markdown = orchestrator.render_report(report)
        print(markdown)
        if args.save:
            json_path, md_path = orchestrator.write_report(report)
            print(f"\nsaved: {json_path}\n       {md_path}")
    finally:
        orchestrator.shutdown()
    return 0


def cmd_liquidate(args: argparse.Namespace) -> int:
    from src.orchestrator import Orchestrator

    if not args.yes:
        confirm = input(
            "This liquidates every open position and marks the experiment COMPLETED. "
            "Continue? [y/N] "
        )
        if confirm.strip().lower() not in {"y", "yes"}:
            print("aborted.")
            return 1

    orchestrator = Orchestrator(SETTINGS)
    try:
        orchestrator.bootstrap()
        report = orchestrator.finalize()
        print(orchestrator.render_report(report))
    finally:
        orchestrator.shutdown()
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    from src.storage.db_manager import open_database

    if not args.yes:
        confirm = input(
            f"This permanently erases every trade, decision and equity record in "
            f"{SETTINGS.storage.db_path}. Type 'RESET' to confirm: "
        )
        if confirm.strip() != "RESET":
            print("aborted.")
            return 1

    database = open_database(SETTINGS)
    try:
        database.reset()
        print(f"database reset: {SETTINGS.storage.db_path}")
    finally:
        database.close()
    return 0


def cmd_doctor(_args: argparse.Namespace) -> int:
    """Pre-flight check: config, exchange reachability, indicators, sentiment, DB."""
    from src.data_ingestion.price_fetcher import MarketDataError, PriceFetcher
    from src.data_ingestion.sentiment_scraper import SentimentScraper
    from src.storage.db_manager import open_database

    _print_config()
    failures = 0
    warnings: list[str] = []

    print("[1/4] database ...", end=" ", flush=True)
    try:
        database = open_database(SETTINGS)
        database.get_state("schema_version")
        database.close()
        print("OK")
    except Exception as exc:  # noqa: BLE001
        failures += 1
        print(f"FAIL ({exc})")

    print("[2/4] exchange / markets ...", end=" ", flush=True)
    fetcher = None
    try:
        fetcher = PriceFetcher(SETTINGS)
        fetcher.load_markets()
        print(f"OK ({len(fetcher.exchange.markets or {})} markets)")
    except Exception as exc:  # noqa: BLE001
        failures += 1
        print(f"FAIL ({exc})")

    print("[3/4] market data + indicators ...")
    if fetcher is not None:
        for asset in SETTINGS.enabled_assets:
            try:
                snapshot = fetcher.snapshot(asset)
                print(
                    f"        {asset.symbol:<12} px={snapshot.price:<14.6f} "
                    f"RSI={snapshot.rsi:<6.2f} "
                    f"EMA{snapshot.ema_fast_period}={snapshot.ema_fast:<14.6f} "
                    f"EMA{snapshot.ema_slow_period}={snapshot.ema_slow:<14.6f} "
                    f"ATR%={snapshot.atr_pct:.2%} [{snapshot.regime}]"
                )
            except MarketDataError as exc:
                failures += 1
                print(f"        {asset.symbol:<12} FAIL ({exc})")
        fetcher.close()
    else:
        failures += 1
        print("        skipped (no exchange client)")

    print("[4/4] sentiment ...")
    if not SETTINGS.sentiment.enabled:
        print("        skipped (SENTIMENT_ENABLED=false; trading on technicals alone)")
    else:
        scraper = SentimentScraper(SETTINGS)
        dead_assets: list[str] = []
        try:
            for asset in SETTINGS.enabled_assets:
                snapshot = scraper.get_sentiment(asset.symbol, asset.sentiment_key)
                flag = " [degraded]" if snapshot.degraded else ""
                print(
                    f"        {asset.symbol:<12} score={snapshot.score:+.3f} "
                    f"({snapshot.label}) "
                    f"sources={','.join(snapshot.sources) or 'none'}{flag}"
                )
                if not snapshot.sources:
                    dead_assets.append(asset.symbol)
        finally:
            scraper.close()

        # Sentiment is an optional input, so an outage is not a hard failure - but it
        # must never be reported as a clean bill of health.
        if dead_assets and len(dead_assets) == len(SETTINGS.enabled_assets):
            warnings.append(
                "no sentiment source is reachable: every decision will run on "
                "technicals alone (weights renormalised over trend + momentum)"
            )
        elif dead_assets:
            warnings.append(
                f"sentiment unavailable for {', '.join(dead_assets)}"
            )

    print()
    if failures:
        print(f"doctor finished with {failures} failure(s).")
        for warning in warnings:
            print(f"  warning: {warning}")
        return 1

    if warnings:
        print(f"doctor finished: core checks passed, {len(warnings)} warning(s).")
        for warning in warnings:
            print(f"  warning: {warning}")
        print(
            "\n  If a sentiment host is blocked on your network, either set\n"
            "  SENTIMENT_ENABLED=false to skip the HTTP calls entirely, or lower\n"
            "  SENTIMENT_HTTP_TIMEOUT / SENTIMENT_MAX_RETRIES so a dead source costs\n"
            "  less time. Failures are already negative-cached for\n"
            f"  {SETTINGS.sentiment.failure_cooldown_seconds}s, so each cycle makes at "
            "most one attempt per source."
        )
        return 0

    print("doctor finished: all checks passed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="30-day crypto paper-trading simulation engine (CCXT / Bybit testnet)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="override LOG_LEVEL for this invocation",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("run", help="start the 30-day experiment").set_defaults(
        func=cmd_run
    )
    subparsers.add_parser("once", help="run a single cycle and exit").set_defaults(
        func=cmd_once
    )
    subparsers.add_parser("status", help="show the live portfolio").set_defaults(
        func=cmd_status
    )

    report_parser = subparsers.add_parser("report", help="print the P&L report")
    report_parser.add_argument(
        "--save", action="store_true", help="also write the report into reports/"
    )
    report_parser.set_defaults(func=cmd_report)

    liquidate_parser = subparsers.add_parser(
        "liquidate", help="close every position and finalise the experiment"
    )
    liquidate_parser.add_argument("--yes", action="store_true", help="skip confirmation")
    liquidate_parser.set_defaults(func=cmd_liquidate)

    reset_parser = subparsers.add_parser("reset", help="wipe the database (destructive)")
    reset_parser.add_argument("--yes", action="store_true", help="skip confirmation")
    reset_parser.set_defaults(func=cmd_reset)

    subparsers.add_parser("doctor", help="run pre-flight checks").set_defaults(
        func=cmd_doctor
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1

    configure_logging(args.log_level or SETTINGS.log_level, SETTINGS.log_file)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        LOGGER.error("configuration error: %s", exc)
        return 2
    except KeyboardInterrupt:
        LOGGER.warning("interrupted by user")
        return 130
    except Exception as exc:  # noqa: BLE001 - top-level guard
        LOGGER.exception("fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
