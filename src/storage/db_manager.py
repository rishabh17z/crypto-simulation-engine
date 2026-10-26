"""
SQLite persistence layer.

Design notes
------------
* **Single writer connection, serialised by an ``RLock``.** APScheduler runs jobs on a
  thread pool, and the deadline-watch job can fire while a 4-hour cycle is mid-write.
  The connection is opened with ``check_same_thread=False`` and every statement goes
  through :meth:`Database._execute`, which holds the lock for the whole transaction.
* **WAL journal + ``busy_timeout``.** WAL lets an external reader (a dashboard, ``sqlite3``
  CLI) read while the engine writes; ``busy_timeout`` plus an explicit retry loop absorbs
  ``database is locked`` / ``database is busy`` from any other process touching the file.
* **Everything is append-only except trade closure.** A trade row is written once on
  entry and updated exactly once on exit, which keeps the audit trail intact.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from src.common import get_logger, retry_on, safe_float, to_iso, utc_now

LOGGER = get_logger(__name__)

# sqlite3.OperationalError covers "database is locked", "database is busy" and
# "disk I/O error"; only the lock family is worth retrying, filtered in _is_transient.
TRANSIENT_DB_ERRORS = (sqlite3.OperationalError,)

SCHEMA_VERSION = 1

SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS engine_state (
        key         TEXT PRIMARY KEY,
        value       TEXT NOT NULL,
        updated_at  TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS trades (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_uid        TEXT    NOT NULL UNIQUE,
        cycle_uid        TEXT,
        timestamp        TEXT    NOT NULL,          -- entry time, ISO-8601 UTC
        closed_at        TEXT,                      -- exit time, NULL while open
        asset            TEXT    NOT NULL,
        action           TEXT    NOT NULL,          -- BUY | SELL (the signal)
        side             TEXT    NOT NULL,          -- long | short
        entry_price      REAL    NOT NULL,          -- simulated fill, slippage applied
        exit_price       REAL,
        position_size    REAL    NOT NULL,          -- base units
        notional         REAL    NOT NULL,          -- quote units at entry
        stop_loss        REAL    NOT NULL,
        take_profit      REAL    NOT NULL,
        fees             REAL    NOT NULL DEFAULT 0,-- cumulative, both sides
        slippage_cost    REAL    NOT NULL DEFAULT 0,
        realized_pnl     REAL,                      -- net of fees and slippage
        gross_pnl        REAL,
        return_pct       REAL,
        status           TEXT    NOT NULL,          -- OPEN | CLOSED
        exit_reason      TEXT,                      -- TAKE_PROFIT | STOP_LOSS | SIGNAL_FLIP
                                                    -- | END_OF_EXPERIMENT | MANUAL
        confidence       REAL,
        rsi              REAL,
        ema_fast         REAL,
        ema_slow         REAL,
        atr              REAL,
        sentiment_score  REAL,
        rationale        TEXT,
        exchange_order_id TEXT,
        metadata         TEXT                       -- JSON blob, free-form
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS decisions (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        cycle_uid        TEXT    NOT NULL,
        timestamp        TEXT    NOT NULL,
        asset            TEXT    NOT NULL,
        action           TEXT    NOT NULL,          -- BUY | SELL | HOLD
        confidence       REAL    NOT NULL,
        score            REAL,
        price            REAL,
        rsi              REAL,
        ema_fast         REAL,
        ema_slow         REAL,
        atr              REAL,
        sentiment_score  REAL,
        take_profit      REAL,
        stop_loss        REAL,
        executed         INTEGER NOT NULL DEFAULT 0,
        skip_reason      TEXT,
        rationale        TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS equity_snapshots (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        cycle_uid        TEXT,
        timestamp        TEXT    NOT NULL,
        equity           REAL    NOT NULL,
        cash             REAL    NOT NULL,
        unrealized_pnl   REAL    NOT NULL,
        realized_pnl_cum REAL    NOT NULL,
        fees_cum         REAL    NOT NULL,
        open_positions   INTEGER NOT NULL,
        exposure_pct     REAL    NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cycles (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        cycle_uid        TEXT    NOT NULL UNIQUE,
        started_at       TEXT    NOT NULL,
        finished_at      TEXT,
        day_index        INTEGER,
        assets_evaluated INTEGER NOT NULL DEFAULT 0,
        orders_opened    INTEGER NOT NULL DEFAULT 0,
        orders_closed    INTEGER NOT NULL DEFAULT 0,
        status           TEXT    NOT NULL,          -- RUNNING | OK | ERROR
        error            TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)",
    "CREATE INDEX IF NOT EXISTS idx_trades_asset ON trades(asset)",
    "CREATE INDEX IF NOT EXISTS idx_trades_closed_at ON trades(closed_at)",
    "CREATE INDEX IF NOT EXISTS idx_decisions_cycle ON decisions(cycle_uid)",
    "CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity_snapshots(timestamp)",
)

# Engine-state keys with first-class meaning.
KEY_START_AT = "experiment_started_at"
KEY_END_AT = "experiment_ends_at"
KEY_STATUS = "experiment_status"          # RUNNING | COMPLETED
KEY_CASH = "cash_balance"
KEY_SCHEMA = "schema_version"
KEY_LAST_CYCLE = "last_cycle_at"
KEY_CYCLE_COUNT = "cycle_count"


def _is_transient(error: sqlite3.OperationalError) -> bool:
    message = str(error).lower()
    return any(
        token in message
        for token in ("locked", "busy", "cannot start a transaction within")
    )


@dataclass(frozen=True, slots=True)
class TradeRecord:
    """Typed view over a ``trades`` row (only the fields callers actually need)."""

    trade_uid: str
    asset: str
    action: str
    side: str
    entry_price: float
    position_size: float
    notional: float
    stop_loss: float
    take_profit: float
    fees: float
    status: str
    timestamp: str
    exit_price: float | None = None
    realized_pnl: float | None = None
    exit_reason: str | None = None
    closed_at: str | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "TradeRecord":
        raw_meta = row["metadata"] if "metadata" in row.keys() else None
        try:
            meta = json.loads(raw_meta) if raw_meta else None
        except json.JSONDecodeError:
            meta = None
        return cls(
            trade_uid=row["trade_uid"],
            asset=row["asset"],
            action=row["action"],
            side=row["side"],
            entry_price=safe_float(row["entry_price"]),
            position_size=safe_float(row["position_size"]),
            notional=safe_float(row["notional"]),
            stop_loss=safe_float(row["stop_loss"]),
            take_profit=safe_float(row["take_profit"]),
            fees=safe_float(row["fees"]),
            status=row["status"],
            timestamp=row["timestamp"],
            exit_price=row["exit_price"],
            realized_pnl=row["realized_pnl"],
            exit_reason=row["exit_reason"],
            closed_at=row["closed_at"],
            metadata=meta,
        )


class Database:
    """Thread-safe SQLite gateway for the simulation engine."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        busy_timeout_ms: int = 10_000,
        journal_mode: str = "WAL",
        synchronous: str = "NORMAL",
        write_retries: int = 6,
        write_retry_delay: float = 0.35,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._busy_timeout_ms = busy_timeout_ms
        self._journal_mode = journal_mode
        self._synchronous = synchronous
        self._write_retries = max(1, write_retries)
        self._write_retry_delay = write_retry_delay
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None

    # -- lifecycle ---------------------------------------------------------------------

    def connect(self) -> "Database":
        with self._lock:
            if self._conn is not None:
                return self
            conn = sqlite3.connect(
                self.db_path,
                timeout=self._busy_timeout_ms / 1000.0,
                check_same_thread=False,
                isolation_level=None,  # explicit transaction control
            )
            conn.row_factory = sqlite3.Row
            conn.execute(f"PRAGMA busy_timeout = {int(self._busy_timeout_ms)}")
            conn.execute(f"PRAGMA journal_mode = {self._journal_mode}")
            conn.execute(f"PRAGMA synchronous = {self._synchronous}")
            conn.execute("PRAGMA foreign_keys = ON")
            self._conn = conn
            LOGGER.info(
                "SQLite ready at %s (journal=%s, busy_timeout=%dms)",
                self.db_path,
                self._journal_mode,
                self._busy_timeout_ms,
            )
            self._create_schema()
            return self

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.commit()
                finally:
                    self._conn.close()
                    self._conn = None
                    LOGGER.debug("SQLite connection closed")

    def __enter__(self) -> "Database":
        return self.connect()

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.connect()
        assert self._conn is not None
        return self._conn

    # -- low-level execution -----------------------------------------------------------

    def _execute_once(
        self, sql: str, params: Sequence[Any] | Mapping[str, Any] = ()
    ) -> sqlite3.Cursor:
        with self._lock:
            cursor = self.conn.execute(sql, params)
            return cursor

    def _execute(
        self, sql: str, params: Sequence[Any] | Mapping[str, Any] = ()
    ) -> sqlite3.Cursor:
        """Execute with a bounded retry loop over lock contention."""
        import time as _time

        last: sqlite3.OperationalError | None = None
        for attempt in range(1, self._write_retries + 1):
            try:
                return self._execute_once(sql, params)
            except sqlite3.OperationalError as exc:
                if not _is_transient(exc) or attempt == self._write_retries:
                    raise
                last = exc
                delay = self._write_retry_delay * attempt
                LOGGER.warning(
                    "SQLite contention (%s/%s): %s - retrying in %.2fs",
                    attempt,
                    self._write_retries,
                    exc,
                    delay,
                )
                _time.sleep(delay)
        raise last if last else RuntimeError("unreachable")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Serialised IMMEDIATE transaction; rolls back on any exception."""
        with self._lock:
            conn = self.conn
            self._execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")

    def _create_schema(self) -> None:
        with self.transaction():
            for statement in SCHEMA_STATEMENTS:
                self._execute_once(statement)
            self._execute_once(
                "INSERT INTO engine_state(key, value, updated_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO NOTHING",
                (KEY_SCHEMA, str(SCHEMA_VERSION), to_iso(utc_now())),
            )

    # -- engine state ------------------------------------------------------------------

    def set_state(self, key: str, value: Any) -> None:
        payload = value if isinstance(value, str) else json.dumps(value)
        with self.transaction():
            self._execute_once(
                "INSERT INTO engine_state(key, value, updated_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
                "updated_at=excluded.updated_at",
                (key, payload, to_iso(utc_now())),
            )

    def get_state(self, key: str, default: Any = None) -> Any:
        row = self._execute(
            "SELECT value FROM engine_state WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def get_state_float(self, key: str, default: float) -> float:
        return safe_float(self.get_state(key, default), default)

    def get_state_int(self, key: str, default: int) -> int:
        return int(safe_float(self.get_state(key, default), default))

    # -- trades ------------------------------------------------------------------------

    def insert_trade(self, trade: Mapping[str, Any]) -> int:
        columns = (
            "trade_uid", "cycle_uid", "timestamp", "asset", "action", "side",
            "entry_price", "position_size", "notional", "stop_loss", "take_profit",
            "fees", "slippage_cost", "status", "confidence", "rsi", "ema_fast",
            "ema_slow", "atr", "sentiment_score", "rationale", "exchange_order_id",
            "metadata",
        )
        values = []
        for column in columns:
            value = trade.get(column)
            if column == "metadata" and value is not None and not isinstance(value, str):
                value = json.dumps(value, default=str)
            values.append(value)

        placeholders = ",".join("?" for _ in columns)
        with self.transaction():
            cursor = self._execute_once(
                f"INSERT INTO trades({','.join(columns)}) VALUES({placeholders})",
                values,
            )
            trade_id = int(cursor.lastrowid or 0)
        LOGGER.debug("trade row %s inserted (%s)", trade_id, trade.get("trade_uid"))
        return trade_id

    def close_trade(
        self,
        trade_uid: str,
        *,
        exit_price: float,
        fees: float,
        slippage_cost: float,
        realized_pnl: float,
        gross_pnl: float,
        return_pct: float,
        exit_reason: str,
        closed_at: str | None = None,
        exchange_order_id: str | None = None,
    ) -> bool:
        """Finalise an open trade. Returns False if the row was already closed."""
        with self.transaction():
            cursor = self._execute_once(
                """
                UPDATE trades
                   SET exit_price = ?, fees = ?, slippage_cost = ?, realized_pnl = ?,
                       gross_pnl = ?, return_pct = ?, exit_reason = ?, closed_at = ?,
                       status = 'CLOSED',
                       exchange_order_id = COALESCE(?, exchange_order_id)
                 WHERE trade_uid = ? AND status = 'OPEN'
                """,
                (
                    exit_price, fees, slippage_cost, realized_pnl, gross_pnl,
                    return_pct, exit_reason, closed_at or to_iso(utc_now()),
                    exchange_order_id, trade_uid,
                ),
            )
            updated = cursor.rowcount
        if not updated:
            LOGGER.warning("close_trade: %s was not OPEN (no-op)", trade_uid)
        return bool(updated)

    def open_trades(self) -> list[TradeRecord]:
        rows = self._execute(
            "SELECT * FROM trades WHERE status = 'OPEN' ORDER BY id ASC"
        ).fetchall()
        return [TradeRecord.from_row(row) for row in rows]

    def closed_trades(self) -> list[TradeRecord]:
        rows = self._execute(
            "SELECT * FROM trades WHERE status = 'CLOSED' ORDER BY closed_at ASC"
        ).fetchall()
        return [TradeRecord.from_row(row) for row in rows]

    def all_trades(self) -> list[sqlite3.Row]:
        return self._execute("SELECT * FROM trades ORDER BY id ASC").fetchall()

    def open_trade_for_asset(self, asset: str) -> TradeRecord | None:
        row = self._execute(
            "SELECT * FROM trades WHERE asset = ? AND status = 'OPEN' "
            "ORDER BY id DESC LIMIT 1",
            (asset,),
        ).fetchone()
        return TradeRecord.from_row(row) if row else None

    # -- decisions ---------------------------------------------------------------------

    def log_decision(self, decision: Mapping[str, Any]) -> None:
        columns = (
            "cycle_uid", "timestamp", "asset", "action", "confidence", "score",
            "price", "rsi", "ema_fast", "ema_slow", "atr", "sentiment_score",
            "take_profit", "stop_loss", "executed", "skip_reason", "rationale",
        )
        placeholders = ",".join("?" for _ in columns)
        with self.transaction():
            self._execute_once(
                f"INSERT INTO decisions({','.join(columns)}) VALUES({placeholders})",
                [decision.get(column) for column in columns],
            )

    def decisions(self, limit: int = 200) -> list[sqlite3.Row]:
        return self._execute(
            "SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

    # -- equity ------------------------------------------------------------------------

    def record_equity(
        self,
        *,
        equity: float,
        cash: float,
        unrealized_pnl: float,
        realized_pnl_cum: float,
        fees_cum: float,
        open_positions: int,
        exposure_pct: float,
        cycle_uid: str | None = None,
        timestamp: str | None = None,
    ) -> None:
        with self.transaction():
            self._execute_once(
                """
                INSERT INTO equity_snapshots(
                    cycle_uid, timestamp, equity, cash, unrealized_pnl,
                    realized_pnl_cum, fees_cum, open_positions, exposure_pct
                ) VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    cycle_uid, timestamp or to_iso(utc_now()), equity, cash,
                    unrealized_pnl, realized_pnl_cum, fees_cum, open_positions,
                    exposure_pct,
                ),
            )

    def equity_curve(self) -> list[sqlite3.Row]:
        return self._execute(
            "SELECT timestamp, equity, cash, unrealized_pnl, realized_pnl_cum, "
            "fees_cum, open_positions, exposure_pct "
            "FROM equity_snapshots ORDER BY id ASC"
        ).fetchall()

    # -- cycles ------------------------------------------------------------------------

    def start_cycle(self, cycle_uid: str, day_index: int) -> None:
        with self.transaction():
            self._execute_once(
                "INSERT INTO cycles(cycle_uid, started_at, day_index, status) "
                "VALUES(?,?,?,'RUNNING') ON CONFLICT(cycle_uid) DO NOTHING",
                (cycle_uid, to_iso(utc_now()), day_index),
            )

    def finish_cycle(
        self,
        cycle_uid: str,
        *,
        assets_evaluated: int,
        orders_opened: int,
        orders_closed: int,
        status: str = "OK",
        error: str | None = None,
    ) -> None:
        with self.transaction():
            self._execute_once(
                """
                UPDATE cycles
                   SET finished_at = ?, assets_evaluated = ?, orders_opened = ?,
                       orders_closed = ?, status = ?, error = ?
                 WHERE cycle_uid = ?
                """,
                (
                    to_iso(utc_now()), assets_evaluated, orders_opened,
                    orders_closed, status, error, cycle_uid,
                ),
            )

    # -- aggregates --------------------------------------------------------------------

    def realized_totals(self) -> tuple[float, float]:
        """``(realized_pnl, fees)`` across closed trades, plus fees already paid on
        still-open entries so the equity series never under-reports costs."""
        row = self._execute(
            "SELECT COALESCE(SUM(realized_pnl),0) AS pnl FROM trades "
            "WHERE status = 'CLOSED'"
        ).fetchone()
        fee_row = self._execute(
            "SELECT COALESCE(SUM(fees),0) AS fees FROM trades"
        ).fetchone()
        return safe_float(row["pnl"]), safe_float(fee_row["fees"])

    def trade_stats(self) -> dict[str, Any]:
        row = self._execute(
            """
            SELECT COUNT(*)                                            AS trades,
                   COALESCE(SUM(realized_pnl), 0)                      AS net_pnl,
                   COALESCE(SUM(gross_pnl), 0)                         AS gross_pnl,
                   COALESCE(SUM(fees), 0)                              AS fees,
                   COALESCE(SUM(slippage_cost), 0)                     AS slippage,
                   COALESCE(SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END), 0) AS wins,
                   COALESCE(SUM(CASE WHEN realized_pnl <= 0 THEN 1 ELSE 0 END), 0) AS losses,
                   COALESCE(SUM(CASE WHEN realized_pnl > 0 THEN realized_pnl ELSE 0 END), 0)
                                                                       AS gross_profit,
                   COALESCE(SUM(CASE WHEN realized_pnl < 0 THEN -realized_pnl ELSE 0 END), 0)
                                                                       AS gross_loss,
                   COALESCE(MAX(realized_pnl), 0)                      AS best,
                   COALESCE(MIN(realized_pnl), 0)                      AS worst
              FROM trades
             WHERE status = 'CLOSED'
            """
        ).fetchone()
        return {key: row[key] for key in row.keys()}

    def per_asset_stats(self) -> list[sqlite3.Row]:
        return self._execute(
            """
            SELECT asset,
                   COUNT(*)                                             AS trades,
                   COALESCE(SUM(realized_pnl), 0)                       AS net_pnl,
                   COALESCE(SUM(fees), 0)                               AS fees,
                   COALESCE(SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END), 0) AS wins
              FROM trades
             WHERE status = 'CLOSED'
             GROUP BY asset
             ORDER BY net_pnl DESC
            """
        ).fetchall()

    def exit_reason_breakdown(self) -> list[sqlite3.Row]:
        return self._execute(
            "SELECT COALESCE(exit_reason,'UNKNOWN') AS exit_reason, COUNT(*) AS n, "
            "COALESCE(SUM(realized_pnl),0) AS net_pnl FROM trades "
            "WHERE status='CLOSED' GROUP BY exit_reason ORDER BY n DESC"
        ).fetchall()

    def action_breakdown(self) -> list[sqlite3.Row]:
        return self._execute(
            "SELECT action, COUNT(*) AS n FROM decisions GROUP BY action"
        ).fetchall()

    def reset(self) -> None:
        """Destructive: wipe all simulation data but keep the schema."""
        with self.transaction():
            for table in ("trades", "decisions", "equity_snapshots", "cycles",
                          "engine_state"):
                self._execute_once(f"DELETE FROM {table}")
            self._execute_once(
                "INSERT INTO engine_state(key, value, updated_at) VALUES(?,?,?)",
                (KEY_SCHEMA, str(SCHEMA_VERSION), to_iso(utc_now())),
            )
        LOGGER.warning("database reset: all trades, decisions and equity history erased")


@retry_on(TRANSIENT_DB_ERRORS, attempts=3, base_delay=0.5, logger=LOGGER)
def open_database(settings: Any) -> Database:
    """Factory used by the orchestrator; retries once or twice on a cold-start lock."""
    storage = settings.storage
    return Database(
        storage.db_path,
        busy_timeout_ms=storage.busy_timeout_ms,
        journal_mode=storage.journal_mode,
        synchronous=storage.synchronous,
        write_retries=storage.write_retries,
        write_retry_delay=storage.write_retry_delay,
    ).connect()


__all__ = [
    "Database",
    "KEY_CASH",
    "KEY_CYCLE_COUNT",
    "KEY_END_AT",
    "KEY_LAST_CYCLE",
    "KEY_START_AT",
    "KEY_STATUS",
    "SCHEMA_VERSION",
    "TradeRecord",
    "open_database",
]
