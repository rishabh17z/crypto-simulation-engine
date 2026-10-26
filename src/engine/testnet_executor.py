"""
Execution layer: risk-checked position sizing, simulated fills, and (optionally) mirrored
Bybit testnet orders.

Accounting model
----------------
Cash is treated as free collateral at 1x notional - no leverage is simulated, so a
position can never lose more than its notional.

    open   cash -= (notional + entry_fee)
    close  cash += (notional + gross_pnl - exit_fee)
    equity  = cash + SUM(position marked value)

Every fill crosses the spread by ``slippage_rate`` in the adverse direction and pays
``fee_rate`` on both sides, so the simulation's reported P&L is net of costs by
construction.

Risk controls (all enforced here, not in the strategy)
------------------------------------------------------
* per-trade allocation capped at ``max_risk_per_trade`` of current equity
* aggregate open notional capped at ``max_portfolio_exposure`` of equity
* ``max_open_positions`` and one position per asset (no pyramiding)
* an order without BOTH a stop-loss and a take-profit is rejected outright
* equity circuit breaker: new entries stop below ``equity_stop`` x initial corpus
* orders below the exchange/asset minimum notional are skipped, never truncated to dust
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

import ccxt

from config.settings import AssetConfig, Settings
from src.common import clamp, get_logger, new_uid, safe_float, to_iso, utc_now
from src.data_ingestion.price_fetcher import MarketSnapshot, PriceFetcher
from src.engine.decision_logic import Action, ExitReason, TradeDecision
from src.storage.db_manager import KEY_CASH, Database, TradeRecord

LOGGER = get_logger(__name__)


class ExecutionError(RuntimeError):
    """Raised when an order cannot be simulated at all (bad inputs, broken state)."""


@dataclass(slots=True)
class Position:
    """An open simulated position."""

    trade_uid: str
    symbol: str
    exchange_symbol: str
    side: str                      # "long" | "short"
    action: str                    # the originating signal: BUY | SELL
    entry_price: float             # simulated fill, slippage included
    size: float                    # base units
    notional: float                # entry_price * size
    stop_loss: float
    take_profit: float
    opened_at: datetime
    entry_fee: float = 0.0
    entry_slippage_cost: float = 0.0
    confidence: float = 0.0
    exchange_order_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def direction(self) -> int:
        return 1 if self.side == "long" else -1

    def gross_pnl(self, mark_price: float) -> float:
        return (mark_price - self.entry_price) * self.size * self.direction

    def marked_value(self, mark_price: float) -> float:
        """Collateral + unrealised P&L, i.e. what this position contributes to equity."""
        return self.notional + self.gross_pnl(mark_price)

    def unrealized_pct(self, mark_price: float) -> float:
        return self.gross_pnl(mark_price) / self.notional if self.notional else 0.0


@dataclass(frozen=True, slots=True)
class Fill:
    """A simulated fill after slippage."""

    price: float
    fee: float
    slippage_cost: float
    reference_price: float


@dataclass(frozen=True, slots=True)
class TradeResult:
    """Outcome of a closed position."""

    trade_uid: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    gross_pnl: float
    fees: float
    slippage_cost: float
    realized_pnl: float
    return_pct: float
    exit_reason: str
    closed_at: datetime

    def summary(self) -> str:
        return (
            f"{self.symbol} {self.side} closed @ {self.exit_price:.6f} "
            f"({self.exit_reason}): net {self.realized_pnl:+.2f} USDT "
            f"({self.return_pct:+.2%}), fees {self.fees:.2f}"
        )


@dataclass(frozen=True, slots=True)
class PortfolioState:
    """Point-in-time portfolio marks."""

    equity: float
    cash: float
    unrealized_pnl: float
    exposure: float
    open_positions: int

    @property
    def exposure_pct(self) -> float:
        return self.exposure / self.equity if self.equity else 0.0


class TestnetExecutor:
    """Simulated broker with an optional real testnet mirror."""

    def __init__(
        self,
        settings: Settings,
        database: Database,
        price_fetcher: PriceFetcher | None = None,
    ) -> None:
        self.settings = settings
        self.db = database
        self.fetcher = price_fetcher
        self.sim = settings.simulation
        self.risk = settings.risk
        self._cash: float = settings.risk.initial_corpus
        self.positions: dict[str, Position] = {}
        self._mirror_orders = (
            settings.exchange.place_testnet_orders
            and settings.exchange.has_credentials
            and price_fetcher is not None
        )
        if settings.exchange.place_testnet_orders and not self._mirror_orders:
            LOGGER.warning(
                "PLACE_TESTNET_ORDERS=true but credentials/exchange client are missing: "
                "running in pure-simulation mode"
            )

    # ---------------------------------------------------------------------------------
    # State
    # ---------------------------------------------------------------------------------

    def bootstrap(self) -> None:
        """Restore cash and open positions so a restart resumes mid-experiment."""
        stored_cash = self.db.get_state(KEY_CASH)
        if stored_cash is None:
            self._cash = self.risk.initial_corpus
            self.db.set_state(KEY_CASH, f"{self._cash:.8f}")
            LOGGER.info("initialised virtual corpus: %.2f USDT", self._cash)
        else:
            self._cash = safe_float(stored_cash, self.risk.initial_corpus)
            LOGGER.info("restored cash balance: %.2f USDT", self._cash)

        self.positions.clear()
        for record in self.db.open_trades():
            try:
                self.positions[record.asset] = self._position_from_record(record)
            except Exception:  # noqa: BLE001 - a corrupt row must not block startup
                LOGGER.exception("could not restore open trade %s", record.trade_uid)
        if self.positions:
            LOGGER.info(
                "restored %d open position(s): %s",
                len(self.positions),
                ", ".join(f"{p.symbol}:{p.side}" for p in self.positions.values()),
            )

    def _position_from_record(self, record: TradeRecord) -> Position:
        meta = record.metadata or {}
        try:
            asset = self.settings.asset(record.asset)
            exchange_symbol = asset.exchange_symbol
        except KeyError:
            exchange_symbol = str(meta.get("exchange_symbol") or record.asset)
        opened_at = utc_now()
        try:
            opened_at = datetime.fromisoformat(record.timestamp)
        except ValueError:
            LOGGER.warning("unparseable entry timestamp on %s", record.trade_uid)
        return Position(
            trade_uid=record.trade_uid,
            symbol=record.asset,
            exchange_symbol=exchange_symbol,
            side=record.side,
            action=record.action,
            entry_price=record.entry_price,
            size=record.position_size,
            notional=record.notional,
            stop_loss=record.stop_loss,
            take_profit=record.take_profit,
            opened_at=opened_at,
            entry_fee=record.fees,
            entry_slippage_cost=safe_float(meta.get("entry_slippage_cost"), 0.0),
            confidence=safe_float(meta.get("confidence"), 0.0),
            metadata=dict(meta),
        )

    @property
    def cash(self) -> float:
        return self._cash

    def _set_cash(self, value: float) -> None:
        self._cash = round(value, 8)
        self.db.set_state(KEY_CASH, f"{self._cash:.8f}")

    def portfolio(self, prices: Mapping[str, float]) -> PortfolioState:
        """Mark the book. Missing prices fall back to the position's entry price."""
        exposure = 0.0
        unrealized = 0.0
        for position in self.positions.values():
            mark = safe_float(prices.get(position.symbol), 0.0) or position.entry_price
            exposure += position.notional
            unrealized += position.gross_pnl(mark)
        return PortfolioState(
            equity=self._cash + exposure + unrealized,
            cash=self._cash,
            unrealized_pnl=unrealized,
            exposure=exposure,
            open_positions=len(self.positions),
        )

    def equity(self, prices: Mapping[str, float]) -> float:
        return self.portfolio(prices).equity

    # ---------------------------------------------------------------------------------
    # Fill simulation
    # ---------------------------------------------------------------------------------

    def simulate_fill(self, reference_price: float, order_side: str, size: float) -> Fill:
        """
        Apply adverse slippage and the taker fee.

        ``order_side`` is the *exchange* side: a buy fills above the reference, a sell
        below it. That holds for opening a long, opening a short, and both closes.
        """
        if reference_price <= 0 or size <= 0:
            raise ExecutionError(
                f"cannot simulate a fill at price={reference_price} size={size}"
            )
        direction = 1.0 if order_side == "buy" else -1.0
        fill_price = reference_price * (1.0 + direction * self.sim.slippage_rate)
        notional = fill_price * size
        return Fill(
            price=fill_price,
            fee=notional * self.sim.fee_rate,
            slippage_cost=abs(fill_price - reference_price) * size,
            reference_price=reference_price,
        )

    # ---------------------------------------------------------------------------------
    # Sizing
    # ---------------------------------------------------------------------------------

    def _round_amount(self, exchange_symbol: str, size: float) -> float:
        """Snap to the market's amount precision when the exchange is reachable."""
        if self.fetcher is None:
            return size
        try:
            return float(self.fetcher.exchange.amount_to_precision(exchange_symbol, size))
        except Exception:  # noqa: BLE001 - precision is a refinement, not a requirement
            return size

    def size_position(
        self,
        decision: TradeDecision,
        asset: AssetConfig,
        equity: float,
        fill_price: float,
    ) -> tuple[float, float, str | None]:
        """
        Resolve ``(size, notional, rejection_reason)``.

        ``notional`` mode (default) allocates exactly ``max_risk_per_trade`` of equity as
        position notional - the literal reading of "never allocate more than the
        configured risk percentage". ``risk`` mode instead sizes so that a stop-out
        costs ``max_risk_per_trade`` of equity, then re-applies the exposure cap.
        """
        weight = clamp(asset.weight, 0.0, 1.0)
        if weight <= 0:
            return 0.0, 0.0, "asset weight is zero"

        budget = equity * self.risk.max_risk_per_trade * weight
        if self.risk.position_sizing_mode == "risk":
            stop_pct = decision.stop_distance_pct
            if stop_pct <= 0:
                return 0.0, 0.0, "stop distance is zero; cannot size by risk"
            budget = min(budget / stop_pct, equity * self.risk.max_portfolio_exposure)

        current_exposure = sum(p.notional for p in self.positions.values())
        exposure_headroom = max(
            equity * self.risk.max_portfolio_exposure - current_exposure, 0.0
        )
        if exposure_headroom <= 0:
            return 0.0, 0.0, (
                f"portfolio exposure cap reached "
                f"({current_exposure:.2f}/{equity * self.risk.max_portfolio_exposure:.2f})"
            )

        cash_headroom = max(self._cash / (1.0 + self.sim.fee_rate), 0.0)
        notional = min(budget, exposure_headroom, cash_headroom)

        floor = max(self.risk.min_order_notional, asset.min_notional)
        if notional < floor:
            return 0.0, 0.0, (
                f"sized notional {notional:.2f} below minimum {floor:.2f} "
                f"(budget {budget:.2f}, exposure headroom {exposure_headroom:.2f}, "
                f"cash headroom {cash_headroom:.2f})"
            )

        size = self._round_amount(asset.exchange_symbol, notional / fill_price)
        if size <= 0:
            return 0.0, 0.0, "size rounded to zero at the market's amount precision"

        notional = size * fill_price
        if notional > self._cash / (1.0 + self.sim.fee_rate) + 1e-9:
            return 0.0, 0.0, (
                f"insufficient cash: need {notional * (1 + self.sim.fee_rate):.2f}, "
                f"have {self._cash:.2f}"
            )
        return size, notional, None

    # ---------------------------------------------------------------------------------
    # Pre-trade gates
    # ---------------------------------------------------------------------------------

    def pre_trade_check(self, decision: TradeDecision, equity: float) -> str | None:
        """Return a rejection reason, or ``None`` when the order may proceed."""
        if not decision.is_actionable:
            return "decision is HOLD"
        if self.risk.require_stop_loss and (
            decision.stop_loss is None or decision.take_profit is None
        ):
            return "order rejected: stop-loss and take-profit are mandatory"
        if decision.action is Action.SELL and not self.risk.allow_shorts:
            return "short entries disabled"
        # The equity circuit breaker is a global halt, so it is reported ahead of the
        # per-asset capacity gates: "we have stopped trading" is more actionable than
        # "this particular slot is full".
        floor_equity = self.risk.initial_corpus * self.risk.equity_stop
        if equity < floor_equity:
            return (
                f"equity circuit breaker: {equity:.2f} < "
                f"{floor_equity:.2f} ({self.risk.equity_stop:.0%} of corpus)"
            )
        if decision.symbol in self.positions:
            return f"position already open on {decision.symbol}"
        if len(self.positions) >= self.risk.max_open_positions:
            return (
                f"max open positions reached "
                f"({len(self.positions)}/{self.risk.max_open_positions})"
            )
        # Direction sanity - never trust an upstream level blindly.
        direction = 1.0 if decision.action is Action.BUY else -1.0
        if (decision.stop_loss - decision.reference_price) * direction >= 0:
            return "stop-loss is on the wrong side of the entry price"
        if (decision.take_profit - decision.reference_price) * direction <= 0:
            return "take-profit is on the wrong side of the entry price"
        return None

    # ---------------------------------------------------------------------------------
    # Open / close
    # ---------------------------------------------------------------------------------

    def open_position(
        self,
        decision: TradeDecision,
        snapshot: MarketSnapshot,
        *,
        equity: float,
        cycle_uid: str | None = None,
        sentiment_score: float | None = None,
    ) -> tuple[Position | None, str | None]:
        """Open a simulated position. Returns ``(position, rejection_reason)``."""
        rejection = self.pre_trade_check(decision, equity)
        if rejection:
            LOGGER.info("%s: order not placed - %s", decision.symbol, rejection)
            return None, rejection

        asset = self.settings.asset(decision.symbol)
        order_side = "buy" if decision.action is Action.BUY else "sell"
        reference = decision.reference_price or snapshot.price

        probe = self.simulate_fill(reference, order_side, 1.0)
        size, notional, reason = self.size_position(decision, asset, equity, probe.price)
        if reason:
            LOGGER.info("%s: order not placed - %s", decision.symbol, reason)
            return None, reason

        fill = self.simulate_fill(reference, order_side, size)
        notional = fill.price * size
        total_debit = notional + fill.fee
        if total_debit > self._cash + 1e-9:
            reason = f"insufficient cash at fill time ({total_debit:.2f} > {self._cash:.2f})"
            LOGGER.warning("%s: %s", decision.symbol, reason)
            return None, reason

        # Re-derive the protective levels from the *actual* fill so the risk distance
        # the strategy asked for is what the position really carries.
        stop_loss, take_profit = self._shift_levels(decision, reference, fill.price)

        position = Position(
            trade_uid=new_uid(decision.symbol.split("/")[0].lower()),
            symbol=decision.symbol,
            exchange_symbol=asset.exchange_symbol,
            side=decision.side,
            action=decision.action.value,
            entry_price=fill.price,
            size=size,
            notional=notional,
            stop_loss=stop_loss,
            take_profit=take_profit,
            opened_at=utc_now(),
            entry_fee=fill.fee,
            entry_slippage_cost=fill.slippage_cost,
            confidence=decision.confidence,
            metadata={
                "exchange_symbol": asset.exchange_symbol,
                "confidence": decision.confidence,
                "score": decision.score,
                "entry_slippage_cost": fill.slippage_cost,
                "reference_price": reference,
                "decision_source": decision.source,
                "sizing_mode": self.risk.position_sizing_mode,
                "components": dict(decision.components),
                "regime": snapshot.regime,
            },
        )

        order_id = self._mirror_entry(position)
        position.exchange_order_id = order_id
        position.metadata["exchange_order_id"] = order_id

        self.db.insert_trade(
            {
                "trade_uid": position.trade_uid,
                "cycle_uid": cycle_uid,
                "timestamp": to_iso(position.opened_at),
                "asset": position.symbol,
                "action": position.action,
                "side": position.side,
                "entry_price": position.entry_price,
                "position_size": position.size,
                "notional": position.notional,
                "stop_loss": position.stop_loss,
                "take_profit": position.take_profit,
                "fees": position.entry_fee,
                "slippage_cost": position.entry_slippage_cost,
                "status": "OPEN",
                "confidence": position.confidence,
                "rsi": snapshot.rsi,
                "ema_fast": snapshot.ema_fast,
                "ema_slow": snapshot.ema_slow,
                "atr": snapshot.atr,
                "sentiment_score": sentiment_score,
                "rationale": decision.rationale,
                "exchange_order_id": order_id,
                "metadata": position.metadata,
            }
        )

        self._set_cash(self._cash - total_debit)
        self.positions[position.symbol] = position
        LOGGER.info(
            "OPEN %s %s %.8f @ %.6f (notional %.2f, fee %.4f) SL=%.6f TP=%.6f "
            "| cash %.2f",
            position.side.upper(), position.symbol, position.size, position.entry_price,
            position.notional, position.entry_fee, position.stop_loss,
            position.take_profit, self._cash,
        )
        return position, None

    def _shift_levels(
        self, decision: TradeDecision, reference: float, fill_price: float
    ) -> tuple[float, float]:
        """Translate SL/TP by the slippage actually incurred, preserving distances."""
        assert decision.stop_loss is not None and decision.take_profit is not None
        offset = fill_price - reference
        return decision.stop_loss + offset, decision.take_profit + offset

    def close_position(
        self,
        position: Position,
        reference_price: float,
        reason: str | ExitReason,
        *,
        closed_at: datetime | None = None,
    ) -> TradeResult:
        """Close a position at ``reference_price`` (slippage and exit fee applied)."""
        order_side = "sell" if position.side == "long" else "buy"
        fill = self.simulate_fill(reference_price, order_side, position.size)

        gross_pnl = position.gross_pnl(fill.price)
        total_fees = position.entry_fee + fill.fee
        total_slippage = position.entry_slippage_cost + fill.slippage_cost
        realized = gross_pnl - total_fees
        return_pct = realized / position.notional if position.notional else 0.0
        reason_text = reason.value if isinstance(reason, ExitReason) else str(reason)
        stamp = closed_at or utc_now()

        exit_order_id = self._mirror_exit(position)

        self.db.close_trade(
            position.trade_uid,
            exit_price=fill.price,
            fees=total_fees,
            slippage_cost=total_slippage,
            realized_pnl=realized,
            gross_pnl=gross_pnl,
            return_pct=return_pct,
            exit_reason=reason_text,
            closed_at=to_iso(stamp),
            exchange_order_id=exit_order_id,
        )

        # Release collateral, credit P&L, debit the exit fee.
        self._set_cash(self._cash + position.notional + gross_pnl - fill.fee)
        self.positions.pop(position.symbol, None)

        result = TradeResult(
            trade_uid=position.trade_uid,
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=fill.price,
            size=position.size,
            gross_pnl=gross_pnl,
            fees=total_fees,
            slippage_cost=total_slippage,
            realized_pnl=realized,
            return_pct=return_pct,
            exit_reason=reason_text,
            closed_at=stamp,
        )
        LOGGER.info("CLOSE %s | cash %.2f", result.summary(), self._cash)
        return result

    # ---------------------------------------------------------------------------------
    # Protective-level monitoring
    # ---------------------------------------------------------------------------------

    def check_protective_levels(
        self, position: Position, snapshot: MarketSnapshot | None, mark_price: float
    ) -> tuple[float, ExitReason] | None:
        """
        Decide whether the stop or target was hit since the last check.

        Two sources, in order:
        1. The last **closed candle's** high/low - but only when that candle closed
           after the position opened, otherwise it describes a bar the position never
           lived through.
        2. The live mark price.

        When a single bar touched both levels the outcome is ambiguous without tick
        data, so ``conservative_fills`` resolves it against the position (stop first).
        """
        if snapshot is not None and snapshot.candle_time >= position.opened_at:
            high, low = snapshot.high, snapshot.low
            if position.side == "long":
                stop_hit, target_hit = low <= position.stop_loss, high >= position.take_profit
            else:
                stop_hit, target_hit = high >= position.stop_loss, low <= position.take_profit

            if stop_hit and target_hit:
                if self.sim.conservative_fills:
                    return position.stop_loss, ExitReason.STOP_LOSS
                return position.take_profit, ExitReason.TAKE_PROFIT
            if stop_hit:
                return position.stop_loss, ExitReason.STOP_LOSS
            if target_hit:
                return position.take_profit, ExitReason.TAKE_PROFIT

        if mark_price > 0:
            if position.side == "long":
                if mark_price <= position.stop_loss:
                    return mark_price, ExitReason.STOP_LOSS
                if mark_price >= position.take_profit:
                    return mark_price, ExitReason.TAKE_PROFIT
            else:
                if mark_price >= position.stop_loss:
                    return mark_price, ExitReason.STOP_LOSS
                if mark_price <= position.take_profit:
                    return mark_price, ExitReason.TAKE_PROFIT
        return None

    def update_positions(
        self,
        snapshots: Mapping[str, MarketSnapshot],
        prices: Mapping[str, float] | None = None,
    ) -> list[TradeResult]:
        """Sweep every open position for stop/target hits. Returns the closures."""
        prices = dict(prices or {})
        results: list[TradeResult] = []
        for symbol, position in list(self.positions.items()):
            snapshot = snapshots.get(symbol)
            mark = safe_float(prices.get(symbol), 0.0) or (
                snapshot.price if snapshot else 0.0
            )
            if mark <= 0:
                LOGGER.warning("no mark price for open position %s; skipping sweep", symbol)
                continue
            hit = self.check_protective_levels(position, snapshot, mark)
            if hit is None:
                LOGGER.debug(
                    "%s %s open: mark %.6f, unrealised %+.2f%%",
                    symbol, position.side, mark, position.unrealized_pct(mark) * 100,
                )
                continue
            exit_price, reason = hit
            results.append(self.close_position(position, exit_price, reason))
        return results

    def close_by_signal(
        self, symbol: str, mark_price: float, detail: str
    ) -> TradeResult | None:
        """Close on a strategy-driven regime flip."""
        position = self.positions.get(symbol)
        if position is None:
            return None
        LOGGER.info("signal exit on %s: %s", symbol, detail)
        return self.close_position(position, mark_price, ExitReason.SIGNAL_FLIP)

    def liquidate_all(
        self,
        prices: Mapping[str, float],
        reason: str | ExitReason = ExitReason.END_OF_EXPERIMENT,
    ) -> list[TradeResult]:
        """Flatten the book - used at the day-30 boundary and by ``--liquidate``."""
        if not self.positions:
            LOGGER.info("liquidation requested: no open positions")
            return []

        LOGGER.warning(
            "liquidating %d open position(s): %s", len(self.positions), reason
        )
        results: list[TradeResult] = []
        for symbol, position in list(self.positions.items()):
            mark = safe_float(prices.get(symbol), 0.0)
            if mark <= 0 and self.fetcher is not None:
                mark = self.fetcher.fetch_last_price(position.exchange_symbol)
            if mark <= 0:
                mark = position.entry_price
                LOGGER.error(
                    "%s: no live price for liquidation; marking at entry %.6f "
                    "(P&L understated for this position)",
                    symbol, mark,
                )
            results.append(self.close_position(position, mark, reason))
        return results

    # ---------------------------------------------------------------------------------
    # Optional testnet mirror
    # ---------------------------------------------------------------------------------

    def _mirror_entry(self, position: Position) -> str | None:
        if not self._mirror_orders:
            return None
        params: dict[str, Any] = {}
        if self.settings.exchange.market_type == "swap":
            # Bybit accepts protective levels inline on the entry order.
            params.update(
                {
                    "stopLoss": f"{position.stop_loss:.10f}".rstrip("0"),
                    "takeProfit": f"{position.take_profit:.10f}".rstrip("0"),
                }
            )
        return self._place_order(
            position.exchange_symbol,
            "buy" if position.side == "long" else "sell",
            position.size,
            params,
        )

    def _mirror_exit(self, position: Position) -> str | None:
        if not self._mirror_orders:
            return None
        params: dict[str, Any] = {}
        if self.settings.exchange.market_type == "swap":
            params["reduceOnly"] = True
        return self._place_order(
            position.exchange_symbol,
            "sell" if position.side == "long" else "buy",
            position.size,
            params,
        )

    def _place_order(
        self, symbol: str, side: str, amount: float, params: Mapping[str, Any]
    ) -> str | None:
        """
        Best-effort market order against the testnet matching engine.

        Failures are logged and swallowed: the authoritative P&L is the internal
        simulation, and a testnet outage must not desynchronise it.
        """
        if self.fetcher is None:
            return None
        try:
            order = self.fetcher.exchange.create_order(
                symbol, "market", side, amount, None, dict(params)
            )
        except (ccxt.InsufficientFunds, ccxt.InvalidOrder) as exc:
            LOGGER.error("testnet rejected %s %s %s: %s", side, amount, symbol, exc)
            return None
        except ccxt.AuthenticationError as exc:
            LOGGER.error("testnet authentication failed (%s); disabling order mirror", exc)
            self._mirror_orders = False
            return None
        except (ccxt.NetworkError, ccxt.ExchangeError) as exc:
            LOGGER.warning("testnet order mirror failed for %s: %s", symbol, exc)
            return None
        except Exception:  # noqa: BLE001
            LOGGER.exception("unexpected error mirroring an order for %s", symbol)
            return None

        order_id = str(order.get("id") or "") or None
        LOGGER.info("testnet order %s accepted (%s %s %s)", order_id, side, amount, symbol)
        return order_id

    # ---------------------------------------------------------------------------------
    # Reporting helper
    # ---------------------------------------------------------------------------------

    def open_positions_table(self, prices: Mapping[str, float]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for position in self.positions.values():
            mark = safe_float(prices.get(position.symbol), 0.0) or position.entry_price
            rows.append(
                {
                    "symbol": position.symbol,
                    "side": position.side,
                    "size": position.size,
                    "entry": position.entry_price,
                    "mark": mark,
                    "notional": position.notional,
                    "stop_loss": position.stop_loss,
                    "take_profit": position.take_profit,
                    "unrealized_pnl": position.gross_pnl(mark),
                    "unrealized_pct": position.unrealized_pct(mark),
                    "opened_at": to_iso(position.opened_at),
                }
            )
        return rows


__all__ = [
    "ExecutionError",
    "Fill",
    "PortfolioState",
    "Position",
    "TestnetExecutor",
    "TradeResult",
]
