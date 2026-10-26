# crypto-simulation-engine

A modular Python engine for a **30-day crypto paper-trading experiment** via CCXT
(Bybit Testnet by default, or live public data from any CCXT exchange such as KuCoin).
It fetches OHLCV and sentiment every 4 hours, scores each instrument,
opens risk-bounded simulated positions with mandatory stop-loss / take-profit, charges
realistic fees and slippage, logs everything to SQLite, and liquidates the book with a
full P&L report the moment day 30 is reached.

> Simulated results only. Fees and slippage are *modelled*; testnet liquidity does not
> reflect production markets. Nothing here is investment advice.

**Highlights**

- 4-hourly cycle: RSI / EMA50 / EMA200 / ATR + Fear & Greed sentiment -> composite score
- Long and short simulated positions, mandatory stop-loss / take-profit, fees + slippage
- Capped book (max positions, exposure); entries ranked strongest-signal-first
- Restart-safe: state lives in SQLite, a restart resumes the same 30-day window
- Read-only web dashboard with equity curve, live positions and per-cycle decisions

---

## Layout

```
crypto-simulation-engine/
├── config/
│   ├── settings.py              # all tunables; env-driven, validated on import
│   └── assets.yaml              # the tradable universe
├── src/
│   ├── common.py                # logging, retry/backoff, UTC + numeric helpers
│   ├── data_ingestion/
│   │   ├── price_fetcher.py     # CCXT OHLCV + RSI(14), EMA(50/200), ATR(14)
│   │   └── sentiment_scraper.py # Fear & Greed + CryptoPanic -> score in [-1, 1]
│   ├── engine/
│   │   ├── decision_logic.py    # BUY/SELL/HOLD + TP/SL (heuristic or LLM prompt)
│   │   └── testnet_executor.py  # sizing, simulated fills, fees/slippage, testnet orders
│   ├── storage/
│   │   └── db_manager.py        # SQLite schema, WAL, concurrent-write handling
│   └── orchestrator.py          # 4h scheduler, 30-day timer, liquidation + report
├── tests/test_engine.py         # offline regression tests (no network)
├── dashboard.py                 # read-only web dashboard (stdlib HTTP server)
├── requirements.txt
└── main.py                      # CLI entry point
```

The tests cover the paths a live run cannot trigger on demand (stop-outs, shorts,
restart recovery).

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # optional: Bybit TESTNET keys, or EXCHANGE_ID=kucoin USE_SANDBOX=false
python main.py doctor       # verify config, exchange, indicators, sentiment, DB
python main.py once         # run a single cycle end to end
python main.py run          # start the 30-day experiment
```

Keys are optional: without them the engine uses public testnet market data and runs as a
pure simulation. They are required only to mirror orders onto the testnet book
(`PLACE_TESTNET_ORDERS=true`).

### CLI

| Command | What it does |
| --- | --- |
| `run` | Starts the blocking scheduler: a cycle every `CYCLE_HOURS`, a deadline check every 15 min, auto-liquidation and report at day 30. |
| `once` | One cycle, then exit. Ideal for cron, CI or a first look. |
| `status` | Experiment clock, equity, cash, exposure and every open position. |
| `report [--save]` | Rebuilds the P&L report from SQLite; `--save` writes JSON + Markdown to `reports/`. |
| `liquidate [--yes]` | Flattens the book and finalises the experiment early. |
| `doctor` | Pre-flight: database, market reachability, indicator warm-up, sentiment sources. |
| `reset [--yes]` | **Destructive.** Wipes all trades, decisions and equity history. |

`run` is restart-safe. The experiment window, cash balance and open positions live in
SQLite, so stopping and restarting resumes mid-experiment — and **does not extend the
30-day deadline**.

---

## Dashboard

```bash
python dashboard.py            # http://127.0.0.1:8050, refreshes every 60s
```

A read-only web view of the running experiment: equity curve, P&L, cycle count and
time remaining, open positions marked with live public tickers, the latest cycle's
decisions (including skipped signals) and closed trades. It opens SQLite with
`mode=ro`, so it can run alongside the engine without writing to or locking the DB.

## How a cycle works

1. **Market data** — OHLCV per asset; the in-progress candle is discarded so indicators
   only ever see closed bars. RSI(14) and ATR(14) use Wilder smoothing; EMA 50/200 use
   the standard recursive form. A single asset's failure degrades that asset only.
2. **Protective sweep** — every open position is checked against the last closed
   candle's high/low *and* the live price. If one bar touched both the stop and the
   target, `conservative_fills` resolves it against the position. A candle that closed
   *before* the position opened can never trigger an exit.
3. **Sentiment** — Fear & Greed (market-wide) and CryptoPanic (per currency, recency-
   weighted with a 24h half-life) are combined into `[-1, 1]`. Weights renormalise over
   whatever answered. A dead source is negative-cached for
   `SENTIMENT_FAILURE_COOLDOWN`, so an outage costs one retry ladder per cycle rather
   than one per asset.
4. **Decision** — composite score in `[-1, 1]`:

   ```
   score = 0.45 * trend + 0.25 * momentum + 0.30 * sentiment
   trend    = 0.70 * tanh(ema_spread / 2%) + 0.30 * tanh(price_vs_ema50 / 3%)
   momentum = clip((RSI - 50) / 25, -1, 1)
   ```

   `score >= +0.30` → BUY, `score <= -0.30` → SELL, otherwise HOLD. Four vetoes force
   HOLD: ATR% above the volatility circuit breaker, RSI exhaustion (never chase a
   vertical move), a confidence floor, and shorts-disabled.

   **Missing sentiment is not neutral sentiment.** A score of 0.0 from a source that
   answered means "the market is balanced" and keeps its 30% weight. No source
   answering means "we know nothing", and that weight is redistributed over trend and
   momentum (0.643 / 0.357). Treating absence as neutrality would cap the attainable
   score at ±0.70 against an unchanged ±0.30 threshold — a silent 30% handicap that
   makes the engine look inert during an outage instead of trading the signals it still
   has. Availability is read from `sources`, never from the score, and the rationale
   recorded in `decisions` says which mode applied.
5. **Execution** — sized, gated and filled (below).
6. **Equity mark** — an `equity_snapshots` row per cycle, which is what the drawdown and
   Sharpe figures are computed from.

### Risk controls

All enforced in `testnet_executor.py`, never in the strategy:

- **Per-trade allocation** capped at `MAX_RISK_PER_TRADE` (default 5%) of *current*
  equity, scaled by the asset's `weight`.
- **Two sizing modes.** `notional` (default) allocates exactly 5% of equity as position
  notional — the literal reading of the risk cap. `risk` instead sizes so a stop-out
  costs 5% of equity, then re-applies the exposure cap.
- **Mandatory protective levels.** An order without *both* a stop-loss and a take-profit
  is rejected; `require_stop_loss` cannot be configured off. Levels are re-derived from
  the actual fill so the intended risk distance is what the position really carries, and
  their direction is re-validated before the order goes through.
- **Aggregate exposure** capped at `MAX_PORTFOLIO_EXPOSURE` (60%), plus
  `MAX_OPEN_POSITIONS` and one position per asset (no pyramiding).
- **Equity circuit breaker** — new entries stop below `EQUITY_STOP` × initial corpus.
- **Minimum notional** — undersized orders are skipped, never truncated to dust.
- Validation at import time rejects an incoherent config: `min_stop_pct` must exceed
  round-trip costs, the take-profit multiple must exceed the stop multiple, shorts
  require a swap market, and `ohlcv_limit` must cover the EMA-200 warm-up.

### Simulated fills

Cash is free collateral at 1x notional — no leverage, so a position cannot lose more
than its notional.

```
open    cash -= notional + entry_fee
close   cash += notional + gross_pnl - exit_fee
equity  = cash + SUM(position marked value)
```

Every fill crosses the spread by `SLIPPAGE_RATE` adversely and pays `FEE_RATE` on both
sides, so reported P&L is net of costs by construction. A flat round trip is a
guaranteed small loss — which is exactly the point of including costs.

### The 30-day boundary

The window is stamped into `engine_state` on first run. A deadline job runs every 15
minutes independently of the trading cadence, so day 30 is honoured within minutes
rather than at the next 4-hour tick. `finalize()` liquidates everything at market, marks
the experiment `COMPLETED`, writes `reports/final_report_<ts>.{json,md}` plus
`reports/latest_report.md`, and stops the scheduler.

---

## Edge cases handled

| Failure | Behaviour |
| --- | --- |
| Exchange timeout / 5xx / `DDoSProtection` | Jittered exponential backoff (`EXCHANGE_MAX_RETRIES`, capped at 45s); CCXT's own rate limiter stays on. |
| Rate limit (429) | Retried with backoff; CryptoPanic's `Retry-After` is honoured. |
| Bad symbol / auth error | Raised immediately — retrying cannot fix it. |
| One asset's data missing | That asset is skipped; the cycle continues and open positions still get a mark price from the ticker. |
| *All* market data missing | Equity is still snapshotted; no orders are placed. |
| Sentiment source down / firewalled | One retry ladder, then negative-cached for `SENTIMENT_FAILURE_COOLDOWN` (default 600s) so the cost is not paid per asset. Scoring renormalises over trend + momentum, and `doctor` reports a warning rather than a clean pass. |
| CryptoPanic token rejected (401/403) | That source is disabled for the rest of the run; no further attempts. |
| SQLite `database is locked` | WAL journal + `busy_timeout` + a bounded retry loop; every statement is serialised through one lock, and APScheduler jobs use `max_instances=1`. |
| Process restart | Cash, open positions and the deadline are restored from SQLite. |
| Unhandled error inside a cycle | Recorded on the `cycles` row with status `ERROR`; the loop survives. |
| Exchange unreachable at liquidation | Falls back to the ticker, then to entry price, with a loud log that P&L is understated. |
| Double-close of a trade | Guarded by `status = 'OPEN'` in the `UPDATE`; returns `False`. |
| Degenerate stats | Sharpe is `null` below 8 observations; profit factor is `null` with no losses (never `inf`, which is not valid JSON). |

---

## Data model

`data/simulation.sqlite3`, WAL mode:

- **`trades`** — one row per position: `timestamp`, `asset`, `action`, `side`,
  `entry_price`, `exit_price`, `position_size`, `notional`, `stop_loss`, `take_profit`,
  `fees`, `slippage_cost`, `realized_pnl`, `gross_pnl`, `return_pct`, `status`,
  `exit_reason`, the indicator and sentiment values at entry, the rationale, and the
  testnet order id. Written once on entry, updated once on exit.
- **`decisions`** — every decision including HOLD, with the score, its components and
  the skip reason. This is the audit trail that explains what the engine *didn't* do.
- **`equity_snapshots`** — the equity curve: equity, cash, unrealised, cumulative
  realised, cumulative fees, exposure.
- **`cycles`** — per-cycle outcome, counts and any error.
- **`engine_state`** — experiment window, status, cash balance, cycle count.

Useful queries:

```sql
SELECT asset, action, side, entry_price, exit_price, realized_pnl, exit_reason
FROM trades WHERE status='CLOSED' ORDER BY closed_at DESC;

SELECT timestamp, equity, exposure_pct FROM equity_snapshots ORDER BY id;

SELECT asset, action, score, skip_reason FROM decisions
WHERE executed = 0 ORDER BY id DESC LIMIT 20;
```

---

## Configuration

Everything is an environment variable (see `.env.example`); `config/settings.py` holds
the defaults and the validation. The notable ones:

| Variable | Default | Meaning |
| --- | --- | --- |
| `INITIAL_CORPUS` | `10000` | Virtual corpus in USDT. |
| `MAX_RISK_PER_TRADE` | `0.05` | Per-trade cap as a fraction of equity. |
| `POSITION_SIZING_MODE` | `notional` | `notional` or `risk` (see above). |
| `MAX_PORTFOLIO_EXPOSURE` | `0.60` | Aggregate open notional cap. |
| `MAX_OPEN_POSITIONS` | `3` | Concurrent position limit. |
| `FEE_RATE` / `SLIPPAGE_RATE` | `0.001` | 0.1% per side, each. |
| `CYCLE_HOURS` / `EXPERIMENT_DAYS` | `4` / `30` | Cadence and hard deadline. |
| `MARKET_TYPE` | `swap` | `swap` (linear perps, shorts possible) or `spot`. |
| `ALLOW_SHORTS` | `true` | Requires `MARKET_TYPE=swap`. |
| `PLACE_TESTNET_ORDERS` | `false` | Mirror simulated fills onto the testnet book. |
| `USE_SANDBOX` | `true` | Setting this false triggers an interactive confirmation. |
| `SENTIMENT_ENABLED` | `true` | `false` skips sentiment HTTP entirely and trades on technicals alone. |
| `SENTIMENT_HTTP_TIMEOUT` | `12` | Per-request timeout; lower it if a source is blocked on your network. |
| `SENTIMENT_MAX_RETRIES` | `3` | Attempts per source before giving up for the cooldown window. |
| `SENTIMENT_FAILURE_COOLDOWN` | `600` | Seconds a failed source is skipped before being retried. |

Edit `config/assets.yaml` to change the universe — `weight` scales an asset's
allocation, `enabled: false` parks it without deleting it. Spot symbols are rewritten to
Bybit perp notation (`BTC/USDT` → `BTC/USDT:USDT`) automatically when `MARKET_TYPE=swap`.

### Using an LLM instead of the heuristic

`decision_logic.py` ships a provider-agnostic adapter. `build_decision_prompt()` renders
the same feature set the heuristic sees into a strict prompt;
`parse_decision_response()` validates the reply and **rejects** any
risk-inconsistent answer — wrong-side levels, a stop outside the configured band, or a
reward:risk below the minimum — rather than repairing it. A model cannot widen the risk
envelope, and anything malformed raises `ValueError` so the caller falls back to the
heuristic.

---

## Tests

```bash
python tests/test_engine.py     # or: pytest tests/
```

34 offline tests, no network: indicator correctness at the extremes, every decision
branch and veto, stop-distance clamping, both fill directions, fee/slippage accounting to
the cent, stop-loss and take-profit triggering from candle ranges, the both-levels-touched
tie-break, stale-candle rejection, short P&L inversion, each risk gate, liquidation,
restart recovery, 8 concurrent SQLite writers, report consistency, and the
sentiment-outage paths (negative caching, cooldown expiry, weight renormalisation,
absent-vs-neutral).

---

## Operational notes

- **Bybit testnet data is thin.** Illiquid pairs can show absurd ATR (15-45% on 4h),
  which trips the volatility circuit breaker and produces HOLD — correct behaviour on bad
  data, but worth knowing before you conclude the strategy is inert. `doctor` prints ATR%
  per asset so you can see it immediately. If every cycle is all-HOLD, check ATR% against
  `max_atr_pct` (12%) first.
- **`api.alternative.me` is blocked on some networks and ISPs.** The symptom is
  `ConnectTimeout`, not a 4xx. The engine degrades correctly, but if it is permanently
  unreachable for you, set `SENTIMENT_ENABLED=false` to stop paying for the attempt.
- **Amount precision truncates downward.** Bybit's BTC perp steps in 0.001, so a 500
  USDT budget at ~85k becomes 0.005 BTC (~425 USDT). Conservative by design.
- The engine is single-process by design: one writer, one scheduler. Run one instance per
  database file.
