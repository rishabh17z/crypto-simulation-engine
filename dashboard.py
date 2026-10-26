"""
Read-only web dashboard for the live 30-day experiment.

    python dashboard.py            # http://127.0.0.1:8050
    python dashboard.py --port 9000

The SQLite file is opened with ``mode=ro`` so the dashboard can never write to (or
lock) the database the engine is using. Open positions are marked with live public
tickers from the configured exchange; if that fetch fails, entry prices are shown.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import ccxt

from config.settings import SETTINGS

DB_PATH = Path(SETTINGS.storage.db_path)
TICKER_TTL_SECONDS = 60
CYCLES_PER_DAY = 24 / SETTINGS.schedule.cycle_hours


# --------------------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------------------


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def _rows(conn: sqlite3.Connection, sql: str, *params: Any) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


class TickerCache:
    """Live last prices for open positions, cached so page refreshes stay cheap."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._prices: dict[str, float] = {}
        self._fetched_at = 0.0
        self._exchange: ccxt.Exchange | None = None

    def prices(self, assets: dict[str, list[str]]) -> tuple[dict[str, float], str | None]:
        """Last price per asset; each asset lists candidate symbols (perp, then spot)."""
        with self._lock:
            fresh = time.time() - self._fetched_at < TICKER_TTL_SECONDS
            if fresh and all(a in self._prices for a in assets):
                return self._prices, None
            try:
                if self._exchange is None:
                    klass = getattr(ccxt, SETTINGS.exchange.exchange_id)
                    self._exchange = klass({"enableRateLimit": True, "timeout": 10_000})
                    self._exchange.load_markets()
                markets = self._exchange.markets or {}
                # Mirror the engine: use the first candidate the exchange lists.
                resolved = {
                    asset: next((c for c in candidates if c in markets), None)
                    for asset, candidates in assets.items()
                }
                # Some venues (KuCoin) refuse mixed spot/swap batches: one call per type.
                by_type: dict[str, set[str]] = {}
                for sym in filter(None, resolved.values()):
                    by_type.setdefault(markets[sym]["type"], set()).add(sym)
                tickers: dict[str, Any] = {}
                for group in by_type.values():
                    tickers.update(self._exchange.fetch_tickers(sorted(group)))
                self._prices = {
                    asset: float(t.get("last") or t.get("close") or 0)
                    for asset, sym in resolved.items()
                    if sym and (t := tickers.get(sym))
                }
                self._fetched_at = time.time()
                return self._prices, None
            except Exception as exc:  # noqa: BLE001 - fall back to entry prices
                return self._prices, f"{type(exc).__name__}: {exc}"


TICKERS = TickerCache()


def collect() -> dict[str, Any]:
    with _connect() as conn:
        state = {r["key"]: r["value"] for r in _rows(conn, "SELECT key, value FROM engine_state")}
        equity = _rows(
            conn,
            "SELECT timestamp, equity, cash, unrealized_pnl, realized_pnl_cum, fees_cum, "
            "open_positions, exposure_pct FROM equity_snapshots ORDER BY timestamp",
        )
        open_trades = _rows(conn, "SELECT * FROM trades WHERE status = 'OPEN' ORDER BY timestamp")
        closed = _rows(
            conn,
            "SELECT asset, side, timestamp, closed_at, entry_price, exit_price, notional, "
            "realized_pnl, return_pct, exit_reason FROM trades WHERE status = 'CLOSED' "
            "ORDER BY closed_at DESC",
        )
        cycles = _rows(
            conn,
            "SELECT cycle_uid, started_at, status, orders_opened, orders_closed, error "
            "FROM cycles ORDER BY started_at DESC LIMIT 1",
        )
        last_cycle = cycles[0] if cycles else None
        decisions = (
            _rows(
                conn,
                "SELECT asset, action, confidence, score, price, rsi, executed, skip_reason "
                "FROM decisions WHERE cycle_uid = ? ORDER BY confidence DESC",
                last_cycle["cycle_uid"],
            )
            if last_cycle
            else []
        )
        cycle_total = conn.execute("SELECT COUNT(*) FROM cycles").fetchone()[0]
        error_total = conn.execute(
            "SELECT COUNT(*) FROM cycles WHERE status = 'ERROR'"
        ).fetchone()[0]

    # Live marks for open positions.
    candidates = {}
    for t in open_trades:
        meta = json.loads(t.get("metadata") or "{}")
        candidates[t["asset"]] = [c for c in (meta.get("exchange_symbol"), t["asset"]) if c]
    prices, ticker_error = TICKERS.prices(candidates)
    positions = []
    for t in open_trades:
        mark = prices.get(t["asset"]) or t["entry_price"]
        sign = 1 if t["side"] == "long" else -1
        pnl = sign * (mark - t["entry_price"]) * t["position_size"]
        positions.append(
            {
                "asset": t["asset"],
                "side": t["side"],
                "opened": t["timestamp"],
                "entry": t["entry_price"],
                "mark": mark,
                "stop_loss": t["stop_loss"],
                "take_profit": t["take_profit"],
                "notional": t["notional"],
                "pnl": pnl,
                "pnl_pct": pnl / t["notional"] if t["notional"] else 0.0,
                "confidence": t["confidence"],
                "live": t["asset"] in prices,
            }
        )

    corpus = SETTINGS.risk.initial_corpus
    start = state.get("experiment_started_at")
    end = state.get("experiment_ends_at")
    now = datetime.now(timezone.utc)
    days_total = SETTINGS.schedule.experiment_days
    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "corpus": corpus,
        "quote": SETTINGS.exchange.quote_currency,
        "status": state.get("experiment_status", "NOT STARTED"),
        "started_at": start,
        "ends_at": end,
        "cycles_done": cycle_total,
        "cycles_errors": error_total,
        "cycles_expected": int(round(days_total * CYCLES_PER_DAY)),
        "last_cycle": last_cycle,
        "equity": equity,
        "positions": positions,
        "closed": closed,
        "decisions": decisions,
        "ticker_error": ticker_error,
        "max_positions": SETTINGS.risk.max_open_positions,
        "cycle_hours": SETTINGS.schedule.cycle_hours,
    }


# --------------------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        if self.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", PAGE.encode())
        elif self.path == "/api/data":
            try:
                body = json.dumps(collect()).encode()
                self._send(200, "application/json", body)
            except Exception as exc:  # noqa: BLE001 - surface to the page
                self._send(500, "application/json", json.dumps({"error": str(exc)}).encode())
        else:
            self._send(404, "text/plain", b"not found")

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: Any) -> None:  # keep the terminal quiet
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="read-only experiment dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()
    if not DB_PATH.exists():
        raise SystemExit(f"database not found: {DB_PATH}")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"dashboard: http://{args.host}:{args.port}  (read-only, Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Experiment Dashboard</title>
<style>
:root {
  color-scheme: light;
  --surface-0: #f4f3f0; --surface-1: #fcfcfb; --border: #e3e2de;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #7a7974;
  --grid: #ebeae6; --series-1: #2a78d6; --baseline: #9c9b96;
  --good: #0ca30c; --good-text: #006300; --critical: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface-0: #111110; --surface-1: #1a1a19; --border: #2e2e2c;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8f8e86;
    --grid: #2a2a28; --series-1: #3987e5; --baseline: #6b6a64;
    --good-text: #4cc34c; --critical: #e66767;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #111110; --surface-1: #1a1a19; --border: #2e2e2c;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8f8e86;
  --grid: #2a2a28; --series-1: #3987e5; --baseline: #6b6a64;
  --good-text: #4cc34c; --critical: #e66767;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface-0); color: var(--text-primary);
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 24px 16px 48px; }
header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px 16px; margin-bottom: 16px; }
h1 { font-size: 20px; margin: 0; }
h2 { font-size: 15px; margin: 0 0 12px; }
.meta { color: var(--text-secondary); font-size: 13px; }
.pill { display: inline-flex; align-items: center; gap: 6px; padding: 2px 10px; border-radius: 999px;
  border: 1px solid var(--border); font-size: 12px; color: var(--text-secondary); }
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--good); }
.dot.off { background: var(--critical); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(150px, 100%), 1fr)); gap: 12px; margin-bottom: 16px; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }
.tile .label { color: var(--text-secondary); font-size: 12px; }
.tile .value { font-size: 24px; font-weight: 600; margin-top: 2px; font-variant-numeric: tabular-nums; }
.tile .sub { color: var(--text-muted); font-size: 12px; margin-top: 2px; }
.bar { height: 6px; border-radius: 3px; background: var(--grid); margin-top: 8px; overflow: hidden; }
.bar > span { display: block; height: 100%; background: var(--series-1); border-radius: 3px; }
.up { color: var(--good-text); } .down { color: var(--critical); }
section { margin-bottom: 16px; }
#chart { position: relative; }
#chart svg { display: block; width: 100%; height: 280px; }
.tip { position: absolute; pointer-events: none; background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 8px; padding: 8px 10px; font-size: 12px; box-shadow: 0 4px 14px rgba(0,0,0,.12);
  white-space: nowrap; display: none; font-variant-numeric: tabular-nums; }
.tip b { font-size: 13px; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }
th:first-child, td:first-child, th.l, td.l { text-align: left; }
th { color: var(--text-secondary); font-weight: 500; font-size: 12px; }
td.muted, .muted { color: var(--text-muted); }
.empty { color: var(--text-muted); padding: 12px 0; }
details summary { cursor: pointer; color: var(--text-secondary); font-size: 13px; margin-top: 8px; }
.err { background: var(--surface-1); border: 1px solid var(--critical); color: var(--critical);
  border-radius: 8px; padding: 10px 12px; margin-bottom: 16px; display: none; }
</style>
</head>
<body>
<main>
  <header>
    <h1>30-day paper-trading experiment</h1>
    <span class="pill"><span class="dot" id="statusDot"></span><span id="status">loading</span></span>
    <span class="meta" id="window"></span>
  </header>
  <div class="err" id="err"></div>
  <div class="tiles" id="tiles"></div>

  <section class="card">
    <h2>Equity</h2>
    <div id="chart"><svg id="svg" role="img" aria-label="Equity over time"></svg><div class="tip" id="tip"></div></div>
    <details><summary>Show equity as a table</summary><div class="table-wrap" id="equityTable"></div></details>
  </section>

  <section class="card">
    <h2 id="posTitle">Open positions</h2>
    <div class="table-wrap" id="positions"></div>
    <div class="meta" id="tickerNote" style="margin-top:8px"></div>
  </section>

  <section class="card">
    <h2 id="decTitle">Latest cycle decisions</h2>
    <div class="table-wrap" id="decisions"></div>
  </section>

  <section class="card">
    <h2>Closed trades</h2>
    <div class="table-wrap" id="closed"></div>
  </section>
  <div class="meta" id="updated"></div>
</main>
<script>
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fmtMoney = (v, d = 2) => Number(v).toLocaleString(undefined, {minimumFractionDigits: d, maximumFractionDigits: d});
const fmtSigned = (v, d = 2) => (v >= 0 ? "+" : "−") + fmtMoney(Math.abs(v), d);
const fmtPct = (v, d = 2) => (v >= 0 ? "+" : "−") + Math.abs(v * 100).toFixed(d) + "%";
const fmtPrice = (v) => { v = Number(v); const d = v >= 1000 ? 2 : v >= 1 ? 4 : 6; return v.toFixed(d); };
const cls = (v) => v > 0 ? "up" : v < 0 ? "down" : "";
const arrow = (v) => v > 0 ? "▲ " : v < 0 ? "▼ " : "";
const fmtTime = (iso) => iso ? new Date(iso).toLocaleString(undefined, {month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"}) : "—";
function human(ms) {
  if (ms <= 0) return "0h";
  const d = Math.floor(ms / 864e5), h = Math.floor(ms % 864e5 / 36e5), m = Math.floor(ms % 36e5 / 6e4);
  return d ? `${d}d ${h}h` : h ? `${h}h ${m}m` : `${m}m`;
}
function table(cols, rows, empty) {
  if (!rows.length) return `<div class="empty">${empty}</div>`;
  return `<table><thead><tr>${cols.map(c => `<th class="${c.l ? "l" : ""}">${c.h}</th>`).join("")}</tr></thead><tbody>` +
    rows.map(r => `<tr>${cols.map(c => `<td class="${c.l ? "l" : ""} ${c.cls ? c.cls(r) : ""}">${c.f(r)}</td>`).join("")}</tr>`).join("") +
    `</tbody></table>`;
}

function renderTiles(d) {
  const last = d.equity[d.equity.length - 1];
  const eq = last ? last.equity : d.corpus;
  const pnl = eq - d.corpus;
  const start = d.started_at ? new Date(d.started_at) : null, end = d.ends_at ? new Date(d.ends_at) : null;
  const now = new Date();
  const elapsed = start && end ? Math.min(1, Math.max(0, (now - start) / (end - start))) : 0;
  const realized = last ? last.realized_pnl_cum : 0, unreal = last ? last.unrealized_pnl : 0;
  const livePnl = d.positions.reduce((a, p) => a + p.pnl, 0);
  const liveEq = last ? last.equity - last.unrealized_pnl + livePnl : d.corpus;
  const nextCycle = d.last_cycle ? new Date(new Date(d.last_cycle.started_at).getTime() + d.cycle_hours * 36e5) : null;
  const tiles = [
    {label: "Equity · last cycle", value: `${fmtMoney(eq)}`,
     sub: `${d.quote} · live estimate ${fmtMoney(liveEq)}${d.ticker_error ? " *" : ""}`},
    {label: "Total P&L", value: `<span class="${cls(pnl)}">${arrow(pnl)}${fmtSigned(pnl)}</span>`,
     sub: `${fmtPct(pnl / d.corpus)} · realized ${fmtSigned(realized)} · open ${fmtSigned(unreal)} (live ${fmtSigned(livePnl)})`},
    {label: "Cycles", value: `${d.cycles_done} <span class="muted" style="font-size:15px">/ ~${d.cycles_expected}</span>`,
     sub: d.cycles_errors ? `<span class="down">${d.cycles_errors} with errors</span>` : "no errors", bar: d.cycles_done / d.cycles_expected},
    {label: "Time remaining", value: end ? human(end - now) : "—", sub: `${(elapsed * 100).toFixed(0)}% of 30 days elapsed`, bar: elapsed},
    {label: "Exposure", value: last ? `${(last.exposure_pct * 100).toFixed(1)}%` : "—",
     sub: `${last ? last.open_positions : 0} / ${d.max_positions} positions · fees ${fmtMoney(last ? last.fees_cum : 0)}`},
    {label: "Next cycle", value: nextCycle ? (nextCycle > now ? human(nextCycle - now) : "due") : "—",
     sub: nextCycle ? `at ${fmtTime(nextCycle.toISOString())}` : ""},
  ];
  $("tiles").innerHTML = tiles.map(t => `<div class="card tile"><div class="label">${t.label}</div><div class="value">${t.value}</div>` +
    `<div class="sub">${t.sub}</div>${t.bar !== undefined ? `<div class="bar"><span style="width:${(Math.min(1, t.bar) * 100).toFixed(1)}%"></span></div>` : ""}</div>`).join("");
}

function renderChart(d) {
  const svg = $("svg"), tip = $("tip");
  const pts = d.equity.map(e => ({t: new Date(e.timestamp).getTime(), v: e.equity, e}));
  if (d.started_at) pts.unshift({t: new Date(d.started_at).getTime(), v: d.corpus, e: null});
  const W = svg.clientWidth || 800, H = 280, m = {l: 64, r: 16, t: 12, b: 28};
  if (pts.length < 2) { svg.innerHTML = `<text x="${W/2}" y="${H/2}" text-anchor="middle" fill="var(--text-muted)">Waiting for the first cycles…</text>`; return; }
  const t0 = pts[0].t, t1 = Math.max(pts[pts.length - 1].t, d.ends_at ? Math.min(new Date(d.ends_at).getTime(), pts[pts.length - 1].t + 864e5) : 0);
  let lo = Math.min(d.corpus, ...pts.map(p => p.v)), hi = Math.max(d.corpus, ...pts.map(p => p.v));
  const pad = Math.max((hi - lo) * 0.1, d.corpus * 0.001);
  const rawStep = (hi - lo + 2 * pad) / 4, mag = 10 ** Math.floor(Math.log10(rawStep));
  const step = [1, 2, 2.5, 5, 10].map(k => k * mag).find(k => k >= rawStep);
  lo = Math.floor((lo - pad) / step) * step; hi = Math.ceil((hi + pad) / step) * step;
  const x = t => m.l + (t - t0) / (t1 - t0) * (W - m.l - m.r);
  const y = v => m.t + (hi - v) / (hi - lo) * (H - m.t - m.b);
  let g = "";
  for (let v = lo; v <= hi + step / 2; v += step) {
    g += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)"/>` +
         `<text x="${m.l - 8}" y="${y(v) + 4}" text-anchor="end" font-size="11" fill="var(--text-muted)">${fmtMoney(v, 0)}</text>`;
  }
  const days = (t1 - t0) / 864e5, dayStep = days > 10 ? 5 : days > 4 ? 2 : 1;
  const firstDay = new Date(t0); firstDay.setHours(24, 0, 0, 0);
  for (let t = firstDay.getTime(), i = 0; t < t1; t += 864e5, i++) {
    if (i % dayStep) continue;
    g += `<text x="${x(t)}" y="${H - 8}" text-anchor="middle" font-size="11" fill="var(--text-muted)">${new Date(t).toLocaleDateString(undefined, {month: "short", day: "numeric"})}</text>`;
  }
  g += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(d.corpus)}" y2="${y(d.corpus)}" stroke="var(--baseline)" stroke-dasharray="4 4"/>` +
       `<text x="${W - m.r}" y="${y(d.corpus) - 6}" text-anchor="end" font-size="11" fill="var(--text-secondary)">start ${fmtMoney(d.corpus, 0)}</text>`;
  const path = pts.map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)},${y(p.v).toFixed(1)}`).join("");
  const lastP = pts[pts.length - 1];
  g += `<path d="${path}" fill="none" stroke="var(--series-1)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>` +
       `<circle cx="${x(lastP.t)}" cy="${y(lastP.v)}" r="4" fill="var(--series-1)" stroke="var(--surface-1)" stroke-width="2"/>` +
       `<line id="xh" y1="${m.t}" y2="${H - m.b}" stroke="var(--text-muted)" stroke-width="1" visibility="hidden"/>` +
       `<circle id="xhDot" r="5" fill="var(--series-1)" stroke="var(--surface-1)" stroke-width="2" visibility="hidden"/>` +
       `<rect x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent" id="hit"/>`;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML = g;
  const xh = $("xh"), dot = $("xhDot");
  const hide = () => { tip.style.display = "none"; xh.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden"); };
  $("hit").addEventListener("mousemove", (ev) => {
    const r = svg.getBoundingClientRect(), mx = (ev.clientX - r.left) * W / r.width;
    let best = pts[0]; for (const p of pts) if (Math.abs(x(p.t) - mx) < Math.abs(x(best.t) - mx)) best = p;
    const px = x(best.t), py = y(best.v), diff = best.v - d.corpus;
    xh.setAttribute("x1", px); xh.setAttribute("x2", px); xh.setAttribute("visibility", "visible");
    dot.setAttribute("cx", px); dot.setAttribute("cy", py); dot.setAttribute("visibility", "visible");
    tip.innerHTML = `<div class="muted">${fmtTime(new Date(best.t).toISOString())}</div><b>${fmtMoney(best.v)} ${d.quote}</b>` +
      `<div class="${cls(diff)}">${arrow(diff)}${fmtSigned(diff)} (${fmtPct(diff / d.corpus)})</div>` +
      (best.e ? `<div class="muted">${best.e.open_positions} open · exposure ${(best.e.exposure_pct * 100).toFixed(1)}%</div>` : `<div class="muted">experiment start</div>`);
    tip.style.display = "block";
    const cw = $("chart").clientWidth, left = px * r.width / W;
    tip.style.left = Math.min(Math.max(0, left + 12), cw - tip.offsetWidth) + "px";
    tip.style.top = Math.max(0, py * r.height / H - 70) + "px";
  });
  $("hit").addEventListener("mouseleave", hide);
  $("equityTable").innerHTML = table([
    {h: "Time", l: true, f: r => fmtTime(r.timestamp)},
    {h: "Equity", f: r => fmtMoney(r.equity)},
    {h: "vs start", cls: r => cls(r.equity - d.corpus), f: r => fmtSigned(r.equity - d.corpus)},
    {h: "Open P&L", f: r => fmtSigned(r.unrealized_pnl)},
    {h: "Realized", f: r => fmtSigned(r.realized_pnl_cum)},
    {h: "Positions", f: r => r.open_positions},
  ], [...d.equity].reverse(), "No equity snapshots yet.");
}

function render(d) {
  $("err").style.display = "none";
  const running = d.status === "RUNNING";
  $("status").textContent = d.status.toLowerCase();
  $("statusDot").className = "dot" + (running ? "" : " off");
  $("window").textContent = d.started_at ? `${fmtTime(d.started_at)} → ${fmtTime(d.ends_at)}` : "";
  renderTiles(d);
  renderChart(d);
  $("posTitle").textContent = `Open positions (${d.positions.length} / ${d.max_positions})`;
  $("positions").innerHTML = table([
    {h: "Asset", l: true, f: r => esc(r.asset)},
    {h: "Side", l: true, f: r => r.side},
    {h: "Opened", l: true, f: r => fmtTime(r.opened)},
    {h: "Entry", f: r => fmtPrice(r.entry)},
    {h: "Mark", f: r => fmtPrice(r.mark) + (r.live ? "" : " *")},
    {h: "Stop", cls: () => "muted", f: r => fmtPrice(r.stop_loss)},
    {h: "Target", cls: () => "muted", f: r => fmtPrice(r.take_profit)},
    {h: "Size", f: r => fmtMoney(r.notional)},
    {h: "Open P&L", cls: r => cls(r.pnl), f: r => `${arrow(r.pnl)}${fmtSigned(r.pnl)}`},
    {h: "%", cls: r => cls(r.pnl), f: r => fmtPct(r.pnl_pct)},
  ], d.positions, "No open positions.");
  $("tickerNote").textContent = d.ticker_error
    ? `* Live prices unavailable (${d.ticker_error}); showing entry prices.`
    : "Marks are live last prices; open P&L is before exit fees and slippage.";
  $("decTitle").textContent = d.last_cycle ? `Latest cycle decisions · ${fmtTime(d.last_cycle.started_at)}` : "Latest cycle decisions";
  $("decisions").innerHTML = table([
    {h: "Asset", l: true, f: r => esc(r.asset)},
    {h: "Signal", l: true, f: r => r.action},
    {h: "Confidence", f: r => Number(r.confidence).toFixed(2)},
    {h: "Score", f: r => r.score == null ? "—" : fmtSigned(r.score, 3)},
    {h: "RSI", f: r => r.rsi == null ? "—" : Number(r.rsi).toFixed(1)},
    {h: "Price", f: r => r.price == null ? "—" : fmtPrice(r.price)},
    {h: "Outcome", l: true, cls: r => r.executed ? "up" : "muted", f: r => r.executed ? "✓ opened" : esc(r.skip_reason || "—")},
  ], d.decisions, "Held assets are checked for exits only, so no new-entry decisions were logged.");
  $("closed").innerHTML = table([
    {h: "Asset", l: true, f: r => esc(r.asset)},
    {h: "Side", l: true, f: r => r.side},
    {h: "Opened", l: true, f: r => fmtTime(r.timestamp)},
    {h: "Closed", l: true, f: r => fmtTime(r.closed_at)},
    {h: "Entry", f: r => fmtPrice(r.entry_price)},
    {h: "Exit", f: r => fmtPrice(r.exit_price)},
    {h: "P&L", cls: r => cls(r.realized_pnl), f: r => `${arrow(r.realized_pnl)}${fmtSigned(r.realized_pnl)}`},
    {h: "Return", cls: r => cls(r.return_pct), f: r => r.return_pct == null ? "—" : fmtPct(r.return_pct)},
    {h: "Reason", l: true, cls: () => "muted", f: r => esc((r.exit_reason || "").replace(/_/g, " ").toLowerCase())},
  ], d.closed, "No closed trades yet.");
  $("updated").textContent = `Updated ${new Date(d.generated_at).toLocaleTimeString()} · refreshes every 60s · read-only`;
}

let last = null;
async function load() {
  try {
    const res = await fetch("/api/data", {cache: "no-store"});
    const d = await res.json();
    if (!res.ok || d.error) throw new Error(d.error || res.statusText);
    last = d; render(d);
  } catch (e) {
    $("err").textContent = `Could not load data: ${e.message}`;
    $("err").style.display = "block";
  }
}
window.addEventListener("resize", () => last && renderChart(last));
load(); setInterval(load, 60000);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
