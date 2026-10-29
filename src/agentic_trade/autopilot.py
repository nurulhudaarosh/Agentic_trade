"""Budget, guardrails and the autonomous agent loop.

Every order the agent wants to place goes through :func:`Autopilot.gate`, which
sizes it against the user's budget and applies the hard limits (kill switch,
allowed tickers, cash reserve, position cap, single-order cap, daily loss,
drawdown, trades per day). What happens to an order that passes depends on the
permission mode:

* ``advisory`` – never trades; decisions are only logged as suggestions.
* ``approve``  – every trade waits in the approval queue (optionally exits auto-run).
* ``guarded``  – trades run automatically unless they trip a soft check (low
  confidence, large order, LLM / rule disagreement); those wait for approval.
* ``full``     – everything that passes the hard limits runs.

Hard limits apply in every mode and to approvals: an approved proposal is checked
again at the current price before it is sent.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from .study.data import UNIVERSE

ET = ZoneInfo("America/New_York")
PERMISSION_MODES = ("advisory", "approve", "guarded", "full")
DECISION_MODES = ("consensus", "llm", "llm_and_consensus")
SIZING_MODES = ("max_position", "allocation")
ALLOCATION_METHODS = ("inverse_vol", "min_variance", "equal", "signal_tilt")

DEFAULT_BUDGET: dict[str, Any] = {
    "total_budget": 10000.0,      # dollars the agent may manage
    "cash_reserve_pct": 20.0,     # always kept in cash
    "max_position_pct": 25.0,     # per stock, of the budget
    "max_order_usd": 2500.0,      # largest single order
    "max_daily_loss_pct": 2.0,    # of the budget; new buys stop for the day
    "max_drawdown_pct": 10.0,     # of the budget, from the peak; new buys stop
    "max_trades_per_day": 10,
    "allowed_tickers": list(UNIVERSE),
}
DEFAULT_AUTOPILOT: dict[str, Any] = {
    "enabled": False,
    "permission_mode": "approve",
    "decision_mode": "consensus",
    "interval_min": 15,
    "market_hours_only": True,
    "tickers_per_cycle": 2,
    "sizing": "max_position",
    "allocation_method": "inverse_vol",
    "min_confidence": 0.55,
    "auto_max_order_usd": 1000.0,
    "auto_exits": True,
    "proposal_ttl_min": 30,
    "backbone": "gpt-oss-120b",
}
NUMERIC_LIMITS = {
    "total_budget": (100.0, 1e9), "cash_reserve_pct": (0.0, 95.0), "max_position_pct": (1.0, 100.0),
    "max_order_usd": (1.0, 1e9), "max_daily_loss_pct": (0.1, 50.0), "max_drawdown_pct": (0.5, 90.0),
    "max_trades_per_day": (0, 500), "interval_min": (1, 1440), "tickers_per_cycle": (1, len(UNIVERSE)),
    "min_confidence": (0.0, 1.0), "auto_max_order_usd": (0.0, 1e9), "proposal_ttl_min": (1, 1440),
}


class Broker(Protocol):
    def state(self) -> dict[str, Any]: ...
    def price(self, ticker: str) -> float | None: ...
    def place(self, ticker: str, action: str, shares: float, reasoning: str, provider: str, model: str) -> dict[str, Any]: ...
    def decide(self, ticker: str, mode: str, backbone: str) -> dict[str, Any]: ...
    def targets(self, method: str) -> dict[str, float]: ...


def _now() -> datetime:
    return datetime.now(timezone.utc)


def market_open(at: datetime | None = None) -> bool:
    """Regular US session, weekdays 09:30–16:00 ET (exchange holidays are not modelled)."""
    t = (at or _now()).astimezone(ET)
    return t.weekday() < 5 and (9, 30) <= (t.hour, t.minute) < (16, 0)


def _validate(section: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in section.items():
        if key not in defaults:
            raise ValueError(f"unknown setting {key}")
        default = defaults[key]
        if isinstance(default, bool):
            out[key] = bool(value)
        elif isinstance(default, (int, float)):
            try:
                v = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{key} must be a number") from None
            lo, hi = NUMERIC_LIMITS[key]
            if not lo <= v <= hi:
                raise ValueError(f"{key} must be between {lo:g} and {hi:g}")
            out[key] = int(v) if isinstance(default, int) else v
        elif key == "allowed_tickers":
            tickers = [str(t).upper() for t in value or []]
            unknown = sorted(set(tickers) - set(UNIVERSE))
            if unknown:
                raise ValueError(f"unknown tickers: {', '.join(unknown)}")
            out[key] = [t for t in UNIVERSE if t in tickers]
        else:
            choices = {"permission_mode": PERMISSION_MODES, "decision_mode": DECISION_MODES,
                       "sizing": SIZING_MODES, "allocation_method": ALLOCATION_METHODS}.get(key)
            if choices and value not in choices:
                raise ValueError(f"{key} must be one of {', '.join(choices)}")
            out[key] = str(value)
    return out


class Autopilot:
    def __init__(self, state_file: Path, log_file: Path, broker: Broker) -> None:
        self.state_file, self.log_file, self.broker = state_file, log_file, broker
        self.lock = threading.RLock()
        self.cycle_lock = threading.Lock()
        self.budget = dict(DEFAULT_BUDGET)
        self.config = dict(DEFAULT_AUTOPILOT)
        self.kill_switch = False
        self.kill_reason = ""
        self.peak_equity: float | None = None
        self.proposals: list[dict[str, Any]] = []
        self.cursor = 0
        self.budget_saved = False
        self.last_run: str | None = None
        self.next_run: float | None = None
        self.running_cycle = False
        self.log: list[dict[str, Any]] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._load()

    # ---------------------------------------------------------------- persistence
    def _load(self) -> None:
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}
        self.budget.update({k: v for k, v in (data.get("budget") or {}).items() if k in DEFAULT_BUDGET})
        self.budget_saved = bool(data.get("budget_saved"))
        self.config.update({k: v for k, v in (data.get("config") or {}).items() if k in DEFAULT_AUTOPILOT})
        self.kill_switch = bool(data.get("kill_switch", False))
        self.kill_reason = str(data.get("kill_reason", ""))
        self.peak_equity = data.get("peak_equity")
        self.proposals = list(data.get("proposals") or [])
        self.cursor = int(data.get("cursor", 0))
        self.last_run = data.get("last_run")
        try:
            lines = self.log_file.read_text(encoding="utf-8").splitlines()[-300:]
            self.log = [json.loads(line) for line in lines if line.strip()]
        except FileNotFoundError:
            self.log = []

    def _save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {"budget": self.budget, "budget_saved": self.budget_saved, "config": self.config, "kill_switch": self.kill_switch,
                "kill_reason": self.kill_reason, "peak_equity": self.peak_equity,
                "proposals": self.proposals[-200:], "cursor": self.cursor, "last_run": self.last_run}
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.state_file)

    def _record(self, entry: dict[str, Any]) -> dict[str, Any]:
        entry = {"time": _now().isoformat(), **entry}
        self.log.append(entry)
        self.log = self.log[-300:]
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        return entry

    # ---------------------------------------------------------------- configuration
    def update(self, budget: dict[str, Any] | None = None, config: dict[str, Any] | None = None) -> None:
        with self.lock:
            if budget:
                clean = _validate(budget, DEFAULT_BUDGET)
                if "allowed_tickers" in clean and not clean["allowed_tickers"]:
                    raise ValueError("allow at least one ticker")
                self.budget.update(clean)
                self.budget_saved = True
                self.peak_equity = None  # drawdown is measured from the new budget's start
            if config:
                was_enabled = self.config["enabled"]
                self.config.update(_validate(config, DEFAULT_AUTOPILOT))
                if self.config["enabled"] and self.kill_switch:
                    self.config["enabled"] = False
                    raise ValueError("emergency stop is active: resume trading first")
                if self.config["enabled"] and not was_enabled:
                    self.next_run = time.time()
            self._save()

    def engage_kill_switch(self, reason: str = "manual emergency stop") -> int:
        with self.lock:
            self.kill_switch, self.kill_reason = True, reason
            self.config["enabled"] = False
            rejected = 0
            for p in self.proposals:
                if p["status"] == "pending":
                    p.update(status="cancelled", resolved_at=_now().isoformat(), note="emergency stop")
                    rejected += 1
            self._save()
            self._record({"kind": "kill", "status": "stopped", "reason": reason, "cancelled": rejected})
            return rejected

    def release_kill_switch(self) -> None:
        with self.lock:
            self.kill_switch, self.kill_reason = False, ""
            self.peak_equity = None
            self._save()
            self._record({"kind": "kill", "status": "resumed", "reason": "trading resumed by user"})

    # ---------------------------------------------------------------- guardrails
    def _account(self) -> dict[str, Any]:
        st = self.broker.state()
        equity = float(st.get("equity") or 0.0)
        cash = float(st.get("cash") or 0.0)
        holdings: dict[str, float] = {}
        shares: dict[str, float] = {}
        for p in st.get("positions") or []:
            sign = -1.0 if p.get("side") == "Short" else 1.0
            t = str(p.get("ticker", "")).upper()
            holdings[t] = holdings.get(t, 0.0) + sign * float(p.get("current_value") or 0.0)
            shares[t] = shares.get(t, 0.0) + sign * float(p.get("shares") or 0.0)
        risk = st.get("risk") or {}
        with self.lock:
            if equity > 0 and (self.peak_equity is None or equity > self.peak_equity):
                self.peak_equity = equity
                self._save()
        return {"equity": equity, "cash": cash, "holdings": holdings, "shares": shares,
                "daily_pnl": float(risk.get("daily_pnl") or 0.0), "peak": self.peak_equity or equity}

    def trades_today(self) -> int:
        today = _now().astimezone(ET).date()
        return sum(1 for e in self.log if e.get("status") == "executed"
                   and datetime.fromisoformat(e["time"]).astimezone(ET).date() == today)

    def limits(self, acct: dict[str, Any] | None = None) -> dict[str, Any]:
        acct = acct or self._account()
        b = self.budget
        managed = min(float(b["total_budget"]), acct["equity"]) if acct["equity"] > 0 else float(b["total_budget"])
        invested = sum(v for v in acct["holdings"].values() if v > 0)
        reserve = managed * b["cash_reserve_pct"] / 100.0
        spendable = max(0.0, min(acct["cash"], managed - invested - reserve))
        daily_limit = b["total_budget"] * b["max_daily_loss_pct"] / 100.0
        dd_limit = b["total_budget"] * b["max_drawdown_pct"] / 100.0
        drawdown = max(0.0, acct["peak"] - acct["equity"])
        return {
            "managed": managed, "invested": invested, "reserve": reserve, "spendable": spendable,
            "budget_exceeds_equity": acct["equity"] > 0 and b["total_budget"] > acct["equity"],
            "daily_pnl": acct["daily_pnl"], "daily_loss_limit": daily_limit,
            "daily_loss_hit": acct["daily_pnl"] <= -daily_limit,
            "drawdown": drawdown, "drawdown_limit": dd_limit, "drawdown_hit": drawdown >= dd_limit,
            "trades_today": self.trades_today(), "max_trades_per_day": b["max_trades_per_day"],
            "trades_hit": self.trades_today() >= b["max_trades_per_day"],
            "position_cap": managed * b["max_position_pct"] / 100.0,
        }

    def gate(self, ticker: str, action: str, *, notional: float | None = None, shares: float | None = None,
             target_value: float | None = None) -> dict[str, Any]:
        """Size an order and check the hard limits.

        Buy: ``notional`` (requested dollars) or, if omitted, up to the position target.
        Sell: ``shares`` to sell, or the whole long position if omitted. The agent never
        opens shorts. Returns ``{"ok", "order", "blocks", "skip"}``.
        """
        ticker, action = ticker.upper(), action.title()
        acct = self._account()
        lim = self.limits(acct)
        blocks: list[str] = []
        if self.kill_switch:
            blocks.append("Emergency stop is active")
        price = self.broker.price(ticker)
        if not price or price <= 0:
            return {"ok": False, "blocks": blocks + [f"No price for {ticker}"], "order": None, "skip": None}
        held_value = acct["holdings"].get(ticker, 0.0)
        held_shares = acct["shares"].get(ticker, 0.0)

        if action == "Sell":
            if held_shares <= 0:
                return {"ok": False, "blocks": blocks, "order": None, "skip": f"No long {ticker} position to sell (the agent never shorts)"}
            qty = min(held_shares, shares) if shares else held_shares
            # Exits reduce risk, so only the emergency stop blocks them.
            order = {"ticker": ticker, "action": "Sell", "shares": qty, "price": price, "notional": qty * price,
                     "risk_reducing": True}
            return {"ok": not blocks, "order": order, "blocks": blocks, "skip": None}

        if action != "Buy":
            return {"ok": False, "blocks": blocks, "order": None, "skip": "Hold: nothing to do"}
        if ticker not in self.budget["allowed_tickers"]:
            blocks.append(f"{ticker} is not in your allowed tickers")
        if lim["daily_loss_hit"]:
            blocks.append(f"Daily loss limit hit ({lim['daily_pnl']:,.0f} vs -{lim['daily_loss_limit']:,.0f}); no new buys today")
        if lim["drawdown_hit"]:
            blocks.append(f"Drawdown limit hit ({lim['drawdown']:,.0f} below peak, limit {lim['drawdown_limit']:,.0f}); no new buys")
        if lim["trades_hit"]:
            blocks.append(f"Daily trade limit reached ({lim['max_trades_per_day']})")

        target = min(lim["position_cap"], target_value if target_value is not None else lim["position_cap"])
        room = max(0.0, target - max(held_value, 0.0))
        caps = {"position target": room, "cash above reserve": lim["spendable"],
                "single-order cap": float(self.budget["max_order_usd"])}
        if notional is not None:
            caps["requested"] = float(notional)
        size = min(caps.values())
        binding = min(caps, key=caps.get)
        if size < 1.0:
            reason = {"position target": f"{ticker} already at its target (${target:,.0f})",
                      "cash above reserve": f"No cash left above the ${lim['reserve']:,.0f} reserve",
                      "single-order cap": "Single-order cap is below $1", "requested": "Requested amount below $1"}[binding]
            return {"ok": False, "blocks": blocks, "order": None, "skip": reason}
        order = {"ticker": ticker, "action": "Buy", "shares": size / price, "price": price, "notional": size,
                 "limited_by": binding, "risk_reducing": False}
        return {"ok": not blocks, "order": order, "blocks": blocks, "skip": None}

    def escalations(self, decision: dict[str, Any], order: dict[str, Any]) -> list[str]:
        """Soft checks: in guarded mode these send the trade to the approval queue."""
        out = []
        conf = decision.get("confidence")
        if conf is not None and conf < self.config["min_confidence"]:
            out.append(f"Confidence {conf:.2f} below {self.config['min_confidence']:.2f}")
        if not order.get("risk_reducing") and order["notional"] > self.config["auto_max_order_usd"]:
            out.append(f"Order ${order['notional']:,.0f} above the ${self.config['auto_max_order_usd']:,.0f} auto limit")
        if decision.get("disagreement"):
            out.append(decision["disagreement"])
        return out

    # ---------------------------------------------------------------- execution
    def _execute(self, order: dict[str, Any], decision: dict[str, Any], via: str) -> dict[str, Any]:
        res = self.broker.place(order["ticker"], order["action"], order["shares"],
                                f"[{via}] {decision.get('reasoning', '')}".strip(),
                                decision.get("provider", "agent"), decision.get("model", ""))
        ok = bool(res and res.get("success"))
        return self._record({"kind": "order", "ticker": order["ticker"], "action": order["action"],
                             "notional": round(order["notional"], 2), "shares": round(order["shares"], 6),
                             "status": "executed" if ok else "failed", "via": via,
                             "reason": decision.get("reasoning", "") if ok else (res or {}).get("error", "order failed"),
                             "provider": decision.get("provider"), "confidence": decision.get("confidence")})

    def _queue(self, order: dict[str, Any], decision: dict[str, Any], why: list[str], source: str) -> dict[str, Any]:
        ttl = timedelta(minutes=float(self.config["proposal_ttl_min"]))
        proposal = {"id": uuid.uuid4().hex[:10], "created_at": _now().isoformat(), "expires_at": (_now() + ttl).isoformat(),
                    "status": "pending", "source": source, "ticker": order["ticker"], "action": order["action"],
                    "notional": round(order["notional"], 2), "shares": round(order["shares"], 6), "price": order["price"],
                    "decision": {k: decision.get(k) for k in ("action", "confidence", "reasoning", "provider", "model", "disagreement")},
                    "why_approval": why}
        with self.lock:
            self.proposals.append(proposal)
            self._save()
        return self._record({"kind": "order", "ticker": order["ticker"], "action": order["action"],
                             "notional": proposal["notional"], "status": "queued", "via": source,
                             "reason": "; ".join(why) or "approval required", "proposal": proposal["id"],
                             "confidence": decision.get("confidence")})

    def handle_decision(self, ticker: str, decision: dict[str, Any], source: str = "autopilot",
                        target_value: float | None = None) -> dict[str, Any]:
        """Route one agent decision according to the permission mode."""
        action = decision.get("action", "Hold")
        base = {"kind": "decision", "ticker": ticker, "action": action, "confidence": decision.get("confidence"),
                "provider": decision.get("provider"), "via": source}
        if action not in ("Buy", "Sell"):
            return self._record({**base, "status": "hold", "reason": decision.get("reasoning", "")})
        g = self.gate(ticker, action, target_value=target_value)
        if g["skip"]:
            return self._record({**base, "status": "skipped", "reason": g["skip"]})
        if not g["ok"]:
            return self._record({**base, "status": "blocked", "reason": "; ".join(g["blocks"]),
                                 "notional": round(g["order"]["notional"], 2) if g["order"] else None})
        order, mode = g["order"], self.config["permission_mode"]
        if mode == "advisory":
            return self._record({**base, "status": "suggested", "notional": round(order["notional"], 2),
                                 "reason": decision.get("reasoning", "")})
        if mode == "full":
            return self._execute(order, decision, source)
        if mode == "approve":
            if order["risk_reducing"] and self.config["auto_exits"]:
                return self._execute(order, decision, f"{source} · auto-exit")
            return self._queue(order, decision, ["Approval mode: every trade needs your OK"], source)
        why = self.escalations(decision, order)
        if why:
            return self._queue(order, decision, why, source)
        return self._execute(order, decision, source)

    def execute_now(self, ticker: str, decision: dict[str, Any], source: str) -> dict[str, Any]:
        """A trade the user asked for with a click: the click is the approval, hard limits still apply."""
        g = self.gate(ticker, decision.get("action", "Hold"))
        base = {"kind": "decision", "ticker": ticker.upper(), "action": decision.get("action"), "via": source,
                "confidence": decision.get("confidence"), "provider": decision.get("provider")}
        if g["skip"]:
            self._record({**base, "status": "skipped", "reason": g["skip"]})
            return {"error": g["skip"]}
        if not g["ok"]:
            reason = "; ".join(g["blocks"])
            self._record({**base, "status": "blocked", "reason": reason})
            return {"error": f"Blocked by your risk limits: {reason}"}
        entry = self._execute(g["order"], decision, source)
        return {"order": entry} if entry["status"] == "executed" else {"error": entry["reason"]}

    def queue_order(self, ticker: str, action: str, reasoning: str, *, notional: float | None = None,
                    shares: float | None = None, source: str = "rebalance") -> dict[str, Any]:
        """User-initiated plan (e.g. a rebalance): size + hard-check, then wait for approval."""
        g = self.gate(ticker, action, notional=notional, shares=shares)
        if not g["order"] or g["skip"]:
            return {"queued": False, "ticker": ticker, "reason": g["skip"] or "; ".join(g["blocks"])}
        if not g["ok"]:
            return {"queued": False, "ticker": ticker, "reason": "; ".join(g["blocks"])}
        entry = self._queue(g["order"], {"action": action, "reasoning": reasoning, "provider": "allocation", "model": source},
                            ["Rebalance plan: review each order"], source)
        return {"queued": True, "ticker": ticker, "proposal": entry["proposal"], "notional": entry["notional"]}

    def _expire(self) -> None:
        now = _now()
        changed = False
        for p in self.proposals:
            if p["status"] == "pending" and datetime.fromisoformat(p["expires_at"]) <= now:
                p.update(status="expired", resolved_at=now.isoformat())
                self._record({"kind": "order", "ticker": p["ticker"], "action": p["action"], "status": "expired",
                              "notional": p["notional"], "reason": "not approved in time", "proposal": p["id"]})
                changed = True
        if changed:
            self._save()

    def resolve(self, proposal_id: str, approve: bool) -> dict[str, Any]:
        with self.lock:
            self._expire()
            p = next((x for x in self.proposals if x["id"] == proposal_id), None)
            if p is None:
                raise ValueError("proposal not found")
            if p["status"] != "pending":
                raise ValueError(f"proposal is already {p['status']}")
            p["resolved_at"] = _now().isoformat()
            if not approve:
                p["status"] = "rejected"
                self._save()
                self._record({"kind": "order", "ticker": p["ticker"], "action": p["action"], "status": "rejected",
                              "notional": p["notional"], "reason": "rejected by you", "proposal": p["id"]})
                return {"proposal": p}
            # Re-check at today's price; never send more than what was approved.
            if p["action"] == "Buy":
                g = self.gate(p["ticker"], "Buy", notional=p["notional"])
            else:
                g = self.gate(p["ticker"], "Sell", shares=p["shares"])
            if not g["ok"] or g["skip"]:
                p["status"] = "blocked"
                reason = g["skip"] or "; ".join(g["blocks"])
                p["note"] = reason
                self._save()
                self._record({"kind": "order", "ticker": p["ticker"], "action": p["action"], "status": "blocked",
                              "notional": p["notional"], "reason": f"on approval: {reason}", "proposal": p["id"]})
                return {"proposal": p}
            entry = self._execute(g["order"], {**p["decision"], "action": p["action"]}, f"{p['source']} · approved")
            p["status"] = "approved" if entry["status"] == "executed" else "failed"
            p["note"] = entry.get("reason")
            self._save()
            return {"proposal": p, "result": entry}

    # ---------------------------------------------------------------- loop
    def run_cycle(self, manual: bool = False) -> list[dict[str, Any]]:
        if not self.cycle_lock.acquire(blocking=False):
            return [{"status": "busy", "reason": "a cycle is already running"}]
        self.running_cycle = True
        try:
            self._expire()
            if self.kill_switch:
                return [self._record({"kind": "cycle", "status": "blocked", "reason": "emergency stop is active"})]
            tickers = list(self.budget["allowed_tickers"])
            if not tickers:
                return [self._record({"kind": "cycle", "status": "skipped", "reason": "no allowed tickers"})]
            n = min(int(self.config["tickers_per_cycle"]), len(tickers))
            batch = [tickers[(self.cursor + i) % len(tickers)] for i in range(n)]
            with self.lock:
                self.cursor = (self.cursor + n) % len(tickers)
            targets = None
            if self.config["sizing"] == "allocation":
                try:
                    weights = self.broker.targets(self.config["allocation_method"])
                    targets = {t: w * self.limits()["managed"] for t, w in weights.items()}
                except Exception as exc:
                    self._record({"kind": "cycle", "status": "warning", "reason": f"allocation targets unavailable, using max position: {exc}"})
            out = []
            for t in batch:
                try:
                    decision = self.broker.decide(t, self.config["decision_mode"], self.config["backbone"])
                except Exception as exc:
                    out.append(self._record({"kind": "decision", "ticker": t, "status": "error", "reason": f"{type(exc).__name__}: {exc}"}))
                    continue
                tv = targets.get(t, 0.0) if targets is not None else None
                out.append(self.handle_decision(t, decision, "manual cycle" if manual else "autopilot", target_value=tv))
            with self.lock:
                self.last_run = _now().isoformat()
                self._save()
            return out
        finally:
            self.running_cycle = False
            self.cycle_lock.release()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="autopilot", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(5.0):
            try:
                self._expire()
                if not self.config["enabled"] or self.kill_switch:
                    self.next_run = None
                    continue
                if self.next_run is None:
                    self.next_run = time.time()
                if time.time() < self.next_run:
                    continue
                self.next_run = time.time() + float(self.config["interval_min"]) * 60
                if self.config["market_hours_only"] and not market_open():
                    continue
                self.run_cycle()
            except Exception as exc:  # keep the loop alive; surface the error in the log
                self._record({"kind": "cycle", "status": "error", "reason": f"{type(exc).__name__}: {exc}"})

    # ---------------------------------------------------------------- status
    def status(self) -> dict[str, Any]:
        self._expire()
        try:
            lim = self.limits()
            lim_error = None
        except Exception as exc:
            lim, lim_error = None, str(exc)
        return {
            "budget": self.budget, "budget_saved": self.budget_saved, "config": self.config, "kill_switch": self.kill_switch,
            "kill_reason": self.kill_reason, "limits": lim, "limits_error": lim_error,
            "market_open": market_open(), "running_cycle": self.running_cycle,
            "last_run": self.last_run,
            "next_run": datetime.fromtimestamp(self.next_run, timezone.utc).isoformat() if self.next_run and self.config["enabled"] else None,
            "pending": [p for p in self.proposals if p["status"] == "pending"],
            "recent_proposals": [p for p in self.proposals if p["status"] != "pending"][-20:][::-1],
            "log": self.log[-80:][::-1],
            "options": {"permission_modes": PERMISSION_MODES, "decision_modes": DECISION_MODES,
                        "sizing": SIZING_MODES, "allocation_methods": ALLOCATION_METHODS, "universe": list(UNIVERSE)},
        }

