from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agentic_trade import autopilot as ap


class FakeBroker:
    def __init__(self, equity=10000.0, cash=10000.0, positions=None, daily_pnl=0.0, decision=None):
        self.equity, self.cash, self.positions, self.daily_pnl = equity, cash, positions or {}, daily_pnl
        self.decision = decision or {"action": "Buy", "confidence": 0.8, "reasoning": "test", "provider": "rule"}
        self.orders = []

    def state(self):
        return {"equity": self.equity, "cash": self.cash, "risk": {"daily_pnl": self.daily_pnl},
                "positions": [{"ticker": t, "side": "Long", "shares": s, "current_value": s * 100.0}
                              for t, s in self.positions.items()]}

    def price(self, ticker):
        return 100.0

    def place(self, ticker, action, shares, reasoning, provider, model):
        self.orders.append((ticker, action, round(shares, 4)))
        return {"success": True}

    def decide(self, ticker, mode, backbone):
        return dict(self.decision)

    def targets(self, method):
        return {t: 0.1 for t in ap.UNIVERSE}


def make(tmp_path, broker, **config):
    pilot = ap.Autopilot(tmp_path / "a.json", tmp_path / "log.jsonl", broker)
    pilot.update(budget={"total_budget": 10000, "cash_reserve_pct": 20, "max_position_pct": 25,
                         "max_order_usd": 1500, "max_daily_loss_pct": 2, "max_drawdown_pct": 10, "max_trades_per_day": 3})
    if config:
        pilot.update(config=config)
    return pilot


def test_buy_is_sized_by_the_smallest_limit(tmp_path):
    b = FakeBroker()
    pilot = make(tmp_path, b)
    g = pilot.gate("TSLA", "Buy")
    assert g["ok"] and g["order"]["notional"] == pytest.approx(1500) and g["order"]["limited_by"] == "single-order cap"
    b.positions = {"TSLA": 20}  # $2,000 held, cap $2,500
    assert pilot.gate("TSLA", "Buy")["order"]["notional"] == pytest.approx(500)
    b.positions, b.cash = {"AAPL": 79}, 2100.0  # $7,900 invested; $2,000 reserve leaves $100
    assert pilot.gate("TSLA", "Buy")["order"]["notional"] == pytest.approx(100)


def test_hard_limits_block_buys_but_not_exits(tmp_path):
    b = FakeBroker(daily_pnl=-250.0, positions={"TSLA": 5})
    pilot = make(tmp_path, b)
    g = pilot.gate("NVDA", "Buy")
    assert not g["ok"] and "Daily loss" in g["blocks"][0]
    assert pilot.gate("TSLA", "Sell")["ok"]
    pilot.update(budget={"allowed_tickers": ["AAPL"]})
    b.daily_pnl = 0.0
    assert "not in your allowed tickers" in pilot.gate("NVDA", "Buy")["blocks"][0]
    pilot.engage_kill_switch()
    assert not pilot.gate("TSLA", "Sell")["ok"]


def test_drawdown_from_peak(tmp_path):
    b = FakeBroker(equity=10000.0)
    pilot = make(tmp_path, b)
    pilot.gate("TSLA", "Buy")
    b.equity = 8900.0
    assert any("Drawdown" in x for x in pilot.gate("TSLA", "Buy")["blocks"])


def test_permission_modes(tmp_path):
    b = FakeBroker()
    pilot = make(tmp_path, b, permission_mode="advisory")
    assert pilot.handle_decision("TSLA", b.decide("TSLA", "", ""))["status"] == "suggested" and not b.orders

    pilot.update(config={"permission_mode": "approve"})
    entry = pilot.handle_decision("TSLA", b.decide("TSLA", "", ""))
    assert entry["status"] == "queued" and not b.orders
    pilot.resolve(entry["proposal"], approve=True)
    assert b.orders == [("TSLA", "Buy", 15.0)]

    pilot.update(config={"permission_mode": "guarded", "auto_max_order_usd": 1000})
    assert pilot.handle_decision("NVDA", b.decide("NVDA", "", ""))["status"] == "queued"  # $1,500 > auto limit
    pilot.update(config={"auto_max_order_usd": 5000})
    low = {**b.decision, "confidence": 0.3}
    assert pilot.handle_decision("NVDA", low)["status"] == "queued"
    assert pilot.handle_decision("NVDA", b.decision)["status"] == "executed"

    pilot.update(config={"permission_mode": "full"})
    b.positions = {"AMD": 10}
    assert pilot.handle_decision("AMD", {**b.decision, "action": "Sell"})["status"] == "executed"


def test_trade_limit_and_expiry(tmp_path):
    b = FakeBroker()
    pilot = make(tmp_path, b, permission_mode="full")
    for t in ("TSLA", "NVDA", "AAPL"):
        b.cash = 10000.0
        assert pilot.handle_decision(t, b.decision)["status"] == "executed"
    assert pilot.handle_decision("AMD", b.decision)["status"] == "blocked"

    pilot.update(config={"permission_mode": "approve"})
    pilot.update(budget={"max_trades_per_day": 50})
    entry = pilot.handle_decision("META", b.decision)
    pilot.proposals[-1]["expires_at"] = datetime(2000, 1, 1, tzinfo=timezone.utc).isoformat()
    with pytest.raises(ValueError, match="expired"):
        pilot.resolve(entry["proposal"], approve=True)


def test_kill_switch_cancels_queue_and_blocks_enabling(tmp_path):
    b = FakeBroker()
    pilot = make(tmp_path, b)
    pilot.handle_decision("TSLA", b.decision)
    assert pilot.engage_kill_switch() == 1
    with pytest.raises(ValueError, match="emergency stop"):
        pilot.update(config={"enabled": True})
    assert pilot.run_cycle()[0]["status"] == "blocked"
    pilot.release_kill_switch()
    results = pilot.run_cycle(manual=True)
    assert len(results) == 2 and all(r["status"] == "queued" for r in results)


def test_settings_validation_and_persistence(tmp_path):
    b = FakeBroker()
    pilot = make(tmp_path, b, permission_mode="guarded")
    with pytest.raises(ValueError):
        pilot.update(budget={"cash_reserve_pct": 150})
    with pytest.raises(ValueError):
        pilot.update(config={"permission_mode": "yolo"})
    with pytest.raises(ValueError):
        pilot.update(budget={"allowed_tickers": ["XYZ"]})
    again = ap.Autopilot(tmp_path / "a.json", tmp_path / "log.jsonl", b)
    assert again.config["permission_mode"] == "guarded" and again.budget["max_order_usd"] == 1500


def test_market_hours():
    assert ap.market_open(datetime(2026, 10, 28, 15, 0, tzinfo=timezone.utc))      # Wed 11:00 ET
    assert not ap.market_open(datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc))  # 08:00 ET
    assert not ap.market_open(datetime(2026, 10, 31, 15, 0, tzinfo=timezone.utc))  # Saturday
