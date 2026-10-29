from __future__ import annotations

import http.client
import json
import os
import stat
import threading

import pytest

from agentic_trade import credentials, dashboard_server


class _Resp:
    def __init__(self, status: int, payload: dict | None = None) -> None:
        self.status_code, self.ok, self._payload = status, 200 <= status < 300, payload or {}

    def json(self) -> dict:
        return self._payload


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(credentials, "STORE", tmp_path / "credentials.json")
    saved_env = dict(os.environ)
    yield tmp_path / "credentials.json"
    os.environ.clear()
    os.environ.update(saved_env)


def _fake_get(valid: set[str]):
    def get(url, headers=None, timeout=None):
        key = headers.get("APCA-API-KEY-ID") or headers.get("Authorization", "").removeprefix("Bearer ")
        if key not in valid:
            return _Resp(401)
        if "alpaca" in url:
            return _Resp(200, {"status": "ACTIVE", "equity": "1000", "buying_power": "2000"})
        if "openrouter" in url:
            return _Resp(200, {"data": {"is_free_tier": True, "limit": None}})
        return _Resp(200, {"data": [{"id": "m"}]})
    return get


def test_save_applies_env_masks_and_is_private(store, monkeypatch):
    monkeypatch.setattr(credentials.requests, "get", _fake_get({"PKGOODKEY12345"}))
    res = credentials.save("alpaca", {"key": "PKGOODKEY12345", "secret": "s3cr3t-value-xyz"})
    assert res["saved"] and res["test"]["ok"]
    assert os.environ["ALPACA_PAPER_KEY"] == "PKGOODKEY12345"
    assert os.environ["ALPACA_PAPER_SECRET"] == "s3cr3t-value-xyz"
    assert stat.S_IMODE(store.stat().st_mode) == 0o600
    status = json.dumps(credentials.status(), ensure_ascii=False)
    assert "s3cr3t-value-xyz" not in status and "PKGOODKEY12345" not in status
    assert "••••-xyz" in status


def test_rejected_key_is_not_saved(store, monkeypatch):
    monkeypatch.setattr(credentials.requests, "get", _fake_get(set()))
    res = credentials.save("openrouter", {"key": "sk-or-bad-key-0000"})
    assert not res["saved"] and not res["test"]["ok"]
    assert not store.exists()


def test_masked_value_keeps_saved_key_and_remove_restores_env(store, monkeypatch):
    monkeypatch.setattr(credentials.requests, "get", _fake_get({"gsk_first_key_0001", "gsk_second_key_002"}))
    monkeypatch.setitem(credentials._BASE_ENV, "GROQ_API_KEY", "gsk_from_dotenv_99")
    credentials.save("groq", {"keys": ["gsk_first_key_0001"]})
    masked = credentials.status()["providers"][1]["masked"]["keys"][0]
    credentials.save("groq", {"keys": [masked, "gsk_second_key_002"]})
    assert os.environ["GROQ_API_KEY"] == "gsk_first_key_0001"
    assert os.environ["GROQ_API_KEY2"] == "gsk_second_key_002"
    credentials.remove("groq")
    assert os.environ["GROQ_API_KEY"] == "gsk_from_dotenv_99"
    assert "GROQ_API_KEY2" not in os.environ


def test_invalid_input_raises(store):
    with pytest.raises(ValueError):
        credentials.save("alpaca", {"key": "has space", "secret": "x"}, test_first=False)
    with pytest.raises(ValueError):
        credentials.save("nope", {}, test_first=False)


def test_endpoint_refuses_cross_origin(store):
    server = dashboard_server.ThreadedHTTPServer(("127.0.0.1", 0), dashboard_server.DashboardHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        def request(headers):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            conn.request("POST", "/api/credentials/remove", body=json.dumps({"provider": "groq"}),
                         headers={"Content-Type": "application/json", **headers})
            return conn.getresponse().status
        assert request({"Origin": "https://evil.example"}) == 403
        assert request({"Host": "evil.example"}) == 403
        assert request({}) == 200
    finally:
        server.shutdown()
