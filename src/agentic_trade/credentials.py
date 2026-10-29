"""API keys entered in the app's Settings tab.

Keys are stored in ``artifacts/credentials.json`` (git-ignored, owner-only
permissions) and applied to ``os.environ`` under the variable names the rest of
the code already reads, so a saved key is used immediately without a restart.
A key saved in the app overrides the same provider's key from ``.env``; removing
it falls back to ``.env`` again. Keys are never sent back to the browser, only a
masked form.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "artifacts" / "credentials.json"
MAX_GROQ_KEYS = 8
TIMEOUT = 12

# Each field maps to the environment variable(s) the trading / LLM code reads.
# ``env_fallbacks`` are .env names that also count as "configured" for the status view.
PROVIDERS: dict[str, dict[str, Any]] = {
    "alpaca": {
        "name": "Alpaca (paper trading)",
        "kind": "broker",
        "description": "Places paper orders, reads positions and live quotes. Paper account only: live-money trading stays disabled.",
        "signup": "https://app.alpaca.markets/paper/dashboard/overview",
        "fields": [
            {"id": "key", "label": "API Key ID", "env": "ALPACA_PAPER_KEY", "env_fallbacks": ["ALPACA_KEY"], "secret": False},
            {"id": "secret", "label": "Secret Key", "env": "ALPACA_PAPER_SECRET", "env_fallbacks": ["ALPACA_SECRET"], "secret": True},
        ],
    },
    "groq": {
        "name": "Groq (LLM)",
        "kind": "llm",
        "description": "Primary LLM for AI trade decisions. Free tier is rate limited, so add keys from several accounts to rotate between them.",
        "signup": "https://console.groq.com/keys",
        "multi": True,
        "fields": [{"id": "keys", "label": "API key", "env": "GROQ_API_KEY",
                    "env_fallbacks": ["GROQ_KEY", "GROOQ_KEY"], "secret": True}],
    },
    "openrouter": {
        "name": "OpenRouter (LLM)",
        "kind": "llm",
        "description": "Fallback LLM using free models when Groq is rate limited or down.",
        "signup": "https://openrouter.ai/settings/keys",
        "fields": [{"id": "key", "label": "API key", "env": "OPENROUTER_API_KEY",
                    "env_fallbacks": ["OPEN_ROUTER_KEY", "OPEN_ROUTER_kEY", "AGENT_ROUTER_KEY"], "secret": True}],
    },
}

_lock = threading.Lock()


def _groq_env_names() -> list[str]:
    # GROQ_API_KEY, GROQ_API_KEY2, ... all match the key-pool pattern in study.llm_agent.
    return ["GROQ_API_KEY"] + [f"GROQ_API_KEY{i}" for i in range(2, MAX_GROQ_KEYS + 1)]


def _managed_env_names() -> list[str]:
    names = []
    for pid, spec in PROVIDERS.items():
        names += _groq_env_names() if spec.get("multi") else [f["env"] for f in spec["fields"]]
    return names


# What the environment held before the app touched it (i.e. from the shell or .env).
_BASE_ENV = {name: os.environ.get(name) for name in _managed_env_names()}


def mask(value: str | None) -> str:
    if not value:
        return ""
    return "••••" + value[-4:] if len(value) > 12 else "••••"


def _read() -> dict[str, Any]:
    try:
        return json.loads(STORE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write(data: dict[str, Any]) -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STORE.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, STORE)
    os.chmod(STORE, 0o600)


def _env_targets(pid: str, values: dict[str, Any]) -> dict[str, str | None]:
    """Environment variables (name -> value or None) for one provider's saved values."""
    spec = PROVIDERS[pid]
    if spec.get("multi"):
        keys = [k for k in values.get("keys", []) if k][:MAX_GROQ_KEYS]
        names = _groq_env_names()
        return {name: (keys[i] if i < len(keys) else None) for i, name in enumerate(names)}
    return {f["env"]: values.get(f["id"]) or None for f in spec["fields"]}


def apply() -> None:
    """Push saved keys into os.environ; restore the original value where nothing is saved."""
    with _lock:
        data = _read()
        for pid in PROVIDERS:
            saved = (data.get(pid) or {}).get("values") or {}
            for name, value in _env_targets(pid, saved).items():
                value = value or _BASE_ENV.get(name)
                if value:
                    os.environ[name] = value
                else:
                    os.environ.pop(name, None)


def _effective(pid: str) -> dict[str, Any]:
    """Values the code will actually use for a provider, and where they come from."""
    spec = PROVIDERS[pid]
    saved = ((_read().get(pid) or {}).get("values")) or {}
    if spec.get("multi"):
        # The Groq key pool is every GROQ_KEY*-style variable, so .env keys stay in
        # rotation next to the ones saved in the app.
        from .study.llm_agent import GROQ_KEY_VAR

        app_keys = list(saved.get("keys") or [])
        env_keys = [v for n, v in sorted(os.environ.items())
                    if GROQ_KEY_VAR.match(n) and n not in _groq_env_names() and v]
        env_keys += [v for n, v in _BASE_ENV.items() if n in _groq_env_names() and v]
        env_keys = [k for k in dict.fromkeys(env_keys) if k not in app_keys]
        if app_keys:
            return {"source": "app", "values": {"keys": app_keys}, "extra_env_keys": len(env_keys)}
        return {"source": ".env" if env_keys else None, "values": {"keys": env_keys}, "extra_env_keys": 0}
    if all(saved.get(f["id"]) for f in spec["fields"]):
        return {"source": "app", "values": dict(saved)}
    values = {}
    for f in spec["fields"]:
        for name in [f["env"], *f.get("env_fallbacks", [])]:
            if os.environ.get(name):
                values[f["id"]] = os.environ[name]
                break
    complete = all(values.get(f["id"]) for f in spec["fields"])
    return {"source": ".env" if complete else None, "values": values if complete else {}}


def status() -> dict[str, Any]:
    data = _read()
    out = []
    for pid, spec in PROVIDERS.items():
        eff = _effective(pid)
        entry = data.get(pid) or {}
        vals = eff["values"]
        masked = ({"keys": [mask(k) for k in vals.get("keys", [])]} if spec.get("multi")
                  else {f["id"]: mask(vals.get(f["id"])) for f in spec["fields"]})
        out.append({
            "id": pid, "name": spec["name"], "kind": spec["kind"], "description": spec["description"],
            "signup": spec["signup"], "multi": bool(spec.get("multi")),
            "fields": [{"id": f["id"], "label": f["label"], "secret": f["secret"]} for f in spec["fields"]],
            "configured": eff["source"] is not None, "source": eff["source"], "masked": masked,
            "saved_in_app": bool(entry.get("values")), "extra_env_keys": eff.get("extra_env_keys", 0),
            "last_test": entry.get("last_test"),
        })
    return {"providers": out, "store": str(STORE.relative_to(ROOT) if STORE.is_relative_to(ROOT) else STORE),
            "ready": {"broker": any(p["configured"] for p in out if p["kind"] == "broker"),
                      "llm": any(p["configured"] for p in out if p["kind"] == "llm")}}


def _clean_values(pid: str, values: dict[str, Any]) -> dict[str, Any]:
    spec = PROVIDERS[pid]
    if spec.get("multi"):
        keys = values.get("keys") or []
        if not isinstance(keys, list):
            raise ValueError("keys must be a list")
        keys = list(dict.fromkeys(str(k).strip() for k in keys if str(k).strip()))
        if not keys:
            raise ValueError("enter at least one key")
        if len(keys) > MAX_GROQ_KEYS:
            raise ValueError(f"at most {MAX_GROQ_KEYS} keys")
        return {"keys": keys}
    out = {}
    for f in spec["fields"]:
        v = str(values.get(f["id"]) or "").strip()
        if not v:
            raise ValueError(f"{f['label']} is required")
        if any(c.isspace() for c in v) or len(v) > 400:
            raise ValueError(f"{f['label']} does not look like an API key")
        out[f["id"]] = v
    return out


def _merge_masked(pid: str, values: dict[str, Any]) -> dict[str, Any]:
    """Let the browser resend a masked key to mean "keep the saved one"."""
    current = _effective(pid)["values"]
    if PROVIDERS[pid].get("multi"):
        lookup = {mask(k): k for k in current.get("keys", [])}
        return {"keys": [lookup.get(k, k) for k in values.get("keys") or []]}
    return {k: (current.get(k) if v and v == mask(current.get(k)) else v) for k, v in values.items()}


def save(pid: str, values: dict[str, Any], test_first: bool = True) -> dict[str, Any]:
    if pid not in PROVIDERS:
        raise ValueError(f"unknown provider {pid}")
    clean = _clean_values(pid, _merge_masked(pid, values))
    result = test(pid, clean) if test_first else None
    if result is not None and not result["ok"]:
        return {"saved": False, "test": result}
    with _lock:
        data = _read()
        data[pid] = {"values": clean, "updated_at": datetime.now(timezone.utc).isoformat(), "last_test": result}
        _write(data)
    apply()
    return {"saved": True, "test": result}


def remove(pid: str) -> None:
    if pid not in PROVIDERS:
        raise ValueError(f"unknown provider {pid}")
    with _lock:
        data = _read()
        data.pop(pid, None)
        _write(data)
    apply()


def test(pid: str, values: dict[str, Any] | None = None) -> dict[str, Any]:
    """Make one cheap authenticated call to the provider."""
    if pid not in PROVIDERS:
        raise ValueError(f"unknown provider {pid}")
    if values is None:
        values = _effective(pid)["values"]
    else:
        values = _clean_values(pid, _merge_masked(pid, values))
    if not values:
        return _stamp(pid, {"ok": False, "message": "No key configured."}, persist=False)
    try:
        result = {"alpaca": _test_alpaca, "groq": _test_groq, "openrouter": _test_openrouter}[pid](values)
    except requests.RequestException as exc:
        result = {"ok": False, "message": f"Could not reach the provider: {type(exc).__name__}"}
    return _stamp(pid, result, persist=_effective(pid)["source"] == "app")


def _stamp(pid: str, result: dict[str, Any], persist: bool) -> dict[str, Any]:
    result["tested_at"] = datetime.now(timezone.utc).isoformat()
    if persist:
        with _lock:
            data = _read()
            if pid in data:
                data[pid]["last_test"] = result
                _write(data)
    return result


def _http_error(resp: requests.Response) -> str:
    if resp.status_code in (401, 403):
        return "Key was rejected (unauthorised). Check that you copied the whole key."
    if resp.status_code == 429:
        return "Key works but is rate limited right now."
    return f"Provider answered HTTP {resp.status_code}."


def _test_alpaca(values: dict[str, Any]) -> dict[str, Any]:
    resp = requests.get("https://paper-api.alpaca.markets/v2/account", timeout=TIMEOUT,
                        headers={"APCA-API-KEY-ID": values["key"], "APCA-API-SECRET-KEY": values["secret"]})
    if not resp.ok:
        msg = _http_error(resp)
        if resp.status_code in (401, 403):
            msg += " Live-account keys do not work here: create keys on the Paper dashboard."
        return {"ok": False, "message": msg}
    acct = resp.json()
    return {"ok": True, "message": f"Paper account {acct.get('status', '').lower()} · equity ${float(acct.get('equity', 0)):,.2f} · buying power ${float(acct.get('buying_power', 0)):,.2f}"}


def _test_groq(values: dict[str, Any]) -> dict[str, Any]:
    results = []
    for key in values["keys"]:
        resp = requests.get("https://api.groq.com/openai/v1/models", timeout=TIMEOUT,
                            headers={"Authorization": f"Bearer {key}"})
        results.append((key, resp))
    good = [k for k, r in results if r.ok or r.status_code == 429]
    bad = [(k, r) for k, r in results if not (r.ok or r.status_code == 429)]
    if not good:
        return {"ok": False, "message": _http_error(bad[0][1]) + (f" ({len(bad)} keys failed)" if len(bad) > 1 else "")}
    models = next((len(r.json().get("data", [])) for _, r in results if r.ok), None)
    msg = f"{len(good)} of {len(results)} key{'s' if len(results) > 1 else ''} valid"
    if models:
        msg += f" · {models} models available"
    if bad:
        msg += " · failing: " + ", ".join(mask(k) for k, _ in bad)
        return {"ok": False, "message": msg}
    return {"ok": True, "message": msg}


def _test_openrouter(values: dict[str, Any]) -> dict[str, Any]:
    resp = requests.get("https://openrouter.ai/api/v1/key", timeout=TIMEOUT,
                        headers={"Authorization": f"Bearer {values['key']}"})
    if not resp.ok:
        return {"ok": False, "message": _http_error(resp)}
    info = resp.json().get("data", {})
    parts = ["Key valid"]
    if info.get("is_free_tier") is not None:
        parts.append("free tier" if info["is_free_tier"] else "paid credits on account")
    if info.get("limit") is not None:
        parts.append(f"limit ${info['limit']}")
    return {"ok": True, "message": " · ".join(parts)}
