"""Live research-grounded trading agent used by the dashboard application.

The service runs the exact memory architectures evaluated in the study
(``agentic_trade.study``) on the newest data and exposes, per ticker:

* the retrieved memory context of *Ours* (layer, credibility, corroboration, age),
  peer evidence and episodic recall, plus FinMem / Plain-RAG signals for comparison;
* a rule recommendation using the decision threshold the rolling protocol selected;
* a price-only LightGBM second opinion (the strongest learned baseline in the study);
* an LLM decision rendered with the same prompt as the backbone experiment.

Everything is read from ``artifacts/app`` (live snapshot) with a fallback to
``artifacts/study`` (the research run), so the app works offline.
"""
from __future__ import annotations

import json
import threading
import time
import traceback
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .study.backtest import COMPANY, StudyData, run_systems
from .study.data import FINMEM_TICKERS, UNIVERSE
from .study.llm_agent import BACKBONES, Backbone, build_prompt
from .study.memory import MemoryConfig

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "study"
STUDY_DIR = ROOT / "artifacts" / "study"
APP_DIR = ROOT / "artifacts" / "app"
FIGURE_DIR = ROOT / "paper" / "figures"
LIVE_SYSTEMS = {"Plain RAG": MemoryConfig.plain_rag(), "FinMem": MemoryConfig.fin_mem(), "Ours": MemoryConfig.ours()}
LAYER = {0: "news", 1: "10-Q", 2: "10-K"}  # layer keys; deeper layers also hold promoted news
MAIN_STRATEGIES = ["Buy & Hold", "LGBM (price)", "LGBM (price+Ours)", "Plain RAG", "FinMem", "Ours"]


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return str(value)[:10]
    return value


class AgentService:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.job: dict[str, Any] = {"running": False, "log": [], "error": None, "finished_at": None}
        self._signals: pd.DataFrame | None = None
        self._evidence: pd.DataFrame | None = None
        self._text: dict[str, str] | None = None
        self._prices: pd.DataFrame | None = None
        self._price_model: dict[str, float] | None = None
        self._loaded_from: str = ""

    # ------------------------------------------------------------------ loading
    def _source_dir(self) -> Path:
        return APP_DIR if (APP_DIR / "signals.parquet").exists() else STUDY_DIR

    def _load(self, force: bool = False) -> None:
        src = self._source_dir()
        stamp = f"{src}:{(src / 'signals.parquet').stat().st_mtime if (src / 'signals.parquet').exists() else 0}"
        if not force and self._signals is not None and stamp == self._loaded_from:
            return
        signals = pd.read_parquet(src / "signals.parquet")
        signals = signals[signals["system"].isin(LIVE_SYSTEMS)].copy()
        signals["date"] = pd.to_datetime(signals["date"]).dt.date
        evidence = pd.read_parquet(src / "evidence.parquet")
        evidence = evidence[evidence["system"].isin(LIVE_SYSTEMS)].copy()
        evidence["date"] = pd.to_datetime(evidence["date"]).dt.date
        news_path = DATA_DIR / "news_annotated.parquet"
        news = pd.read_parquet(news_path, columns=["id", "headline", "created_at", "url"])
        filings = pd.read_parquet(DATA_DIR / "filings_annotated.parquet", columns=["id", "text", "filed", "form"])
        text = {str(i): h for i, h in zip(news["id"], news["headline"])}
        text.update({str(i): t for i, t in zip(filings["id"], filings["text"])})
        self._signals, self._evidence, self._text = signals, evidence, text
        self._urls = {str(i): u for i, u in zip(news["id"], news["url"])}
        self._prices = pd.read_parquet(DATA_DIR / "prices.parquet")
        self._prices["date"] = pd.to_datetime(self._prices["date"]).dt.date
        self._price_model = None
        self._thresholds = self._live_thresholds(signals)
        self._loaded_from = stamp

    def _live_thresholds(self, signals: pd.DataFrame) -> dict[str, float]:
        from .study.pipeline import _best_threshold

        out = {}
        for label in LIVE_SYSTEMS:
            frame = signals[(signals["system"] == label)].dropna(subset=["next_return"])
            days = sorted(frame["date"].unique())[-42:]
            out[label] = _best_threshold(frame[frame["date"].isin(days)])[0] if days else 0.15
        return out

    def _price_opinion(self) -> dict[str, float]:
        if self._price_model is not None:
            return self._price_model
        from .study.policies import PRICE_FEATURES, fit_predict, price_features

        feats = price_features(self._prices, UNIVERSE)
        feats["date"] = pd.to_datetime(feats["date"]).dt.date
        close = self._prices.pivot_table(index="date", columns="ticker", values="close").sort_index()
        nxt = close.pct_change().shift(-1)
        feats["next_return"] = [nxt.at[d, t] if d in nxt.index else np.nan for d, t in zip(feats["date"], feats["ticker"])]
        feats = feats.dropna(subset=["ret_20"]).reset_index(drop=True)
        feats["split"] = np.where(feats["next_return"].notna(), "train", "live")
        prob = np.mean([fit_predict(feats, PRICE_FEATURES, "lgbm", seed=s).to_numpy() for s in range(3)], axis=0)
        feats["p_up"] = prob
        latest = feats.sort_values("date").groupby("ticker").tail(1)
        self._price_model = {t: float(p) for t, p in zip(latest["ticker"], latest["p_up"])}
        return self._price_model

    # ------------------------------------------------------------------- status
    def status(self) -> dict:
        self._load()
        sig = self._signals
        meta = {}
        meta_path = DATA_DIR / "dataset_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        return _clean({
            "source": "live snapshot" if self._source_dir() == APP_DIR else "research run",
            "latest_decision_date": max(sig["date"]),
            "first_date": min(sig["date"]),
            "last_price_date": max(self._prices["date"]),
            "articles": len(self._text),
            "tickers": UNIVERSE,
            "finmem_tickers": FINMEM_TICKERS,
            "thresholds": self._thresholds,
            "backbones": list(BACKBONES),
            "dataset": meta,
            "job": {k: v for k, v in self.job.items() if k != "log"} | {"log": self.job["log"][-30:]},
        })

    # ------------------------------------------------------------- ticker brief
    def _action(self, signal: float, threshold: float, abstain: bool) -> str:
        if abstain:
            return "Hold"
        return "Buy" if signal > threshold else "Sell" if signal < -threshold else "Hold"

    def brief(self, ticker: str, day: str | None = None, position: float = 0.0) -> dict:
        self._load()
        ticker = ticker.upper()
        sig = self._signals[self._signals["ticker"] == ticker]
        if sig.empty:
            raise ValueError(f"unknown ticker {ticker}")
        target = date.fromisoformat(day) if day else max(sig["date"])
        rows = sig[sig["date"] == target].set_index("system")
        systems = {}
        for label in LIVE_SYSTEMS:
            if label not in rows.index:
                continue
            r = rows.loc[label]
            th = self._thresholds[label]
            systems[label] = {
                "signal": r["signal"], "threshold": th, "abstain": bool(r["abstain"]),
                "action": self._action(float(r["signal"]), th, bool(r["abstain"])),
                "n_retrieved": r["n_retrieved"], "mean_age": r["mean_age"], "redundancy": r["redundancy"],
                "conflict": r["conflict"], "layers": [r["n_shallow"], r["n_intermediate"], r["n_deep"]],
            }
        ours = rows.loc["Ours"] if "Ours" in rows.index else None
        ev = self._evidence[(self._evidence["system"] == "Ours") & (self._evidence["ticker"] == ticker)
                            & (self._evidence["date"] == target)].sort_values("rank")
        memories = []
        for _, e in ev.iterrows():
            iid = str(e["item_id"])
            memories.append({
                "id": iid, "layer": LAYER[int(e["layer"])], "layer_id": int(e["layer"]),
                "text": self._text.get(iid, ""), "url": self._urls.get(iid, ""),
                "date": str(date.fromordinal(target.toordinal() - int(e["age_days"]))),
                "age_days": int(e["age_days"]), "sentiment": e["sentiment"], "credibility": e.get("cred", np.nan),
                "weight": e["weight"], "corroboration": e["corroboration"], "category": e["category"],
                "author": e["author"], "score": e["score"],
            })
        prices = self._prices[self._prices["ticker"] == ticker].sort_values("date")
        prices = prices[prices["date"] <= target]
        close = prices["close"].to_numpy()
        logret = np.diff(np.log(close))
        p_up = self._price_opinion().get(ticker)
        hist = sig[sig["system"] == "Ours"].sort_values("date").tail(90)
        price_hist = prices.tail(90)
        consensus = self._consensus(systems.get("Ours"), p_up, ours)
        return _clean({
            "ticker": ticker, "company": COMPANY.get(ticker, ticker), "date": target,
            "price": close[-1], "change_1d": close[-1] / close[-2] - 1 if len(close) > 1 else 0.0,
            "momentum_3d": float(logret[-3:].sum()) if len(logret) >= 3 else 0.0,
            "change_20d": close[-1] / close[-21] - 1 if len(close) > 21 else None,
            "systems": systems, "memories": memories,
            "peer": {"signal": ours["peer_signal"], "n": ours["peer_n"]} if ours is not None else None,
            "episodic": {"edge": ours["episodic_edge"], "n": ours["episodic_n"]} if ours is not None else None,
            "n_today": ours["n_today"] if ours is not None else 0,
            "price_model": {"p_up": p_up},
            "consensus": consensus,
            "history": {"date": list(hist["date"]), "signal": list(hist["signal"]),
                        "price_date": list(price_hist["date"]), "close": list(price_hist["close"])},
        })

    @staticmethod
    def _consensus(ours: dict | None, p_up: float | None, row: pd.Series | None) -> dict:
        """Combine the memory agent and the price model into one recommendation.

        Research finding (rolling protocol): neither news-memory signals nor price models
        are reliably predictive day to day, so confidence is capped and disagreement
        resolves to Hold."""
        reasons, votes = [], []
        if ours and ours["abstain"]:
            reasons.append("Episodic memory: similar past situations were mis-signalled -> abstain")
            if p_up is not None:
                reasons.append(f"Price model P(up) = {p_up:.2f} (ignored while abstaining)")
            return {"action": "Hold", "confidence": 0.3, "agreement": False, "reasons": reasons}
        if ours:
            votes.append({"Buy": 1, "Sell": -1, "Hold": 0}[ours["action"]])
            reasons.append(f"Memory signal {ours['signal']:+.2f} vs threshold ±{ours['threshold']:.2f} -> {ours['action']}")
        if p_up is not None:
            votes.append(1 if p_up >= 0.53 else -1 if p_up <= 0.47 else 0)
            reasons.append(f"Price model P(up) = {p_up:.2f}")
        nonzero = {v for v in votes if v}
        if len(nonzero) != 1:
            action = "Hold"
            if len(nonzero) == 2:
                reasons.append("Memory and price model disagree -> Hold")
        else:
            action = "Buy" if nonzero.pop() == 1 else "Sell"
        agree = len(votes) == 2 and votes[0] == votes[1] != 0
        confidence = 0.6 if agree else 0.4 if action != "Hold" else 0.3
        return {"action": action, "confidence": confidence, "agreement": agree, "reasons": reasons}

    def scan(self, positions: dict[str, float] | None = None) -> dict:
        self._load()
        out = []
        for t in UNIVERSE:
            try:
                b = self.brief(t, position=(positions or {}).get(t, 0.0))
            except ValueError:
                continue
            o = b["systems"].get("Ours", {})
            f = b["systems"].get("FinMem", {})
            out.append({"ticker": t, "company": b["company"], "date": b["date"], "price": b["price"],
                        "change_1d": b["change_1d"], "change_20d": b["change_20d"],
                        "ours_signal": o.get("signal"), "ours_action": o.get("action"), "abstain": o.get("abstain"),
                        "finmem_signal": f.get("signal"), "finmem_action": f.get("action"),
                        "p_up": b["price_model"]["p_up"], "consensus": b["consensus"]["action"],
                        "confidence": b["consensus"]["confidence"], "n_today": b["n_today"],
                        "episodic_edge": (b["episodic"] or {}).get("edge"),
                        "peer_signal": (b["peer"] or {}).get("signal"),
                        "top_memory": b["memories"][0]["text"] if b["memories"] else ""})
        return _clean({"rows": out})

    # --------------------------------------------------------------- LLM agent
    def llm_decide(self, ticker: str, backbone: str = "gpt-oss-120b", position: float = 0.0,
                   cum3: float = 0.0) -> dict:
        b = self.brief(ticker, position=position)
        memories = [{"id": m["id"], "layer": m["layer_id"], "text": m["text"][:220], "date": m["date"],
                     "cred": m["credibility"] if m["credibility"] is not None else 0.5,
                     "corrob": m["corroboration"]} for m in b["memories"]]
        extras = {}
        if b["peer"] and b["peer"]["n"]:
            extras["peer"] = f"{int(b['peer']['n'])} same-day articles on co-mentioned firms, credibility-weighted sentiment {b['peer']['signal']:+.2f}."
        if b["episodic"] and b["episodic"]["edge"] is not None:
            e = b["episodic"]["edge"]
            verdict = ("news signals in such situations were followed by adverse moves - be cautious" if e < 0
                       else "news signals in such situations were reliable")
            extras["episodic"] = f"in {int(b['episodic']['n'])} similar past situations the signed market-adjusted return was {e:+.2%}; {verdict}."
        prompt = build_prompt("Ours", ticker, b["date"], b["momentum_3d"] or 0.0, cum3, position, memories, extras)
        bb = Backbone(backbone, cache_dir=APP_DIR / "llm_cache")
        bb.min_interval = 0.0
        started = time.time()
        decision = bb.ask(prompt)
        return _clean({"ticker": ticker, "backbone": backbone, "model": bb.model, "provider": bb.provider,
                       "decision": decision, "latency_s": time.time() - started, "prompt": prompt, "brief": b})

    # ------------------------------------------------------------ data refresh
    def refresh(self) -> dict:
        with self.lock:
            if self.job["running"]:
                return {"started": False, "reason": "a refresh is already running"}
            self.job = {"running": True, "log": [], "error": None, "finished_at": None,
                        "started_at": datetime.now(timezone.utc).isoformat()}
        threading.Thread(target=self._refresh_worker, daemon=True).start()
        return {"started": True}

    def _log(self, msg: str) -> None:
        self.job["log"].append(f"{datetime.now().strftime('%H:%M:%S')} {msg}")

    def _refresh_worker(self) -> None:
        try:
            from .study.data import refresh_dataset

            info = refresh_dataset(DATA_DIR, UNIVERSE, log=self._log)
            self._log(f"data ready to {info['last_session']} (+{info['new_articles']} articles)")
            self.rebuild_snapshot()
            self._log("snapshot rebuilt")
        except Exception as exc:  # surfaced in the UI
            self.job["error"] = f"{exc}"
            self._log(traceback.format_exc().splitlines()[-1])
        finally:
            self.job["running"] = False
            self.job["finished_at"] = datetime.now(timezone.utc).isoformat()

    def rebuild_snapshot(self) -> None:
        self._log("replaying memory systems over the full history (Plain RAG, FinMem, Ours)")
        data = StudyData.load(DATA_DIR, UNIVERSE)
        recent = data.days[-130] if len(data.days) > 130 else data.days[0]
        signals, evidence, stats = run_systems(data, list(LIVE_SYSTEMS.values()), list(LIVE_SYSTEMS),
                                               log_evidence=True, include_last=True, evidence_since=recent)
        APP_DIR.mkdir(parents=True, exist_ok=True)
        signals.to_parquet(APP_DIR / "signals.parquet", index=False)
        evidence.to_parquet(APP_DIR / "evidence.parquet", index=False)
        (APP_DIR / "memory_stats.json").write_text(json.dumps(stats, default=float, indent=2))
        self._load(force=True)

    # --------------------------------------------------------------- research
    def research(self, protocol: str = "rolling") -> dict:
        pdir = STUDY_DIR / protocol
        if not (pdir / "summary.json").exists():
            return {"available": False}
        summary = json.loads((pdir / "summary.json").read_text())
        daily = pd.read_parquet(pdir / "daily.parquet")
        test = daily[daily["split"] == "test"]
        curves = {}
        for scope, tickers in (("all", None), ("finmem5", FINMEM_TICKERS)):
            sub = test if tickers is None else test[test["ticker"].isin(tickers)]
            port = sub.pivot_table(index="date", columns="system", values="net", aggfunc="mean").sort_index()
            eq = (1 + port.fillna(0)).cumprod()
            curves[scope] = {"date": [str(d) for d in eq.index],
                             "series": {s: list(eq[s].round(5)) for s in eq.columns
                                        if not s.startswith(("w/o", "FinMem K", "Ours K"))}}
        llm = None
        llm_path = STUDY_DIR / "llm" / "llm_decisions.parquet"
        if llm_path.exists():
            frame = pd.read_parquet(llm_path)
            frame["ret"] = frame["position"] * frame["next_return"]
            rows = []
            for (bb, sysname), g in frame.groupby(["backbone", "system"]):
                port = g.groupby("date")["ret"].mean().sort_index()
                eq = (1 + port).cumprod()
                sd = port.std(ddof=1)
                rows.append({"backbone": bb, "system": sysname, "n": len(g), "days": int(port.size),
                             "cumulative_return": float(eq.iloc[-1] - 1),
                             "sharpe": float(port.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
                             "buy": float((g["action"] == "Buy").mean()), "sell": float((g["action"] == "Sell").mean()),
                             "hold": float((g["action"] == "Hold").mean()),
                             "parse_ok": float(g["status"].str.startswith("ok").mean())})
            llm = rows
        figures = sorted(p.stem for p in FIGURE_DIR.glob("fig*.png")) if FIGURE_DIR.exists() else []
        # Extension models (scripts/run_extensions.py) are only evaluated under the rolling protocol.
        ext_path = STUDY_DIR / "rolling" / "extensions_summary.json"
        extensions = None
        if protocol == "rolling" and ext_path.exists():
            ext = json.loads(ext_path.read_text())
            extensions = {"portfolio": ext["portfolio"],
                          "tests": {k: {f: v.get(f) for f in ("sharpe_diff", "ci_low", "ci_high", "wilcoxon_p")}
                                    for k, v in ext["tests"].items()}}
        return _clean({"available": True, "protocol": protocol, "test_days": summary.get("test_days"),
                       "test_range": summary.get("test_range"), "portfolio": summary["portfolio"],
                       "finmem5": summary["finmem5"], "per_ticker": summary["per_ticker"], "ic": summary["ic"],
                       "tests": summary["tests"].get("Ours", {}), "curves": curves, "llm": llm,
                       "memory_stats": {k: v for k, v in summary.get("memory_stats", {}).items() if not k.startswith("_")},
                       "credibility": summary.get("memory_stats", {}).get("_credibility", {}).get("Ours", []),
                       "figures": figures, "extensions": extensions})


service = AgentService()
