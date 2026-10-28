"""Dataset construction for the one-year study.

Sources
-------
- Alpaca News API (Benzinga backend, the same source FinMem used) -> shallow layer.
- SEC EDGAR XBRL company facts (10-Q / 10-K) -> intermediate / deep layers.
- Alpaca daily bars (split/dividend adjusted, SIP feed) -> prices.

Point-in-time rule: an item becomes visible at the close of the first trading day
whose 16:00 America/New_York close is at or after the item's timestamp. Decisions
are made at the close of day t and earn the close(t) -> close(t+1) return.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

UNIVERSE = ["TSLA", "NFLX", "AMZN", "MSFT", "COIN", "NVDA", "AAPL", "GOOGL", "META", "AMD"]
FINMEM_TICKERS = ["TSLA", "NFLX", "AMZN", "MSFT", "COIN"]
BENCHMARKS = ["SPY", "QQQ"]
NEWS_START = "2025-09-01"
FILINGS_START = "2024-07-01"
END = "2026-10-27"
SEC_USER_AGENT = "AgenticTrade academic research contact@agentic-trade-research.org"

ALPACA_DATA = "https://data.alpaca.markets"


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(".env"))
    except ImportError:
        pass


def _alpaca_headers() -> dict[str, str]:
    _load_env()
    key = os.getenv("ALPACA_KEY") or os.getenv("ALPACA_PAPER_KEY")
    secret = os.getenv("ALPACA_SECRET") or os.getenv("ALPACA_PAPER_SECRET")
    if not key or not secret:
        raise RuntimeError("ALPACA_KEY / ALPACA_SECRET are required to build the dataset")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _get(url: str, params: dict | None = None, headers: dict | None = None, retries: int = 6) -> dict:
    for attempt in range(retries):
        response = requests.get(url, params=params, headers=headers, timeout=60)
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        response.raise_for_status()
        return response.json()
    response.raise_for_status()
    return {}


# --------------------------------------------------------------------------- news
def fetch_news(tickers: list[str], start: str = NEWS_START, end: str = END) -> pd.DataFrame:
    headers = _alpaca_headers()
    rows: dict[int, dict] = {}
    for ticker in tickers:
        token, pages = None, 0
        while True:
            params = {"symbols": ticker, "start": f"{start}T00:00:00Z", "end": f"{end}T23:59:59Z",
                      "limit": 50, "sort": "asc", "include_content": "false"}
            if token:
                params["page_token"] = token
            data = _get(f"{ALPACA_DATA}/v1beta1/news", params, headers)
            for article in data.get("news", []):
                rows[article["id"]] = {
                    "id": int(article["id"]),
                    "created_at": article.get("created_at"),
                    "headline": (article.get("headline") or "").strip(),
                    "summary": (article.get("summary") or "").strip(),
                    "source": article.get("source") or "",
                    "author": article.get("author") or "",
                    "symbols": ",".join(article.get("symbols") or []),
                    "url": article.get("url") or "",
                }
            pages += 1
            token = data.get("next_page_token")
            if not token:
                break
        print(f"[news] {ticker}: {pages} pages, {len(rows)} unique articles so far", flush=True)
    frame = pd.DataFrame(rows.values())
    frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True)
    return frame.sort_values(["created_at", "id"]).reset_index(drop=True)


# -------------------------------------------------------------------------- bars
def fetch_bars(symbols: list[str], start: str = "2025-08-01", end: str = END) -> pd.DataFrame:
    headers = _alpaca_headers()
    frames = []
    for symbol in symbols:
        token, rows = None, []
        while True:
            params = {"timeframe": "1Day", "start": start, "end": end, "adjustment": "all",
                      "feed": "sip", "limit": 10000}
            if token:
                params["page_token"] = token
            data = _get(f"{ALPACA_DATA}/v2/stocks/{symbol}/bars", params, headers)
            rows += data.get("bars") or []
            token = data.get("next_page_token")
            if not token:
                break
        frame = pd.DataFrame(rows).rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low",
                                                   "c": "close", "v": "volume", "vw": "vwap", "n": "trades"})
        frame["date"] = pd.to_datetime(frame["timestamp"]).dt.tz_convert("America/New_York").dt.date
        frame["ticker"] = symbol
        frames.append(frame[["date", "ticker", "open", "high", "low", "close", "volume", "vwap"]])
        print(f"[bars] {symbol}: {len(frame)} days", flush=True)
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------- filings
XBRL_CONCEPTS = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
    "net_income": ["NetIncomeLoss"],
    "operating_income": ["OperatingIncomeLoss"],
    "eps": ["EarningsPerShareDiluted", "EarningsPerShareBasic"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "rnd": ["ResearchAndDevelopmentExpense"],
}


def _sec_get(url: str) -> dict:
    return _get(url, headers={"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"})


def _cik_map() -> dict[str, int]:
    data = _sec_get("https://www.sec.gov/files/company_tickers.json")
    return {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}


def _period_value(facts: dict, concept_names: list[str], accession: str, form: str) -> tuple[float | None, float | None]:
    """Return (current, prior-year comparable) values reported in a given filing."""
    for name in concept_names:
        concept = facts.get("us-gaap", {}).get(name)
        if not concept:
            continue
        units = concept["units"].get("USD") or concept["units"].get("USD/shares") or []
        reported = [u for u in units if u.get("accn") == accession and u.get("form") == form]
        if not reported:
            continue
        # Duration facts: pick the quarter (~90d) for 10-Q, the year (~365d) for 10-K.
        target = 365 if form == "10-K" else 91
        def span(u: dict) -> int:
            if "start" not in u:
                return target
            return (pd.Timestamp(u["end"]) - pd.Timestamp(u["start"])).days
        candidates = [u for u in reported if abs(span(u) - target) < 25]
        if not candidates:
            continue
        candidates.sort(key=lambda u: u["end"], reverse=True)
        current = candidates[0]
        prior_end = pd.Timestamp(current["end"]) - pd.Timedelta(days=365)
        prior = [u for u in candidates[1:] if abs((pd.Timestamp(u["end"]) - prior_end).days) < 20]
        return float(current["val"]), (float(prior[0]["val"]) if prior else None)
    return None, None


def _fmt_money(value: float) -> str:
    if abs(value) >= 1e9:
        return f"${value / 1e9:.2f}B"
    return f"${value / 1e6:.0f}M"


def _growth(current: float | None, prior: float | None) -> float | None:
    if current is None or prior is None or prior == 0:
        return None
    return (current - prior) / abs(prior)


def fetch_filings(tickers: list[str], start: str = FILINGS_START, end: str = END) -> pd.DataFrame:
    ciks = _cik_map()
    rows = []
    for ticker in tickers:
        cik = ciks.get(ticker)
        if cik is None:
            continue
        facts = _sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")["facts"]
        filings: dict[str, tuple[str, str, str, str]] = {}
        for concept in facts.get("us-gaap", {}).values():
            for unit in concept["units"].values():
                for u in unit:
                    if u.get("form") in ("10-Q", "10-K") and start <= u.get("filed", "") <= end:
                        filings[u["accn"]] = (u["form"], u["filed"], str(u.get("fy")), str(u.get("fp")))
        for accession, (form, filed, fy, fp) in sorted(filings.items(), key=lambda item: item[1][1]):
            values = {key: _period_value(facts, names, accession, form) for key, names in XBRL_CONCEPTS.items()}
            revenue, revenue_prior = values["revenue"]
            income, income_prior = values["net_income"]
            op_income, _ = values["operating_income"]
            eps, eps_prior = values["eps"]
            parts = []
            rev_g, inc_g, eps_g = _growth(revenue, revenue_prior), _growth(income, income_prior), _growth(eps, eps_prior)
            if revenue is not None:
                parts.append(f"revenue {_fmt_money(revenue)}" + (f" ({rev_g:+.1%} YoY)" if rev_g is not None else ""))
            if income is not None:
                parts.append(f"net income {_fmt_money(income)}" + (f" ({inc_g:+.1%} YoY)" if inc_g is not None else ""))
            if eps is not None:
                parts.append(f"diluted EPS ${eps:.2f}" + (f" vs ${eps_prior:.2f} a year earlier" if eps_prior is not None else ""))
            if op_income is not None and revenue:
                parts.append(f"operating margin {op_income / revenue:.1%}")
            if not parts:
                continue
            period = "fiscal year" if form == "10-K" else f"fiscal {fp}"
            text = f"{ticker} {form} for {period} {fy} filed {filed}: " + "; ".join(parts) + "."
            growth = [g for g in (rev_g, inc_g, eps_g) if g is not None]
            score = float(np.tanh(3 * np.mean(growth))) if growth else 0.0
            rows.append({"id": f"sec-{accession}", "ticker": ticker, "form": form, "filed": filed,
                         "fiscal_year": fy, "fiscal_period": fp, "text": text,
                         "fundamental_score": score, "revenue_growth": rev_g, "income_growth": inc_g,
                         "eps_growth": eps_g})
        print(f"[sec] {ticker}: {sum(r['ticker'] == ticker for r in rows)} filings", flush=True)
        time.sleep(0.2)
    return pd.DataFrame(rows)


# --------------------------------------------------------------- PIT alignment
def assign_trading_day(timestamps: pd.Series, trading_days: list) -> pd.Series:
    """Map UTC timestamps to the first trading day whose 16:00 ET close is >= the timestamp."""
    days = pd.DatetimeIndex(pd.to_datetime(trading_days))
    closes = days.tz_localize("America/New_York") + pd.Timedelta(hours=16)
    stamps = pd.to_datetime(timestamps, utc=True).dt.tz_convert("America/New_York")
    positions = np.searchsorted(closes.asi8, stamps.astype("int64").to_numpy(), side="left")
    result = [days[p].date() if p < len(days) else None for p in positions]
    return pd.Series(result, index=timestamps.index)


def build_dataset(out_dir: Path, tickers: list[str] = UNIVERSE) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: out_dir / f"{name}.parquet" for name in ("news", "prices", "filings")}
    if not paths["prices"].exists():
        fetch_bars(tickers + BENCHMARKS).to_parquet(paths["prices"], index=False)
    prices = pd.read_parquet(paths["prices"])
    trading_days = sorted(prices.loc[prices["ticker"].eq("SPY"), "date"].unique())

    if not paths["news"].exists():
        news = fetch_news(tickers)
        news["trade_date"] = assign_trading_day(news["created_at"], trading_days)
        news = news.dropna(subset=["trade_date"])
        news["headline"] = news["headline"].fillna("")
        news = news[news["headline"].str.len() > 0]
        news.to_parquet(paths["news"], index=False)
    if not paths["filings"].exists():
        filings = fetch_filings(tickers)
        # SEC "filed" is a date; treat as available after that day's close (conservative).
        # Filings older than the price history map to the first price day, which is
        # fine: they only act as long-horizon memory before the evaluation window.
        filed = pd.to_datetime(filings["filed"]).dt.tz_localize("America/New_York") + pd.Timedelta(hours=20)
        filings["trade_date"] = assign_trading_day(filed.dt.tz_convert("UTC"), trading_days)
        filings.to_parquet(paths["filings"], index=False)
    meta = {"tickers": tickers, "benchmarks": BENCHMARKS, "news_start": NEWS_START, "filings_start": FILINGS_START,
            "end": END, "built_at": pd.Timestamp.utcnow().isoformat()}
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2))
    return paths



def last_completed_session() -> str:
    """SIP bars are only free once >15 min old; the previous calendar day is always safe."""
    return (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).date().isoformat()


def refresh_dataset(data_dir: Path, tickers: list[str] = UNIVERSE, log=print) -> dict:
    """Incrementally extend the dataset to the last completed session.

    Bars and filings are re-downloaded (small); news is fetched from three days before
    the newest stored article, de-duplicated, and only new rows are annotated. Rows are
    appended so that ``news_annotated.parquet`` and ``news_emb.npy`` stay aligned.
    """
    from .annotate import embed, finbert_scores, item_text

    end = last_completed_session()
    log(f"fetching bars to {end}")
    bars = fetch_bars(tickers + BENCHMARKS, end=end)
    bars.to_parquet(data_dir / "prices.parquet", index=False)
    trading_days = sorted(bars.loc[bars["ticker"].eq("SPY"), "date"].unique())

    annotated = pd.read_parquet(data_dir / "news_annotated.parquet")
    emb = np.load(data_dir / "news_emb.npy")
    since = (pd.to_datetime(annotated["created_at"], utc=True).max() - pd.Timedelta(days=3)).date().isoformat()
    log(f"fetching news since {since}")
    fresh = fetch_news(tickers, start=since, end=end)
    fresh = fresh[~fresh["id"].isin(set(annotated["id"]))]
    fresh = fresh[fresh["headline"].fillna("").str.len() > 0].reset_index(drop=True)
    added = 0
    if len(fresh):
        log(f"annotating {len(fresh)} new articles")
        probs = finbert_scores([item_text(h) for h in fresh["headline"]])
        fresh["p_pos"], fresh["p_neg"], fresh["p_neu"] = probs[:, 0], probs[:, 1], probs[:, 2]
        fresh["sentiment"] = fresh["p_pos"] - fresh["p_neg"]
        vectors = embed([item_text(h, s) for h, s in zip(fresh["headline"], fresh["summary"])])
        annotated = pd.concat([annotated, fresh], ignore_index=True)
        emb = np.concatenate([emb, vectors.astype(emb.dtype)])
        added = len(fresh)
    # Re-align every article (new rows, and old rows that were waiting for a session).
    annotated["trade_date"] = assign_trading_day(annotated["created_at"], trading_days)
    keep = annotated["trade_date"].notna().to_numpy()
    annotated.to_parquet(data_dir / "news_annotated.parquet", index=False)
    np.save(data_dir / "news_emb.npy", emb)
    if not keep.all():
        log(f"{int((~keep).sum())} articles arrived after the last close and wait for the next session")

    log("refreshing SEC filings")
    filings = fetch_filings(tickers, end=end)
    filed = pd.to_datetime(filings["filed"]).dt.tz_localize("America/New_York") + pd.Timedelta(hours=20)
    filings["trade_date"] = assign_trading_day(filed.dt.tz_convert("UTC"), trading_days)
    filings = filings.dropna(subset=["trade_date"])
    filings["sentiment"] = filings["fundamental_score"].astype(float)
    from .annotate import embed as _embed

    np.save(data_dir / "filings_emb.npy", _embed(filings["text"].tolist()))
    filings.to_parquet(data_dir / "filings_annotated.parquet", index=False)
    return {"end": end, "new_articles": added, "last_session": str(trading_days[-1]), "filings": len(filings)}
