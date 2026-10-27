#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

# Ensure src is importable when running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path('.env'))
except ImportError:
    pass

DEFAULT_TICKERS = ["TSLA", "NFLX", "AMZN", "MSFT", "COIN"]
DEFAULT_START = "2026-06-01"
DEFAULT_END = "2026-10-24"
DEFAULT_DATA_DIR = Path("data")


def fetch_news(
    tickers: list[str] = DEFAULT_TICKERS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    max_pages_per_ticker: int = 10,
) -> pd.DataFrame:
    key = os.getenv("ALPACA_KEY") or os.getenv("ALPACA_PAPER_KEY")
    secret = os.getenv("ALPACA_SECRET") or os.getenv("ALPACA_PAPER_SECRET")
    if not key or not secret:
        raise RuntimeError("ALPACA_KEY and ALPACA_SECRET (or ALPACA_PAPER_*) are required")

    rows = []
    for ticker in tickers:
        page_token = None
        pages = 0
        while pages < max_pages_per_ticker:
            params: dict[str, str | int] = {
                "symbols": ticker,
                "start": start,
                "end": end,
                "limit": 50,
                "sort": "asc",
            }
            if page_token:
                params["page_token"] = page_token

            response = requests.get(
                "https://data.alpaca.markets/v1beta1/news",
                headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
                params=params,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            articles = data.get("news", [])
            if not articles:
                break

            for article in articles:
                created = pd.to_datetime(article.get("created_at"), utc=True)
                rows.append({
                    "date": created.date().isoformat(),
                    "ticker": ticker,
                    "source_type": "news",
                    "text": article.get("headline", ""),
                    "summary": article.get("summary", ""),
                    "source_name": article.get("source", ""),
                    "source_id": article.get("id", ""),
                })

            pages += 1
            page_token = data.get("next_page_token")
            if not page_token:
                break

    frame = pd.DataFrame(rows, columns=["date", "ticker", "source_type", "text", "summary", "source_name", "source_id"])
    if frame.empty:
        raise RuntimeError("Alpaca returned no news rows for the requested tickers/window")
    return frame.drop_duplicates(subset=["date", "ticker", "text"]).sort_values(["ticker", "date", "source_id"])


def fetch_prices(
    tickers: list[str] = DEFAULT_TICKERS,
    start: str = "2026-05-25",
    end: str = "2026-10-25",
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install yfinance before running fetch_prices: pip install yfinance") from exc

    frames = []
    for ticker in tickers:
        prices = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if prices.empty:
            raise RuntimeError(f"No yfinance prices returned for {ticker}")
        close = prices["Close"]
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]
        frame = close.rename("close").reset_index()
        frame.columns = ["date", "close"]
        frame["date"] = pd.to_datetime(frame["date"]).dt.date.astype(str)
        frame["ticker"] = ticker
        frames.append(frame[["date", "ticker", "close"]])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["date", "ticker"]).sort_values(["ticker", "date"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh market news and price data from live providers")
    parser.add_argument("--tickers", default="TSLA,NFLX,AMZN,MSFT,COIN")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--prices-only", action="store_true")
    parser.add_argument("--news-only", action="store_true")
    args = parser.parse_args()

    tickers = [item.strip().upper() for item in args.tickers.split(",") if item.strip()]
    args.data_dir.mkdir(parents=True, exist_ok=True)

    if not args.prices_only:
        print(f"Fetching news for {tickers}...")
        news = fetch_news(tickers, args.start, args.end, args.max_pages)
        news.to_csv(args.data_dir / "multi_ticker_news.csv", index=False)
        print(f"Saved news_rows={len(news)} tickers={sorted(news.ticker.unique())}")

    if not args.news_only:
        print(f"Fetching prices for {tickers}...")
        prices = fetch_prices(tickers, start=args.start, end=args.end)
        prices.to_csv(args.data_dir / "prices.csv", index=False)
        print(f"Saved price_rows={len(prices)} tickers={sorted(prices.ticker.unique())}")


if __name__ == "__main__":
    main()
