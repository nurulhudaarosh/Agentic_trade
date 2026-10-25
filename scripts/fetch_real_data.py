from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path('.env'))

TICKERS = ["TSLA", "NFLX", "AMZN", "MSFT", "COIN"]
START = "2026-06-01"
END = "2026-10-24"
DATA_DIR = Path("data")


def fetch_news() -> pd.DataFrame:
    key = os.getenv("ALPACA_KEY")
    secret = os.getenv("ALPACA_SECRET")
    if not key or not secret:
        raise RuntimeError("ALPACA_KEY and ALPACA_SECRET are required")
    rows = []
    for ticker in TICKERS:
        response = requests.get(
            "https://data.alpaca.markets/v1beta1/news",
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            params={"symbols": ticker, "start": START, "end": END, "limit": 50, "sort": "asc"},
            timeout=60,
        )
        response.raise_for_status()
        for article in response.json().get("news", []):
            created = pd.to_datetime(article.get("created_at"), utc=True)
            rows.append({
                "date": created.date().isoformat(),
                "ticker": ticker,
                "source_type": "news",
                "text": article.get("headline", ""),
                "summary": article.get("summary", ""),
                "source_id": article.get("id", ""),
            })
    frame = pd.DataFrame(rows, columns=["date", "ticker", "source_type", "text", "summary", "source_id"])
    if frame.empty:
        raise RuntimeError("Alpaca returned no news rows for the requested tickers/window")
    return frame.drop_duplicates(subset=["date", "ticker", "text"]).sort_values(["ticker", "date", "source_id"])


def fetch_prices() -> pd.DataFrame:
    import yfinance as yf
    frames = []
    for ticker in TICKERS:
        prices = yf.download(ticker, start="2026-05-25", end="2026-10-25", auto_adjust=False, progress=False)
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
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    news = fetch_news()
    prices = fetch_prices()
    news.to_csv(DATA_DIR / "multi_ticker_news.csv", index=False)
    prices.to_csv(DATA_DIR / "prices.csv", index=False)
    print(f"news_rows={len(news)} tickers={sorted(news.ticker.unique())}")
    print(f"price_rows={len(prices)} tickers={sorted(prices.ticker.unique())}")


if __name__ == "__main__":
    main()
