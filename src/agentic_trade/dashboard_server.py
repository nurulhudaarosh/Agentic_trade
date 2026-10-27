"""
High-performance Dashboard Server for Agentic Trade Research & Paper Trading.
Provides zero-dependency HTTP server with REST APIs for metrics, equity curves,
memory evidence logs, research benchmarks, and paper trading orders.
"""

from __future__ import annotations

import argparse
import csv
import http.server
import json
import os
import socketserver
import subprocess
import sys
import threading
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None


def find_workspace_root() -> Path:
    """Find the project root containing artifacts or pyproject.toml."""
    curr = Path(__file__).resolve().parent
    for _ in range(4):
        if (curr / "pyproject.toml").exists() or (curr / "artifacts").exists():
            return curr
        curr = curr.parent
    return Path.cwd()


WORKSPACE_ROOT = find_workspace_root()
ARTIFACTS_DIR = WORKSPACE_ROOT / "artifacts"
DATA_DIR = WORKSPACE_ROOT / "data"
DASHBOARD_DIR = WORKSPACE_ROOT / "dashboard"


def clean_val(val: Any) -> Any:
    """Ensure floating NaNs and Infs become None for JSON."""
    if val is None:
        return None
    if isinstance(val, float) and (val != val or val == float("inf") or val == float("-inf")):
        return None
    return val


def clean_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: clean_val(v) for k, v in d.items()}


def get_summary_data() -> Dict[str, Any]:
    """Overall summary data for header and KPI ribbon."""
    summary: Dict[str, Any] = {
        "project": "Agentic Trade Research",
        "description": "Cognitive Trading Memory System (FinMem Base + 4 Architectural Novelties)",
        "datasets": ["offline"],
        "tickers": ["TSLA"],
        "trials": [0, 1, 2],
        "has_offline": (ARTIFACTS_DIR / "offline" / "metrics.json").exists(),
        "has_matrix": (ARTIFACTS_DIR / "research" / "matrix" / "table2_metrics.csv").exists(),
        "has_orders": (ARTIFACTS_DIR / "paper_trading" / "orders.jsonl").exists(),
    }

    # Discover available tickers and datasets from matrix
    matrix_csv = ARTIFACTS_DIR / "research" / "matrix" / "table2_metrics.csv"
    if matrix_csv.exists() and pd is not None:
        try:
            df = pd.read_csv(matrix_csv)
            tickers = sorted(df["ticker"].dropna().unique().tolist())
            summary["tickers"] = tickers
            summary["trials"] = sorted(df["trial"].dropna().unique().astype(int).tolist())
        except Exception:
            pass

    # Wilcoxon summary
    wilcoxon_json = ARTIFACTS_DIR / "research" / "matrix" / "wilcoxon.json"
    if wilcoxon_json.exists():
        try:
            with open(wilcoxon_json) as f:
                summary["wilcoxon"] = json.load(f)
        except Exception:
            summary["wilcoxon"] = None

    return summary


def get_equity_data(dataset: str = "offline", ticker: str = "TSLA", trial: int = 0) -> Dict[str, Any]:
    """
    Returns time-series returns and daily decisions for all systems.
    Systems: ours, fin_mem, plain_rag, no_memory, buy_and_hold.
    """
    systems = ["ours", "fin_mem", "plain_rag", "no_memory", "buy_and_hold"]
    series_map: Dict[str, List[Dict[str, Any]]] = {s: [] for s in systems}
    metrics_map: Dict[str, Dict[str, Any]] = {}
    dates: List[str] = []
    prices: List[float] = []

    if dataset in ["offline", "llm_eval"]:
        target_dir = ARTIFACTS_DIR / dataset
        # Load metrics.json
        metrics_file = target_dir / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    metrics_map = json.load(f)
            except Exception:
                pass

        # Load daily CSVs
        for s in ["ours", "fin_mem", "plain_rag", "no_memory"]:
            daily_file = target_dir / f"{s}_daily.csv"
            if daily_file.exists() and pd is not None:
                try:
                    df = pd.read_csv(daily_file)
                    df["cum_return"] = (1.0 + df["daily_return"]).cumprod() - 1.0
                    for _, row in df.iterrows():
                        series_map[s].append({
                            "date": str(row["date"]),
                            "return": float(row["cum_return"]),
                            "daily_return": float(row["daily_return"]),
                            "decision": str(row["decision"]),
                            "position": float(row["position"]),
                            "price": float(row["price"]),
                        })
                    if s == "ours" and not dates:
                        dates = [str(d) for d in df["date"]]
                        prices = [float(p) for p in df["price"]]
                except Exception:
                    pass

        # Buy and hold series from ours or no_memory
        if series_map["no_memory"]:
            daily_file = target_dir / "no_memory_daily.csv"
            if daily_file.exists() and pd is not None:
                try:
                    df = pd.read_csv(daily_file)
                    df["cum_bnh"] = (1.0 + df["next_return"]).cumprod() - 1.0
                    for _, row in df.iterrows():
                        series_map["buy_and_hold"].append({
                            "date": str(row["date"]),
                            "return": float(row["cum_bnh"]),
                            "daily_return": float(row["next_return"]),
                            "decision": "Hold",
                            "position": 1.0,
                            "price": float(row["price"]),
                        })
                except Exception:
                    pass

    else:
        # Load from research matrix: artifacts/research/matrix/daily_long.csv
        daily_long_csv = ARTIFACTS_DIR / "research" / "matrix" / "daily_long.csv"
        metrics_csv = ARTIFACTS_DIR / "research" / "matrix" / "table2_metrics.csv"

        if metrics_csv.exists() and pd is not None:
            try:
                m_df = pd.read_csv(metrics_csv)
                sub_m = m_df[(m_df["ticker"] == ticker) & (m_df["trial"] == trial)]
                for _, row in sub_m.iterrows():
                    metrics_map[str(row["system"])] = clean_dict(row.to_dict())
            except Exception:
                pass

        if daily_long_csv.exists() and pd is not None:
            try:
                d_df = pd.read_csv(daily_long_csv)
                sub_d = d_df[(d_df["ticker"] == ticker) & (d_df["trial"] == trial)]

                for s in ["ours", "fin_mem", "plain_rag", "no_memory"]:
                    sys_sub = sub_d[sub_d["system"] == s].sort_values("date")
                    if not sys_sub.empty:
                        sys_sub["cum_return"] = (1.0 + sys_sub["daily_return"]).cumprod() - 1.0
                        for _, row in sys_sub.iterrows():
                            series_map[s].append({
                                "date": str(row["date"]),
                                "return": float(row["cum_return"]),
                                "daily_return": float(row["daily_return"]),
                                "decision": str(row["decision"]),
                                "position": float(row["position"]),
                                "price": float(row["price"]),
                            })
                        if s == "ours" and not dates:
                            dates = [str(d) for d in sys_sub["date"]]
                            prices = [float(p) for p in sys_sub["price"]]

                # Buy and hold
                ref = sub_d[sub_d["system"] == "no_memory"].sort_values("date")
                if not ref.empty:
                    ref["cum_bnh"] = (1.0 + ref["next_return"]).cumprod() - 1.0
                    for _, row in ref.iterrows():
                        series_map["buy_and_hold"].append({
                            "date": str(row["date"]),
                            "return": float(row["cum_bnh"]),
                            "daily_return": float(row["next_return"]),
                            "decision": "Hold",
                            "position": 1.0,
                            "price": float(row["price"]),
                        })
            except Exception:
                pass

    return {
        "dataset": dataset,
        "ticker": ticker,
        "trial": trial,
        "dates": dates,
        "prices": prices,
        "series": series_map,
        "metrics": metrics_map,
    }


def get_evidence_data(
    dataset: str = "offline",
    ticker: Optional[str] = None,
    system: Optional[str] = None,
    layer: Optional[str] = None,
    cited: Optional[bool] = None,
    active: Optional[bool] = None,
    outcome_recalled: Optional[bool] = None,
    search: Optional[str] = None,
    page: int = 1,
    limit: int = 50,
) -> Dict[str, Any]:
    """Retrieve filtered evidence log items."""
    file_path = ARTIFACTS_DIR / "offline" / "evidence_log.csv"
    if dataset == "llm_eval":
        file_path = ARTIFACTS_DIR / "llm_eval" / "evidence_log.csv"
    elif dataset != "offline":
        file_path = ARTIFACTS_DIR / "research" / "matrix" / "evidence_log.csv"

    if not file_path.exists():
        return {"items": [], "total": 0, "page": page, "limit": limit}

    items: List[Dict[str, Any]] = []

    with open(file_path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Filters
            if ticker and row.get("ticker", "").strip().upper() != ticker.upper():
                continue
            if system and row.get("system", "").strip().lower() != system.lower():
                continue
            if layer and row.get("layer", "").strip().lower() != layer.lower():
                continue
            if cited is not None:
                is_cited = str(row.get("cited", "")).strip().lower() in ["true", "1", "yes"]
                if is_cited != cited:
                    continue
            if active is not None:
                is_active = str(row.get("active", "")).strip().lower() in ["true", "1", "yes"]
                if is_active != active:
                    continue
            if outcome_recalled is not None:
                is_recalled = str(row.get("outcome_recalled", "")).strip().lower() in ["true", "1", "yes"]
                if is_recalled != outcome_recalled:
                    continue
            if search:
                query = search.strip().lower()
                text = (row.get("text", "") + " " + row.get("topic", "") + " " + row.get("source", "")).lower()
                if query not in text:
                    continue

            # Parsing numeric fields
            rel = None
            try:
                rel = round(float(row.get("reliability", 0.0)), 3)
            except Exception:
                pass

            ret = None
            try:
                ret = round(float(row.get("realized_return", 0.0)), 4)
            except Exception:
                pass

            item = {
                "date": row.get("date", ""),
                "memory_id": row.get("memory_id", ""),
                "layer": row.get("layer", ""),
                "source": row.get("source", ""),
                "text": row.get("text", ""),
                "topic": row.get("topic", ""),
                "related_tickers": row.get("related_tickers", ""),
                "reliability": rel,
                "active": str(row.get("active", "")).lower() in ["true", "1"],
                "rank": row.get("rank", ""),
                "cited": str(row.get("cited", "")).lower() in ["true", "1"],
                "decision": row.get("decision", ""),
                "realized_return": ret,
                "system": row.get("system", ""),
                "ticker": row.get("ticker", ""),
                "outcome_recalled": str(row.get("outcome_recalled", "")).lower() in ["true", "1"],
            }
            items.append(item)

    total = len(items)
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated_items = items[start_idx:end_idx]

    return {
        "items": paginated_items,
        "total": total,
        "page": page,
        "limit": limit,
    }


def get_benchmarks_data() -> Dict[str, Any]:
    """Retrieve Table 2, Table 4, Table 6, Table 7, and Wilcoxon."""
    res: Dict[str, Any] = {
        "table2": [],
        "table4": [],
        "table6": [],
        "table7": [],
        "wilcoxon": None,
    }

    # Table 2
    t2_file = ARTIFACTS_DIR / "research" / "matrix" / "table2_metrics.csv"
    if t2_file.exists() and pd is not None:
        try:
            df = pd.read_csv(t2_file).replace({np.nan: None})
            res["table2"] = df.to_dict(orient="records")
        except Exception:
            pass

    # Table 4
    t4_file = ARTIFACTS_DIR / "research" / "risk_profiles" / "table4_risk_profiles.csv"
    if t4_file.exists() and pd is not None:
        try:
            df = pd.read_csv(t4_file).replace({np.nan: None})
            res["table4"] = df.to_dict(orient="records")
        except Exception:
            pass

    # Table 6
    t6_file = ARTIFACTS_DIR / "research" / "matrix" / "table6_ablation.csv"
    if t6_file.exists() and pd is not None:
        try:
            df = pd.read_csv(t6_file).replace({np.nan: None})
            res["table6"] = df.to_dict(orient="records")
        except Exception:
            pass

    # Table 7
    t7_file = ARTIFACTS_DIR / "research" / "matrix" / "table7_probes.csv"
    if t7_file.exists() and pd is not None:
        try:
            df = pd.read_csv(t7_file).replace({np.nan: None})
            res["table7"] = df.to_dict(orient="records")
        except Exception:
            pass

    # Wilcoxon
    w_file = ARTIFACTS_DIR / "research" / "matrix" / "wilcoxon.json"
    if w_file.exists():
        try:
            with open(w_file) as f:
                res["wilcoxon"] = json.load(f)
        except Exception:
            pass

    return res


def get_orders_data() -> Dict[str, Any]:
    """Retrieve paper trading orders and latest decisions."""
    orders = []
    orders_file = ARTIFACTS_DIR / "paper_trading" / "orders.jsonl"
    if orders_file.exists():
        try:
            with open(orders_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        orders.append(json.loads(line))
        except Exception:
            pass

    decisions = []
    decisions_csv = WORKSPACE_ROOT / "decisions.csv"
    if decisions_csv.exists() and pd is not None:
        try:
            df = pd.read_csv(decisions_csv).replace({np.nan: None})
            decisions = df.to_dict(orient="records")
        except Exception:
            pass

    return {
        "orders": orders[::-1],  # Latest first
        "decisions": decisions,
        "order_count": len(orders),
    }


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler serving index.html and JSON REST API endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        # Keep terminal log clean; only log errors or crucial queries
        if args and str(args[0]).startswith("50"):
            super().log_message(format, *args)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            index_path = DASHBOARD_DIR / "index.html"
            if index_path.exists():
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                with open(index_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"index.html not found.")
                return

        # API: /api/summary
        if path == "/api/summary":
            data = get_summary_data()
            self._send_json(data)
            return

        # API: /api/equity
        if path == "/api/equity":
            dataset = query.get("dataset", ["offline"])[0]
            ticker = query.get("ticker", ["TSLA"])[0]
            trial_str = query.get("trial", ["0"])[0]
            try:
                trial = int(trial_str)
            except ValueError:
                trial = 0
            data = get_equity_data(dataset=dataset, ticker=ticker, trial=trial)
            self._send_json(data)
            return

        # API: /api/evidence
        if path == "/api/evidence":
            dataset = query.get("dataset", ["offline"])[0]
            ticker = query.get("ticker", [None])[0]
            system = query.get("system", [None])[0]
            layer = query.get("layer", [None])[0]
            search = query.get("search", [None])[0]
            cited_str = query.get("cited", [None])[0]
            active_str = query.get("active", [None])[0]
            recalled_str = query.get("outcome_recalled", [None])[0]
            page_str = query.get("page", ["1"])[0]
            limit_str = query.get("limit", ["50"])[0]

            cited = None
            if cited_str is not None:
                cited = cited_str.lower() in ["true", "1"]

            active = None
            if active_str is not None:
                active = active_str.lower() in ["true", "1"]

            recalled = None
            if recalled_str is not None:
                recalled = recalled_str.lower() in ["true", "1"]

            try:
                page = max(1, int(page_str))
            except ValueError:
                page = 1
            try:
                limit = min(200, max(10, int(limit_str)))
            except ValueError:
                limit = 50

            data = get_evidence_data(
                dataset=dataset,
                ticker=ticker,
                system=system,
                layer=layer,
                cited=cited,
                active=active,
                outcome_recalled=recalled,
                search=search,
                page=page,
                limit=limit,
            )
            self._send_json(data)
            return

        # API: /api/benchmarks
        if path == "/api/benchmarks":
            data = get_benchmarks_data()
            self._send_json(data)
            return

        # API: /api/orders
        if path == "/api/orders":
            data = get_orders_data()
            self._send_json(data)
            return

        # Fallback to default static file handling
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/run_offline":
            # Run offline experiment
            try:
                cmd = [sys.executable, "run_experiment.py", "--data-dir", "data", "--output-dir", "artifacts/offline"]
                res = subprocess.run(cmd, cwd=str(WORKSPACE_ROOT), capture_output=True, text=True, timeout=120)
                out = res.stdout + ("\nSTDERR:\n" + res.stderr if res.stderr else "")
                self._send_json({"success": res.returncode == 0, "output": out})
            except Exception as e:
                self._send_json({"success": False, "error": str(e)}, status=500)
            return

        if path == "/api/run_paper_dryrun":
            # Run dry run paper trading
            try:
                cmd = [sys.executable, "-m", "agentic_trade.paper_trader", "--decisions", "decisions.csv", "--budget", "100"]
                res = subprocess.run(cmd, cwd=str(WORKSPACE_ROOT), capture_output=True, text=True, timeout=60)
                out = res.stdout + ("\nSTDERR:\n" + res.stderr if res.stderr else "")
                self._send_json({"success": res.returncode == 0, "output": out})
            except Exception as e:
                self._send_json({"success": False, "error": str(e)}, status=500)
            return

        self.send_response(404)
        self.end_headers()

    def _send_json(self, data: Any, status: int = 200) -> None:
        blob = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(blob)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(blob)


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_server(port: int = 8050, host: str = "127.0.0.1") -> None:
    # Ensure dashboard directory exists
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    # If port is occupied, scan upwards
    actual_port = port
    for p in range(port, port + 20):
        try:
            httpd = ThreadedHTTPServer((host, p), DashboardHandler)
            actual_port = p
            break
        except OSError:
            continue
    else:
        print(f"Error: Could not bind to any port in range {port}-{port+20}")
        sys.exit(1)

    print("\n" + "=" * 65)
    print(" 🚀 AGENTIC TRADE RESEARCH DASHBOARD RUNNING")
    print(f" 🌐 URL: http://{host}:{actual_port}/")
    print(" 📈 System Comparison: Ours vs FinMem vs Plain RAG vs Buy & Hold")
    print(" 🧠 Features: Memory Inspector, Research Tables 2-7, Alpaca Paper")
    print(" ⌨️  Press Ctrl+C to terminate.")
    print("=" * 65 + "\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server...")
        httpd.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic Trade Dashboard Server")
    parser.add_argument("--port", type=int, default=8050, help="Port to listen on (default 8050)")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default 127.0.0.1)")
    args = parser.parse_args()
    start_server(port=args.port, host=args.host)


if __name__ == "__main__":
    main()

