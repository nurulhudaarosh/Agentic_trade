#!/usr/bin/env bash
# Quick launcher for Agentic Trade Interactive Dashboard
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

if [ -d ".venv" ]; then
    .venv/bin/python scripts/run_dashboard.py "$@"
else
    python3 scripts/run_dashboard.py "$@"
fi

