#!/usr/bin/env python3
"""Capture application screenshots for the report with headless Chrome.

Usage: start the dashboard (e.g. on port 8077), then
    python scripts/capture_app_screens.py http://localhost:8077
Images go to paper/figures/app_*.png. The Connections page is skipped on purpose:
it shows the last characters of the user's API keys.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
SCALE = 2  # device pixels per CSS pixel: 2x captures print sharply on A4 (about 350 dpi at 17 cm)

SHOTS = [
    # name, query, hash, width, height, crop-to-content
    ("app_home", "lang=en&theme=light&ticker=TSLA", "home", 1440, 1500),
    ("app_home_bn", "lang=bn&theme=light&ticker=TSLA", "home", 1440, 1500),
    ("app_trade", "lang=en&theme=light&ticker=NVDA", "trade", 1440, 1250),
    ("app_autopilot", "lang=en&theme=light", "autopilot", 1440, 1900),
    ("app_autopilot_bn", "lang=bn&theme=light", "autopilot", 1440, 1900),
    ("app_split", "lang=en&theme=light&panel=split", "insights", 1440, 2000),
    ("app_pattern", "lang=en&theme=light&ticker=TSLA&panel=pattern", "insights", 1440, 2100),
    ("app_whatif", "lang=en&theme=light&panel=whatif", "insights", 1440, 1200),
    ("app_risk", "lang=en&theme=light&ticker=COIN&panel=risk", "insights", 1440, 1100),
    ("app_corr", "lang=en&theme=light&panel=corr", "insights", 1440, 900),
    ("app_algorithms", "lang=en&theme=light&ticker=NVDA&panel=algo", "insights", 1440, 5200),
    ("app_budget", "lang=en&theme=light", "settings", 1440, 1100),
    ("app_memory", "lang=en&theme=light&ticker=NVDA", "agent", 1440, 1500),
    ("app_scanner", "lang=en&theme=light", "scanner", 1440, 1000),
    ("app_research", "lang=en&theme=light", "research", 1440, 3600),
    ("app_mobile_home_bn", "lang=bn&theme=light", "home", 400, 1500),
    ("app_mobile_trade", "lang=en&theme=light&ticker=AAPL", "trade", 400, 1500),
]


def shoot(base: str, name: str, query: str, page: str, w: int, h: int, profile: Path) -> Path:
    path = OUT / f"{name}.png"
    url = f"{base}/?{query}#{page}"
    subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
                    f"--user-data-dir={profile}", f"--window-size={w},{h}", f"--force-device-scale-factor={SCALE}", "--virtual-time-budget=25000",
                    f"--screenshot={path}", url], check=True, capture_output=True, timeout=120)
    return path


def trim(path: Path) -> None:
    """Remove the empty page background below the content."""
    im = Image.open(path).convert("RGB")
    # Ignore the full-height sidebar on desktop shots; look only at the main column.
    left = 260 * SCALE if im.width > 900 * SCALE else 0
    main = im.crop((left, 0, im.width, im.height))
    bg = Image.new("RGB", main.size, main.getpixel((main.width - 2, main.height - 2)))
    box = ImageChops.difference(main, bg).getbbox()
    if box:
        im.crop((0, 0, im.width, min(im.height, box[3] + 24 * SCALE))).save(path, optimize=True)


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8077"
    only = set(sys.argv[2:])
    with tempfile.TemporaryDirectory() as tmp:
        for name, query, page, w, h in SHOTS:
            if only and name not in only:
                continue
            p = shoot(base, name, query, page, w, h, Path(tmp) / name)
            trim(p)
            print(name, Image.open(p).size)
    # The Research lab page is long: keep its top half for the report and the model-zoo card separately.
    research = OUT / "app_research.png"
    if research.exists() and (not only or "app_research" in only):
        im = Image.open(research)
        if im.height > 2700 * SCALE:
            im.crop((250 * SCALE, 2200 * SCALE, im.width, 2700 * SCALE)).save(OUT / "app_zoo.png", optimize=True)
            im.crop((0, 0, im.width, 1560 * SCALE)).save(research, optimize=True)
    # The algorithm board, the experience log and the RL agent live on one long page: save each card.
    algo = OUT / "app_algorithms.png"
    if algo.exists() and (not only or "app_algorithms" in only):
        cards = card_segments(algo)[-3:]
        im = Image.open(algo)
        for name, (top, bottom) in zip(("app_algorithms_board", "app_experience", "app_rl"), cards):
            pad = 8 * SCALE
            im.crop((250 * SCALE, max(0, top - pad), im.width, min(im.height, bottom + pad))).save(OUT / f"{name}.png", optimize=True)


def card_segments(path: Path, min_height: int = 40 * SCALE) -> list[tuple[int, int]]:
    """Vertical extents of the cards in the main column (rows separated by plain background)."""
    import numpy as np

    a = np.asarray(Image.open(path).convert("RGB"))[:, 300 * SCALE:-40 * SCALE].astype(int)
    bg = a[-3, 10]
    gap = np.abs(a - bg).sum(axis=2).max(axis=1) < 12
    segs, start = [], None
    for y, g in enumerate(gap):
        if not g and start is None:
            start = y
        if g and start is not None:
            if y - start > min_height:
                segs.append((start, y))
            start = None
    if start is not None:
        segs.append((start, len(gap)))
    return segs

if __name__ == "__main__":
    main()
