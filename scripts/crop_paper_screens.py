"""Prepare application screenshots for the A4 report.

Desktop captures (1440 px) include the navigation sidebar, which repeats on every page and
shrinks the content when the image is scaled to the text width. This script removes it and
splits very tall captures at a blank gap between cards, so that every part can be printed at
full width. Outputs go next to the originals as ``<name>_paper.png`` or ``<name>_paper_<k>.png``.

    python scripts/crop_paper_screens.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

FIGURES = Path(__file__).resolve().parents[1] / "paper" / "figures"
SCALE = 2  # device pixel ratio of the captures; must match SCALE in capture_app_screens.py
SIDEBAR_EDGE = 248  # first CSS-pixel column right of the sidebar border in 1440 px wide captures
KEEP_SIDEBAR = {"app_home"}  # one page keeps the navigation to show the layout
# A part may be at most this tall relative to its width (printed width 17.5 cm, about 21 cm of height).
MAX_RATIO = 1.2
SHRINK_TOLERANCE = 1.04  # split when the page height would shrink the printed width by more than 4%

SHOTS = [
    "app_home", "app_home_bn", "app_trade", "app_autopilot", "app_budget", "app_memory",
    "app_scanner", "app_research", "app_zoo", "app_pattern", "app_split", "app_autopilot_bn",
    "app_whatif", "app_risk", "app_algorithms_board", "app_experience", "app_rl",
]


def _gap_rows(img: np.ndarray) -> np.ndarray:
    """Rows that carry no content across the inner 90% of the width (gaps between cards or table rows)."""
    w = img.shape[1]
    gray = img[:, int(0.05 * w):int(0.95 * w)].mean(axis=2)
    return np.where(gray.std(axis=1) < 1.5)[0]


def _gap_runs(img: np.ndarray, min_len: int = 12) -> list[tuple[int, int]]:
    """Blank runs tall enough to be a gap between cards or table rows, not between two text lines."""
    rows = _gap_rows(img)
    runs, start = [], None
    for i, r in enumerate(rows):
        if start is None:
            start = r
        if i + 1 == len(rows) or rows[i + 1] != r + 1:
            if r - start + 1 >= min_len:
                runs.append((int(start), int(r)))
            start = None
    return runs


def _cut_points(img: np.ndarray, parts: int, scale: int = 1) -> list[int]:
    h = img.shape[0]
    centres = [(lo + hi) // 2 for lo, hi in _gap_runs(img, 12 * scale)]
    cuts = []
    for k in range(1, parts):
        target = h * k / parts
        near = [c for c in centres if abs(c - target) < 0.45 * h / parts]
        if not near:
            print(f"  warning: no blank gap near row {int(target)}, cutting through content")
            cuts.append(int(target))
            continue
        cuts.append(min(near, key=lambda c: abs(c - target)))
    return cuts


def process(name: str) -> list[str]:
    src = FIGURES / f"{name}.png"
    img = Image.open(src).convert("RGB")
    if img.width == 1440 * SCALE and name not in KEEP_SIDEBAR:
        img = img.crop((SIDEBAR_EDGE * SCALE, 0, img.width, img.height))
    arr = np.asarray(img)
    ratio = img.height / img.width
    parts = 1 if ratio <= MAX_RATIO * SHRINK_TOLERANCE else int(np.ceil(ratio / MAX_RATIO))
    for old in FIGURES.glob(f"{name}_paper*.png"):
        old.unlink()
    if parts == 1:
        out = FIGURES / f"{name}_paper.png"
        img.save(out, optimize=True)
        return [out.name]
    bounds = [0, *_cut_points(arr, parts, SCALE), img.height]
    names = []
    for k, (top, bottom) in enumerate(zip(bounds[:-1], bounds[1:]), start=1):
        out = FIGURES / f"{name}_paper_{k}.png"
        img.crop((0, top, img.width, bottom)).save(out, optimize=True)
        names.append(out.name)
    return names


def main() -> None:
    for name in SHOTS:
        print(name, "->", ", ".join(process(name)))


if __name__ == "__main__":
    main()
