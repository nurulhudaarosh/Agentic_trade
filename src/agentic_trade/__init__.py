"""Reproducible memory-based trading research package."""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from .engine import ExperimentConfig, ExperimentEngine, run_experiment

__all__ = ["ExperimentConfig", "ExperimentEngine", "run_experiment"]
