"""
Benchmark orchestration utilities.

This module provides a lightweight harness to run experiments with
reproducible seeds, timing hooks, structured metric collection, and
artifact tracking. Each benchmark run is stored in an isolated
directory so downstream tools (like the report generator) can render
HTML/Markdown summaries.
"""

from __future__ import annotations

import json
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


@dataclass
class BenchmarkRecord:
    """Structured record representing a single benchmark run."""

    name: str
    seed: int
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds")
    )
    metrics: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    artifacts: List[Dict[str, str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


class BenchmarkRun:
    """Convenience wrapper to manage benchmark bookkeeping."""

    def __init__(
        self,
        name: str,
        *,
        seed: int = 0,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        output_dir: str | Path = "results/benchmarks",
    ) -> None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.name = name
        self.record = BenchmarkRecord(
            name=name,
            seed=seed,
            config=config or {},
            metadata=metadata or {},
        )
        self.output_root = Path(output_dir)
        self.run_dir = self.output_root / f"{name}-{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        set_global_seed(seed)

    @contextmanager
    def time_section(self, label: str):
        """Context manager to time a named section."""

        start = time.perf_counter()
        try:
            yield
        finally:
            # Always record the timing even if the section raises, so partial
            # failures can still be inspected in the saved run metadata.
            self.record.timings[label] = time.perf_counter() - start

    def add_metric(self, key: str, value: Any) -> None:
        """Record a scalar or JSON-serializable metric."""

        self.record.metrics[key] = value

    def add_metrics(self, metrics: Dict[str, Any]) -> None:
        """Bulk update metrics."""

        self.record.metrics.update(metrics)

    def add_artifact(self, path: str | Path, description: Optional[str] = None) -> str:
        """Track an artifact path (e.g., plot image or config dump)."""

        artifact_path = Path(path)
        if not artifact_path.is_absolute():
            artifact_path = self.run_dir / artifact_path
        artifact_path = artifact_path.resolve()
        self.record.artifacts.append(
            {
                "path": str(artifact_path),
                "description": description or "",
            }
        )
        return str(artifact_path)

    def add_note(self, text: str) -> None:
        """Add a textual note to the run record."""

        self.record.notes.append(text)

    def save_json(self, name: str, payload: Dict[str, Any]) -> str:
        """Save a JSON payload inside the run directory and register it."""

        path = self.run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self.add_artifact(path, description=f"JSON dump: {name}")
        return str(path)

    def finalize(self) -> Dict[str, Any]:
        """Persist the run record to disk and return it."""

        run_payload = self.to_dict()
        with (self.run_dir / "run.json").open("w", encoding="utf-8") as f:
            json.dump(run_payload, f, indent=2)
        return run_payload

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary for the run."""

        return {
            "name": self.record.name,
            "seed": self.record.seed,
            "config": self.record.config,
            "metadata": self.record.metadata,
            "started_at": self.record.started_at,
            "metrics": self.record.metrics,
            "timings": self.record.timings,
            "artifacts": self.record.artifacts,
            "notes": self.record.notes,
            "run_dir": str(self.run_dir),
        }


def load_run(path: str | Path) -> Dict[str, Any]:
    """Load a previously saved benchmark run JSON."""

    run_path = Path(path)
    if run_path.is_dir():
        run_path = run_path / "run.json"
    with run_path.open("r", encoding="utf-8") as f:
        return json.load(f)

