"""Hyperparameter sweeps for BQNN benchmarks.

This module provides a small, dependency-light sweep runner that plugs into the
existing :class:`bqnn.benchmark.BenchmarkRun` harness.

Design goals
- Reproducible: explicit seeds and per-trial seed offsets
- Simple: plain Python + PyTorch; no external sweep frameworks
- Useful artifacts: JSON/CSV tables + a few aggregated plots
"""

from __future__ import annotations

import csv
import itertools
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import torch

from .benchmark import BenchmarkRun, set_global_seed
from .data import get_synthetic_dataset
from .inference import evaluate_full
from .model import BQNNModel
from .training import train_epoch
from .utils.plots import aggregate_metrics, plot_metric_bar, plot_metric_curve, save_figure


SearchType = Literal["grid", "random"]


@dataclass(frozen=True)
class SweepConfig:
    """Configuration for a sweep run."""

    name: str = "bqnn_sweep"
    output_dir: str | Path = "results/sweeps"

    # Dataset / model sizing
    n_features: int = 16
    n_hidden: int = 8
    n_classes: int = 2
    n_qubits: Optional[int] = None  # defaults to n_hidden

    # Dataset sizes
    n_train: int = 1024
    n_test: int = 256
    batch_size: int = 64

    # Training loop
    epochs: int = 5
    device: Optional[str] = None

    # Optional PennyLane shots (if the underlying model is shot-based)
    shots: Optional[int] = None

    # Seeds (train/test differ by default to reduce leakage)
    train_seed: int = 42
    test_seed: int = 12345

    # Extra info you might want to persist (git hash, machine, notes, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SweepSpace:
    """Discrete search space for simple sweeps."""

    a: Sequence[float] = (0.0, 0.5, 1.0)
    lr: Sequence[float] = (1e-3, 5e-4)
    noise_pairs: Sequence[int] = (0, 2)
    noise_angle: Sequence[float] = (0.0, 0.05)

    def grid(self) -> Iterable[Dict[str, Any]]:
        for a, lr, n, ang in itertools.product(self.a, self.lr, self.noise_pairs, self.noise_angle):
            yield {"a": float(a), "lr": float(lr), "noise_pairs": int(n), "noise_angle": float(ang)}


def _select_candidates(
    space: SweepSpace,
    *,
    search: SearchType,
    num_samples: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    candidates = list(space.grid())

    if search == "random":
        rng = random.Random(seed)
        if num_samples is None or num_samples >= len(candidates):
            rng.shuffle(candidates)
            return candidates
        return rng.sample(candidates, num_samples)

    return candidates


def _prepare_loaders(cfg: SweepConfig, *, seed_offset: int = 0):
    train_loader = get_synthetic_dataset(
        n_samples=cfg.n_train,
        n_features=cfg.n_features,
        n_classes=cfg.n_classes,
        batch_size=cfg.batch_size,
        seed=cfg.train_seed + seed_offset,
    )
    test_loader = get_synthetic_dataset(
        n_samples=cfg.n_test,
        n_features=cfg.n_features,
        n_classes=cfg.n_classes,
        batch_size=cfg.batch_size,
        seed=cfg.test_seed + seed_offset,
    )
    return train_loader, test_loader


def _run_trial(
    cfg: SweepConfig,
    params: Dict[str, Any],
    train_loader,
    test_loader,
    *,
    device: str,
) -> Dict[str, Any]:
    model = BQNNModel(
        n_features=cfg.n_features,
        n_hidden=cfg.n_hidden,
        n_classes=cfg.n_classes,
        n_qubits=cfg.n_qubits or cfg.n_hidden,
        a=float(params["a"]),
        shots=cfg.shots,
    )
    model.set_noise(int(params["noise_pairs"]), angle=float(params["noise_angle"]))

    optimizer = torch.optim.Adam(model.parameters(), lr=float(params["lr"]))
    last_stats: Dict[str, float] = {}

    for _ in range(cfg.epochs):
        last_stats = train_epoch(
            model,
            train_loader,
            device=device,
            lr=float(params["lr"]),
            optimizer=optimizer,
        )

    metrics = evaluate_full(model, test_loader, device=device, n_classes=cfg.n_classes)
    return {
        **params,
        **metrics,
        "final_loss": float(last_stats.get("loss", 0.0)),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return str(path)

    # Deterministic column order: params first, then metrics
    preferred = [
        "seed",
        "a",
        "lr",
        "noise_pairs",
        "noise_angle",
        "accuracy",
        "macro_f1",
        "final_loss",
        "mean_confidence",
        "confidence_std",
        "n_samples",
    ]
    keys = list(dict.fromkeys(preferred + sorted({k for r in rows for k in r.keys()})))

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in keys})
    return str(path)


def _best_row(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    if not rows:
        return {}
    return max(rows, key=lambda r: float(r.get(key, 0.0)))


def _generate_plots(rows: List[Dict[str, Any]], run_dir: Path) -> List[str]:
    run_dir = Path(run_dir)
    artifacts: List[str] = []

    acc_by_a = aggregate_metrics(rows, "a", "accuracy")
    f1_by_a = aggregate_metrics(rows, "a", "macro_f1")
    acc_by_noise = aggregate_metrics(rows, "noise_pairs", "accuracy")
    f1_by_noise = aggregate_metrics(rows, "noise_pairs", "macro_f1")

    figures = [
        (plot_metric_curve(acc_by_a, param_label="a", metric_label="Accuracy"), run_dir / "acc_vs_a"),
        (plot_metric_curve(f1_by_a, param_label="a", metric_label="Macro F1"), run_dir / "f1_vs_a"),
        (plot_metric_bar(acc_by_noise, param_label="Noise pairs", metric_label="Accuracy"), run_dir / "acc_vs_noise_pairs"),
        (plot_metric_bar(f1_by_noise, param_label="Noise pairs", metric_label="Macro F1"), run_dir / "f1_vs_noise_pairs"),
    ]

    for fig, base in figures:
        try:
            artifacts.extend(save_figure(fig, str(base)))
        finally:
            import matplotlib.pyplot as plt
            plt.close(fig)

    return artifacts


def run_sweep(
    cfg: SweepConfig,
    space: SweepSpace,
    *,
    search: SearchType = "grid",
    num_samples: Optional[int] = None,
    seed: int = 0,
    make_plots: bool = True,
) -> List[Dict[str, Any]]:
    """Run a sweep and persist a BenchmarkRun directory.

    Returns the per-trial rows (dicts). Artifacts are written into the created
    run directory inside ``cfg.output_dir``.
    """

    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

    runner = BenchmarkRun(
        cfg.name,
        seed=seed,
        config={
            "search": search,
            "num_samples": num_samples,
            "space": asdict(space),
            "sweep_config": asdict(cfg),
        },
        metadata=cfg.metadata,
        output_dir=cfg.output_dir,
    )

    candidates = _select_candidates(space, search=search, num_samples=num_samples, seed=seed)
    set_global_seed(seed)

    rows: List[Dict[str, Any]] = []
    for idx, params in enumerate(candidates):
        trial_seed = seed + idx
        set_global_seed(trial_seed)

        # Keep dataset fixed across trials by default, but allow easy diversification:
        train_loader, test_loader = _prepare_loaders(cfg)

        label = f"trial_{idx}_a{params['a']}_lr{params['lr']}_n{params['noise_pairs']}"
        with runner.time_section(label):
            out = _run_trial(cfg, params, train_loader, test_loader, device=device)

        out = {"seed": trial_seed, **out}
        rows.append(out)

    payload = {
        "results": rows,
        "best_by_accuracy": _best_row(rows, "accuracy"),
        "best_by_macro_f1": _best_row(rows, "macro_f1"),
    }

    # Save core tables
    runner.save_json("sweep_results.json", payload)
    csv_path = Path(runner.run_dir) / "sweep_results.csv"
    runner.add_artifact(_write_csv(csv_path, rows), description="Flat sweep results table")

    # Plots
    if make_plots and rows:
        for art in _generate_plots(rows, runner.run_dir):
            runner.add_artifact(art, description="Sweep plot")

    runner.add_metrics(
        {
            "best_accuracy": float(payload["best_by_accuracy"].get("accuracy", 0.0)) if payload["best_by_accuracy"] else 0.0,
            "best_macro_f1": float(payload["best_by_macro_f1"].get("macro_f1", 0.0)) if payload["best_by_macro_f1"] else 0.0,
            "n_trials": len(rows),
        }
    )

    runner.finalize()
    return rows
