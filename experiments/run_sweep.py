"""Command-line runner for BQNN hyperparameter sweeps.

Typical usage (grid):
    python -m experiments.run_sweep --name demo_sweep \
        --a 0.0 0.2 0.5 1.0 \
        --lr 1e-3 5e-4 \
        --noise-pairs 0 2 4 \
        --noise-angle 0.0 0.05 \
        --epochs 5

Random search (sample 12 configs):
    python -m experiments.run_sweep --search random --num-samples 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bqnn.sweeps import SweepConfig, SweepSpace, run_sweep


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run BQNN sweeps (grid or random)")
    p.add_argument("--name", default="bqnn_sweep", help="Run name (becomes a folder name)")
    p.add_argument("--output-dir", default="results/sweeps", help="Parent directory for runs")

    p.add_argument("--a", nargs="+", type=float, default=[0.0, 0.5, 1.0], help="Quantumness values")
    p.add_argument("--lr", nargs="+", type=float, default=[1e-3, 5e-4], help="Learning rate values")
    p.add_argument("--noise-pairs", nargs="+", type=int, default=[0, 2], help="Noise pair counts")
    p.add_argument("--noise-angle", nargs="+", type=float, default=[0.0, 0.05], help="Noise angles")

    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--n-train", type=int, default=1024)
    p.add_argument("--n-test", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)

    p.add_argument("--n-features", type=int, default=16)
    p.add_argument("--n-hidden", type=int, default=8)
    p.add_argument("--n-classes", type=int, default=2)
    p.add_argument("--n-qubits", type=int, default=None)

    p.add_argument("--device", default=None, help="Force device (cpu/cuda)")
    p.add_argument("--shots", type=int, default=None, help="Optional shot count (if model supports it)")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--search", choices=["grid", "random"], default="grid")
    p.add_argument("--num-samples", type=int, default=None, help="For random search")
    p.add_argument("--no-plots", action="store_true", help="Skip plot generation")

    return p


def main() -> None:
    args = build_parser().parse_args()

    cfg = SweepConfig(
        name=args.name,
        output_dir=Path(args.output_dir),
        n_features=args.n_features,
        n_hidden=args.n_hidden,
        n_classes=args.n_classes,
        n_qubits=args.n_qubits,
        n_train=args.n_train,
        n_test=args.n_test,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        shots=args.shots,
    )
    space = SweepSpace(
        a=args.a,
        lr=args.lr,
        noise_pairs=args.noise_pairs,
        noise_angle=args.noise_angle,
    )

    rows = run_sweep(
        cfg,
        space,
        search=args.search,
        num_samples=args.num_samples,
        seed=args.seed,
        make_plots=not args.no_plots,
    )

    # Human-friendly summary
    if rows:
        best = max(rows, key=lambda r: float(r.get("accuracy", 0.0)))
        print(f"Completed {len(rows)} trials. Best accuracy: {best.get('accuracy'):.4f}")
        print("Best config:", {k: best[k] for k in ("a","lr","noise_pairs","noise_angle") if k in best})
    else:
        print("No sweep trials were run (empty search space).")


if __name__ == "__main__":
    main()
