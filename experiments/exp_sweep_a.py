"""
Sweep the quantumness parameter `a` and measure accuracy on a synthetic task.

This experiment demonstrates how BQNN behavior interpolates between
classical (a=0) and quantum (a>0) regimes.

Usage:
    python -m experiments.exp_sweep_a
"""

import torch

from bqnn import (
    get_synthetic_dataset,
    ClassicalBinarizedNet,
    BQNNModel,
    BenchmarkRun,
    train_epoch,
    evaluate_accuracy,
)
from bqnn.utils.plots import plot_accuracy_vs_a, plot_quantum_classical_comparison, save_figure
from bqnn.utils.report import save_report


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"BQNN Quantumness Sweep Experiment")
    print(f"Device: {device}")
    print("=" * 50)

    n_features = 16
    n_hidden = 8
    n_classes = 2
    a_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    runner = BenchmarkRun(
        name="exp_sweep_a",
        seed=42,
        config={
            "n_features": n_features,
            "n_hidden": n_hidden,
            "n_classes": n_classes,
            "epochs": 5,
            "a_values": a_values,
            "train_samples": 1024,
            "test_samples": 256,
        },
        metadata={"description": "Quantumness sweep on synthetic data"},
    )

    # Use different seeds for train/test to avoid data leakage
    train_loader = get_synthetic_dataset(
        n_samples=1024, n_features=16, batch_size=64, seed=runner.record.seed
    )
    test_loader = get_synthetic_dataset(
        n_samples=256, n_features=16, batch_size=64, seed=12345
    )

    # Classical baseline
    baseline_model = ClassicalBinarizedNet(
        n_features=n_features,
        n_hidden=n_hidden,
        n_classes=n_classes,
    )
    
    print("\nTraining classical baseline...")
    # Create persistent optimizer for proper momentum
    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=1e-3)
    with runner.time_section("baseline_training"):
        for epoch in range(5):
            stats = train_epoch(baseline_model, train_loader, device=device, lr=1e-3, optimizer=optimizer)
            print(f"  [Baseline] epoch {epoch+1}: loss={stats['loss']:.4f}")

    base_metrics = evaluate_accuracy(baseline_model, test_loader, device=device)
    baseline_acc = base_metrics["accuracy"]
    print(f"  Baseline test accuracy: {baseline_acc:.3f}")

    # BQNN sweeps over quantumness parameter a
    acc_list = []

    for a in a_values:
        print(f"\nTraining BQNN with a={a}...")
        model = BQNNModel(
            n_features=n_features,
            n_hidden=n_hidden,
            n_classes=n_classes,
            n_qubits=n_hidden,
            a=a,
        )
        model.set_noise(n_pairs=0, angle=0.1)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        with runner.time_section(f"train_a_{a}"):
            for epoch in range(5):
                stats = train_epoch(model, train_loader, device=device, lr=1e-3, optimizer=optimizer)
                print(f"  [BQNN a={a}] epoch {epoch+1}: loss={stats['loss']:.4f}")

        metrics = evaluate_accuracy(model, test_loader, device=device)
        acc = metrics["accuracy"]
        print(f"  a={a}: accuracy={acc:.3f}")
        acc_list.append(acc)

    # Results summary
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"{'a':>8} | {'Accuracy':>10} | {'vs Baseline':>12}")
    print("-" * 35)
    for a, acc in zip(a_values, acc_list):
        delta = acc - baseline_acc
        sign = "+" if delta >= 0 else ""
        print(f"{a:>8.2f} | {acc:>10.3f} | {sign}{delta:>11.3f}")
    print("-" * 35)
    print(f"{'baseline':>8} | {baseline_acc:>10.3f} |")

    # Find optimal a
    best_idx = acc_list.index(max(acc_list))
    best_a = a_values[best_idx]
    print(f"\nBest quantumness: a={best_a} (acc={acc_list[best_idx]:.3f})")

    # Plotting
    try:
        fig1 = plot_accuracy_vs_a(a_values, acc_list, baseline_acc=baseline_acc)
        fig2 = plot_quantum_classical_comparison(a_values, acc_list, baseline_acc)

        acc_paths = save_figure(fig1, runner.run_dir / "sweep_a_accuracy")
        comp_paths = save_figure(fig2, runner.run_dir / "sweep_a_comparison")
        for p in acc_paths + comp_paths:
            runner.add_artifact(p, description="Quantumness sweep plot")

        import matplotlib.pyplot as plt
        plt.show()
    except Exception as e:
        print(f"\nPlotting skipped: {e}")

    runner.add_metrics(
        {
            "baseline_accuracy": round(baseline_acc, 4),
            "best_a": best_a,
            "best_accuracy": round(acc_list[best_idx], 4),
            "accuracy_by_a": {str(a): round(acc, 4) for a, acc in zip(a_values, acc_list)},
        }
    )

    run_payload = runner.finalize()
    save_report(runner.run_dir, run_data=run_payload)


if __name__ == "__main__":
    main()
