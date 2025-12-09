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
    train_epoch,
    evaluate_accuracy,
)
from bqnn.utils.plots import plot_accuracy_vs_a, plot_quantum_classical_comparison, save_figure


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"BQNN Quantumness Sweep Experiment")
    print(f"Device: {device}")
    print("=" * 50)

    # Use different seeds for train/test to avoid data leakage
    train_loader = get_synthetic_dataset(
        n_samples=1024, n_features=16, batch_size=64, seed=42
    )
    test_loader = get_synthetic_dataset(
        n_samples=256, n_features=16, batch_size=64, seed=12345
    )

    n_features = 16
    n_hidden = 8
    n_classes = 2

    # Classical baseline
    baseline_model = ClassicalBinarizedNet(
        n_features=n_features,
        n_hidden=n_hidden,
        n_classes=n_classes,
    )
    
    print("\nTraining classical baseline...")
    # Create persistent optimizer for proper momentum
    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=1e-3)
    for epoch in range(5):
        stats = train_epoch(baseline_model, train_loader, device=device, lr=1e-3, optimizer=optimizer)
        print(f"  [Baseline] epoch {epoch+1}: loss={stats['loss']:.4f}")
    
    base_metrics = evaluate_accuracy(baseline_model, test_loader, device=device)
    baseline_acc = base_metrics["accuracy"]
    print(f"  Baseline test accuracy: {baseline_acc:.3f}")

    # BQNN sweeps over quantumness parameter a
    as_list = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    acc_list = []

    for a in as_list:
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
    for a, acc in zip(as_list, acc_list):
        delta = acc - baseline_acc
        sign = "+" if delta >= 0 else ""
        print(f"{a:>8.2f} | {acc:>10.3f} | {sign}{delta:>11.3f}")
    print("-" * 35)
    print(f"{'baseline':>8} | {baseline_acc:>10.3f} |")

    # Find optimal a
    best_idx = acc_list.index(max(acc_list))
    print(f"\nBest quantumness: a={as_list[best_idx]} (acc={acc_list[best_idx]:.3f})")

    # Plotting
    try:
        fig1 = plot_accuracy_vs_a(as_list, acc_list, baseline_acc=baseline_acc)
        fig2 = plot_quantum_classical_comparison(as_list, acc_list, baseline_acc)
        
        save_figure(fig1, "results/sweep_a_accuracy")
        save_figure(fig2, "results/sweep_a_comparison")
        
        import matplotlib.pyplot as plt
        plt.show()
    except Exception as e:
        print(f"\nPlotting skipped: {e}")


if __name__ == "__main__":
    main()
