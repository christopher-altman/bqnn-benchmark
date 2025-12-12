"""
Sweep injected noise depth (U U† pairs) vs accuracy on synthetic data.

This experiment characterizes the noise robustness of BQNNs by measuring
how accuracy degrades as circuit noise increases.

Usage:
    python -m experiments.exp_noise_threshold
"""

import torch
import numpy as np

from bqnn import (
    get_synthetic_dataset,
    BQNNModel,
    BenchmarkRun,
    train_epoch,
    evaluate_accuracy,
)
from bqnn.utils.plots import plot_heatmap, save_figure
from bqnn.utils.report import save_report


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("BQNN Noise Robustness Experiment")
    print("=" * 60)
    print(f"Device: {device}")

    n_features = 16
    n_hidden = 8
    n_classes = 2

    runner = BenchmarkRun(
        name="exp_noise_threshold",
        seed=123,
        config={
            "n_features": n_features,
            "n_hidden": n_hidden,
            "n_classes": n_classes,
            "epochs": 3,
            "train_samples": 1024,
            "test_samples": 256,
        },
        metadata={"description": "Noise robustness sweep"},
    )

    # Use different seeds for proper train/test separation
    train_loader = get_synthetic_dataset(
        n_samples=1024, n_features=16, batch_size=64, seed=runner.record.seed
    )
    test_loader = get_synthetic_dataset(
        n_samples=256, n_features=16, batch_size=64, seed=12345
    )

    # Parameter grids
    a_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    noise_pairs_values = [0, 1, 2, 4, 8]

    print(f"\nSweeping:")
    print(f"  Quantumness a: {a_values}")
    print(f"  Noise pairs:   {noise_pairs_values}")
    print(f"  Total configs: {len(a_values) * len(noise_pairs_values)}")

    acc_grid = []
    results_table = []

    for a in a_values:
        row_accs = []
        for n_pairs in noise_pairs_values:
            print(f"\n[a={a}, noise_pairs={n_pairs}]")
            
            model = BQNNModel(
                n_features=n_features,
                n_hidden=n_hidden,
                n_classes=n_classes,
                n_qubits=n_hidden,
                a=a,
            )
            model.set_noise(n_pairs=n_pairs, angle=0.15)

            # Train with persistent optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            with runner.time_section(f"train_a_{a}_noise_{n_pairs}"):
                for epoch in range(3):
                    stats = train_epoch(model, train_loader, device=device, lr=1e-3, optimizer=optimizer)
                    print(f"  epoch {epoch+1}: loss={stats['loss']:.4f}")
            
            metrics = evaluate_accuracy(model, test_loader, device=device)
            acc = metrics["accuracy"]
            print(f"  accuracy={acc:.3f}")
            
            row_accs.append(acc)
            acc_grid.append(acc)
            results_table.append({
                "a": a,
                "noise_pairs": n_pairs,
                "accuracy": acc,
            })
        
    # Results summary
    print("\n" + "=" * 60)
    print("RESULTS MATRIX")
    print("=" * 60)
    
    # Print header
    header_label = "a \\ noise"
    header = f"{header_label:>10}"
    for np_val in noise_pairs_values:
        header += f" | {np_val:>6}"
    print(header)
    print("-" * len(header))
    
    # Print rows
    acc_matrix = np.array(acc_grid).reshape(len(a_values), len(noise_pairs_values))
    for i, a in enumerate(a_values):
        row = f"{a:>10.2f}"
        for j in range(len(noise_pairs_values)):
            row += f" | {acc_matrix[i, j]:>6.3f}"
        print(row)
    
    # Analysis
    print("\n" + "-" * 40)
    print("ANALYSIS")
    print("-" * 40)
    
    # Find best configuration
    best_idx = np.argmax(acc_grid)
    best_a = a_values[best_idx // len(noise_pairs_values)]
    best_noise = noise_pairs_values[best_idx % len(noise_pairs_values)]
    print(f"Best configuration: a={best_a}, noise_pairs={best_noise}")
    print(f"Best accuracy: {acc_grid[best_idx]:.3f}")
    
    # Noise degradation analysis
    print("\nNoise degradation (accuracy drop from 0 to max noise):")
    for i, a in enumerate(a_values):
        clean_acc = acc_matrix[i, 0]  # noise_pairs=0
        noisy_acc = acc_matrix[i, -1]  # max noise
        degradation = clean_acc - noisy_acc
        print(f"  a={a}: {clean_acc:.3f} → {noisy_acc:.3f} (Δ={degradation:+.3f})")
    
    # Find noise threshold (first noise level where accuracy < 0.6)
    print("\nNoise threshold analysis (acc < 0.6):")
    for i, a in enumerate(a_values):
        threshold = None
        for j, np_val in enumerate(noise_pairs_values):
            if acc_matrix[i, j] < 0.6:
                threshold = np_val
                break
        if threshold is not None:
            print(f"  a={a}: threshold at noise_pairs={threshold}")
        else:
            print(f"  a={a}: robust across all noise levels")

    # Plotting
    try:
        fig = plot_heatmap(
            xs=a_values,
            ys=noise_pairs_values,
            values=acc_grid,
            xlabel="Noise Pairs",
            ylabel="Quantumness a",
            title="BQNN Accuracy: Quantumness vs Noise Depth",
            annotate=True,
        )

        paths = save_figure(fig, runner.run_dir / "noise_heatmap")
        for p in paths:
            runner.add_artifact(p, description="Noise vs quantumness heatmap")

        import matplotlib.pyplot as plt
        plt.show()
    except Exception as e:
        print(f"\nPlotting skipped: {e}")

    runner.add_metrics(
        {
            "best_accuracy": float(acc_grid[best_idx]),
            "best_a": best_a,
            "best_noise_pairs": best_noise,
            "accuracy_matrix_shape": [len(a_values), len(noise_pairs_values)],
        }
    )
    runner.save_json("results_table.json", {"rows": results_table})
    run_payload = runner.finalize()
    save_report(runner.run_dir, run_data=run_payload)


if __name__ == "__main__":
    main()
