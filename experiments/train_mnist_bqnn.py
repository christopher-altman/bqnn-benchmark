"""
Train classical baseline and BQNN on a tiny MNIST subset (digits 0 and 1).

This experiment provides a realistic benchmark on actual image data,
comparing classical binarized networks against quantum-enhanced variants.

Usage:
    python -m experiments.train_mnist_bqnn
"""

import torch

from bqnn import BenchmarkRun
from bqnn.data import get_mnist_small
from bqnn.classical_reference import ClassicalBinarizedNet
from bqnn.model import BQNNModel
from bqnn.training import train_model, TrainingConfig
from bqnn.inference import evaluate_accuracy, evaluate_full
from bqnn.utils.report import save_report


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("BQNN vs Classical on Tiny MNIST (0 vs 1)")
    print("=" * 60)
    print(f"Device: {device}")

    size = 4  # 4x4 MNIST → 16 features → 8 qubits
    n_features = size * size
    n_hidden = 8
    n_classes = 2

    runner = BenchmarkRun(
        name="train_mnist_bqnn",
        seed=7,
        config={
            "size": size,
            "n_hidden": n_hidden,
            "n_classes": n_classes,
            "epochs": 5,
            "train_samples": 2000,
            "test_samples": 600,
        },
        metadata={"description": "MNIST 0/1 benchmark"},
    )

    # Load data with proper train/test splits
    print("\nLoading MNIST data...")
    train_loader = get_mnist_small(
        n_samples=2000,
        batch_size=64,
        size=size,
        classes=(0, 1),
        train=True,   # MNIST training split
        seed=42,
    )
    test_loader = get_mnist_small(
        n_samples=600,
        batch_size=64,
        size=size,
        classes=(0, 1),
        train=False,  # MNIST test split - proper evaluation
        seed=12345,
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # =========================================
    # Classical baseline
    # =========================================
    print("\n" + "-" * 40)
    print("Training Classical Binarized Baseline")
    print("-" * 40)
    
    baseline = ClassicalBinarizedNet(
        n_features=n_features,
        n_hidden=n_hidden,
        n_classes=n_classes,
    )
    
    config = TrainingConfig(lr=1e-3, grad_clip=1.0)
    with runner.time_section("baseline_training"):
        history = train_model(
            baseline, train_loader, test_loader,
            n_epochs=5, config=config, device=device, verbose=True
        )

    base_metrics = evaluate_full(baseline, test_loader, device=device)
    print(f"\nBaseline Results:")
    print(f"  Accuracy: {base_metrics['accuracy']:.3f}")
    print(f"  Macro F1: {base_metrics['macro_f1']:.3f}")

    # =========================================
    # BQNN model (no noise)
    # =========================================
    print("\n" + "-" * 40)
    print("Training BQNN (a=0.5, no noise)")
    print("-" * 40)
    
    a = 0.5
    model = BQNNModel(
        n_features=n_features,
        n_hidden=n_hidden,
        n_classes=n_classes,
        n_qubits=n_hidden,
        a=a,
    )
    model.set_noise(n_pairs=0, angle=0.1)
    
    print(f"Circuit info: {model.get_circuit_info()}")

    with runner.time_section("bqnn_training_clean"):
        history = train_model(
            model, train_loader, test_loader,
            n_epochs=5, config=config, device=device, verbose=True
        )

    bqnn_metrics = evaluate_full(model, test_loader, device=device)
    print(f"\nBQNN Results:")
    print(f"  Accuracy: {bqnn_metrics['accuracy']:.3f}")
    print(f"  Macro F1: {bqnn_metrics['macro_f1']:.3f}")

    # =========================================
    # BQNN with noise injection
    # =========================================
    print("\n" + "-" * 40)
    print("Training BQNN with U U† Noise (a=0.5, noise_pairs=4)")
    print("-" * 40)
    
    model_noisy = BQNNModel(
        n_features=n_features,
        n_hidden=n_hidden,
        n_classes=n_classes,
        n_qubits=n_hidden,
        a=a,
    )
    model_noisy.set_noise(n_pairs=4, angle=0.15)
    
    print(f"Circuit info: {model_noisy.get_circuit_info()}")

    with runner.time_section("bqnn_training_noisy"):
        history = train_model(
            model_noisy, train_loader, test_loader,
            n_epochs=5, config=config, device=device, verbose=True
        )
    
    bqnn_noisy_metrics = evaluate_full(model_noisy, test_loader, device=device)
    print(f"\nBQNN-Noisy Results:")
    print(f"  Accuracy: {bqnn_noisy_metrics['accuracy']:.3f}")
    print(f"  Macro F1: {bqnn_noisy_metrics['macro_f1']:.3f}")

    # =========================================
    # Final comparison
    # =========================================
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"{'Model':<25} | {'Accuracy':>10} | {'F1':>8}")
    print("-" * 50)
    print(f"{'Classical Baseline':<25} | {base_metrics['accuracy']:>10.3f} | {base_metrics['macro_f1']:>8.3f}")
    print(f"{'BQNN (a=0.5)':<25} | {bqnn_metrics['accuracy']:>10.3f} | {bqnn_metrics['macro_f1']:>8.3f}")
    print(f"{'BQNN + Noise':<25} | {bqnn_noisy_metrics['accuracy']:>10.3f} | {bqnn_noisy_metrics['macro_f1']:>8.3f}")
    print("-" * 50)
    
    # Quantum advantage check
    quantum_advantage = bqnn_metrics['accuracy'] - base_metrics['accuracy']
    noise_robustness = bqnn_noisy_metrics['accuracy'] - base_metrics['accuracy']
    
    print(f"\nQuantum advantage (clean): {quantum_advantage:+.3f}")
    print(f"Quantum advantage (noisy): {noise_robustness:+.3f}")
    
    if quantum_advantage > 0:
        print("✓ BQNN outperforms classical baseline")
    else:
        print("✗ Classical baseline is competitive")
    
    if noise_robustness > -0.05:
        print("✓ BQNN is noise-robust (maintains within 5% of baseline)")
    else:
        print("✗ Significant performance degradation under noise")

    runner.add_metrics(
        {
            "baseline_accuracy": round(base_metrics["accuracy"], 4),
            "baseline_macro_f1": round(base_metrics["macro_f1"], 4),
            "bqnn_accuracy": round(bqnn_metrics["accuracy"], 4),
            "bqnn_macro_f1": round(bqnn_metrics["macro_f1"], 4),
            "bqnn_noisy_accuracy": round(bqnn_noisy_metrics["accuracy"], 4),
            "bqnn_noisy_macro_f1": round(bqnn_noisy_metrics["macro_f1"], 4),
        }
    )
    run_payload = runner.finalize()
    save_report(runner.run_dir, run_data=run_payload)


if __name__ == "__main__":
    main()
