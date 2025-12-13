"""
Data loading utilities for BQNN experiments.

Provides synthetic binary classification data and tiny MNIST.
"""

from typing import Tuple, Optional
import torch
from torch.utils.data import TensorDataset, DataLoader


def get_synthetic_dataset(
    n_samples: int = 1024,
    n_features: int = 16,
    n_classes: int = 2,
    batch_size: int = 32,
    seed: Optional[int] = None,
    rule_bits: int = 3,
) -> DataLoader:
    """
    Generate a simple synthetic dataset of binary features with a linear
    decision rule. This keeps the pipeline functional without external
    downloads.

    Features: in {0, 1}^n_features
    Labels: parity-like rule on first `rule_bits` bits mod n_classes.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Dimensionality of feature vectors
        n_classes: Number of output classes
        batch_size: Batch size for DataLoader
        seed: Random seed (IMPORTANT: use different seeds for train/test!)
        rule_bits: Number of bits used in classification rule
        
    Returns:
        DataLoader yielding (features, labels) batches
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    x = torch.randint(0, 2, (n_samples, n_features)).float()
    rule_sum = x[:, :rule_bits].sum(dim=1) % n_classes
    y = rule_sum.long()

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def get_mnist_small(
    n_samples: int = 2000,
    batch_size: int = 64,
    size: int = 4,
    classes: Tuple[int, ...] = (0, 1),
    train: bool = True,
    seed: Optional[int] = None,
) -> DataLoader:
    """
    Load MNIST, downsample to (size x size), binarize to {0,1}, and
    optionally restrict to a small set of classes (default: digits 0 and 1).

    This keeps the quantum circuit width manageable:
      - size=4 -> 16 pixels -> 16 qubits
      
    Args:
        n_samples: Maximum number of samples
        batch_size: Batch size for DataLoader
        size: Image dimension after downsampling (size x size)
        classes: Tuple of digit classes to include
        train: Use training split (True) or test split (False)
        seed: Random seed for shuffling
        
    Returns:
        DataLoader yielding (features, labels) batches
    """
    try:
        from torchvision.datasets import MNIST
        from torchvision import transforms
    except Exception as e:
        raise ImportError(
            "get_mnist_small requires torchvision; install it with `pip install torchvision`."
        ) from e

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float()),
    ])

    ds = MNIST(root="./data", train=train, download=True, transform=transform)

    # filter by class
    idx = [i for i, (_, y) in enumerate(ds) if int(y) in classes]
    
    # Shuffle indices if seed provided
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(idx), generator=gen).tolist()
        idx = [idx[p] for p in perm]
    
    idx = idx[:n_samples]
    xs = []
    ys = []
    for i in idx:
        x_i, y_i = ds[i]
        xs.append(x_i)
        ys.append(y_i)

    x = torch.stack(xs, dim=0)  # [N, 1, H, W]
    y = torch.tensor(ys, dtype=torch.long)

    # remap labels -> 0..len(classes)-1
    class_to_idx = {c: j for j, c in enumerate(classes)}
    y = torch.tensor([class_to_idx[int(lbl)] for lbl in y], dtype=torch.long)

    x = x.view(x.size(0), -1)  # flatten

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return loader


def get_train_test_loaders(
    dataset: str = "synthetic",
    n_train: int = 1024,
    n_test: int = 256,
    n_features: int = 16,
    batch_size: int = 64,
    **kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """
    Convenience function to get train and test loaders with proper seeds.
    
    Args:
        dataset: Either "synthetic" or "mnist"
        n_train: Number of training samples
        n_test: Number of test samples
        n_features: Feature dimension (only for synthetic)
        batch_size: Batch size
        **kwargs: Additional arguments passed to dataset functions
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    if dataset == "synthetic":
        train_loader = get_synthetic_dataset(
            n_samples=n_train,
            n_features=n_features,
            batch_size=batch_size,
            seed=42,  # Fixed train seed
            **kwargs,
        )
        test_loader = get_synthetic_dataset(
            n_samples=n_test,
            n_features=n_features,
            batch_size=batch_size,
            seed=12345,  # Different test seed
            **kwargs,
        )
    elif dataset == "mnist":
        size = kwargs.pop("size", 4)
        classes = kwargs.pop("classes", (0, 1))
        train_loader = get_mnist_small(
            n_samples=n_train,
            batch_size=batch_size,
            size=size,
            classes=classes,
            train=True,  # MNIST train split
            seed=42,
            **kwargs,
        )
        test_loader = get_mnist_small(
            n_samples=n_test,
            batch_size=batch_size,
            size=size,
            classes=classes,
            train=False,  # MNIST test split - FIXED
            seed=12345,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return train_loader, test_loader
