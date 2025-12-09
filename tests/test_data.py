"""
Unit tests for data loading and generation utilities.
"""

import pytest
import torch
from bqnn.data import (
    get_synthetic_dataset,
    get_train_test_loaders,
)


class TestGetSyntheticDataset:
    """Test synthetic dataset generation."""
    
    def test_basic_generation(self):
        """Test that synthetic dataset is generated correctly."""
        loader = get_synthetic_dataset(
            n_samples=64,
            n_features=16,
            n_classes=2,
            batch_size=8,
            seed=42,
        )
        
        # Check loader properties
        assert loader is not None
        assert len(loader) > 0  # Should have batches
        
        # Get first batch
        x, y = next(iter(loader))
        
        # Check shapes
        assert x.shape[1] == 16, f"Expected 16 features, got {x.shape[1]}"
        assert x.shape[0] <= 8, f"Batch size should be <= 8"
        
        # Check data types
        assert x.dtype == torch.float32
        assert y.dtype == torch.long
        
        # Check value ranges
        assert torch.all((x == 0) | (x == 1)), "Features should be binary {0, 1}"
        assert torch.all((y >= 0) & (y < 2)), "Labels should be in [0, n_classes)"
    
    def test_different_parameters(self):
        """Test generation with various parameter combinations."""
        configs = [
            {"n_samples": 32, "n_features": 8, "n_classes": 2},
            {"n_samples": 64, "n_features": 16, "n_classes": 3},
            {"n_samples": 128, "n_features": 32, "n_classes": 4},
        ]
        
        for config in configs:
            loader = get_synthetic_dataset(
                n_samples=config["n_samples"],
                n_features=config["n_features"],
                n_classes=config["n_classes"],
                batch_size=16,
                seed=42,
            )
            
            x, y = next(iter(loader))
            
            assert x.shape[1] == config["n_features"]
            assert torch.all((y >= 0) & (y < config["n_classes"]))
    
    def test_seed_reproducibility(self):
        """Test that same seed produces same dataset (ignoring shuffle order)."""
        # Note: DataLoader has shuffle=True, so batch order varies
        # We collect all data and verify the dataset itself is the same
        loader1 = get_synthetic_dataset(
            n_samples=64,
            n_features=16,
            batch_size=16,
            seed=42,
        )
        loader2 = get_synthetic_dataset(
            n_samples=64,
            n_features=16,
            batch_size=16,
            seed=42,
        )
        
        # Collect all data from both loaders
        all_x1, all_y1 = [], []
        for x, y in loader1:
            all_x1.append(x)
            all_y1.append(y)
        x1 = torch.cat(all_x1)
        y1 = torch.cat(all_y1)
        
        all_x2, all_y2 = [], []
        for x, y in loader2:
            all_x2.append(x)
            all_y2.append(y)
        x2 = torch.cat(all_x2)
        y2 = torch.cat(all_y2)
        
        # Sort by creating a unique key for each sample
        # Convert feature vectors to strings for sorting
        def sort_key(x_tensor, y_tensor):
            # Create sortable keys: combine y and x values
            keys = []
            for i in range(len(y_tensor)):
                key = (y_tensor[i].item(), tuple(x_tensor[i].tolist()))
                keys.append(key)
            indices = sorted(range(len(keys)), key=lambda i: keys[i])
            return torch.tensor(indices)
        
        idx1 = sort_key(x1, y1)
        idx2 = sort_key(x2, y2)
        
        # Should have same data (possibly in different order due to shuffle)
        assert torch.allclose(x1[idx1], x2[idx2]), "Same seed should produce same features"
        assert torch.equal(y1[idx1], y2[idx2]), "Same seed should produce same labels"
    
    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different data."""
        loader1 = get_synthetic_dataset(
            n_samples=64,
            n_features=16,
            batch_size=64,
            seed=42,
        )
        loader2 = get_synthetic_dataset(
            n_samples=64,
            n_features=16,
            batch_size=64,
            seed=12345,
        )
        
        x1, y1 = next(iter(loader1))
        x2, y2 = next(iter(loader2))
        
        # Should be different (with high probability)
        assert not torch.equal(x1, x2) or not torch.equal(y1, y2), \
            "Different seeds should produce different data"
    
    def test_batch_size_respected(self):
        """Test that batch size is correctly applied."""
        batch_size = 8
        n_samples = 32
        
        loader = get_synthetic_dataset(
            n_samples=n_samples,
            n_features=16,
            batch_size=batch_size,
            seed=42,
        )
        
        batch_sizes = []
        for x, y in loader:
            batch_sizes.append(x.shape[0])
        
        # All batches except possibly the last should be full
        assert all(bs <= batch_size for bs in batch_sizes)
        assert batch_sizes[0] == batch_size  # First batch should be full
    
    def test_total_samples(self):
        """Test that total number of samples is correct."""
        n_samples = 100
        loader = get_synthetic_dataset(
            n_samples=n_samples,
            n_features=16,
            batch_size=16,
            seed=42,
        )
        
        total = 0
        for x, y in loader:
            total += x.shape[0]
        
        assert total == n_samples, f"Expected {n_samples} total samples, got {total}"
    
    def test_rule_bits_parameter(self):
        """Test that rule_bits parameter affects label generation."""
        # With rule_bits=1, labels only depend on first bit
        loader = get_synthetic_dataset(
            n_samples=100,
            n_features=16,
            n_classes=2,
            batch_size=100,
            seed=42,
            rule_bits=1,
        )
        
        x, y = next(iter(loader))
        
        # Check that labels correspond to first bit parity
        expected_labels = x[:, 0].long()
        assert torch.equal(y, expected_labels), \
            "With rule_bits=1 and n_classes=2, labels should equal first feature"
    
    def test_no_seed_randomness(self):
        """Test that without seed, data varies across calls."""
        loader1 = get_synthetic_dataset(
            n_samples=64,
            n_features=16,
            batch_size=64,
            seed=None,
        )
        loader2 = get_synthetic_dataset(
            n_samples=64,
            n_features=16,
            batch_size=64,
            seed=None,
        )
        
        x1, y1 = next(iter(loader1))
        x2, y2 = next(iter(loader2))
        
        # Should very likely be different without fixed seed
        # (there's a tiny chance they're the same, but we accept this test flakiness)
        assert not torch.equal(x1, x2) or not torch.equal(y1, y2)


class TestGetTrainTestLoaders:
    """Test train/test loader generation."""
    
    def test_synthetic_split(self):
        """Test that train and test loaders are created with different data."""
        train_loader, test_loader = get_train_test_loaders(
            dataset="synthetic",
            n_train=64,
            n_test=32,
            n_features=16,
            batch_size=16,
        )
        
        # Check loaders exist
        assert train_loader is not None
        assert test_loader is not None
        
        # Get samples
        x_train, y_train = next(iter(train_loader))
        x_test, y_test = next(iter(test_loader))
        
        # Check shapes
        assert x_train.shape[1] == 16
        assert x_test.shape[1] == 16
        
        # Train and test should be different (different seeds)
        assert not torch.equal(x_train[:min(len(x_train), len(x_test))], 
                              x_test[:min(len(x_train), len(x_test))])
    
    def test_train_test_sizes(self):
        """Test that train and test sets have correct sizes."""
        n_train = 100
        n_test = 50
        
        train_loader, test_loader = get_train_test_loaders(
            dataset="synthetic",
            n_train=n_train,
            n_test=n_test,
            n_features=16,
            batch_size=10,
        )
        
        # Count samples
        train_count = sum(x.shape[0] for x, _ in train_loader)
        test_count = sum(x.shape[0] for x, _ in test_loader)
        
        assert train_count == n_train
        assert test_count == n_test
    
    def test_different_seeds_for_train_test(self):
        """Test that train and test use different seeds."""
        # This is critical to prevent data leakage
        train_loader, test_loader = get_train_test_loaders(
            dataset="synthetic",
            n_train=64,
            n_test=64,
            n_features=16,
            batch_size=64,
        )
        
        x_train, y_train = next(iter(train_loader))
        x_test, y_test = next(iter(test_loader))
        
        # Should be different data
        assert not torch.equal(x_train, x_test), \
            "Train and test data should be different"
    
    def test_synthetic_with_kwargs(self):
        """Test passing additional kwargs to synthetic dataset."""
        train_loader, test_loader = get_train_test_loaders(
            dataset="synthetic",
            n_train=64,
            n_test=32,
            n_features=16,
            batch_size=16,
            n_classes=3,
            rule_bits=2,
        )
        
        x_train, y_train = next(iter(train_loader))
        x_test, y_test = next(iter(test_loader))
        
        # Check n_classes was respected
        assert torch.all((y_train >= 0) & (y_train < 3))
        assert torch.all((y_test >= 0) & (y_test < 3))
    
    def test_reproducibility_across_calls(self):
        """Test that loaders produce same dataset across calls (ignoring shuffle)."""
        train1, test1 = get_train_test_loaders(
            dataset="synthetic",
            n_train=64,
            n_test=32,
            n_features=16,
            batch_size=16,
        )
        
        train2, test2 = get_train_test_loaders(
            dataset="synthetic",
            n_train=64,
            n_test=32,
            n_features=16,
            batch_size=16,
        )
        
        # Collect all data to account for shuffling
        all_x1, all_y1 = [], []
        for x, y in train1:
            all_x1.append(x)
            all_y1.append(y)
        x_train1 = torch.cat(all_x1)
        y_train1 = torch.cat(all_y1)
        
        all_x2, all_y2 = [], []
        for x, y in train2:
            all_x2.append(x)
            all_y2.append(y)
        x_train2 = torch.cat(all_x2)
        y_train2 = torch.cat(all_y2)
        
        # Create stable sort keys from data
        def sort_key(x_tensor, y_tensor):
            keys = []
            for i in range(len(y_tensor)):
                key = (y_tensor[i].item(), tuple(x_tensor[i].tolist()))
                keys.append(key)
            indices = sorted(range(len(keys)), key=lambda i: keys[i])
            return torch.tensor(indices)
        
        idx1 = sort_key(x_train1, y_train1)
        idx2 = sort_key(x_train2, y_train2)
        
        # Should be identical datasets (fixed seeds, though shuffled differently)
        assert torch.equal(x_train1[idx1], x_train2[idx2])
        assert torch.equal(y_train1[idx1], y_train2[idx2])
    
    def test_output_format(self):
        """Test that output format is correct (tuple of loaders)."""
        result = get_train_test_loaders(
            dataset="synthetic",
            n_train=32,
            n_test=16,
            n_features=8,
            batch_size=8,
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        train_loader, test_loader = result
        
        # Should be DataLoaders
        assert hasattr(train_loader, "__iter__")
        assert hasattr(test_loader, "__iter__")
    
    def test_invalid_dataset_name(self):
        """Test that invalid dataset name raises error."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_train_test_loaders(
                dataset="invalid_name",
                n_train=32,
                n_test=16,
            )
