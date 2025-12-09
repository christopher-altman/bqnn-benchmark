"""
Training utilities for BQNN models.

Provides training loops with proper optimizer persistence, gradient tracking,
learning rate scheduling, and early stopping.
"""

from typing import Dict, Optional, Callable, List
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler

from .utils.metrics import compute_gradient_stats


@dataclass
class TrainingConfig:
    """Configuration for training."""
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: Optional[float] = 1.0
    scheduler: Optional[str] = None  # 'cosine', 'step', 'plateau'
    scheduler_params: Dict = field(default_factory=dict)
    early_stopping_patience: Optional[int] = None
    track_gradients: bool = False


class Trainer:
    """
    Stateful trainer for BQNN models.
    
    Maintains optimizer and scheduler state across epochs to ensure
    proper momentum accumulation and learning rate scheduling.
    
    Args:
        model: PyTorch model to train
        config: TrainingConfig instance
        device: Target device
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None,
        device: str = "cpu",
    ):
        self.model = model
        self.config = config or TrainingConfig()
        self.device = device
        
        self.model.to(device)
        
        # Initialize optimizer (persistent across epochs)
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        
        # Initialize scheduler if requested
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.epoch = 0
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.history: Dict[str, List[float]] = {
            "loss": [],
            "grad_mean": [],
            "grad_max": [],
        }
    
    def _create_scheduler(self) -> Optional[_LRScheduler]:
        """Create learning rate scheduler based on config."""
        if self.config.scheduler is None:
            return None
        
        params = self.config.scheduler_params
        
        if self.config.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=params.get("T_max", 100),
                eta_min=params.get("eta_min", 1e-6),
            )
        elif self.config.scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=params.get("step_size", 10),
                gamma=params.get("gamma", 0.5),
            )
        elif self.config.scheduler == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=params.get("factor", 0.5),
                patience=params.get("patience", 5),
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")
    
    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """
        Train for a single epoch.
        
        Args:
            loader: DataLoader for training data
            
        Returns:
            Dictionary with training statistics
        """
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        n_batches = 0
        grad_stats_list = []
        
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = criterion(logits, y)
            loss.backward()
            
            # Track gradients for barren plateau detection
            if self.config.track_gradients:
                grad_stats_list.append(compute_gradient_stats(self.model))
            
            # Gradient clipping
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / max(1, n_batches)
        
        # Update scheduler
        if self.scheduler is not None:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(avg_loss)
            else:
                self.scheduler.step()
        
        # Early stopping check
        if self.config.early_stopping_patience is not None:
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        
        # Aggregate gradient stats
        result = {"loss": avg_loss}
        if grad_stats_list:
            result["grad_mean"] = sum(g["grad_mean"] for g in grad_stats_list) / len(grad_stats_list)
            result["grad_max"] = max(g["grad_max"] for g in grad_stats_list)
        
        # Update history
        self.history["loss"].append(avg_loss)
        if "grad_mean" in result:
            self.history["grad_mean"].append(result["grad_mean"])
            self.history["grad_max"].append(result["grad_max"])
        
        self.epoch += 1
        return result
    
    def should_stop(self) -> bool:
        """Check if early stopping criteria met."""
        if self.config.early_stopping_patience is None:
            return False
        return self.patience_counter >= self.config.early_stopping_patience
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
    lr: float = 1e-3,
    optimizer: Optional[optim.Optimizer] = None,
) -> Dict[str, float]:
    """
    Train for a single epoch (stateless version for backward compatibility).
    
    WARNING: Creating a new optimizer each epoch loses momentum state.
    For proper training, use the Trainer class instead.
    
    Args:
        model: Model to train
        loader: Training data loader
        device: Target device
        lr: Learning rate (only used if optimizer is None)
        optimizer: Optional pre-existing optimizer
        
    Returns:
        Dictionary with "loss" key
    """
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    return {"loss": avg_loss}


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    n_epochs: int = 10,
    config: Optional[TrainingConfig] = None,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Full training loop with validation.
    
    Args:
        model: Model to train
        train_loader: Training data
        val_loader: Optional validation data
        n_epochs: Number of epochs
        config: Training configuration
        device: Target device
        verbose: Print progress
        
    Returns:
        Training history dictionary
    """
    from .inference import evaluate_accuracy
    
    trainer = Trainer(model, config, device)
    
    for epoch in range(n_epochs):
        stats = trainer.train_epoch(train_loader)
        
        if val_loader is not None:
            val_metrics = evaluate_accuracy(model, val_loader, device)
            stats["val_accuracy"] = val_metrics["accuracy"]
            
            if "val_accuracy" not in trainer.history:
                trainer.history["val_accuracy"] = []
            trainer.history["val_accuracy"].append(val_metrics["accuracy"])
        
        if verbose:
            msg = f"Epoch {epoch+1}/{n_epochs}: loss={stats['loss']:.4f}"
            if "val_accuracy" in stats:
                msg += f", val_acc={stats['val_accuracy']:.3f}"
            msg += f", lr={trainer.get_lr():.2e}"
            print(msg)
        
        if trainer.should_stop():
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
    
    return trainer.history
