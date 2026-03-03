"""
Utility Functions for VideoMAE
==============================
Common utilities for training, evaluation, and visualization.

Author: VLoad Project
Date: 2026
"""

import os
import random
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Device Management
# =============================================================================

def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_info() -> Dict[str, Any]:
    """Get device information."""
    info = {"device": str(get_device())}
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        info["gpu_count"] = torch.cuda.device_count()
    
    return info


# =============================================================================
# Model Utilities
# =============================================================================

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def freeze_module(module: nn.Module) -> None:
    """Freeze a module's parameters."""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    """Unfreeze a module's parameters."""
    for param in module.parameters():
        param.requires_grad = True


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate."""
    for param_group in optimizer.param_groups:
        return param_group["lr"]


# =============================================================================
# Checkpoint Management
# =============================================================================

class CheckpointManager:
    """Manage model checkpoints with automatic best model tracking."""
    
    def __init__(
        self,
        output_dir: str,
        max_checkpoints: int = 5,
        mode: str = "max",
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            output_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            mode: "max" for accuracy-like metrics, "min" for loss-like
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.mode = mode
        self.best_metric = float("-inf") if mode == "max" else float("inf")
        self.checkpoints: List[Tuple[str, float]] = []
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metric: float,
        scheduler: Optional[Any] = None,
        extra: Optional[Dict] = None,
    ) -> bool:
        """
        Save checkpoint if it's good enough.
        
        Returns:
            True if this is the new best checkpoint
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metric": metric,
        }
        
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        if extra is not None:
            checkpoint.update(extra)
        
        # Check if best
        is_best = False
        if self.mode == "max" and metric > self.best_metric:
            is_best = True
            self.best_metric = metric
        elif self.mode == "min" and metric < self.best_metric:
            is_best = True
            self.best_metric = metric
        
        # Save checkpoint
        ckpt_path = self.output_dir / f"checkpoint_epoch{epoch}.pt"
        torch.save(checkpoint, ckpt_path)
        self.checkpoints.append((str(ckpt_path), metric))
        
        # Save best
        if is_best:
            best_path = self.output_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
        
        # Cleanup old checkpoints
        self._cleanup()
        
        return is_best
    
    def _cleanup(self) -> None:
        """Remove old checkpoints."""
        if len(self.checkpoints) > self.max_checkpoints:
            # Sort by metric (keep best ones)
            reverse = self.mode == "max"
            self.checkpoints.sort(key=lambda x: x[1], reverse=reverse)
            
            # Remove worst ones
            while len(self.checkpoints) > self.max_checkpoints:
                ckpt_path, _ = self.checkpoints.pop()
                if Path(ckpt_path).exists() and "best" not in ckpt_path:
                    os.remove(ckpt_path)
    
    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> int:
        """Load best checkpoint. Returns epoch."""
        best_path = self.output_dir / "checkpoint_best.pt"
        
        if not best_path.exists():
            raise FileNotFoundError(f"No best checkpoint found at {best_path}")
        
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        return checkpoint.get("epoch", 0)


# =============================================================================
# Metrics
# =============================================================================

class MetricTracker:
    """Track and log metrics during training."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.current: Dict[str, float] = {}
    
    def update(self, name: str, value: float) -> None:
        """Update a metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
        self.current[name] = value
    
    def get_current(self, name: str) -> float:
        """Get current value of a metric."""
        return self.current.get(name, 0.0)
    
    def get_best(self, name: str, mode: str = "max") -> float:
        """Get best value of a metric."""
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        
        if mode == "max":
            return max(self.metrics[name])
        return min(self.metrics[name])
    
    def get_history(self, name: str) -> List[float]:
        """Get full history of a metric."""
        return self.metrics.get(name, [])
    
    def save(self, path: str) -> None:
        """Save metrics to JSON."""
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=2)
    
    def load(self, path: str) -> None:
        """Load metrics from JSON."""
        with open(path, "r") as f:
            self.metrics = json.load(f)


class AverageMeter:
    """Compute and store the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# =============================================================================
# Evaluation
# =============================================================================

def compute_accuracy(
    preds: torch.Tensor,
    labels: torch.Tensor,
    topk: Tuple[int, ...] = (1,),
) -> List[float]:
    """
    Compute top-k accuracy.
    
    Args:
        preds: Predictions (B, C) or (B,)
        labels: Ground truth labels (B,)
        topk: Tuple of k values
        
    Returns:
        List of accuracies for each k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)
        
        if preds.dim() == 1:
            preds = preds.unsqueeze(1)
        
        _, pred = preds.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append((correct_k / batch_size).item())
        
        return res


def compute_confusion_matrix(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        preds: Predictions (B,)
        labels: Ground truth labels (B,)
        num_classes: Number of classes
        
    Returns:
        Confusion matrix (num_classes, num_classes)
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    
    for p, l in zip(preds, labels):
        cm[l, p] += 1
    
    return cm


def compute_metrics_from_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute various metrics from confusion matrix.
    
    Returns:
        Dictionary with accuracy, per-class precision/recall/f1
    """
    num_classes = cm.shape[0]
    
    # Overall accuracy
    accuracy = np.diag(cm).sum() / cm.sum()
    
    # Per-class metrics
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
    
    metrics = {
        "accuracy": accuracy,
        "macro_precision": precision.mean(),
        "macro_recall": recall.mean(),
        "macro_f1": f1.mean(),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
    }
    
    if class_names:
        for i, name in enumerate(class_names):
            metrics[f"{name}_precision"] = precision[i]
            metrics[f"{name}_recall"] = recall[i]
            metrics[f"{name}_f1"] = f1[i]
    
    return metrics


# =============================================================================
# Logging
# =============================================================================

def setup_logging(output_dir: str, name: str = "training") -> Path:
    """Setup logging directory and return log path."""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{name}_{timestamp}.log"
    
    return log_path


def log_config(config: Dict[str, Any], output_dir: str) -> None:
    """Save configuration to file."""
    config_path = Path(output_dir) / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)


# =============================================================================
# Mixed Precision
# =============================================================================

def get_autocast_context(device: torch.device, enabled: bool = True):
    """Get autocast context for mixed precision training."""
    if not enabled:
        return torch.cuda.amp.autocast(enabled=False)
    
    if device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=torch.float16)
    elif device.type == "mps":
        return torch.autocast(device_type="mps", dtype=torch.float16)
    else:
        return torch.cuda.amp.autocast(enabled=False)


# =============================================================================
# Data Utilities
# =============================================================================

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for video batches."""
    collated = {}
    
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        else:
            collated[key] = values
    
    return collated
