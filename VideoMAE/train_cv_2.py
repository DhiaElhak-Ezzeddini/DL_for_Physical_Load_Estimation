import os
import json
import argparse
import copy
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models import (
    create_videomae_for_classification,
    load_pretrained_encoder,
    get_model_info,
)
from dataset import (
    create_cv_fold_dataloaders,
    create_test_dataloader,
)
from utils import (
    set_seed,
    AverageMeter,
    compute_accuracy,
    compute_confusion_matrix,
    compute_metrics_from_confusion_matrix,
)


# =============================================================================
# Distributed Utilities (same as train.py for consistency)
# =============================================================================

def setup_distributed():
    """Initialize distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", init_method="env://",
                                world_size=world_size, rank=rank)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True
    return 0, 1, 0, False


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _is_main(rank: int) -> bool:
    return rank == 0


# =============================================================================
# Training / Validation for a single fold
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    label_smoothing: float = 0.1,
    show_progress: bool = True,
) -> Tuple[float, float]:
    """Train one epoch.  Returns (loss, accuracy)."""
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    pbar = tqdm(dataloader, desc=f"  Epoch {epoch} [Train]", disable=not show_progress)
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        preds = outputs.logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()

        loss_meter.update(loss.item(), pixel_values.size(0))
        acc_meter.update(acc.item(), pixel_values.size(0))
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}")

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: Optional[List[str]] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Validate model with multi-clip logit averaging.

    When the validation dataset provides multiple clips per video, this
    function collects the raw logits for every clip, groups them by
    video_idx, and averages the logits before making the final per-video
    prediction.  This is the standard evaluation protocol used in
    VideoMAE, TimeSformer, and other video classification methods.

    Returns dict with loss, accuracy, confusion matrix,
    per-class precision / recall / F1.
    """
    model.eval()
    loss_meter = AverageMeter()

    # Accumulate per-clip results keyed by video index
    # video_logits: {video_idx: [logit_tensor, ...]}
    # video_labels: {video_idx: label}
    video_logits: Dict[int, List[torch.Tensor]] = {}
    video_labels: Dict[int, int] = {}

    for batch in tqdm(dataloader, desc="  Validating", disable=not show_progress):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        video_idxs = batch["video_idx"]  # stays on CPU, used as dict keys

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss_meter.update(outputs.loss.mean().item(), pixel_values.size(0))

        logits = outputs.logits.cpu()  # (B, num_classes)

        for i in range(logits.size(0)):
            vid = video_idxs[i].item()
            if vid not in video_logits:
                video_logits[vid] = []
                video_labels[vid] = labels[i].item()
            video_logits[vid].append(logits[i])

    # Average logits per video and produce one prediction per video
    all_preds = []
    all_labels = []
    for vid in sorted(video_logits.keys()):
        avg_logit = torch.stack(video_logits[vid]).mean(dim=0)  # (num_classes,)
        pred = avg_logit.argmax().item()
        all_preds.append(pred)
        all_labels.append(video_labels[vid])

    all_preds_t = torch.tensor(all_preds, dtype=torch.long)
    all_labels_t = torch.tensor(all_labels, dtype=torch.long)

    num_videos = len(video_logits)
    accuracy = (all_preds_t == all_labels_t).float().mean().item()
    cm = compute_confusion_matrix(all_preds_t, all_labels_t, num_classes)
    metrics = compute_metrics_from_confusion_matrix(cm, class_names)
    metrics["loss"] = loss_meter.avg
    metrics["accuracy"] = accuracy
    metrics["confusion_matrix"] = cm.tolist()
    metrics["num_videos"] = num_videos

    return metrics


# =============================================================================
# Save / Load helpers
# =============================================================================

def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: Dict[str, Any],
    path: str,
) -> None:
    """Save checkpoint (handles DDP-wrapped models)."""
    model_to_save = model.module if hasattr(model, "module") else model
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": {k: v for k, v in metrics.items() if k != "confusion_matrix"},
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    try:
        torch.save(checkpoint, path, _use_new_zipfile_serialization=False)
    except RuntimeError as e:
        if "file write failed" in str(e) or "PytorchStreamWriter failed" in str(e):
            print(f"  WARNING: Could not save checkpoint: {e}")
        else:
            raise


# =============================================================================
# Single Fold Training
# =============================================================================

def train_single_fold(
    fold_idx: int,
    train_operators: List[str],
    val_operator: str,
    args: argparse.Namespace,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
    is_distributed: bool = False,
) -> Dict[str, Any]:
    """
    Train and evaluate a single LOOCV fold.

    Args:
        fold_idx: Fold number (1-indexed).
        train_operators: Operators used for training in this fold.
        val_operator: The single held-out operator for validation.
        args: Parsed CLI arguments.
        device: Torch device.
        rank, world_size, local_rank, is_distributed: DDP params.

    Returns:
        Dictionary with best validation metrics for this fold.
    """
    show = _is_main(rank)
    fold_dir = Path(args.output_dir) / f"fold_{fold_idx}_val_{val_operator}"
    if show:
        fold_dir.mkdir(parents=True, exist_ok=True)

    if is_distributed:
        dist.barrier()

    # ── Data ──────────────────────────────────────────────────────────────
    class_names = args.class_names.split(",") if args.class_names else None

    train_loader, val_loader = create_cv_fold_dataloaders(
        data_dir=args.data_dir,
        train_operators=train_operators,
        val_operators=[val_operator],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_frames=args.num_frames,
        image_size=args.image_size,
        class_names=class_names,
        num_clips=args.num_clips,
        val_num_clips=args.val_num_clips,
        sampling_strategy=args.sampling_strategy,
        activity_filter=args.activity_filter,
        rank=rank,
        world_size=world_size,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = create_videomae_for_classification(
        num_classes=args.num_classes,
        model_name=args.model_name,
        num_frames=args.num_frames,
        image_size=args.image_size,
        from_pretrained=not args.pretrained_path,
        freeze_encoder=args.freeze_encoder,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
    )

    if args.pretrained_path:
        model = load_pretrained_encoder(model, args.pretrained_path)

    model = model.to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ── Optimizer / Scheduler ─────────────────────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_metrics: Dict[str, Any] = {}
    patience_counter = 0

    for epoch in range(args.epochs):
        if is_distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            label_smoothing=args.label_smoothing,
            show_progress=show,
        )

        val_metrics = validate(
            model, val_loader, device,
            num_classes=args.num_classes,
            class_names=class_names,
            show_progress=show,
        )

        scheduler.step()

        val_acc = val_metrics["accuracy"]
        val_loss = val_metrics["loss"]

        if show:
            print(
                f"  Fold {fold_idx} | Epoch {epoch:3d} | "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )

        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_metrics = copy.deepcopy(val_metrics)
            best_metrics["best_epoch"] = epoch
            patience_counter = 0

            if show:
                _save_checkpoint(
                    model, optimizer, scheduler, epoch, val_metrics,
                    str(fold_dir / "best_model.pt"),
                )
                print(f"    -> New best model (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1

        # Early stopping — synchronize decision across all ranks to prevent
        # DDP deadlock (each rank may compute different val_acc on its data
        # subset, leading to different patience counters and one rank exiting
        # the loop while the other still expects a DDP forward pass).
        should_stop = False
        if args.patience > 0 and patience_counter >= args.patience:
            should_stop = True

        if is_distributed:
            stop_tensor = torch.tensor([1 if should_stop else 0],
                                       dtype=torch.long, device=device)
            dist.broadcast(stop_tensor, src=0)
            should_stop = stop_tensor.item() == 1

        if should_stop:
            if show:
                print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
            break

    # ── Summary ───────────────────────────────────────────────────────────
    best_metrics["fold"] = fold_idx
    best_metrics["val_operator"] = val_operator
    best_metrics["train_operators"] = train_operators

    if show:
        _print_fold_summary(fold_idx, val_operator, best_metrics, class_names)
        # Save fold metrics
        metrics_path = fold_dir / "metrics.json"
        _save_metrics_json(best_metrics, str(metrics_path))

    return best_metrics


def _print_fold_summary(
    fold_idx: int,
    val_operator: str,
    metrics: Dict[str, Any],
    class_names: Optional[List[str]],
) -> None:
    """Print a concise summary for a completed fold."""
    print(f"\n  ╔══ Fold {fold_idx} Summary (val={val_operator}) ══")
    print(f"  ║ Best epoch : {metrics.get('best_epoch', '?')}")
    print(f"  ║ Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  ║ Macro F1   : {metrics.get('macro_f1', 0):.4f}")
    if class_names:
        for name in class_names:
            p = metrics.get(f"{name}_precision", 0)
            r = metrics.get(f"{name}_recall", 0)
            f1 = metrics.get(f"{name}_f1", 0)
            print(f"  ║   {name:>8s}: P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
    print(f"  ╚{'═' * 44}")


def _save_metrics_json(metrics: Dict[str, Any], path: str) -> None:
    """Save metrics dict to JSON (handles numpy types)."""
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = {k: _convert(v) for k, v in metrics.items()}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)


# =============================================================================
# Aggregation of Cross-Validation Results
# =============================================================================

def aggregate_cv_results(
    fold_results: List[Dict[str, Any]],
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Aggregate metrics across all CV folds.

    Returns a summary dict with per-metric mean and std, plus
    per-fold details.
    """
    accs = [r["accuracy"] for r in fold_results]
    f1s = [r.get("macro_f1", 0) for r in fold_results]
    precisions = [r.get("macro_precision", 0) for r in fold_results]
    recalls = [r.get("macro_recall", 0) for r in fold_results]

    summary: Dict[str, Any] = {
        "num_folds": len(fold_results),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "macro_precision_mean": float(np.mean(precisions)),
        "macro_precision_std": float(np.std(precisions)),
        "macro_recall_mean": float(np.mean(recalls)),
        "macro_recall_std": float(np.std(recalls)),
        "per_fold_accuracy": accs,
        "per_fold_macro_f1": f1s,
    }

    # Per-class aggregation
    if class_names:
        for name in class_names:
            vals_p = [r.get(f"{name}_precision", 0) for r in fold_results]
            vals_r = [r.get(f"{name}_recall", 0) for r in fold_results]
            vals_f1 = [r.get(f"{name}_f1", 0) for r in fold_results]
            summary[f"{name}_precision_mean"] = float(np.mean(vals_p))
            summary[f"{name}_precision_std"] = float(np.std(vals_p))
            summary[f"{name}_recall_mean"] = float(np.mean(vals_r))
            summary[f"{name}_recall_std"] = float(np.std(vals_r))
            summary[f"{name}_f1_mean"] = float(np.mean(vals_f1))
            summary[f"{name}_f1_std"] = float(np.std(vals_f1))

    return summary


def print_cv_summary(
    summary: Dict[str, Any],
    class_names: Optional[List[str]] = None,
) -> None:
    """Print a formatted cross-validation summary."""
    n = summary["num_folds"]
    print("\n" + "=" * 64)
    print(f"  LEAVE-ONE-OPERATOR-OUT CROSS-VALIDATION RESULTS  ({n} folds)")
    print("=" * 64)
    print(f"  Accuracy       : {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
    print(f"  Macro F1       : {summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f}")
    print(f"  Macro Precision: {summary['macro_precision_mean']:.4f} ± {summary['macro_precision_std']:.4f}")
    print(f"  Macro Recall   : {summary['macro_recall_mean']:.4f} ± {summary['macro_recall_std']:.4f}")
    print("-" * 64)
    print("  Per-fold accuracy:", [f"{a:.4f}" for a in summary["per_fold_accuracy"]])
    print("  Per-fold macro F1:", [f"{f:.4f}" for f in summary["per_fold_macro_f1"]])

    if class_names:
        print("-" * 64)
        print("  Per-class results (mean ± std):")
        for name in class_names:
            p_m = summary.get(f"{name}_precision_mean", 0)
            p_s = summary.get(f"{name}_precision_std", 0)
            r_m = summary.get(f"{name}_recall_mean", 0)
            r_s = summary.get(f"{name}_recall_std", 0)
            f_m = summary.get(f"{name}_f1_mean", 0)
            f_s = summary.get(f"{name}_f1_std", 0)
            print(
                f"    {name:>8s}:  P={p_m:.3f}±{p_s:.3f}  "
                f"R={r_m:.3f}±{r_s:.3f}  F1={f_m:.3f}±{f_s:.3f}"
            )

    print("=" * 64)


# =============================================================================
# Final Test Evaluation
# =============================================================================

def evaluate_on_test(
    args: argparse.Namespace,
    best_fold_dir: str,
    test_operators: List[str],
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
) -> Dict[str, Any]:
    """
    Load the best model from the best fold and evaluate on the held-out
    test operator(s).
    """
    show = _is_main(rank)
    class_names = args.class_names.split(",") if args.class_names else None

    # Load best model
    ckpt_path = Path(best_fold_dir) / "best_model.pt"
    if not ckpt_path.exists():
        if show:
            print(f"  No checkpoint found at {ckpt_path}, skipping test evaluation.")
        return {}

    model = create_videomae_for_classification(
        num_classes=args.num_classes,
        model_name=args.model_name,
        num_frames=args.num_frames,
        image_size=args.image_size,
        from_pretrained=False,
        freeze_encoder=False,
        dropout=args.dropout,
    )

    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    test_loader = create_test_dataloader(
        data_dir=args.data_dir,
        test_operators=test_operators,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_frames=args.num_frames,
        image_size=args.image_size,
        class_names=class_names,
        num_clips=args.val_num_clips,
        sampling_strategy=args.sampling_strategy,
        activity_filter=args.activity_filter,
        rank=rank,
        world_size=world_size,
    )

    metrics = validate(
        model, test_loader, device,
        num_classes=args.num_classes,
        class_names=class_names,
        show_progress=show,
    )

    if show:
        print(f"\n{'=' * 64}")
        print(f"  HELD-OUT TEST RESULTS  (test_operators={test_operators})")
        print(f"{'=' * 64}")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Macro F1 : {metrics.get('macro_f1', 0):.4f}")
        if class_names:
            for name in class_names:
                p = metrics.get(f"{name}_precision", 0)
                r = metrics.get(f"{name}_recall", 0)
                f1 = metrics.get(f"{name}_f1", 0)
                print(f"    {name:>8s}: P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
        print(f"{'=' * 64}")

    return metrics


# =============================================================================
# Main Orchestrator
# =============================================================================

def run_loocv(args: argparse.Namespace) -> None:
    """
    Run full Leave-One-Operator-Out Cross-Validation.

    - CV operators: ope1 .. ope7  (configurable via --cv_operators)
    - Test operator: ope8          (configurable via --test_operator)
    - 7 folds: each fold holds out one CV operator for validation
    """
    # ── Setup ─────────────────────────────────────────────────────────────
    rank, world_size, local_rank, is_distributed = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    show = _is_main(rank)

    set_seed(args.seed)

    if show:
        print("=" * 64)
        print("  LEAVE-ONE-OPERATOR-OUT CROSS-VALIDATION")
        print("=" * 64)
        print(f"  Device            : {device}")
        print(f"  Distributed       : {is_distributed}")
        if is_distributed:
            print(f"  World size        : {world_size}")
        print(f"  CV operators      : {args.cv_operators}")
        print(f"  Test operator     : {args.test_operator}")
        print(f"  Num folds         : {len(args.cv_operators)}")
        print(f"  Epochs per fold   : {args.epochs}")
        print(f"  Early stop patience: {args.patience}")
        print(f"  Model             : {args.model_name}")
        print(f"  Learning rate     : {args.learning_rate}")
        print(f"  Batch size        : {args.batch_size}")
        if args.activity_filter:
            print(f"  Activity filter   : {args.activity_filter}")
        print(f"  Train clips/video : {args.num_clips}")
        print(f"  Val clips/video   : {args.val_num_clips}")
        print(f"  Label smoothing   : {args.label_smoothing}")
        print(f"  Dropout           : {args.dropout}")
        print("=" * 64)

    output_dir = Path(args.output_dir)
    if show:
        output_dir.mkdir(parents=True, exist_ok=True)

    if is_distributed:
        dist.barrier()

    # ── Run folds ─────────────────────────────────────────────────────────
    cv_operators = list(args.cv_operators)
    fold_results: List[Dict[str, Any]] = []

    for fold_idx, val_op in enumerate(cv_operators, start=1):
        train_ops = [op for op in cv_operators if op != val_op]

        if show:
            print(f"\n{'━' * 64}")
            print(f"  FOLD {fold_idx}/{len(cv_operators)} — "
                  f"Validate on: {val_op}  |  Train on: {train_ops}")
            print(f"{'━' * 64}")

        # Ensure reproducibility per fold but different seed per fold
        set_seed(args.seed + fold_idx)

        fold_metrics = train_single_fold(
            fold_idx=fold_idx,
            train_operators=train_ops,
            val_operator=val_op,
            args=args,
            device=device,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            is_distributed=is_distributed,
        )
        fold_results.append(fold_metrics)

    # ── Aggregate ─────────────────────────────────────────────────────────
    class_names = args.class_names.split(",") if args.class_names else None

    if show:
        summary = aggregate_cv_results(fold_results, class_names)
        print_cv_summary(summary, class_names)

        # Save aggregated results
        _save_metrics_json(summary, str(output_dir / "cv_summary.json"))
        _save_metrics_json(
            {"folds": fold_results},
            str(output_dir / "cv_all_folds.json"),
        )

    # ── Held-out test ─────────────────────────────────────────────────────
    if args.test_operator and show:
        # Pick the fold with the best validation accuracy
        best_fold = max(fold_results, key=lambda r: r["accuracy"])
        best_fold_idx = best_fold["fold"]
        best_val_op = best_fold["val_operator"]
        best_fold_dir = output_dir / f"fold_{best_fold_idx}_val_{best_val_op}"

        print(f"\n  Best fold: {best_fold_idx} (val_acc={best_fold['accuracy']:.4f})")
        print(f"  Evaluating on held-out test operator: {args.test_operator}")

        test_metrics = evaluate_on_test(
            args=args,
            best_fold_dir=str(best_fold_dir),
            test_operators=[args.test_operator],
            device=device,
            rank=rank,
            world_size=world_size,
        )

        if test_metrics:
            _save_metrics_json(test_metrics, str(output_dir / "test_results.json"))

    # ── Cleanup ───────────────────────────────────────────────────────────
    cleanup_distributed()

    if show:
        print("\nDone!  All results saved to:", output_dir)


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leave-One-Operator-Out Cross-Validation for VideoMAE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ──────────────────────────────────────────────────────────────
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset root (e.g. ../Augmented_Dataset_2)")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Number of frames per clip")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Input spatial resolution")
    parser.add_argument("--class_names", type=str, default="empty,light,heavy",
                        help="Comma-separated class names")
    parser.add_argument("--num_clips", type=int, default=1,
                        help="Number of clips per video for training (1 = no duplication)")
    parser.add_argument("--val_num_clips", type=int, default=8,
                        help="Number of clips per video for validation/test (logit averaging)")
    parser.add_argument("--sampling_strategy", type=str, default="multi_clip",
                        choices=["multi_clip", "uniform"],
                        help="Frame sampling strategy")
    parser.add_argument("--activity_filter", type=str, default=None,
                        choices=["walk", "carry"],
                        help="If set, only use videos from this activity subfolder (e.g., walk or carry)")

    # ── Cross-validation operators ────────────────────────────────────────
    parser.add_argument("--cv_operators", type=str,
                        default="ope1,ope2,ope3,ope4,ope5,ope6,ope7",
                        help="Comma-separated operators for LOOCV folds")
    parser.add_argument("--test_operator", type=str, default="ope8",
                        help="Held-out operator for final testing (set to '' to skip)")

    # ── Model ─────────────────────────────────────────────────────────────
    parser.add_argument("--model_name", type=str, default="videomae-base",
                        help="Model architecture name")
    parser.add_argument("--num_classes", type=int, default=3,
                        help="Number of classification classes")
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="Path to custom pretrained checkpoint")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze encoder (linear probing mode)")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Classifier head dropout")

    # ── Training ──────────────────────────────────────────────────────────
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Max epochs per fold")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help="Weight decay")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (0 = disable)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Data loading workers")

    # ── Output ────────────────────────────────────────────────────────────
    parser.add_argument("--output_dir", type=str, default="./checkpoints_cv",
                        help="Root output directory for all folds")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Parse cv_operators from comma-separated string to list
    args.cv_operators = [op.strip() for op in args.cv_operators.split(",") if op.strip()]

    return args


# =============================================================================
# Entry Point
# =============================================================================

def main():
    args = parse_args()
    run_loocv(args)


if __name__ == "__main__":
    main()
