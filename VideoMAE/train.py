import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models import (
    create_videomae_for_pretraining,
    create_videomae_for_classification,
    load_pretrained_encoder,
    get_model_info,
)
from dataset import (
    create_pretraining_dataloader,
    create_classification_dataloaders,
    create_pretraining_dataloader_distributed,
    create_classification_dataloaders_distributed,
)


# =============================================================================
# Distributed Training Utilities
# =============================================================================

def setup_distributed():
    """Initialize distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Initialize process group with NCCL backend
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        # Set device for this process
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank, True
    else:
        # Single GPU training
        return 0, 1, 0, False


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """Check if current process is the main process."""
    return rank == 0


def setup_pretrain_logger(output_dir: Path, filename: str = "pretrain.log") -> logging.Logger:
    """Create a logger that writes to console and a pretraining log file."""
    # Use output directory in the logger name to keep handlers scoped per run
    logger = logging.getLogger(f"videomae_pretrain_{output_dir}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(message)s")

    output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(output_dir / filename)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


# =============================================================================
# Training Utilities
# =============================================================================

class AverageMeter:
    """Compute and store average values."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    loss: float,
    save_path: str,
    is_best: bool = False,
) -> None:
    """Save training checkpoint."""
    # Handle DataParallel wrapped models
    model_to_save = model.module if hasattr(model, 'module') else model
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    try:
        # Use legacy format (pickle) instead of zip to avoid write issues
        torch.save(checkpoint, save_path, _use_new_zipfile_serialization=False)
        
        if is_best:
            best_path = save_path.replace(".pt", "_best.pt")
            torch.save(checkpoint, best_path, _use_new_zipfile_serialization=False)
    except RuntimeError as e:
        if "file write failed" in str(e) or "PytorchStreamWriter failed" in str(e):
            print(f"WARNING: Failed to save checkpoint (disk full or read-only): {e}")
            print("Continuing training without saving...")
        else:
            raise


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    checkpoint_path: str,
) -> int:
    """Load training checkpoint. Returns start epoch."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return checkpoint.get("epoch", 0)


# =============================================================================
# Pretraining
# =============================================================================

def generate_tube_mask(
    batch_size: int,
    num_frames: int,
    image_size: int,
    patch_size: int,
    tubelet_size: int,
    mask_ratio: float,
    device: torch.device,
) -> torch.BoolTensor:
    """
    Generate tube masking for VideoMAE pretraining.
    
    A tube mask is generated over the spatial grid and replicated identically
    across all temporal positions, so each masked spatial location is masked
    at every timestep.  This forces the model to reconstruct entire
    space-time tubes rather than interpolating from adjacent frames.
    
    Args:
        batch_size: Batch size
        num_frames: Number of frames
        image_size: Image size
        patch_size: Spatial patch size
        tubelet_size: Temporal tubelet size
        mask_ratio: Ratio of patches to mask
        device: Device to create tensor on
        
    Returns:
        Boolean mask tensor (batch_size, num_patches) where True = masked
    """
    # Calculate number of patches
    num_patches_per_frame = (image_size // patch_size) ** 2  # 196 for 224/16
    num_temporal_patches = num_frames // tubelet_size         # 8 for 16/2
    
    # Number of spatial positions to mask (applied to every temporal position)
    num_spatial_masked = int(num_patches_per_frame * mask_ratio)
    
    # Generate random masks for each sample in batch
    masks = []
    for _ in range(batch_size):
        # Random permutation over spatial positions only
        noise = torch.rand(num_patches_per_frame)
        ids_shuffle = torch.argsort(noise)
        
        # Create spatial mask: True for masked positions
        spatial_mask = torch.zeros(num_patches_per_frame, dtype=torch.bool)
        spatial_mask[ids_shuffle[:num_spatial_masked]] = True
        
        # Repeat the same spatial mask across all temporal positions
        # Layout: [t0_s0, t0_s1, ..., t0_s195, t1_s0, ..., t7_s195]
        mask = spatial_mask.unsqueeze(0).expand(num_temporal_patches, -1).reshape(-1)
        masks.append(mask)
    
    return torch.stack(masks).to(device)


def train_pretrain_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    grad_accum_steps: int = 1,
    mask_ratio: float = 0.9,
    is_main_process: bool = True,
) -> float:
    """Train one epoch for pretraining."""
    model.train()
    loss_meter = AverageMeter()
    
    # Get model config for mask generation (handle DDP/DataParallel)
    config = model.module.config if hasattr(model, 'module') else model.config
    
    # Only show progress bar on main process
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", disable=not is_main_process)
    optimizer.zero_grad()
    
    for step, batch in enumerate(pbar):
        pixel_values = batch["pixel_values"].to(device)
        batch_size = pixel_values.size(0)
        
        # Generate tube mask
        bool_masked_pos = generate_tube_mask(
            batch_size=batch_size,
            num_frames=config.num_frames,
            image_size=config.image_size,
            patch_size=config.patch_size,
            tubelet_size=config.tubelet_size,
            mask_ratio=mask_ratio,
            device=device,
        )
        
        # Forward pass with mask
        outputs = model(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)
        # Handle DataParallel: loss may be a vector (one per GPU), take mean
        loss = outputs.loss.mean() / grad_accum_steps
        
        # Backward pass
        loss.backward()
        
        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        loss_meter.update(loss.item() * grad_accum_steps, pixel_values.size(0))
        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})
    
    return loss_meter.avg


def pretrain(args: argparse.Namespace) -> None:
    """Run pretraining with optional distributed training."""
    # Setup distributed training
    rank, world_size, local_rank, is_distributed = setup_distributed()
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # Create output directory and logger (only main process)
    output_dir = Path(args.output_dir)
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_pretrain_logger(output_dir) if is_main_process(rank) else None
    log = logger.info if logger else print

    if is_main_process(rank):
        log(f"Using device: {device}")
        log(f"Distributed training: {is_distributed}")
        if is_distributed:
            log(f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")
            log("Using NCCL backend for distributed training")
    
    # Synchronize before continuing
    if is_distributed:
        dist.barrier()
    
    # Create model
    model = create_videomae_for_pretraining(
        model_name=args.model_name,
        num_frames=args.num_frames,
        image_size=args.image_size,
        mask_ratio=args.mask_ratio,
        from_pretrained=args.from_pretrained,
    )
    model = model.to(device)
    
    # Wrap model with DDP if distributed
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if is_main_process(rank):
            log(f"Using DistributedDataParallel with {world_size} GPUs (NCCL backend)")
    
    if is_main_process(rank):
        log(f"Model info: {get_model_info(model.module if is_distributed else model)}")
    
    # Create dataloader with distributed sampler
    if is_distributed:
        dataloader = create_pretraining_dataloader_distributed(
            video_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_frames=args.num_frames,
            image_size=args.image_size,
            num_clips=args.num_clips,
            rank=rank,
            world_size=world_size,
            activity_filter=args.activity_filter,
        )
    else:
        dataloader = create_pretraining_dataloader(
            video_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_frames=args.num_frames,
            image_size=args.image_size,
            num_clips=args.num_clips,
            activity_filter=args.activity_filter,
        )
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.01,
    )
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.resume)
        if is_main_process(rank):
            log(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    best_loss = float("inf")
    
    for epoch in range(start_epoch, args.epochs):
        # Set epoch for distributed sampler
        if is_distributed and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
        
        loss = train_pretrain_epoch(
            model, dataloader, optimizer, device, epoch,
            grad_accum_steps=args.grad_accum_steps,
            mask_ratio=args.mask_ratio,
            is_main_process=is_main_process(rank),
        )
        
        scheduler.step()
        
        if is_main_process(rank):
            log(f"Epoch {epoch}: loss={loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")
        
        # Save only the best model (only main process)
        if is_main_process(rank) and loss < best_loss:
            best_loss = loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, loss,
                str(output_dir / "best_model.pt"),
                is_best=True,
            )
            log(f"New best model saved (loss={loss:.4f})")

    if is_main_process(rank):
        log(f"Pretraining complete! Best loss: {best_loss:.4f}")
        log(f"Best model saved to {output_dir / 'best_model.pt'}")
    
    # Cleanup
    cleanup_distributed()


# =============================================================================
# Fine-tuning
# =============================================================================

def train_finetune_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    label_smoothing: float = 0.1,
    is_main_process: bool = True,
) -> tuple:
    """Train one epoch for classification with label smoothing."""
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    # Create loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", disable=not is_main_process)
    
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass without labels to get logits
        outputs = model(pixel_values=pixel_values)
        
        # Compute loss with label smoothing
        loss = criterion(outputs.logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Compute accuracy
        preds = outputs.logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        
        loss_meter.update(loss.item(), pixel_values.size(0))
        acc_meter.update(acc.item(), pixel_values.size(0))
        
        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "acc": f"{acc_meter.avg:.4f}",
        })
    
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    is_main_process: bool = True,
) -> tuple:
    """Validate model."""
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    for batch in tqdm(dataloader, desc="Validating", disable=not is_main_process):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(pixel_values=pixel_values, labels=labels)
        
        preds = outputs.logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        
        # Handle DDP: take mean for safety
        loss_meter.update(outputs.loss.mean().item(), pixel_values.size(0))
        acc_meter.update(acc.item(), pixel_values.size(0))
    
    return loss_meter.avg, acc_meter.avg


def finetune(args: argparse.Namespace) -> None:
    """Run fine-tuning for classification with optional distributed training."""
    # Setup distributed training
    rank, world_size, local_rank, is_distributed = setup_distributed()
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    if is_main_process(rank):
        print(f"Using device: {device}")
        print(f"Distributed training: {is_distributed}")
        if is_distributed:
            print(f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")
            print(f"Using NCCL backend for distributed training")
        if args.activity_filter:
            print(f"Activity filter: {args.activity_filter}")
    
    # Create output directory (only main process)
    output_dir = Path(args.output_dir)
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Synchronize before continuing
    if is_distributed:
        dist.barrier()
    
    # Parse validation operators
    val_operators = args.val_operators.split(",") if args.val_operators else None
    
    # Create dataloaders with distributed sampler
    if is_distributed:
        train_loader, val_loader = create_classification_dataloaders_distributed(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_frames=args.num_frames,
            image_size=args.image_size,
            val_operators=val_operators,
            class_names=args.class_names.split(",") if args.class_names else None,
            num_clips=args.num_clips,
            sampling_strategy=args.sampling_strategy,
            activity_filter=args.activity_filter,
            rank=rank,
            world_size=world_size,
        )
    else:
        train_loader, val_loader = create_classification_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_frames=args.num_frames,
            image_size=args.image_size,
            val_operators=val_operators,
            class_names=args.class_names.split(",") if args.class_names else None,
            num_clips=args.num_clips,
            sampling_strategy=args.sampling_strategy,
            activity_filter=args.activity_filter,
        )
    
    # Create model
    model = create_videomae_for_classification(
        num_classes=args.num_classes,
        model_name=args.model_name,
        num_frames=args.num_frames,
        image_size=args.image_size,
        from_pretrained=not args.pretrained_path,  # Use HF pretrained if no custom checkpoint
        freeze_encoder=args.freeze_encoder,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
    )
    
    # Load custom pretrained weights if provided
    if args.pretrained_path:
        model = load_pretrained_encoder(model, args.pretrained_path)
    
    model = model.to(device)
    
    # Wrap model with DDP if distributed
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if is_main_process(rank):
            print(f"Using DistributedDataParallel with {world_size} GPUs (NCCL backend)")
    
    if is_main_process(rank):
        print(f"\nModel info: {get_model_info(model.module if is_distributed else model)}")
    
    # Optimizer and scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.01,
    )
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.resume)
        if is_main_process(rank):
            print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    best_acc = 0.0
    
    for epoch in range(start_epoch, args.epochs):
        # Set epoch for distributed sampler
        if is_distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        train_loss, train_acc = train_finetune_epoch(
            model, train_loader, optimizer, device, epoch,
            label_smoothing=args.label_smoothing,
            is_main_process=is_main_process(rank),
        )
        
        val_loss, val_acc = validate(model, val_loader, device, is_main_process=is_main_process(rank))
        
        scheduler.step()
        
        if is_main_process(rank):
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        # Save only the best model (only main process)
        if is_main_process(rank) and val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                str(output_dir / "best_model.pt"),
                is_best=True,
            )
            print(f"  -> New best model saved (acc={val_acc:.4f})")
    
    if is_main_process(rank):
        print(f"\nFine-tuning complete! Best accuracy: {best_acc:.4f}")
        print(f"Best model saved to {output_dir / 'best_model.pt'}")
    
    # Cleanup
    cleanup_distributed()


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VideoMAE Training")
    
    # Mode
    parser.add_argument("--mode", type=str, default="finetune",choices=["pretrain", "finetune"],help="Training mode")
    
    # Data
    parser.add_argument("--data_dir", type=str, required=True,help="Path to dataset directory")
    parser.add_argument("--num_frames", type=int, default=16,help="Number of frames per video")
    parser.add_argument("--image_size", type=int, default=224,help="Input image size")
    parser.add_argument("--val_operators", type=str, default="ope7,ope8",help="default: 'ope7,ope8'")
    parser.add_argument("--class_names", type=str, default=None,help="Comma-separated class names 'empty,light,heavy' ")
    parser.add_argument("--num_clips", type=int, default=8,help="Number of clips to extract per video (default: 8)")
    parser.add_argument("--sampling_strategy", type=str, default="multi_clip",choices=["multi_clip", "uniform"],
                        help="Frame sampling strategy: 'multi_clip' (8 clips from middle 60%%) "
                             "or 'uniform' (16 frames uniformly from middle 60%%)")
    parser.add_argument("--activity_filter", type=str, default=None, choices=["walk", "carry"],
                        help="Only include videos from this activity subfolder (walk or carry) "
                            "for pretraining or fine-tuning; matched against dataset path parts.")
    
    # Model
    parser.add_argument("--model_name", type=str, default="videomae-base",help="Model configuration name")
    parser.add_argument("--num_classes", type=int, default=3,help="Number of classes (for fine-tuning)")
    parser.add_argument("--from_pretrained", action="store_true",help="Load pretrained weights from HuggingFace")
    parser.add_argument("--pretrained_path", type=str, default=None,help="Path to custom pretrained checkpoint")
    parser.add_argument("--freeze_encoder", action="store_true",help="Freeze encoder during fine-tuning")
    parser.add_argument("--mask_ratio", type=float, default=0.9,help="Mask ratio for pretraining")
    parser.add_argument("--label_smoothing", type=float, default=0.1,help="Label smoothing factor (0.0-0.3 recommended)")
    parser.add_argument("--dropout", type=float, default=0.5,help="Dropout rate for classifier head")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=8,help="Batch size")
    parser.add_argument("--epochs", type=int, default=100,help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05,help="Weight decay")
    parser.add_argument("--grad_accum_steps", type=int, default=1,help="Gradient accumulation steps")
    parser.add_argument("--num_workers", type=int, default=4,help="Number of data loading workers")
    
    # Checkpointing
    parser.add_argument("--output_dir", type=str, default="./checkpoints",help="Output directory for checkpoints")
    parser.add_argument("--resume", type=str, default=None,help="Path to checkpoint to resume from")
    parser.add_argument("--save_every", type=int, default=5,help="Save checkpoint every N epochs")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("VideoMAE Training")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Data: {args.data_dir}")
    print(f"Model: {args.model_name}")
    print("=" * 60)
    
    if args.mode == "pretrain":
        pretrain(args)
    else:
        finetune(args)


if __name__ == "__main__":
    main()
