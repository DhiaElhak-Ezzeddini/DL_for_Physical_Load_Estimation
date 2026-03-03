"""
VideoMAE Models using Hugging Face Transformers
================================================
Clean wrapper around HF VideoMAE for pretraining and fine-tuning.

Author: VLoad Project
Date: 2026
"""

from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
from transformers import (
    VideoMAEForPreTraining,
    VideoMAEForVideoClassification,
    VideoMAEConfig,
    VideoMAEModel
)


# =============================================================================
# Model Configurations
# =============================================================================

MODEL_CONFIGS = {
    "videomae-small": {
        "hidden_size": 384,
        "num_hidden_layers": 12,
        "num_attention_heads": 6,
        "intermediate_size": 1536,
    },
    "videomae-base": {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
    },
    "videomae-large": {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
    },
}

PRETRAINED_MODELS = {
    "videomae-base": "MCG-NJU/videomae-base",
    "videomae-base-finetuned-kinetics": "MCG-NJU/videomae-base-finetuned-kinetics",
    "videomae-large": "MCG-NJU/videomae-large",
    "videomae-large-finetuned-kinetics": "MCG-NJU/videomae-large-finetuned-kinetics",
    "videomae-base-short": "MCG-NJU/videomae-base-short",
    "videomae-base-short-finetuned-kinetics": "MCG-NJU/videomae-base-short-finetuned-kinetics",
}


# =============================================================================
# VideoMAE for Pretraining
# =============================================================================

def create_videomae_for_pretraining(
    model_name: str = "videomae-base",
    num_frames: int = 16,
    image_size: int = 224,
    patch_size: int = 16,
    tubelet_size: int = 2,
    mask_ratio: float = 0.9,
    decoder_num_heads: int = 6,
    decoder_hidden_size: int = 384,
    decoder_num_layers: int = 4,
    norm_pix_loss: bool = True,
    from_pretrained: bool = False,
) -> VideoMAEForPreTraining:
    """
    Create VideoMAE model for self-supervised pretraining.
    
    Args:
        model_name: Model configuration name or HF model ID
        num_frames: Number of input frames
        image_size: Input image size
        patch_size: Spatial patch size (square patches)
        tubelet_size: Temporal tubelet size
        mask_ratio: Ratio of patches to mask (0.9 = 90% masked)
        decoder_num_heads: Number of decoder attention heads
        decoder_hidden_size: Decoder hidden dimension
        decoder_num_layers: Number of decoder layers
        norm_pix_loss: Whether to normalize pixel targets
        from_pretrained: Load pretrained weights
        
    Returns:
        VideoMAEForPreTraining model
    """
    if from_pretrained and model_name in PRETRAINED_MODELS:
        print(f"Loading pretrained model: {PRETRAINED_MODELS[model_name]}")
        model = VideoMAEForPreTraining.from_pretrained(
            PRETRAINED_MODELS[model_name]
        )
        return model
    
    # Get base config
    base_config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["videomae-base"])
    
    config = VideoMAEConfig(
        image_size=image_size,
        patch_size=patch_size,  # Must be int, not tuple
        num_channels=3,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        hidden_size=base_config["hidden_size"],
        num_hidden_layers=base_config["num_hidden_layers"],
        num_attention_heads=base_config["num_attention_heads"],
        intermediate_size=base_config["intermediate_size"],
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        decoder_num_attention_heads=decoder_num_heads,
        decoder_hidden_size=decoder_hidden_size,
        decoder_num_hidden_layers=decoder_num_layers,
        decoder_intermediate_size=decoder_hidden_size * 4,
        mask_ratio=mask_ratio,
        norm_pix_loss=norm_pix_loss,
    )
    
    model = VideoMAEForPreTraining(config)
    print(f"Created VideoMAE for pretraining: {model_name}")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Num layers: {config.num_hidden_layers}")
    print(f"  - Num frames: {num_frames}, Image size: {image_size}")
    print(f"  - Mask ratio: {mask_ratio}")
    
    return model


# =============================================================================
# VideoMAE for Classification
# =============================================================================

def create_videomae_for_classification(
    num_classes: int,
    model_name: str = "videomae-base",
    num_frames: int = 16,
    image_size: int = 224,
    from_pretrained: bool = True,
    pretrained_model_name: Optional[str] = None,
    freeze_encoder: bool = False,
    dropout: float = 0.5,
    label_smoothing: float = 0.1,
) -> VideoMAEForVideoClassification:
    """
    Create VideoMAE model for video classification.
    
    Args:
        num_classes: Number of output classes
        model_name: Model configuration name
        num_frames: Number of input frames
        image_size: Input image size
        from_pretrained: Load pretrained encoder weights
        pretrained_model_name: Specific pretrained model to load
        freeze_encoder: Whether to freeze encoder weights
        dropout: Dropout rate for classifier (higher = more regularization)
        label_smoothing: Label smoothing factor (0.1-0.3 reduces overconfidence)
        
    Returns:
        VideoMAEForVideoClassification model
    """
    pretrained_name = pretrained_model_name or PRETRAINED_MODELS.get(
        model_name, "MCG-NJU/videomae-base"
    )
    
    if from_pretrained:
        print(f"Loading pretrained model: {pretrained_name}")
        model = VideoMAEForVideoClassification.from_pretrained(
            pretrained_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    else:
        base_config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["videomae-base"])
        config = VideoMAEConfig(
            image_size=image_size,
            num_channels=3,
            num_frames=num_frames,
            hidden_size=base_config["hidden_size"],
            num_hidden_layers=base_config["num_hidden_layers"],
            num_attention_heads=base_config["num_attention_heads"],
            intermediate_size=base_config["intermediate_size"],
            num_labels=num_classes,
        )
        model = VideoMAEForVideoClassification(config)
    
    # Apply higher dropout to classifier head to prevent overfitting
    if hasattr(model, 'classifier') and hasattr(model.classifier, 'dropout'):
        model.classifier.dropout = nn.Dropout(dropout)
    
    # Store label smoothing for loss computation
    model.config.label_smoothing = label_smoothing
    
    if freeze_encoder:
        print("Freezing encoder weights...")
        for param in model.videomae.parameters():
            param.requires_grad = False
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Created VideoMAE classifier: {num_classes} classes")
    print(f"  - Total params: {total_params / 1e6:.1f}M")
    print(f"  - Trainable params: {trainable_params / 1e6:.1f}M")
    print(f"  - Dropout: {dropout}, Label smoothing: {label_smoothing}")
    
    return model


# =============================================================================
# Load from Custom Checkpoint
# =============================================================================

def load_pretrained_encoder(
    model: VideoMAEForVideoClassification,
    checkpoint_path: str,
    strict: bool = False,
) -> VideoMAEForVideoClassification:
    """
    Load pretrained encoder weights from a pretraining checkpoint.
    
    Args:
        model: VideoMAE classification model
        checkpoint_path: Path to pretraining checkpoint
        strict: Whether to enforce strict state dict matching
        
    Returns:
        Model with loaded encoder weights
    """
    print(f"Loading encoder from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    # Filter to encoder weights only
    encoder_state_dict = {}
    for k, v in state_dict.items():
        # Remove prefix if present
        if k.startswith("videomae."):
            encoder_state_dict[k] = v
        elif k.startswith("encoder."):
            new_key = k.replace("encoder.", "videomae.encoder.")
            encoder_state_dict[new_key] = v
        elif not k.startswith("decoder") and not k.startswith("classifier"):
            encoder_state_dict[f"videomae.{k}"] = v
    
    # Load weights
    missing, unexpected = model.load_state_dict(encoder_state_dict, strict=strict)
    
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    
    print("Encoder weights loaded successfully!")
    return model


# =============================================================================
# Feature Extractor Wrapper
# =============================================================================

class VideoMAEFeatureExtractor(nn.Module):
    """
    Wrapper to extract features from VideoMAE encoder.
    Useful for downstream tasks with custom heads.
    """
    
    def __init__(
        self,
        model_name: str = "videomae-base",
        from_pretrained: bool = True,
        pool_type: str = "mean",
    ):
        """
        Initialize feature extractor.
        
        Args:
            model_name: Model name or HF model ID
            from_pretrained: Load pretrained weights
            pool_type: Pooling strategy ("mean", "cls")
        """
        super().__init__()
        
        pretrained_name = PRETRAINED_MODELS.get(model_name, model_name)
        
        if from_pretrained:
            self.encoder = VideoMAEModel.from_pretrained(pretrained_name)
        else:
            base_config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["videomae-base"])
            config = VideoMAEConfig(**base_config)
            self.encoder = VideoMAEModel(config)
        
        self.pool_type = pool_type
        self.hidden_size = self.encoder.config.hidden_size
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract features from video.
        
        Args:
            pixel_values: Video tensor (B, C, T, H, W) or (B, T, C, H, W)
            
        Returns:
            Features (B, hidden_size)
        """
        outputs = self.encoder(pixel_values=pixel_values)
        hidden_states = outputs.last_hidden_state  # (B, N, D)
        
        if self.pool_type == "mean":
            features = hidden_states.mean(dim=1)
        elif self.pool_type == "cls":
            features = hidden_states[:, 0]
        else:
            features = hidden_states.mean(dim=1)
        
        return features


# =============================================================================
# Custom Classification Head
# =============================================================================

class VideoMAEWithCustomHead(nn.Module):
    """
    VideoMAE encoder with custom classification head.
    Provides more flexibility than VideoMAEForVideoClassification.
    """
    
    def __init__(
        self,
        num_classes: int,
        model_name: str = "videomae-base",
        from_pretrained: bool = True,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        
        self.feature_extractor = VideoMAEFeatureExtractor(
            model_name=model_name,
            from_pretrained=from_pretrained,
        )
        
        hidden_size = self.feature_extractor.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )
        
        if freeze_encoder:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            pixel_values: Video tensor
            
        Returns:
            Logits (B, num_classes)
        """
        features = self.feature_extractor(pixel_values)
        logits = self.classifier(features)
        return logits


# =============================================================================
# Utility Functions
# =============================================================================

def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """Get model information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
        "total_params_m": f"{total_params / 1e6:.2f}M",
        "trainable_params_m": f"{trainable_params / 1e6:.2f}M",
    }


def freeze_model(model: nn.Module, freeze: bool = True) -> None:
    """Freeze or unfreeze model parameters."""
    for param in model.parameters():
        param.requires_grad = not freeze
