import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Lightweight LoRA adapter (no peft dependency)
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """LoRA adapter wrapping an existing nn.Linear (frozen)."""

    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 32.0):
        super().__init__()
        self.original = original
        self.scaling = alpha / rank
        self.lora_A = nn.Linear(original.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, original.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.original(x) + self.lora_B(self.lora_A(x)) * self.scaling


def apply_lora_to_model(model: nn.Module, target_names=("qkv", "proj"), rank: int = 8, alpha: float = 32.0):
    """Walk *model* and replace matching nn.Linear layers with LoRALinear wrappers.

    Returns the number of LoRA-injected layers.
    """
    count = 0
    for parent_name, parent_module in list(model.named_modules()):
        for attr_name, child in list(parent_module.named_children()):
            if isinstance(child, nn.Linear) and any(t in attr_name for t in target_names):
                lora = LoRALinear(child, rank=rank, alpha=alpha)
                setattr(parent_module, attr_name, lora)
                count += 1
    return count


# ---------------------------------------------------------------------------
# Enhanced Temporal Encoding Modules (Priority 1 Improvements)
# ---------------------------------------------------------------------------

class TemporalConv3DEncoder(nn.Module):
    """3D Convolutional encoder for temporal-spatial feature extraction.

    Preserves temporal structure while downsampling spatial dimensions.
    Input: [B, C, T, H, W] where T is number of frames
    Output: [B, C_out, T, H', W'] where H', W' are downsampled
    """

    def __init__(self, in_channels=2048, out_channels=512, num_frames=3):
        super().__init__()

        # 3D Conv layers: preserve temporal dimension, downsample spatial
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 1024, kernel_size=(3, 3, 3),
                     stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.GroupNorm(32, 1024),
            nn.GELU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(1024, out_channels, kernel_size=(3, 3, 3),
                     stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
        )

        # Output: [B, 512, T, H/4, W/4]
        # For 37×37 input: [B, 512, 3, 9, 9]
 
    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W] - Temporal-spatial features
        Returns:
            [B, C_out, T, H', W'] - Downsampled features
        """
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MotionAwareTemporalBlock(nn.Module):
    """Temporal-only attention block that matches DynamicVGGT MTA equations (6)-(8).

    Input is `[B, T, K, D]`, where temporal attention is computed independently for
    each token slot `K` along the frame axis `T`.
    """

    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal self-attention over the frame axis only.

        Args:
            x: [B, T, K, D]
        Returns:
            [B, T, K, D]
        """
        bsz, num_frames, num_slots, dim = x.shape

        x_norm = self.norm1(x)
        x_time = x_norm.permute(0, 2, 1, 3).reshape(bsz * num_slots, num_frames, dim)
        attn_out, _ = self.attn(x_time, x_time, x_time, need_weights=False)
        attn_out = attn_out.reshape(bsz, num_slots, num_frames, dim).permute(0, 2, 1, 3)

        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class CausalTemporalAttention(nn.Module):
    """Causal temporal attention for modeling t-2, t-1 → t dependencies.

    Each frame can only attend to itself and previous frames (causal mask).
    This enforces temporal causality: future frames cannot influence past frames.
    """

    def __init__(self, embed_dim=512, num_heads=8, num_frames=3):
        super().__init__()

        self.num_frames = num_frames
        self.embed_dim = embed_dim

        # Multi-head attention
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )

        # Layer norm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1),
        )

    def _generate_causal_mask(self, num_frames, tokens_per_frame, device):
        """Generate causal mask: frame t can only see frames 0...t.

        Returns:
            mask: [total_tokens, total_tokens] bool tensor
                  True = cannot attend, False = can attend
        """
        total_tokens = num_frames * tokens_per_frame
        mask = torch.ones(total_tokens, total_tokens, dtype=torch.bool, device=device)

        for t in range(num_frames):
            start_t = t * tokens_per_frame
            end_t = (t + 1) * tokens_per_frame

            # Frame t can see all previous frames (0...t-1) and itself
            for prev_t in range(t + 1):
                start_prev = prev_t * tokens_per_frame
                end_prev = (prev_t + 1) * tokens_per_frame
                mask[start_t:end_t, start_prev:end_prev] = False  # False = can attend

        return mask

    def forward(self, x, tokens_per_frame):
        """
        Args:
            x: [B, T*N, D] - Temporal tokens (T frames, N tokens per frame)
            tokens_per_frame: int - Number of tokens per frame
        Returns:
            [B, T*N, D] - Attended features with temporal dependencies
        """
        B, total_tokens, D = x.shape
        num_frames = total_tokens // tokens_per_frame

        # Generate causal mask
        attn_mask = self._generate_causal_mask(num_frames, tokens_per_frame, x.device)

        # Self-attention with causal mask
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + attn_out

        # FFN
        x = x + self.ffn(self.norm2(x))

        return x


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal frames.

    Combines fixed sinusoidal encoding with learnable scaling.
    """

    def __init__(self, num_frames, embed_dim):
        super().__init__()

        # Generate sinusoidal encoding
        position = torch.arange(num_frames).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() *
                            -(math.log(10000.0) / embed_dim))

        pe = torch.zeros(num_frames, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

        # Learnable scaling
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, frame_idx):
        """
        Args:
            frame_idx: int - Frame index (0, 1, 2, ...)
        Returns:
            [embed_dim] - Positional encoding for this frame
        """
        return self.pe[frame_idx] * self.scale


# Add AD-FFgsStudio to python path
_root_path = Path(__file__).resolve().parents[3]
_ad_ffgs_path = _root_path / "third_party" / "AD-FFgsStudio"
if str(_ad_ffgs_path) not in sys.path:
    sys.path.append(str(_ad_ffgs_path))

try:
    from models.vggt3dgs_model import VGGT3DGSModel
except ImportError:
    VGGT3DGSModel = None
    logging.warning("VGGT3DGSModel not available. Gaussian features will be disabled.")







class GaussianFeatureEncoder(nn.Module):
    """
    Adapter for integrating VGGT (Transformer-based 3DGS) features into GaussianDream models.
    """
    def __init__(self, use_gaussian: bool, action_expert_width: int, use_lgpd: bool = False,
                 num_frames: int = 3, inference_num_frames: int = 1,
                 use_single_frame_mode: bool = False,
                 temporal_context_offsets: tuple[int, ...] | None = None,
                 unfreeze_encoder: bool = False, unfreeze_decoder_only: bool = True,
                 use_lora: bool = True, lora_rank: int = 8, lora_alpha: float = 32.0,
                 lora_targets=("qkv", "proj")):
        """
        Adapter for integrating VGGT (Transformer-based 3DGS) features into GaussianDream models.
        LGPD support has been removed; this adapter only handles Gaussian and MTA features.
        """
        super().__init__()
        self.use_gaussian = use_gaussian
        self.encoder = None
        self.proj = None
        self.temporal_context_offsets = tuple(temporal_context_offsets or tuple(range(-(num_frames - 1), 1)))

        # Single-frame mode: override num_frames to 1 for both training and inference
        if use_single_frame_mode:
            self.num_frames = 1
            self.inference_num_frames = 1
            self.temporal_context_offsets = (0,)
            logging.info("Single-frame mode enabled: using 1 frame for both training and inference")
        else:
            self.num_frames = num_frames
            self.inference_num_frames = inference_num_frames

        self._temporal_context_labels = [
            "t" if offset == 0 else (f"t+{offset}" if offset > 0 else f"t{offset}")
            for offset in self.temporal_context_offsets
        ]

        if self.use_gaussian and VGGT3DGSModel is not None:
            logging.info("Initializing VGGT 3DGS Components in Adapter...")

            try:
                # Initialize VGGT Model
                # Parameters based on vggt3dgs_model.py defaults or typical values
                self.encoder = VGGT3DGSModel(sh_degree=4, min_depth=1.5, max_depth=100.0)
                
                # Freeze/Unfreeze Encoder based on configuration
                # NOTE: gs_head/gs_feathead/depth_head are no longer run in forward()
                # (only aggregator is used), so we freeze the entire encoder by default
                # and only selectively unfreeze LoRA in the aggregator.
                if unfreeze_encoder:
                    # Unfreeze entire encoder for end-to-end training
                    for param in self.encoder.parameters():
                        param.requires_grad = True
                    self.encoder.train()
                    logging.info("VGGT encoder is UNFROZEN - will be trained end-to-end")
                else:
                    # Freeze entire encoder (gs_head/depth_head are skipped in forward)
                    for param in self.encoder.parameters():
                        param.requires_grad = False
                    self.encoder.eval()
                    logging.info("VGGT encoder is FROZEN")

                # Unfreeze LoRA parameters in encoder backbone (if use_lora=True)
                # VGGT pretrained model already has LoRA layers (lora_down, lora_up)
                if use_lora and not unfreeze_encoder:
                    aggregator = self.encoder.aggregator
                    n_lora = 0
                    for name, param in aggregator.named_parameters():
                        # Unfreeze LoRA parameters (lora_down, lora_up)
                        if 'lora_down' in name or 'lora_up' in name:
                            param.requires_grad = True
                            n_lora += 1

                    lora_params = sum(
                        p.numel() for p in aggregator.parameters() if p.requires_grad
                    )
                    total_params = sum(p.numel() for p in aggregator.parameters())
                    logging.info(
                        f"LoRA unfrozen: {n_lora} parameters, rank={lora_rank}, "
                        f"trainable={lora_params:,} / {total_params:,} "
                        f"({100*lora_params/total_params:.2f}%)"
                    )

                # === Priority 1 & 2 Improvements: Enhanced Temporal Encoding + Multi-Scale Features ===

                # Configuration
                self.gaussian_feat_dim = 2048  # VGGT output dimension
                self.tokens_per_frame = 256  # Keep static/current Gaussian tokens at 16×16
                self.mta_tokens_per_frame = 1024  # Future MTA slots use 32×32 for finer motion detail

                # Single-frame mode: disable temporal encoding to avoid "single frame through temporal modules"
                # This ensures a clean ablation: spatial-only feature extraction without temporal machinery
                if use_single_frame_mode:
                    self.use_enhanced_temporal = False  # Force disable temporal modules in single-frame mode
                    self.use_multi_scale = False  # Also disable multi-scale (requires temporal modules)
                    logging.info("Single-frame mode: disabling temporal and multi-scale modules for clean spatial-only ablation")
                else:
                    self.use_enhanced_temporal = True  # Enable enhanced temporal encoding
                    self.use_multi_scale = True  # Enable multi-scale feature extraction

                if self.use_enhanced_temporal:
                    # === Multi-Scale Feature Extraction (Priority 2) ===
                    if self.use_multi_scale:
                        # Extract features from layers [11, 17, 23] instead of just [23]
                        self.layer_indices = [11, 17, 23]

                        # Project each layer to unified dimension (512)
                        self.layer_projs = nn.ModuleList([
                            nn.Linear(2048, 512) for _ in self.layer_indices
                        ])

                        # FPN-style fusion (bottom-up)
                        self.fusion_blocks = nn.ModuleList([
                            nn.Sequential(
                                nn.Linear(512, 512),
                                nn.LayerNorm(512),
                                nn.GELU(),
                            ) for _ in range(len(self.layer_indices) - 1)
                        ])

                        # Project fused features back to 2048 for downstream spatial pooling
                        self.fused_to_2048 = nn.Linear(512, 2048)

                        logging.info(
                            f"Multi-scale feature extraction enabled:\n"
                            f"  - Layers: {self.layer_indices}\n"
                            f"  - Fusion: FPN-style bottom-up\n"
                            f"  - Projection: 512 → 2048 for spatial pooling"
                        )
                    else:
                        self.layer_indices = [23]  # Only last layer
                        self.layer_projs = None
                        self.fusion_blocks = None
                        self.fused_to_2048 = None

                    # Simplified temporal path: remove 3D conv + causal temporal attention.
                    # Keep per-frame 16x16 spatial pooling so each frame still contributes 256 tokens.
                    self.tokens_per_frame = 256

                    # Frame positional encoding (sinusoidal + learnable)
                    # Match projected token width so per-frame offsets can be added to frame_embs.
                    self.frame_pos_encoding = nn.ModuleList([
                        SinusoidalPositionalEncoding(self.num_frames, action_expert_width)
                        for _ in range(self.num_frames)
                    ])
                    self.frame_embeddings = nn.Parameter(
                        torch.randn(self.num_frames, action_expert_width) * 0.02
                    )
                    self.temporal_offset_proj = nn.Linear(1, action_expert_width)

                    # Dedicated projections for exposing per-layer MTA patch tokens.
                    # Main Gaussian embeddings stay at 16×16, while MTA features use a 32×32 pool
                    # so future motion-query tokens can operate at higher spatial resolution.
                    # DynamicVGGT-style MTA consumes denser backbone taps than the static
                    # Gaussian multi-scale branch, so keep a separate 12-stage AA pairing here.
                    self.mta_layer_pairs = [(2 * idx, 2 * idx + 1) for idx in range(12)]
                    self.mta_layer_keys = tuple(pair_high for _, pair_high in self.mta_layer_pairs)
                    self.mta_layer_pool = nn.AdaptiveAvgPool2d((32, 32))
                    self.mta_pair_input_projs = nn.ModuleList([
                        nn.Linear(2048, 512) for _ in self.mta_layer_pairs
                    ])
                    self.mta_layer_projs = nn.ModuleList([
                        nn.Linear(512, 512) for _ in self.mta_layer_pairs
                    ])

                    # Projection to action_expert_width
                    self.proj = nn.Linear(2048, action_expert_width)

                    logging.info(
                        f"Simplified temporal encoding enabled:\n"
                        f"  - Removed 3D Conv and causal temporal attention\n"
                        f"  - Spatial resolution: 37×37 → 16×16 per frame\n"
                        f"  - Tokens per frame: {self.tokens_per_frame}\n"
                        f"  - Total tokens: {self.num_frames * self.tokens_per_frame}\n"
                        f"  - Frame positional encoding: {self.num_frames} frames"
                    )
                else:
                    # Original implementation (fallback)
                    # In single-frame mode, use 16×16 pooling to match future query expectations (256 tokens)
                    # In multi-frame mode, use 10×10 pooling for backward compatibility (100 tokens per frame)
                    if use_single_frame_mode:
                        self.pool = nn.AdaptiveAvgPool2d((16, 16))
                        self.tokens_per_frame = 256  # 16×16 = 256 tokens
                        logging.info(f"Using spatial-only encoding with 16×16 pooling (256 tokens, no temporal modules)")
                    else:
                        self.pool = nn.AdaptiveAvgPool2d((10, 10))
                        self.tokens_per_frame = 100  # 10×10 = 100 tokens per frame
                        logging.info(f"Using original temporal encoding with {self.num_frames} frames, 100 tokens/frame")

                    self.proj = nn.Linear(self.gaussian_feat_dim, action_expert_width)

                    # Frame positional encoding
                    # In single-frame mode, disable frame embeddings for clean spatial-only ablation
                    if use_single_frame_mode:
                        self.use_frame_pos_encoding = False
                        self.frame_embeddings = None
                        self.temporal_offset_proj = None
                    else:
                        # Multi-frame mode: use frame positional encoding
                        self.use_frame_pos_encoding = True
                        self.frame_embeddings = nn.Parameter(
                            torch.randn(self.num_frames, action_expert_width) * 0.02
                        )
                        self.temporal_offset_proj = nn.Linear(1, action_expert_width)

            except Exception as e:
                logging.error(f"Failed to initialize VGGT components: {e}")
                self.use_gaussian = False
                self.encoder = None
        else:
            self.use_gaussian = False


    def _frame_offset_embedding(self, frame_idx: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Project configured temporal offset into token space for frame identity."""
        if frame_idx < len(self.temporal_context_offsets):
            offset_value = float(self.temporal_context_offsets[frame_idx])
        else:
            offset_value = float(frame_idx - len(self.temporal_context_offsets) + 1)
        offset_tensor = torch.tensor([[offset_value]], device=device, dtype=dtype)
        if getattr(self, "temporal_offset_proj", None) is None:
            return torch.zeros(width, device=device, dtype=dtype)
        return self.temporal_offset_proj(offset_tensor).reshape(width)

    def encode(self, observation, device, batch_size):
        """Pure encoding step to get Gaussian features."""
        inputs = self.prepare_inputs(observation, device, batch_size)
        if inputs is None:
            return None
        return self.forward(inputs)[0]

    def _select_observation_image(
        self,
        images_dict,
        view_type: str,
        preferred_camera_name: str | None = None,
    ):
        """Resolve the image tensor to encode for VGGT."""
        if preferred_camera_name is not None:
            if preferred_camera_name in images_dict:
                return preferred_camera_name, images_dict[preferred_camera_name]

            preferred_lower = preferred_camera_name.lower()
            for key, value in images_dict.items():
                if key.lower() == preferred_lower:
                    return key, value

            available_keys = list(images_dict.keys())
            logging.warning(
                "Preferred camera %r not found for VGGT encoding. Available keys: %s",
                preferred_camera_name,
                available_keys,
            )
            return None, None

        selected_images = {}
        for key, value in images_dict.items():
            key_lower = key.lower()
            is_wrist_key = "wrist" in key_lower or "bravo" in key_lower
            if view_type == "agent":
                if is_wrist_key:
                    continue
                if key == "image" or any(prefix in key_lower for prefix in ["base", "agent", "sideview"]) or \
                   "_image" in key_lower or "_rgb" in key_lower:
                    selected_images[key] = value
            else:
                if is_wrist_key:
                    selected_images[key] = value

        if not selected_images:
            available_keys = list(images_dict.keys())
            logging.warning(f"No {view_type} view images found for VGGT encoding. Available keys: {available_keys}")
            return None, None

        selected_key = list(selected_images.keys())[0]
        return selected_key, selected_images[selected_key]

    def prepare_inputs(
        self,
        observation,
        device,
        batch_size,
        is_training: bool = None,
        view_type: str = "agent",
        current_frame_only: bool = False,
        preferred_camera_name: str | None = None,
        force_num_frames: int | None = None,
    ):
        """Helper to prepare inputs for VGGT encoder from observation.
        VGGT expects [Batch_size, S (frames), 3, H, W]

        Strategy:
        - Agent path (default): use packed context frames from agent view only.
        - Wrist path: optionally select wrist view and, when requested, only keep the current frame.
        - Exact-camera path: when preferred_camera_name is set, select that camera directly.

        Args:
            is_training: If True, uses num_frames. If False, uses inference_num_frames.
                        If None, auto-detects from model.training.
            view_type: Which logical camera family to use: "agent" or "wrist".
            current_frame_only: If True, keep only the current frame (last available context slot).
            preferred_camera_name: Optional exact camera key to use instead of view_type routing.
            force_num_frames: Optional explicit frame-count override. When shrinking a temporal stack,
                keeps the most recent frames so the current frame is preserved.
        """
        if not self.use_gaussian:
            return None

        if force_num_frames is not None:
            force_num_frames = int(force_num_frames)
            if force_num_frames <= 0:
                raise ValueError(f"force_num_frames must be positive, got {force_num_frames}")

        view_type = str(view_type).lower()
        if view_type not in {"agent", "wrist"}:
            raise ValueError(f"Unsupported view_type={view_type!r}; expected 'agent' or 'wrist'")

        # Observation is a dataclass with .images attribute (dict)
        if not hasattr(observation, 'images'):
            # Fallback: try to_dict() if it's an Observation object
            if hasattr(observation, 'to_dict'):
                obs_dict = observation.to_dict()
                images_dict = obs_dict.get('image', {})
            else:
                logging.warning("Observation does not have 'images' attribute and cannot convert to dict.")
                return None
        else:
            images_dict = observation.images

        selected_key, img = self._select_observation_image(
            images_dict,
            view_type=view_type,
            preferred_camera_name=preferred_camera_name,
        )
        if img is None:
            return None
        selected_route = f"camera={selected_key}" if preferred_camera_name is not None else f"{view_type} view"
        logging.debug(f"Using {selected_route} for VGGT: shape={img.shape}, ndim={img.ndim}")
        # Determine number of frames to use
        # Auto-detect training mode if not specified
        if is_training is None:
            is_training = self.training if hasattr(self, 'training') else True

        if force_num_frames is not None:
            target_num_frames = force_num_frames
        elif is_training:
            target_num_frames = self.num_frames
        else:
            target_num_frames = self.inference_num_frames
        if current_frame_only:
            target_num_frames = 1
            is_training = False

        # Use the image size defined in the encoder if available, otherwise default to 518 (VGGT standard)
        target_size = getattr(self.encoder, "img_size", 518)

        # Handle Temporal Dimension
        # Goal: Extract packed context frames from the front of the temporal slots.
        # Layout is expected to be [context..., future...], so current frame is always the
        # last context slot rather than inferred from total T.
        if img.ndim == 5:  # [B, T, C, H, W] or [B, T, H, W, C]
            T = img.shape[1]

            # Check if channel last
            if img.shape[-1] == 3:  # [B, T, H, W, C]
                img = img.permute(0, 1, 4, 2, 3)  # [B, T, C, H, W]

            if current_frame_only:
                if T <= 0:
                    logging.error(f"No temporal frames available for {view_type} current-frame selection")
                    return None
                current_idx = T - 1
                img = img[:, current_idx:current_idx + 1]
            elif T >= target_num_frames:
                if force_num_frames is not None:
                    img = img[:, T - target_num_frames:T]
                else:
                    img = img[:, :target_num_frames]
            else:
                if not is_training:
                    if T > 0 and target_num_frames > T:
                        logging.warning(
                            f"Inference: padding VGGT {view_type} view from T={T} to T={target_num_frames} "
                            f"by repeating frames (MTA expects same temporal length as training)."
                        )
                        pad_n = target_num_frames - T
                        repeat_idx = max(0, T - 1)
                        img = torch.cat(
                            [img, img[:, repeat_idx : repeat_idx + 1].repeat(1, pad_n, 1, 1, 1)],
                            dim=1,
                        )
                    else:
                        logging.debug(
                            f"Inference: Using {T} available frames (requested {target_num_frames})"
                        )
                        target_num_frames = T
                else:
                    if T == 0:
                        logging.error("No temporal frames available for GaussianFeatureEncoder.prepare_inputs")
                        return None
                    logging.warning(
                        f"Only {T} frames available, requested {target_num_frames}. Padding from earliest context frame."
                    )
                    first_frame = img[:, :1]
                    padding = first_frame.repeat(1, target_num_frames - T, 1, 1, 1)
                    img = torch.cat([padding, img], dim=1)

        elif img.ndim == 4:  # [B, C, H, W] or [B, H, W, C] - single frame
            # Check if channel last
            if img.shape[-1] == 3:  # [B, H, W, C]
                img = img.permute(0, 3, 1, 2)  # [B, C, H, W]

            # Single frame handling
            if not is_training or current_frame_only:
                img = img.unsqueeze(1)  # [B, 1, C, H, W]
                if target_num_frames > 1:
                    logging.warning(
                        f"Inference: {view_type} view is single-frame but model expects T={target_num_frames}; "
                        f"repeating the current frame for VGGT/MTA (prefer feeding a true multi-frame stack)."
                    )
                    img = img.repeat(1, target_num_frames, 1, 1, 1)
                else:
                    logging.debug(f"Inference: Using single frame for VGGT ({view_type})")
            else:
                # Training mode
                if target_num_frames == 1:
                    # Single-frame mode: this is expected, no warning needed
                    img = img.unsqueeze(1)  # [B, 1, C, H, W]
                else:
                    # Multi-frame mode but got single frame: this can happen near episode boundaries
                    # (e.g., not enough history frames). Repeat the frame without spamming warnings.
                    logging.debug(
                        f"Single frame input detected during training. "
                        f"Image shape: {img.shape}, Expected 5D [B, T, H, W, C] with T>={target_num_frames}. "
                        f"Repeating frame {target_num_frames} times for VGGT. "
                        f"Check data_loader delta_timestamps configuration if this is frequent."
                    )
                    img = img.unsqueeze(1).repeat(1, target_num_frames, 1, 1, 1)  # [B, target_num_frames, C, H, W]
        else:
            logging.error(f"Unexpected image shape: {img.shape}. Expected 4D [B, C, H, W] or 5D [B, T, C, H, W]")
            return None

        # Now img is [B, target_num_frames, C, H, W]
        # Process each frame: normalize and resize
        processed_frames = []
        for t in range(img.shape[1]):
            frame = img[:, t]  # [B, C, H, W]

            # Normalize to [0, 1]. GaussianDream observations are often float tensors in
            # [-1, 1], while some loaders may still produce uint8 or float [0, 255].
            if frame.dtype == torch.uint8:
                frame = frame.to(torch.float32) / 255.0
            else:
                frame = frame.to(torch.float32)
                frame_min = float(frame.detach().amin().item())
                frame_max = float(frame.detach().amax().item())
                if frame_min < -0.1:
                    frame = (frame + 1.0) / 2.0
                elif frame_max > 1.5:
                    frame = frame / 255.0

            # Resize
            if frame.shape[-2:] != (target_size, target_size):
                frame = F.interpolate(frame, size=(target_size, target_size),
                                    mode='bilinear', align_corners=False)

            processed_frames.append(frame)

        # Stack frames: [B, target_num_frames, C, H, W]
        imgs_stacked = torch.stack(processed_frames, dim=1)

        # Fix NaN: Check and sanitize input images before passing to VGGT encoder
        # This is critical because NaN in input will propagate through the entire model
        if torch.isnan(imgs_stacked).any() or torch.isinf(imgs_stacked).any():
            import warnings
            warnings.warn("NaN/Inf detected in VGGT input images! Replacing with zeros. This may indicate data loading issues.")
            imgs_stacked = torch.where(
                torch.isnan(imgs_stacked) | torch.isinf(imgs_stacked),
                torch.zeros_like(imgs_stacked),
                imgs_stacked
            )

        # Clamp to valid range [0, 1] for normalized images
        imgs_stacked = torch.clamp(imgs_stacked, min=0.0, max=1.0)

        mode_str = "training" if is_training else "inference"
        logging.debug(f"VGGT input shape: {imgs_stacked.shape} ({view_type} view, {target_num_frames} frames, {mode_str})")

        return imgs_stacked.to(device)

    def _extract_layer_mta_features(self, aggregated_tokens_list) -> Optional[Dict[int, torch.Tensor]]:
        """Extract pooled per-layer patch tokens for MTA.

        Returns a dict mapping VGGT layer index -> [B, T, 1024, 512].
        """
        if not (
            hasattr(self, 'use_multi_scale')
            and self.use_multi_scale
            and hasattr(self, 'mta_pair_input_projs')
            and self.mta_pair_input_projs is not None
        ):
            return None

        mta_features: Dict[int, torch.Tensor] = {}
        patch_start_idx_val = None
        if hasattr(self.encoder, 'aggregator') and hasattr(self.encoder.aggregator, 'patch_start_idx'):
            patch_start_idx_val = self.encoder.aggregator.patch_start_idx

        for idx, (layer_lo_idx, layer_hi_idx) in enumerate(self.mta_layer_pairs):
            if layer_hi_idx >= len(aggregated_tokens_list):
                logging.warning(
                    f"Requested MTA AA layer pair ({layer_lo_idx}, {layer_hi_idx}) but encoder only returned "
                    f"{len(aggregated_tokens_list)} layers. Stopping MTA feature extraction."
                )
                break

            layer_tokens_lo = aggregated_tokens_list[layer_lo_idx]  # [B, T, N, 2048]
            layer_tokens_hi = aggregated_tokens_list[layer_hi_idx]  # [B, T, N, 2048]
            if patch_start_idx_val is not None and layer_tokens_lo.shape[2] > 1369:
                layer_tokens_lo = layer_tokens_lo[:, :, patch_start_idx_val:]
            if patch_start_idx_val is not None and layer_tokens_hi.shape[2] > 1369:
                layer_tokens_hi = layer_tokens_hi[:, :, patch_start_idx_val:]

            if layer_tokens_lo.shape != layer_tokens_hi.shape:
                logging.warning(
                    f"MTA pair shape mismatch for layers ({layer_lo_idx}, {layer_hi_idx}): "
                    f"{tuple(layer_tokens_lo.shape)} vs {tuple(layer_tokens_hi.shape)}. Skipping pair."
                )
                continue

            layer_tokens = 0.5 * (layer_tokens_lo + layer_tokens_hi)

            bsz, num_frames, n_patches, _ = layer_tokens.shape
            if n_patches != 1369:
                logging.warning(
                    f"Expected 1369 patch tokens for MTA at pair ({layer_lo_idx}, {layer_hi_idx}), got {n_patches}. Skipping pair."
                )
                continue

            layer_tokens = self.mta_pair_input_projs[idx](layer_tokens)  # [B, T, 1369, 512]
            layer_tokens_2d = layer_tokens.permute(0, 1, 3, 2).reshape(bsz * num_frames, 512, 37, 37)
            layer_tokens_pooled = self.mta_layer_pool(layer_tokens_2d)  # [B*T, 512, 32, 32]
            mta_grid_size = layer_tokens_pooled.shape[-1]
            layer_tokens_flat = layer_tokens_pooled.reshape(bsz, num_frames, 512, mta_grid_size, mta_grid_size)
            layer_tokens_flat = layer_tokens_flat.permute(0, 1, 3, 4, 2).reshape(
                bsz, num_frames, mta_grid_size * mta_grid_size, 512
            )
            layer_tokens_flat = self.mta_layer_projs[idx](layer_tokens_flat)
            mta_features[layer_hi_idx] = layer_tokens_flat

        return mta_features if mta_features else None

    def forward(self, gaussian_inputs, text_embedding=None, return_gaussian_params=False, return_raw_tokens=False, return_mta_features=False, raw_tokens_layer_idx: int | None = None, return_unprojected_raw_tokens: bool = False, step=None, visualize=False):

        """
        Processes gaussian inputs and returns embeddings.
        Input:
            gaussian_inputs: [B, S, 3, H, W]
            text_embedding: [B, D] Optional text embedding for LGPD.
            return_gaussian_params: If True, also return decoded Gaussian parameters from VGGT.
            return_raw_tokens: If True, also return raw tokens before pooling (for VAE supervision).
            raw_tokens_layer_idx: If set, return patch tokens from that exact VGGT layer index before temporal pooling.
            return_unprojected_raw_tokens: If True, return frozen raw encoder-layer patch tokens without self.proj.
            step: Current training step (for visualization)
            visualize: If True and step % 100 == 0, save visualization
        Returns:
            gaussian_embs: [B, N, D] - Token embeddings for World Model
            g_mask: [B, N] - Mask for tokens
            gaussian_params (optional): Dict with depth, rot, scale, opacity, sh if return_gaussian_params=True
            raw_tokens (optional): [B, S, 1369, D] - Raw patch tokens before pooling if return_raw_tokens=True
            mta_features (optional): Dict[layer_idx, [B, T, 1024, 512]]
        """
        if not self.use_gaussian or gaussian_inputs is None:
            return (None, None) if not return_gaussian_params else (None, None, None)

        # Fix NaN: Check input before encoding
        if torch.isnan(gaussian_inputs).any() or torch.isinf(gaussian_inputs).any():
            import warnings
            warnings.warn("NaN/Inf detected in gaussian_inputs before VGGT encoding! This will cause NaN outputs.")
            # Replace NaN/Inf with zeros (black images) as fallback
            gaussian_inputs = torch.where(
                torch.isnan(gaussian_inputs) | torch.isinf(gaussian_inputs),
                torch.zeros_like(gaussian_inputs),
                gaussian_inputs
            )
        
        # Only run VGGT aggregator (backbone) — skip depth_head/gs_head
        # since depth & Gaussian params are predicted by the independent
        # GaussianDecoder (world_model) later.  This saves significant
        # GPU memory and compute.
        encoder_trainable = self.training and any(p.requires_grad for p in self.encoder.parameters())

        if encoder_trainable:
            self.encoder.train()
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                aggregated_tokens_list, patch_start_idx = self.encoder.aggregator(
                    gaussian_inputs.to(torch.bfloat16)
                )
        else:
            self.encoder.eval()
            with torch.no_grad():
                torch.cuda.empty_cache()
                with torch.inference_mode():
                    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                        aggregated_tokens_list, patch_start_idx = self.encoder.aggregator(
                            gaussian_inputs.to(torch.bfloat16)
                        )
                torch.cuda.empty_cache()

        # Check if we got valid features
        if aggregated_tokens_list is None or (isinstance(aggregated_tokens_list, (list, tuple)) and len(aggregated_tokens_list) == 0):
            logging.warning("No features extracted from Gaussian Encoder.")
            return (None, None) if not return_gaussian_params else (None, None, None)

        # VGGT depth/gs heads are no longer run here — Gaussian params
        # are predicted by the independent GaussianDecoder (world_model).
        gaussian_params_dict = None
        if return_gaussian_params:
            logging.warning("return_gaussian_params=True but VGGT heads are skipped. Returning None.")

        mta_features = self._extract_layer_mta_features(aggregated_tokens_list) if return_mta_features else None

        # Visualize if requested and step is a multiple of 100
        # Note: LGPD gate will be available after LGPD is applied below
        visualize_gate = False
        if visualize and step is not None and step % 100 == 0:
            visualize_gate = True
        
        # Process tokens with temporal awareness
        # Strategy: Enhanced temporal encoding with 3D Conv + Causal Attention

        # === Priority 3: Multi-scale Feature Extraction ===
        if raw_tokens_layer_idx is not None:
            if not isinstance(aggregated_tokens_list, (list, tuple)):
                raise ValueError("raw_tokens_layer_idx requires aggregated_tokens_list to be a list/tuple of per-layer features")
            resolved_layer_idx = raw_tokens_layer_idx
            if resolved_layer_idx < 0:
                resolved_layer_idx = len(aggregated_tokens_list) + resolved_layer_idx
            if resolved_layer_idx < 0 or resolved_layer_idx >= len(aggregated_tokens_list):
                raise ValueError(
                    f"raw_tokens_layer_idx={raw_tokens_layer_idx} is out of range for {len(aggregated_tokens_list)} VGGT layers"
                )
            raw_tokens = aggregated_tokens_list[resolved_layer_idx]
            B, S, N_patches, D = raw_tokens.shape
            if hasattr(self.encoder, 'aggregator') and hasattr(self.encoder.aggregator, 'patch_start_idx'):
                patch_start_idx_val = self.encoder.aggregator.patch_start_idx
                if N_patches > 1369:
                    raw_tokens = raw_tokens[:, :, patch_start_idx_val:]
                    B, S, N_patches, D = raw_tokens.shape
        elif hasattr(self, 'use_multi_scale') and self.use_multi_scale:
            # Extract features from layers [11, 17, 23]
            multi_scale_features = []
            for idx, layer_idx in enumerate(self.layer_indices):
                layer_tokens = aggregated_tokens_list[layer_idx]  # [B, S, N_patches, 2048]

                # Remove camera + register tokens
                if hasattr(self.encoder, 'aggregator') and hasattr(self.encoder.aggregator, 'patch_start_idx'):
                    patch_start_idx_val = self.encoder.aggregator.patch_start_idx
                    if layer_tokens.shape[2] > 1369:
                        layer_tokens = layer_tokens[:, :, patch_start_idx_val:]

                # Project to 512 dims
                layer_tokens = self.layer_projs[idx](layer_tokens)  # [B, S, 1369, 512]
                multi_scale_features.append(layer_tokens)

            # FPN-style bottom-up fusion: [11] <- [17] <- [23]
            fused_features = multi_scale_features[-1]  # Start with layer 23
            for i in range(len(multi_scale_features) - 2, -1, -1):
                # Fuse current layer with previous
                fused_features = fused_features + multi_scale_features[i]
                if i > 0:  # Apply fusion block except for the last iteration
                    B_f, S_f, N_f, D_f = fused_features.shape
                    fused_features = self.fusion_blocks[i](
                        fused_features.reshape(B_f * S_f * N_f, D_f)
                    ).reshape(B_f, S_f, N_f, D_f)

            # Project fused features back to 2048 for temporal_conv
            B_f, S_f, N_f, D_f = fused_features.shape
            fused_features = self.fused_to_2048(
                fused_features.reshape(B_f * S_f * N_f, D_f)
            ).reshape(B_f, S_f, N_f, 2048)

            raw_tokens = fused_features  # [B, S, 1369, 2048]
            B, S, N_patches, D = raw_tokens.shape
        else:
            # Original single-layer extraction
            if isinstance(aggregated_tokens_list, (list, tuple)):
                raw_tokens = aggregated_tokens_list[-1]
            else:
                raw_tokens = aggregated_tokens_list

            B, S, N_patches, D = raw_tokens.shape

            # Remove camera + register tokens (if not already removed)
            if hasattr(self.encoder, 'aggregator') and hasattr(self.encoder.aggregator, 'patch_start_idx'):
                patch_start_idx_val = self.encoder.aggregator.patch_start_idx
                if N_patches > 1369:  # Has camera + register tokens
                    raw_tokens = raw_tokens[:, :, patch_start_idx_val:]
                    B, S, N_patches, D = raw_tokens.shape

        # Initialize variables to ensure they're defined in all code paths
        gaussian_embs = None
        tokens_flat = None  # Initialize to avoid UnboundLocalError

        # === Priority 1: Simplified per-frame spatial encoding ===
        if hasattr(self, 'use_enhanced_temporal') and self.use_enhanced_temporal:
            frame_tokens_list = []
            patch_h = int(np.sqrt(N_patches))
            patch_w = N_patches // patch_h if patch_h > 0 else 1
            if patch_h * patch_w != N_patches:
                raise ValueError(f"Expected square patch grid, got N_patches={N_patches}")

            for frame_idx in range(S):
                frame_tokens = raw_tokens[:, frame_idx, :, :]  # [B, 1369, D]

                # [B, 1369, D] -> [B, D, 37, 37]
                tokens_2d = frame_tokens.permute(0, 2, 1).reshape(B, D, patch_h, patch_w)

                # Pool each frame independently to 16x16 -> 256 tokens/frame
                tokens_pooled = F.adaptive_avg_pool2d(tokens_2d, (16, 16))  # [B, D, 16, 16]

                # [B, D, 16, 16] -> [B, 256, D]
                tokens_flat = tokens_pooled.reshape(B, D, 16 * 16).permute(0, 2, 1)

                # Project to action_expert_width
                frame_embs = self.proj(tokens_flat)  # [B, 256, action_expert_width]

                # Add frame positional encoding
                offset_emb = self._frame_offset_embedding(
                    frame_idx, frame_embs.shape[-1], frame_embs.device, frame_embs.dtype
                )
                frame_pos = self.frame_pos_encoding[frame_idx](frame_idx)
                frame_pos = frame_pos + self.frame_embeddings[frame_idx] + offset_emb
                frame_pos = frame_pos.unsqueeze(0).unsqueeze(0)  # [1, 1, D]

                frame_embs = frame_embs + frame_pos
                frame_tokens_list.append(frame_embs)

            # [B, 256, D] * S -> [B, S*256, D]
            gaussian_embs = torch.cat(frame_tokens_list, dim=1)

            if step is not None and step % 100 == 0:
                logging.info(
                    f"[Enhanced Temporal] Step {step}: "
                    f"tokens shape={gaussian_embs.shape}, "
                    f"frames={S}, tokens_per_frame={self.tokens_per_frame}"
                )

        # === Fallback: Original Per-frame Processing ===
        elif hasattr(self, 'use_frame_pos_encoding') and self.use_frame_pos_encoding:
            # Process each frame separately to maintain temporal structure
            frame_tokens_list = []
            for frame_idx in range(S):
                # Extract tokens for this frame: [B, N_patches, D]
                frame_tokens = raw_tokens[:, frame_idx, :, :]

                # Reshape for pooling: [B, N_patches, D] -> [B, D, sqrt(N), sqrt(N)]
                patch_h = int(np.sqrt(N_patches))
                patch_w = N_patches // patch_h if patch_h > 0 else 1

                # Reshape to 2D: [B, N_patches, D] -> [B, D, patch_h, patch_w]
                if patch_h * patch_w == N_patches:
                    tokens_2d = frame_tokens.permute(0, 2, 1).view(B, D, patch_h, patch_w)
                else:
                    # Fallback: reshape to approximate grid
                    spatial_size = int(np.sqrt(N_patches))
                    tokens_reshaped = frame_tokens.permute(0, 2, 1)  # [B, D, N_patches]
                    if spatial_size * spatial_size != N_patches:
                        spatial_size = int(np.ceil(np.sqrt(N_patches)))
                        padding = spatial_size * spatial_size - N_patches
                        tokens_reshaped = F.pad(tokens_reshaped, (0, padding), mode='constant', value=0)
                    tokens_2d = tokens_reshaped.view(B, D, spatial_size, spatial_size)

                # Pool to reduce tokens: [B, D, H, W] -> [B, D, 10, 10]
                tokens_pooled = self.pool(tokens_2d)  # [B, D, 10, 10]

                # Flatten: [B, D, 10, 10] -> [B, 100, D]
                tokens_final = tokens_pooled.view(B, -1, D)  # [B, 100, D]

                # Project to action_expert_width
                frame_embs = self.proj(tokens_final)  # [B, 100, action_expert_width]

                if hasattr(self, 'frame_embeddings') and self.frame_embeddings is not None:
                    frame_emb = self.frame_embeddings[frame_idx]
                    offset_emb = self._frame_offset_embedding(frame_idx, frame_emb.shape[-1], frame_emb.device, frame_emb.dtype)
                    frame_emb = frame_emb + offset_emb
                    frame_emb = frame_emb.unsqueeze(0).unsqueeze(0).expand(B, 100, -1)
                    frame_embs = frame_embs + frame_emb

                frame_tokens_list.append(frame_embs)

            # Concatenate frames: [B, 100, D] * S -> [B, S*100, D]
            # This preserves temporal order: tokens from frame 0, then frame 1, then frame 2
            gaussian_embs = torch.cat(frame_tokens_list, dim=1)  # [B, S*100, action_expert_width]

        else:
            # Option 2: Original approach (flatten all frames together)
            # This loses temporal structure but maintains backward compatibility
            try:
                tokens_flat = raw_tokens.view(B, S * N_patches, D)  # [B, S*N_patches, D]
            except Exception as e:
                logging.error(f"Failed to reshape raw_tokens: {e}, raw_tokens.shape={raw_tokens.shape}, expected shape=[{B}, {S}, {N_patches}, {D}]")
                raise
            
            # Reshape for pooling: [B, S*N_patches, D] -> [B, D, sqrt(N), sqrt(N)]
            spatial_size = int(np.sqrt(S * N_patches))
            if spatial_size * spatial_size != S * N_patches:
                spatial_size = int(np.ceil(np.sqrt(S * N_patches)))
                padding = spatial_size * spatial_size - S * N_patches
                tokens_reshaped = tokens_flat.permute(0, 2, 1)  # [B, D, S*N_patches]
                tokens_reshaped = F.pad(tokens_reshaped, (0, padding), mode='constant', value=0)
            else:
                tokens_reshaped = tokens_flat.permute(0, 2, 1)  # [B, D, S*N_patches]
            
            tokens_2d = tokens_reshaped.view(B, D, spatial_size, spatial_size)

            # Pool to reduce tokens
            # Single-frame mode: [B, D, spatial_size, spatial_size] -> [B, D, 16, 16] (256 tokens)
            # Multi-frame mode: [B, D, spatial_size, spatial_size] -> [B, D, 10, 10] (100 tokens)
            tokens_pooled = self.pool(tokens_2d)

            # Flatten back
            # Single-frame mode: [B, D, 16, 16] -> [B, 256, D]
            # Multi-frame mode: [B, D, 10, 10] -> [B, 100, D]
            # Use reshape instead of view to handle non-contiguous tensors
            tokens_final = tokens_pooled.reshape(B, D, -1).permute(0, 2, 1)  # [B, D, N] -> [B, N, D]

            # Project to action_expert_width
            # Single-frame mode: [B, 256, action_expert_width]
            # Multi-frame mode: [B, 100, action_expert_width]
            gaussian_embs = self.proj(tokens_final)
        
        # Create mask (all tokens are valid)
        # Single-frame mode: 256 tokens
        # Multi-frame mode with frame pos encoding: S*100 tokens
        # Multi-frame mode without frame pos encoding: 100 tokens
        g_mask = torch.ones(B, gaussian_embs.shape[1], dtype=torch.bool, device=gaussian_embs.device)
        
        # LGPD is disabled in this configuration; pass through tokens unchanged.
        
        if return_unprojected_raw_tokens:
            return gaussian_embs, g_mask, raw_tokens.detach().to(torch.float32)

        # Prepare return values
        if return_gaussian_params and return_raw_tokens and return_mta_features:
            raw_tokens_proj = None
            if hasattr(self, 'proj') and self.proj is not None:
                B, S, N, D = raw_tokens.shape
                raw_tokens_flat = raw_tokens.view(B * S, N, D)
                raw_tokens_proj_flat = self.proj(raw_tokens_flat)
                raw_tokens_proj = raw_tokens_proj_flat.view(B, S, N, -1)
            return gaussian_embs, g_mask, gaussian_params_dict, raw_tokens_proj, mta_features
        elif return_gaussian_params and return_raw_tokens:
            # Project raw_tokens to action_expert_width for VAE supervision
            # raw_tokens: [B, S, 1369, D] where D is VGGT embed_dim (2048)
            # VAE expects tokens in action_expert_width dimension
            raw_tokens_proj = None
            if hasattr(self, 'proj') and self.proj is not None:
                # Project each frame's tokens: [B, S, 1369, D] -> [B, S, 1369, action_expert_width]
                B, S, N, D = raw_tokens.shape
                raw_tokens_flat = raw_tokens.view(B * S, N, D)
                raw_tokens_proj_flat = self.proj(raw_tokens_flat)  # [B*S, 1369, action_expert_width]
                raw_tokens_proj = raw_tokens_proj_flat.view(B, S, N, -1)  # [B, S, 1369, action_expert_width]
            return gaussian_embs, g_mask, gaussian_params_dict, raw_tokens_proj
        elif return_gaussian_params and return_mta_features:
            return gaussian_embs, g_mask, gaussian_params_dict, mta_features
        elif return_gaussian_params:
            return gaussian_embs, g_mask, gaussian_params_dict
        elif return_raw_tokens and return_mta_features:
            raw_tokens_proj = None
            if hasattr(self, 'proj') and self.proj is not None:
                B, S, N, D = raw_tokens.shape
                raw_tokens_flat = raw_tokens.view(B * S, N, D)
                raw_tokens_proj_flat = self.proj(raw_tokens_flat)
                raw_tokens_proj = raw_tokens_proj_flat.view(B, S, N, -1)
            return gaussian_embs, g_mask, raw_tokens_proj, mta_features
        elif return_raw_tokens:
            # Project raw_tokens to action_expert_width for VAE supervision
            raw_tokens_proj = None
            if hasattr(self, 'proj') and self.proj is not None:
                B, S, N, D = raw_tokens.shape
                raw_tokens_flat = raw_tokens.view(B * S, N, D)
                raw_tokens_proj_flat = self.proj(raw_tokens_flat)
                raw_tokens_proj = raw_tokens_proj_flat.view(B, S, N, -1)
            return gaussian_embs, g_mask, raw_tokens_proj
        elif return_mta_features:
            return gaussian_embs, g_mask, mta_features
        else:
            return gaussian_embs, g_mask
