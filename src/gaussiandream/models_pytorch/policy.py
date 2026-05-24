import logging
import math
from types import SimpleNamespace

import numpy as np
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import gaussiandream.models.gemma as _gemma
from gaussiandream.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import gaussiandream.models_pytorch.preprocessing_pytorch as _preprocessing
from gaussiandream.models_pytorch.encoder import GaussianFeatureEncoder
from gaussiandream.models_pytorch.decoder import GaussianDecoder
# Import Gaussian Renderer
from gaussiandream.models_pytorch.gaussian_renderer import (
    GaussianRenderer,
    build_orbit_camera_params,
    build_sweep_camera_params,
    build_projected_velocity_map,
    compute_multi_view_rendering_loss,
    visualize_rendering_comparison,
)


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def _resolve_future_prediction_offsets(config, default_horizon: int) -> list[int]:
    """Resolve future rollout offsets in frame steps for logging and supervision labels."""
    raw_offsets = getattr(config, "future_prediction_offsets", None)
    if raw_offsets:
        return [max(1, int(value)) for value in raw_offsets]
    return list(range(1, max(1, int(default_horizon)) + 1))




def _resolve_temporal_context_offsets(config, use_single_frame_mode: bool) -> list[int]:
    """Resolve temporal context offsets in frame steps for context packing and labels."""
    if use_single_frame_mode:
        return [0]
    raw_offsets = getattr(config, "temporal_context_offsets", None)
    if raw_offsets:
        offsets = [int(value) for value in raw_offsets]
    else:
        offsets = [-2, -1, 0]
    if not offsets:
        offsets = [0]
    if offsets[-1] != 0:
        raise ValueError(f"temporal_context_offsets must end with 0, got {tuple(offsets)}")
    return offsets


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)

def sample_beta(alpha: float, beta: float, bsize: int, device) -> Tensor:
    """Sample Beta-distributed timesteps."""
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class MotionAwareTemporalBlock(nn.Module):
    """Mix tokens within each frame, then model each token slot over time."""

    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.1, max_temporal_frames: int = 8):
        super().__init__()
        self.max_temporal_frames = max_temporal_frames
        self.token_norm = nn.LayerNorm(embed_dim)
        self.token_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.temporal_bias = nn.Parameter(torch.zeros(max_temporal_frames, max_temporal_frames))
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply frame-wise joint self-attention followed by per-slot temporal self-attention.

        Args:
            x: [B, T, K, D]
        Returns:
            [B, T, K, D]
        """
        if x.ndim != 4:
            raise ValueError(f"Expected MTA tokens with shape [B, T, K, D], got {tuple(x.shape)}")
        bsz, num_frames, num_slots, dim = x.shape
        if num_frames > self.max_temporal_frames:
            raise ValueError(
                f"MotionAwareTemporalBlock supports up to {self.max_temporal_frames} temporal frames, got {num_frames}"
            )

        token_input = self.token_norm(x).reshape(bsz * num_frames, num_slots, dim)
        token_out, _ = self.token_attn(token_input, token_input, token_input, need_weights=False)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(
                "MTA token mixing shapes: input=%s output=%s",
                tuple(token_input.shape),
                tuple(token_out.shape),
            )
        token_out = token_out.reshape(bsz, num_frames, num_slots, dim)
        x = x + token_out

        temporal_input = self.norm1(x).permute(0, 2, 1, 3).reshape(bsz * num_slots, num_frames, dim)
        temporal_bias = self.temporal_bias[:num_frames, :num_frames].to(
            device=temporal_input.device,
            dtype=temporal_input.dtype,
        )
        temporal_out, _ = self.attn(
            temporal_input,
            temporal_input,
            temporal_input,
            attn_mask=temporal_bias,
            need_weights=False,
        )
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(
                "MTA temporal attention shapes: input=%s output=%s",
                tuple(temporal_input.shape),
                tuple(temporal_out.shape),
            )
        temporal_out = temporal_out.reshape(bsz, num_slots, num_frames, dim).permute(0, 2, 1, 3)
        x = x + temporal_out
        x = x + self.mlp(self.norm2(x))
        return x


class MotionAwareTemporalEncoder(nn.Module):
    """Stacked MTA encoder following paper equations (5)-(9)."""

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        max_temporal_frames: int = 8,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            MotionAwareTemporalBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                max_temporal_frames=max_temporal_frames,
            )
            for _ in range(num_layers)
        ])

    def forward(self, motion_tokens: Tensor, layer_features: list[Tensor]) -> Tensor:
        """Run layer-wise MTA and return updated motion tokens from the final layer.

        Args:
            motion_tokens: [B, T, M, D]
            layer_features: three tensors of shape [B, T, N, D] corresponding to layers 11/17/23
        Returns:
            final motion tokens: [B, T, M, D]
        """
        if motion_tokens.ndim != 4:
            raise ValueError(f"Expected motion tokens with shape [B, T, M, D], got {tuple(motion_tokens.shape)}")
        if len(layer_features) != len(self.blocks):
            raise ValueError(f"Expected {len(self.blocks)} layer features, got {len(layer_features)}")

        prev_patch = None
        current_motion = motion_tokens
        motion_dim = motion_tokens.shape[2]
        for block_idx, (block, patch_tokens) in enumerate(zip(self.blocks, layer_features, strict=True)):
            if patch_tokens.ndim != 4:
                raise ValueError(f"Expected patch tokens with shape [B, T, N, D], got {tuple(patch_tokens.shape)}")
            if patch_tokens.shape[:2] != current_motion.shape[:2] or patch_tokens.shape[-1] != current_motion.shape[-1]:
                raise ValueError(
                    "MTA motion/patch shape mismatch: "
                    f"motion={tuple(current_motion.shape)}, patch={tuple(patch_tokens.shape)}"
                )
            if prev_patch is not None:
                if prev_patch.shape != patch_tokens.shape:
                    raise ValueError(
                        "MTA adjacent patch feature shape mismatch: "
                        f"prev={tuple(prev_patch.shape)}, current={tuple(patch_tokens.shape)}"
                    )
                patch_input = patch_tokens + prev_patch
            else:
                patch_input = patch_tokens

            block_input = torch.cat([current_motion, patch_input], dim=2)
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(
                    "MTA block %d input shapes: motion=%s patch=%s concat=%s",
                    block_idx,
                    tuple(current_motion.shape),
                    tuple(patch_input.shape),
                    tuple(block_input.shape),
                )
            block_output = block(block_input)
            current_motion = block_output[:, :, :motion_dim, :]
            prev_patch = block_output[:, :, motion_dim:, :]
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(
                    "MTA block %d output shapes: block=%s motion=%s patch=%s",
                    block_idx,
                    tuple(block_output.shape),
                    tuple(current_motion.shape),
                    tuple(prev_patch.shape),
                )

        if current_motion.shape != motion_tokens.shape:
            raise RuntimeError(
                f"MTA output shape changed unexpectedly: expected {tuple(motion_tokens.shape)}, got {tuple(current_motion.shape)}"
            )
        return current_motion


class PolicyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )
        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        # torch.compile disabled due to graph break issues with VGGT

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False
        self.state_norm_stats = getattr(config, "state_norm_stats", None)
        self.state_use_quantile_norm = bool(getattr(config, "state_use_quantile_norm", False))

        # Can be set via VIS_SAVE_DIR environment variable, or defaults to ./visualizations/rendering
        import os
        self.vis_save_dir = os.environ.get("VIS_SAVE_DIR", "./visualizations/rendering_independent_decoder_test0501_lpips_future_5_frame")

        # --- 3D Gaussian Integration ---
        use_gaussian = getattr(config, "use_gaussian", False)
        use_single_frame_mode = getattr(config, "use_single_frame_mode", False)
        self.temporal_context_offsets = _resolve_temporal_context_offsets(config, use_single_frame_mode)
        self.temporal_context_count = len(self.temporal_context_offsets)
        # Typically Gaussian features join the prefix, so they must match the VLM width
        # Disable LGPD for now to test training stability
        # Option to unfreeze VGGT encoder/decoder for reconstruction loss training
        unfreeze_vggt_encoder = getattr(config, "unfreeze_vggt_encoder", False)
        unfreeze_vggt_decoder_only = getattr(config, "unfreeze_vggt_decoder_only", True)  # Default: train decoder only
        use_lora = getattr(config, "use_lora", False)  # Default: keep VGGT fully frozen
        # Match training frame count at inference so MTA layers stay [B,T,...] with T==temporal_context_count.
        _infer_frames = 1 if use_single_frame_mode else self.temporal_context_count
        self.gaussian_adapter = GaussianFeatureEncoder(
            use_gaussian,
            paligemma_config.width,
            # LGPD support has been removed; GaussianFeatureEncoder only handles
            # Gaussian + MTA features now.
            num_frames=self.temporal_context_count,
            inference_num_frames=_infer_frames,
            use_single_frame_mode=use_single_frame_mode,
            temporal_context_offsets=tuple(self.temporal_context_offsets),
            unfreeze_encoder=unfreeze_vggt_encoder,
            unfreeze_decoder_only=unfreeze_vggt_decoder_only,
            use_lora=use_lora,
        )

        self.action_warmup_steps = int(getattr(config, "action_warmup_steps", 0))
        self._stage2_world_loss_multiplier = float(getattr(config, "stage2_world_loss_multiplier", 0.0))
        self._stage2_keep_world_model_trainable = bool(getattr(config, "stage2_keep_world_model_trainable", False))
        self._stage4_freeze_world_model = bool(getattr(config, "stage4_freeze_world_model", False))
        self._stage4_disable_world_model_losses = bool(getattr(config, "stage4_disable_world_model_losses", False))
        self._active_training_stage: int | None = None

        # Current frame reconstruction loss weight
        self.current_frame_recon_loss_weight = getattr(config, "current_frame_recon_loss_weight", 0.5)
        self._base_current_frame_recon_loss_weight = float(self.current_frame_recon_loss_weight)
        self._base_render_loss_weight = float(getattr(config, "render_loss_weight", 0.1))
        self._base_depth_loss_weight = float(getattr(config, "depth_loss_weight", 0.1))
        self._base_future_depth_aux_loss_weight = float(getattr(config, "future_depth_aux_loss_weight", 0.0))
        self._base_flow_loss_weight = float(getattr(config, "flow_loss_weight", 0.0))
        self._world_loss_enabled = True
        self._world_supervision_enabled = True
        self._world_loss_multiplier = 1.0

        # --- World Model Tokens in Prefix (NEW Architecture) ---
        # Add future query tokens to prefix for unified VLM processing
        self.use_world_tokens_in_prefix = getattr(config, "use_world_model", False) and use_gaussian
        self.future_prediction_offsets = _resolve_future_prediction_offsets(
            config, 5 if self.use_world_tokens_in_prefix else 1
        )
        self.future_prediction_horizon = len(self.future_prediction_offsets)
        self.use_velocity_future_gaussians = False
        self.use_shared_motion_query_velocity = False
        self.world_model = None
        self.gaussian_renderer = None
        self.lpips_fn = None
        self.lpips_weight = getattr(config, "lpips_weight", 0.1)
        if self.use_world_tokens_in_prefix:
            # Keep current/static Gaussian template tokens at 16×16, but raise
            # future motion-query tokens to 32×32 for finer spatial dynamics.
            self.static_gaussian_token_count = 256
            self.future_token_count = 1024
            self.future_grid_size = 32

            # World tokens removed — redundant with Gaussian tokens (768)
            self.world_token_count = 0
            self.world_token_proj = None

            # Future query tokens: learnable embeddings predicted by VLM
            # These represent "delta queries" for predicting changes from current frame
            self.future_query_tokens = nn.Parameter(
                torch.randn(1, self.future_token_count, paligemma_config.width) * 0.02
            )

            # === Future Motion Query Tokens ===
            # 1024 future query tokens (32×32) for higher-resolution motion rollouts.
            self.future_spatial_pos = nn.Parameter(
                torch.randn(self.future_grid_size, self.future_grid_size, paligemma_config.width) * 0.02
            )

            # Optional: Sinusoidal spatial encoding (more stable)
            self.use_sinusoidal_spatial = True
            if self.use_sinusoidal_spatial:
                self.register_buffer(
                    'future_spatial_sinusoidal',
                    self._create_2d_sinusoidal_encoding(
                        self.future_grid_size,
                        self.future_grid_size,
                        paligemma_config.width
                    )
                )

            self.future_query_to_mta = nn.Linear(paligemma_config.width, 512)
            self.future_mta_layer_keys = tuple(getattr(self.gaussian_adapter, "mta_layer_keys", (11, 17, 23)))
            self.future_mta_encoder = MotionAwareTemporalEncoder(
                embed_dim=512,
                num_heads=8,
                num_layers=len(self.future_mta_layer_keys),
                max_temporal_frames=max(1, self.temporal_context_count),
            )
            self.mta_to_future = nn.Linear(512, paligemma_config.width)

            _vel_g = bool(getattr(config, "use_velocity_future_gaussians", False))
            self.use_velocity_future_gaussians = _vel_g
            self.use_shared_motion_query_velocity = _vel_g
            self.use_future_motion_gate = bool(getattr(config, "use_future_motion_gate", False))

            self.world_model = GaussianDecoder(
                token_dim=paligemma_config.width,
                input_num_tokens=self.static_gaussian_token_count,
                future_input_num_tokens=self.future_token_count,
                use_incremental_depth=getattr(config, "use_incremental_depth", True),
                future_prediction_horizon=self.future_prediction_horizon,
                use_velocity_future_gaussians=_vel_g,
                velocity_world_model_scale=float(getattr(config, "velocity_world_model_scale", 2.0)),
                num_motion_slots=int(getattr(config, "num_motion_slots", 8)),
                slot_assignment_temperature=float(getattr(config, "slot_assignment_temperature", 1.0)),
                slot_translation_scale=float(
                    getattr(config, "slot_translation_scale", getattr(config, "velocity_world_model_scale", 2.0))
                ),
                slot_rotation_scale=float(getattr(config, "slot_rotation_scale", 1.0)),
                use_future_depth_aux=bool(getattr(config, "use_future_depth_aux", False)),
                future_depth_aux_downsample=int(getattr(config, "future_depth_aux_downsample", 2)),
                use_future_motion_gate=self.use_future_motion_gate,
            )
            logging.info(
                "Future velocity motion gate %s",
                "enabled" if self.use_future_motion_gate else "disabled",
            )

            # Initialize Gaussian Renderer (sh_degree=1 for DC + 1st order SH)
            try:
                self.gaussian_renderer = GaussianRenderer(image_size=224, sh_degree=1, scale_factor=1.0)
                logging.info("Gaussian Renderer initialized with sh_degree=1 (DC + 1st order) for World Model supervision.")
            except ImportError:
                self.gaussian_renderer = None
                logging.warning("Gaussian Renderer not available. Skipping rendering loss.")

            # Initialize LPIPS perceptual loss (optional)
            if getattr(config, "use_lpips", False):
                try:
                    import lpips
                    # Initialize LPIPS and move to the same device as the model
                    self.lpips_fn = lpips.LPIPS(net='vgg')
                    # Set to eval mode and freeze parameters
                    self.lpips_fn.eval()
                    for param in self.lpips_fn.parameters():
                        param.requires_grad = False
                    logging.info(f"LPIPS perceptual loss initialized with weight={self.lpips_weight}")
                except ImportError:
                    logging.warning("lpips package not installed. Install with: pip install lpips")
        else:
            self.static_gaussian_token_count = 0
            self.future_token_count = 0
            self.future_grid_size = 0
            self.world_token_count = 0
            self.world_token_proj = None
            self.future_query_tokens = None
            self.future_query_to_mta = None
            self.future_mta_encoder = None
            self.mta_to_future = None
            self.future_mta_layer_keys = ()
            self.future_spatial_pos = None
            self.use_sinusoidal_spatial = False

        # Initialize render loss weight (can be changed dynamically for staged training)
        self.render_loss_weight = self._base_render_loss_weight  # 降低render loss权重，让action loss主导
        self.depth_loss_weight = self._base_depth_loss_weight
        self.use_future_depth_aux = bool(getattr(config, "use_future_depth_aux", False))
        self.future_depth_aux_loss_weight = self._base_future_depth_aux_loss_weight
        self.future_depth_aux_downsample = max(1, int(getattr(config, "future_depth_aux_downsample", 2)))
        self.flow_loss_weight = self._base_flow_loss_weight
        self.flow_loss_type = str(getattr(config, "flow_loss_type", "smooth_l1")).lower()
        self.flow_first_horizon_only = bool(getattr(config, "flow_first_horizon_only", True))
        raw_flow_horizon_weights = getattr(config, "flow_horizon_weights", None)
        if raw_flow_horizon_weights:
            self.flow_horizon_weights = [float(weight) for weight in raw_flow_horizon_weights]
        else:
            self.flow_horizon_weights = None
        raw_flow_channel_weights = getattr(config, "flow_loss_channel_weights", (1.0, 1.0, 1.0))
        self.flow_loss_channel_weights = tuple(float(weight) for weight in raw_flow_channel_weights)
        self.future_delta_reg_weight = getattr(config, "future_delta_reg_weight", 1e-4)
        self.slot_entropy_loss_weight = float(getattr(config, "slot_entropy_loss_weight", 0.0))
        self.slot_balance_loss_weight = float(getattr(config, "slot_balance_loss_weight", 0.0))
        self.slot_transform_reg_weight = float(getattr(config, "slot_transform_reg_weight", 0.0))
        self.future_motion_loss_gain = float(getattr(config, "future_motion_loss_gain", 2.0))
        self.future_motion_rgb_threshold = float(getattr(config, "future_motion_rgb_threshold", 0.03))
        self.future_motion_depth_threshold = float(getattr(config, "future_motion_depth_threshold", 0.01))
        self.future_motion_depth_weight = float(getattr(config, "future_motion_depth_weight", 0.5))
        self.future_motion_loss_min_weight = float(getattr(config, "future_motion_loss_min_weight", 0.35))
        self.future_motion_loss_max_weight = float(getattr(config, "future_motion_loss_max_weight", 4.0))
        self.future_motion_blur_kernel = max(1, int(getattr(config, "future_motion_blur_kernel", 9)))
        if self.future_motion_blur_kernel % 2 == 0:
            self.future_motion_blur_kernel += 1
        self.future_horizon_curriculum_steps = int(getattr(config, "future_horizon_curriculum_steps", 0))
        self.future_horizon_early_min_weight = float(getattr(config, "future_horizon_early_min_weight", 0.2))
        self._action_loss_enabled = True  # Can be toggled by freeze/unfreeze_action_expert


        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/gaussiandream/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PolicyModel model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PolicyModel model")

    def freeze_world_model(self):
        """Freeze world model (Gaussian Decoder) for stage 1 training (action-only)."""
        if self.world_model is not None:
            for param in self.world_model.parameters():
                param.requires_grad = False
            logging.info("Froze World Model (Gaussian Decoder) - Stage 1: Action-only training")

        if hasattr(self, 'gaussian_adapter') and self.gaussian_adapter is not None:
            for param in self.gaussian_adapter.parameters():
                param.requires_grad = False
            logging.info("Froze Gaussian Adapter")

    def unfreeze_world_model(self):
        """Unfreeze world model for stage 2 training (joint training)."""
        if self.world_model is not None:
            for param in self.world_model.parameters():
                param.requires_grad = True
            logging.info("Unfroze World Model (Gaussian Decoder) - Stage 2: Joint training")

        if hasattr(self, 'gaussian_adapter') and self.gaussian_adapter is not None:
            for param in self.gaussian_adapter.parameters():
                param.requires_grad = True
            logging.info("Unfroze Gaussian Adapter")

    def freeze_world_model_decoder(self):
        """Freeze the full Gaussian decoder without touching the Gaussian adapter."""
        if self.world_model is None:
            return
        for param in self.world_model.parameters():
            param.requires_grad = False
        logging.info("Froze full world-model decoder")

    def unfreeze_world_model_decoder(self):
        """Unfreeze the full Gaussian decoder before applying per-head stage settings."""
        if self.world_model is None:
            return
        for param in self.world_model.parameters():
            param.requires_grad = True
        logging.info("Unfroze full world-model decoder")

    def _set_module_requires_grad(self, module: nn.Module | None, enabled: bool) -> bool:
        """Enable or disable gradients for a module if it exists."""
        if module is None:
            return False
        for param in module.parameters():
            param.requires_grad = enabled
        return True

    def freeze_static_head(self):
        """Freeze the world-model static head."""
        if self.world_model is None:
            return
        if self._set_module_requires_grad(getattr(self.world_model, "static_head", None), False):
            logging.info("Froze world-model static_head")

    def unfreeze_static_head(self):
        """Unfreeze the world-model static head."""
        if self.world_model is None:
            return
        if self._set_module_requires_grad(getattr(self.world_model, "static_head", None), True):
            logging.info("Unfroze world-model static_head")

    def freeze_velocity_head(self):
        """Freeze the world-model velocity head."""
        if self.world_model is None:
            return
        if self._set_module_requires_grad(getattr(self.world_model, "velocity_head", None), False):
            logging.info("Froze world-model velocity_head")

    def unfreeze_velocity_head(self):
        """Unfreeze the world-model velocity head."""
        if self.world_model is None:
            return
        if self._set_module_requires_grad(getattr(self.world_model, "velocity_head", None), True):
            logging.info("Unfroze world-model velocity_head")

    def freeze_shared_backbone(self):
        """Freeze the world-model shared backbone."""
        if self.world_model is None:
            return
        if self._set_module_requires_grad(getattr(self.world_model, "shared_backbone", None), False):
            logging.info("Froze world-model shared_backbone")

    def unfreeze_shared_backbone(self):
        """Unfreeze the world-model shared backbone."""
        if self.world_model is None:
            return
        if self._set_module_requires_grad(getattr(self.world_model, "shared_backbone", None), True):
            logging.info("Unfroze world-model shared_backbone")

    def set_world_model_image_fusion(self, enabled: bool):
        if self.world_model is None:
            return
        setter = getattr(self.world_model, "set_image_fusion_enabled", None)
        if setter is not None:
            setter(enabled)
            logging.info("Set world-model image fusion to %s", "enabled" if enabled else "disabled")

    def set_world_supervision_enabled(self, enabled: bool, *, render_weight: float):
        """Enable or disable world-model supervision losses for the current stage."""
        self._world_supervision_enabled = bool(enabled)
        self._world_loss_enabled = bool(enabled)
        self._world_loss_multiplier = 1.0 if enabled else 0.0
        if enabled:
            self.render_loss_weight = float(render_weight)
            self.depth_loss_weight = float(self._base_depth_loss_weight)
            self.future_depth_aux_loss_weight = float(self._base_future_depth_aux_loss_weight)
            self.flow_loss_weight = float(self._base_flow_loss_weight)
            self.current_frame_recon_loss_weight = float(self._base_current_frame_recon_loss_weight)
        else:
            self.render_loss_weight = 0.0
            self.depth_loss_weight = 0.0
            self.future_depth_aux_loss_weight = 0.0
            self.flow_loss_weight = 0.0
            self.current_frame_recon_loss_weight = 0.0
        logging.info(
            "Set world supervision to %s | render=%.4f depth=%.4f future_depth=%.4f flow=%.4f current_recon=%.4f",
            "enabled" if enabled else "disabled",
            self.render_loss_weight,
            self.depth_loss_weight,
            self.future_depth_aux_loss_weight,
            self.flow_loss_weight,
            self.current_frame_recon_loss_weight,
        )

    def freeze_prefix_vlm_backbone(self):
        """Freeze the full PaliGemma prefix backbone while keeping custom world-token modules trainable."""
        self._set_module_requires_grad(self.paligemma_with_expert.paligemma, False)
        logging.info("Froze full prefix VLM backbone (entire PaliGemma module)")

    def unfreeze_prefix_vlm_backbone(self):
        """Unfreeze the full PaliGemma prefix backbone."""
        self._set_module_requires_grad(self.paligemma_with_expert.paligemma, True)
        logging.info("Unfroze full prefix VLM backbone (entire PaliGemma module)")

    def apply_world_model_stage(
        self,
        stage: int,
        render_weight: float,
        *,
        freeze_velocity_head: bool = True,
        freeze_static_head: bool = True,
    ):
        """Apply the simplified 2-stage trainability schedule for action/world-model branches."""
        if stage == 1:
            # Stage 1: representation/world-model only, image fusion always on, action off.
            self.set_world_loss_enabled(True, render_weight=render_weight)
            self.freeze_action_expert()
            self.unfreeze_prefix_vlm_backbone()
            self.set_world_model_image_fusion(True)
            self.unfreeze_shared_backbone()
            self.unfreeze_static_head()
            self.unfreeze_velocity_head()
        elif stage == 2:
            # Stage 2: action-focused fine-tuning. Keep image fusion on and optionally
            # retain weak world losses so spatial structure does not collapse.
            self.set_world_loss_enabled(
                self._stage2_world_loss_multiplier > 0.0,
                render_weight=render_weight,
                loss_multiplier=self._stage2_world_loss_multiplier,
            )
            self.unfreeze_prefix_vlm_backbone()
            self.set_world_model_image_fusion(True)
            self.unfreeze_action_expert()
            if self._stage2_keep_world_model_trainable:
                self.unfreeze_shared_backbone()
                self.unfreeze_static_head()
                self.unfreeze_velocity_head()
            elif self._stage4_freeze_world_model:
                self.freeze_world_model_decoder()
            else:
                self.freeze_shared_backbone()
                self.freeze_static_head()
                self.freeze_velocity_head()
            if self._stage4_disable_world_model_losses:
                self.set_world_supervision_enabled(False, render_weight=0.0)
        else:
            raise ValueError(f"Unsupported stage: {stage}")


    def get_stage_trainability_summary(self) -> dict[str, bool]:
        """Return whether key training groups currently require gradients."""
        def _module_trainable(module: nn.Module | None) -> bool:
            return bool(module is not None and any(param.requires_grad for param in module.parameters()))

        return {
            "prefix_paligemma": _module_trainable(self.paligemma_with_expert.paligemma),
            "prefix_language_model": _module_trainable(
                getattr(self.paligemma_with_expert.paligemma, "language_model", None)
            ),
            "prefix_vision_tower": _module_trainable(
                getattr(self.paligemma_with_expert.paligemma, "vision_tower", None)
            ),
            "prefix_multimodal_projector": _module_trainable(
                getattr(self.paligemma_with_expert.paligemma, "multi_modal_projector", None)
            ),
            "action_expert": _module_trainable(self.paligemma_with_expert.gemma_expert),
            "shared_backbone": _module_trainable(getattr(self.world_model, "shared_backbone", None)),
            "static_head": _module_trainable(getattr(self.world_model, "static_head", None)),
            "velocity_head": _module_trainable(getattr(self.world_model, "velocity_head", None)),
        }

    def _get_action_mlp_modules(self):
        """Get the time/state MLP modules based on pi05 mode."""
        if self.pi05:
            return [self.time_mlp_in, self.time_mlp_out]
        else:
            return [self.state_proj, self.action_time_mlp_in, self.action_time_mlp_out]

    def freeze_action_expert(self):
        """Freeze action expert and related projections during world-only training."""
        for param in self.paligemma_with_expert.gemma_expert.parameters():
            param.requires_grad = False
        for module in [self.action_in_proj, self.action_out_proj] + self._get_action_mlp_modules():
            for param in module.parameters():
                param.requires_grad = False
        self._action_loss_enabled = False
        logging.info("Froze Action Expert + projections, disabled action loss")

    def unfreeze_action_expert(self):
        """Unfreeze action expert for joint world+action training."""
        for param in self.paligemma_with_expert.gemma_expert.parameters():
            param.requires_grad = True
        for module in [self.action_in_proj, self.action_out_proj] + self._get_action_mlp_modules():
            for param in module.parameters():
                param.requires_grad = True
        self._action_loss_enabled = True
        logging.info("Unfroze Action Expert + projections, enabled action loss")

    def set_world_loss_enabled(
        self,
        enabled: bool,
        render_weight: float | None = None,
        loss_multiplier: float | None = None,
    ):
        """Enable or disable world-model supervision while preserving configured base weights."""
        self._world_loss_enabled = bool(enabled)
        self._world_supervision_enabled = bool(enabled)
        if enabled:
            multiplier = 1.0 if loss_multiplier is None else max(0.0, float(loss_multiplier))
        else:
            multiplier = 0.0
        self._world_loss_multiplier = multiplier
        target_render_weight = self._base_render_loss_weight if render_weight is None else float(render_weight)
        self.render_loss_weight = target_render_weight * multiplier
        self.depth_loss_weight = self._base_depth_loss_weight * multiplier
        self.future_depth_aux_loss_weight = self._base_future_depth_aux_loss_weight * multiplier
        self.flow_loss_weight = self._base_flow_loss_weight * multiplier
        self.current_frame_recon_loss_weight = self._base_current_frame_recon_loss_weight * multiplier
        logging.info(
            "Set world-model supervision %s (multiplier=%.3f) | current=%.4f render=%.4f depth=%.4f flow=%.4f future_depth_aux=%.4f",
            "enabled" if enabled else "disabled",
            multiplier,
            self.current_frame_recon_loss_weight,
            self.render_loss_weight,
            self.depth_loss_weight,
            self.flow_loss_weight,
            self.future_depth_aux_loss_weight,
        )

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if getattr(self, "gradient_checkpointing_enabled", False) and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _get_training_stage(self, step: int | None) -> int:
        """Map training steps onto the simplified 2-stage PyTorch schedule."""
        if step is None:
            return 2
        stage1_steps = int(getattr(self.config, "stage1_steps", 0))
        if stage1_steps > 0 and step < stage1_steps:
            return 1
        return 2

    def _apply_training_stage_if_needed(self, step: int | None) -> int:
        stage = self._get_training_stage(step)
        if self._active_training_stage == stage:
            return stage
        render_weight = float(getattr(self.config, f"stage{stage}_render_weight", self.render_loss_weight))
        self.apply_world_model_stage(
            stage,
            render_weight,
            freeze_velocity_head=bool(getattr(self.config, "stage1_freeze_velocity_head", True)),
            freeze_static_head=bool(getattr(self.config, "stage2_freeze_static_head", True)),
        )
        self._active_training_stage = stage
        logging.info("Applied training stage %s with render_weight=%s", stage, render_weight)
        return stage

    def _create_2d_sinusoidal_encoding(self, height, width, embed_dim):
        """Create 2D sinusoidal positional encoding for spatial grid.

        Args:
            height: Grid height (e.g., 16)
            width: Grid width (e.g., 16)
            embed_dim: Embedding dimension

        Returns:
            [height, width, embed_dim] - 2D positional encoding
        """
        # Create position indices
        y_pos = torch.arange(height).unsqueeze(1).float()  # [H, 1]
        x_pos = torch.arange(width).unsqueeze(0).float()   # [1, W]

        # Expand to grid
        y_grid = y_pos.expand(height, width)  # [H, W]
        x_grid = x_pos.expand(height, width)  # [H, W]

        # Frequency bands
        half_dim = embed_dim // 4  # Split embed_dim into 4 parts (y_sin, y_cos, x_sin, x_cos)
        div_term = torch.exp(torch.arange(0, half_dim).float() * -(math.log(10000.0) / half_dim))

        # Compute sinusoidal encoding for y and x
        pe = torch.zeros(height, width, embed_dim)

        # Y position encoding (first half of embed_dim)
        pe[:, :, 0:half_dim] = torch.sin(y_grid.unsqueeze(-1) * div_term)
        pe[:, :, half_dim:2*half_dim] = torch.cos(y_grid.unsqueeze(-1) * div_term)

        # X position encoding (second half of embed_dim)
        pe[:, :, 2*half_dim:3*half_dim] = torch.sin(x_grid.unsqueeze(-1) * div_term)
        pe[:, :, 3*half_dim:4*half_dim] = torch.cos(x_grid.unsqueeze(-1) * div_term)

        return pe


    def _temporal_context_labels(self) -> list[str]:
        labels: list[str] = []
        for offset in self.temporal_context_offsets:
            if offset == 0:
                labels.append("t")
            elif offset > 0:
                labels.append(f"t+{offset}")
            else:
                labels.append(f"t{offset}")
        return labels

    def _temporal_context_future_indices(self, time_dim: int, use_single_frame_mode: bool) -> tuple[list[int], list[int], int]:
        """Pick context / future frame indices from packed temporal slots.

        Expected packed layout:
          - single-frame mode: [t, future...]
          - multi-frame mode: [context..., future...]
            where context is configured by temporal_context_offsets and ends at offset 0.
        """
        context_count = 1 if use_single_frame_mode else self.temporal_context_count
        future_count = len(self.future_prediction_offsets) if self.use_world_tokens_in_prefix else 0

        def _legacy_consecutive() -> tuple[list[int], list[int], int]:
            context_frames = 1 if use_single_frame_mode else min(context_count, time_dim)
            future_steps = max(0, min(self.future_prediction_horizon, time_dim - context_frames))
            context_start = max(0, time_dim - future_steps - context_frames)
            ctx = list(range(context_start, context_start + context_frames))
            fut = list(range(context_start + context_frames, context_start + context_frames + future_steps))
            return ctx, fut, ctx[-1]

        if not self.use_world_tokens_in_prefix:
            return _legacy_consecutive()

        expected_t = context_count + future_count
        if time_dim >= expected_t:
            context_indices = list(range(context_count))
            future_indices = list(range(context_count, expected_t))
            return context_indices, future_indices, context_indices[-1]

        if not getattr(self, "_warned_future_offset_fallback", False):
            context_desc = ", ".join(self._temporal_context_labels()[:context_count])
            future_desc = ", ".join(f"t+{offset}" for offset in self.future_prediction_offsets)
            logging.warning(
                "World model: packed temporal slots expect time_dim >= %s for [%s, %s], got %s. "
                "Using consecutive tail fallback.",
                expected_t,
                context_desc,
                future_desc,
                time_dim,
            )
            self._warned_future_offset_fallback = True
        return _legacy_consecutive()

    def _prepare_attention_masks_4d(self, att_2d_masks: torch.Tensor) -> torch.Tensor:
        """Convert boolean pairwise attention mask [B, N, N] to additive 4D mask [B, 1, N, N]."""
        if att_2d_masks.ndim != 3:
            raise ValueError(f"Expected att_2d_masks with shape [B, N, N], got {att_2d_masks.shape}")
        min_dtype = torch.finfo(torch.float32).min
        additive_mask = torch.zeros_like(att_2d_masks, dtype=torch.float32)
        additive_mask = additive_mask.masked_fill(~att_2d_masks, min_dtype)
        return additive_mask[:, None, :, :]

    def _get_current_frame_index(self, image_tensor: torch.Tensor) -> int:
        if image_tensor.ndim == 5:
            return max(0, image_tensor.shape[1] - 1)
        return 0

    def _preprocess_observation(self, observation, train: bool = True):
        """Helper method to preprocess observation."""
        use_single_frame_mode = getattr(self.config, "use_single_frame_mode", False)

        # --- Handle Future Split for 3DGS World Model ---
        future_observation = None
        motion_weight_observation = None
        # Check if first image has T dimension (ndim=5 for B,T,H,W,C now due to model.py fix)
        if observation.images:
            # Inspect first image to detect time dimension
            img_val = next(iter(observation.images.values()))
            # DEBUG PRINT
            if train and torch.rand(1).item() < 0.01:
                 print(f"DEBUG: _preprocess_observation. img_val ndim={img_val.ndim}, shape={img_val.shape}, use_single_frame_mode={use_single_frame_mode}")

            if img_val.ndim == 5:
                # [B, T, H, W, C]
                time_dim = img_val.shape[1]
                context_indices, future_indices, idx_curr = self._temporal_context_future_indices(
                    time_dim, use_single_frame_mode
                )

                # DEBUG PRINT
                if train and torch.rand(1).item() < 0.01:
                     print(
                         f"DEBUG: Found Time Dim {time_dim}. "
                         f"context_indices={context_indices}, future_indices={future_indices}, "
                         f"offsets={self.future_prediction_offsets if self.use_world_tokens_in_prefix else 'n/a'}"
                     )

                if future_indices:
                    curr_imgs, fut_imgs = {}, {}
                    # Save original temporal images for visualization before slicing
                    raw_temporal_images = {}
                    if use_single_frame_mode:
                        raw_temporal_labels = ["t"] + [f"t+{offset}" for offset in self.future_prediction_offsets]
                    else:
                        raw_temporal_labels = self._temporal_context_labels() + [
                            f"t+{offset}" for offset in self.future_prediction_offsets
                        ]

                    def _select_time_slices(tensor, indices, *, collapse_single=False):
                        if tensor is None or tensor.ndim < 2:
                            return tensor
                        selected = tensor[:, indices]
                        if collapse_single and len(indices) == 1:
                            return selected[:, 0]
                        return selected

                    # Handle single-frame mode vs multi-frame mode
                    for k, v in observation.images.items():
                        raw_ix = sorted(set(context_indices + future_indices))
                        raw_temporal_images[k] = _select_time_slices(v, raw_ix)
                        curr_imgs[k] = _select_time_slices(v, context_indices, collapse_single=use_single_frame_mode)
                        fut_imgs[k] = _select_time_slices(v, future_indices, collapse_single=len(future_indices) == 1)

                    curr_state = observation.state
                    fut_state = observation.state

                    # Handle state [B, T, D]
                    if observation.state is not None:
                        if observation.state.ndim == 3:
                            # State has temporal dimension [B, T, D]
                            if observation.state.shape[1] == time_dim:
                                curr_state = _select_time_slices(
                                    observation.state,
                                    context_indices,
                                    collapse_single=use_single_frame_mode,
                                )
                                fut_state = _select_time_slices(
                                    observation.state, future_indices, collapse_single=len(future_indices) == 1
                                )
                        elif observation.state.ndim == 2:
                            curr_state = observation.state
                            if len(future_indices) == 1:
                                fut_state = observation.state
                            else:
                                fut_state = observation.state.unsqueeze(1).expand(-1, len(future_indices), -1)

                    # Clone observation for future
                    # Note: We must also slice the masks and prompts to match the single-step batch dimension,
                    # otherwise jaxtyping will complain about mismatched *b dimensions (e.g. mask [B, T] vs image [B, H, W, C])

                    # 1. Slice Image Masks
                    fut_masks = {}
                    curr_masks = {}
                    for k, v in observation.image_masks.items():
                        if v.ndim == 2: # [B, T]
                            curr_masks[k] = _select_time_slices(v, context_indices, collapse_single=use_single_frame_mode)
                            fut_masks[k] = _select_time_slices(v, future_indices, collapse_single=len(future_indices) == 1)
                        else: # [B] - assume valid for all steps
                            curr_masks[k] = v
                            fut_masks[k] = v

                    # 2. Slice Prompts
                    curr_prompt = observation.tokenized_prompt
                    fut_prompt = observation.tokenized_prompt
                    if curr_prompt is not None:
                        if curr_prompt.ndim == 3: # [B, T, L]
                            curr_prompt = _select_time_slices(
                                curr_prompt, context_indices, collapse_single=use_single_frame_mode
                            )
                            fut_prompt = _select_time_slices(
                                observation.tokenized_prompt,
                                future_indices,
                                collapse_single=len(future_indices) == 1,
                            )
                        elif curr_prompt.ndim == 2: # [B, L]
                            if not use_single_frame_mode:
                                curr_prompt = curr_prompt.unsqueeze(1).expand(-1, len(context_indices), -1)
                            if len(future_indices) > 1:
                                fut_prompt = observation.tokenized_prompt.unsqueeze(1).expand(-1, len(future_indices), -1)

                    curr_prompt_mask = observation.tokenized_prompt_mask
                    fut_prompt_mask = observation.tokenized_prompt_mask
                    if curr_prompt_mask is not None:
                        if curr_prompt_mask.ndim == 3: # [B, T, L]
                            curr_prompt_mask = _select_time_slices(
                                curr_prompt_mask, context_indices, collapse_single=use_single_frame_mode
                            )
                            fut_prompt_mask = _select_time_slices(
                                observation.tokenized_prompt_mask,
                                future_indices,
                                collapse_single=len(future_indices) == 1,
                            )
                        elif curr_prompt_mask.ndim == 2: # [B, L]
                            if not use_single_frame_mode:
                                curr_prompt_mask = curr_prompt_mask.unsqueeze(1).expand(-1, len(context_indices), -1)
                            if len(future_indices) > 1:
                                fut_prompt_mask = observation.tokenized_prompt_mask.unsqueeze(1).expand(
                                    -1, len(future_indices), -1
                                )

                    # Handle depth data if available
                    fut_depth = None
                    curr_depth = None
                    if hasattr(observation, 'depth') and observation.depth is not None:
                        # observation.depth: [B, T, 1, H, W] or [B, 1, H, W]
                        if observation.depth.ndim == 5:  # [B, T, 1, H, W]
                            if observation.depth.shape[1] == time_dim:
                                curr_depth = _select_time_slices(
                                    observation.depth, context_indices, collapse_single=use_single_frame_mode
                                )
                                fut_depth = _select_time_slices(
                                    observation.depth, future_indices, collapse_single=len(future_indices) == 1
                                )
                        elif observation.depth.ndim == 4:  # [B, 1, H, W]
                            # Single frame depth, use as is for both
                            curr_depth = observation.depth
                            if len(future_indices) == 1:
                                fut_depth = observation.depth
                            else:
                                fut_depth = observation.depth.unsqueeze(1).expand(-1, len(future_indices), -1, -1, -1)
                        else:
                            logging.warning(f"Unexpected depth shape: {observation.depth.shape}, ndim={observation.depth.ndim}")

                    future_observation = observation.replace(
                        images=fut_imgs,
                        state=fut_state,
                        image_masks=fut_masks,
                        tokenized_prompt=fut_prompt,
                        tokenized_prompt_mask=fut_prompt_mask,
                        depth=fut_depth  # Set depth directly to avoid type check error
                    )
                    # Update current observation
                    observation = observation.replace(
                        images=curr_imgs,
                        state=curr_state,
                        image_masks=curr_masks,
                        tokenized_prompt=curr_prompt,
                        tokenized_prompt_mask=curr_prompt_mask,
                        depth=curr_depth  # Set depth for current frame
                    )

                    # Also preprocess future observation (normalization etc)
                    future_observation = _preprocessing.preprocess_observation_pytorch(future_observation, train=False)

                    # Motion weights should use the unaugmented current observation.
                    motion_weight_observation = _preprocessing.preprocess_observation_pytorch(observation, train=False)

        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        # Store the preprocessed observation for later use in World Model
        # This ensures that when _prepare_gaussian_inputs is called later, it uses the observation
        # with preserved temporal dimension, not the original one
        preprocessed_observation = observation
        if motion_weight_observation is not None:
            preprocessed_observation.motion_weight_observation = motion_weight_observation

        if 'raw_temporal_images' in locals() and raw_temporal_images:
            preprocessed_observation.raw_temporal_images = raw_temporal_images
            preprocessed_observation.raw_temporal_labels = raw_temporal_labels

        future_motion_prior = self._build_temporal_motion_prior_maps(
            self._get_motion_weight_source_observation(preprocessed_observation)
        )
        if future_motion_prior:
            preprocessed_observation.future_motion_prior = future_motion_prior
            if future_observation is not None:
                future_observation.future_motion_prior = future_motion_prior

        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
            future_observation,
            preprocessed_observation  # Return preprocessed observation for World Model
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def _get_future_horizon_loss_weights(
        self, step: int | None, horizon: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Anneal future-rollout horizon weights from near-heavy to uniform.

        Stage 1 (step < stage1_steps): only supervise t+1 (first horizon),
        later horizons get zero weight. From Stage 2 onward, use curriculum.
        """
        if horizon <= 0:
            return torch.zeros(0, device=device, dtype=dtype)

        # If there's only one horizon, always use weight 1.0
        if horizon == 1:
            return torch.ones(1, device=device, dtype=dtype)

        # Optional stage-1 gating: only supervise t+1 before stage1_steps
        stage1_steps = getattr(self.config, "stage1_steps", 0)
        if step is not None and stage1_steps > 0 and step < stage1_steps:
            weights = torch.zeros(horizon, device=device, dtype=torch.float32)
            weights[0] = 1.0  # t+1
            return weights.to(dtype=dtype)

        # Default curriculum over horizons
        uniform = torch.ones(horizon, device=device, dtype=torch.float32)
        if step is None or self.future_horizon_curriculum_steps <= 0:
            return uniform.to(dtype=dtype)

        tail_weight = min(max(self.future_horizon_early_min_weight, 1e-3), 1.0)
        early = torch.linspace(1.0, tail_weight, horizon, device=device, dtype=torch.float32)
        early = early / early.mean().clamp_min(1e-6)

        progress = min(max(step or 0, 0), self.future_horizon_curriculum_steps) / float(
            self.future_horizon_curriculum_steps
        )
        weights = (1.0 - progress) * early + progress * uniform
        return weights.to(dtype=dtype)

    def _log_future_rollout_diagnostics(
        self,
        step: int | None,
        future_seed_tokens: torch.Tensor,
        z_future_pred_tokens: torch.Tensor,
        per_step_delta: torch.Tensor,
        horizon_weights: torch.Tensor,
    ) -> None:
        """Log future-token / rollout diagnostics on the visualization cadence."""
        if step is None or step % 400 != 0:
            return

        import torch.distributed as dist

        if dist.is_initialized() and dist.get_rank() != 0:
            return

        delta_fp32 = per_step_delta.float()
        future_fp32 = z_future_pred_tokens.float()
        delta_rms_by_h = delta_fp32.pow(2).mean(dim=(0, 2, 3)).sqrt()

        inter_horizon_l2 = torch.zeros(0, device=delta_fp32.device, dtype=torch.float32)
        t1_tH_l2 = torch.zeros((), device=delta_fp32.device, dtype=torch.float32)
        if delta_fp32.shape[1] > 1:
            inter_horizon_l2 = (delta_fp32[:, 1:] - delta_fp32[:, :-1]).pow(2).mean(dim=(0, 2, 3)).sqrt()
            t1_tH_l2 = (delta_fp32[:, -1] - delta_fp32[:, 0]).pow(2).mean().sqrt()

        seed_to_token_l2 = (future_fp32 - future_seed_tokens.float()).pow(2).mean().sqrt()
        weights_str = ", ".join(f"{value:.3f}" for value in horizon_weights.detach().cpu().tolist())
        delta_rms_str = ", ".join(f"{value:.6f}" for value in delta_rms_by_h.detach().cpu().tolist())
        inter_l2_str = ", ".join(f"{value:.6f}" for value in inter_horizon_l2.detach().cpu().tolist()) or "N/A"
        logging.info(
            f"Step {step}: Future Rollout Diagnostics | "
            f"motion_token_abs_mean={future_fp32.abs().mean().item():.6f}, "
            f"motion_token_rms={future_fp32.pow(2).mean().sqrt().item():.6f}, "
            f"scaled_delta_abs_mean={delta_fp32.abs().mean().item():.6f}, "
            f"scaled_delta_rms={delta_fp32.pow(2).mean().sqrt().item():.6f}, "
            f"delta_rms_by_h=[{delta_rms_str}], "
            f"inter_horizon_l2=[{inter_l2_str}], "
            f"t1_tH_l2={t1_tH_l2.item():.6f}, "
            f"seed_to_token_l2={seed_to_token_l2.item():.6f}, "
            f"horizon_weights=[{weights_str}]"
        )

    def _log_future_rollout_pixel_lpips(self, step: int | None, rendered_obs_seq: list[dict[str, torch.Tensor]]) -> None:
        """Log pixel-space LPIPS between the first and last rendered future predictions."""
        if step is None or step % 400 != 0 or len(rendered_obs_seq) < 2:
            return

        import torch.distributed as dist

        if dist.is_initialized() and dist.get_rank() != 0:
            return

        first_render = rendered_obs_seq[0]
        last_render = rendered_obs_seq[-1]
        shared_keys = list(set(first_render.keys()) & set(last_render.keys()))
        if not shared_keys:
            return

        view_key = "agent_image" if "agent_image" in shared_keys else shared_keys[0]
        if self.lpips_fn is None:
            logging.info(
                f"Step {step}: Future Rollout Pixel LPIPS(t1,t{len(rendered_obs_seq)})[{view_key}] unavailable (LPIPS disabled)"
            )
            return

        pred_t1 = torch.clamp(first_render[view_key], 0.0, 1.0)
        pred_tH = torch.clamp(last_render[view_key], 0.0, 1.0)
        with torch.no_grad():
            lpips_value = self.lpips_fn(pred_t1 * 2.0 - 1.0, pred_tH * 2.0 - 1.0).mean()
        logging.info(
            f"Step {step}: Future Rollout Pixel LPIPS(t1,t{len(rendered_obs_seq)})[{view_key}] = {lpips_value.item():.6f}"
        )

    def _get_temporal_observation_length(self, observation) -> int:
        """Infer the temporal length stored in a future observation sequence."""
        if observation is None:
            return 0

        if hasattr(observation, "images"):
            for value in observation.images.values():
                if value.ndim == 5:
                    return value.shape[1]
        if hasattr(observation, "depth") and observation.depth is not None and observation.depth.ndim == 5:
            return observation.depth.shape[1]
        if hasattr(observation, "state") and observation.state is not None and observation.state.ndim == 3:
            return observation.state.shape[1]
        return 0

    def _slice_temporal_observation(self, observation, index: int):
        """Extract one horizon from a temporally-stacked observation container."""
        if observation is None:
            return None

        images = {}
        for key, value in observation.images.items():
            if value.ndim == 5:
                images[key] = value[:, index]
            else:
                images[key] = value

        image_masks = {}
        for key, value in observation.image_masks.items():
            if value.ndim == 2:
                image_masks[key] = value[:, index]
            else:
                image_masks[key] = value

        state = observation.state[:, index] if observation.state is not None and observation.state.ndim == 3 else observation.state
        tokenized_prompt = (
            observation.tokenized_prompt[:, index]
            if observation.tokenized_prompt is not None and observation.tokenized_prompt.ndim == 3
            else observation.tokenized_prompt
        )
        tokenized_prompt_mask = (
            observation.tokenized_prompt_mask[:, index]
            if observation.tokenized_prompt_mask is not None and observation.tokenized_prompt_mask.ndim == 3
            else observation.tokenized_prompt_mask
        )
        depth = observation.depth[:, index] if getattr(observation, "depth", None) is not None and observation.depth.ndim == 5 else getattr(observation, "depth", None)
        flow_3d = (
            observation.flow_3d[:, index]
            if getattr(observation, "flow_3d", None) is not None and observation.flow_3d.ndim == 5
            else getattr(observation, "flow_3d", None)
        )
        flow_valid_mask = (
            observation.flow_valid_mask[:, index]
            if getattr(observation, "flow_valid_mask", None) is not None and observation.flow_valid_mask.ndim == 4
            else getattr(observation, "flow_valid_mask", None)
        )

        sliced_observation = SimpleNamespace(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
            token_ar_mask=getattr(observation, "token_ar_mask", None),
            token_loss_mask=getattr(observation, "token_loss_mask", None),
            depth=depth,
            flow_3d=flow_3d,
            flow_valid_mask=flow_valid_mask,
        )
        motion_weight_observation = getattr(observation, "motion_weight_observation", None)
        if motion_weight_observation is not None:
            sliced_observation.motion_weight_observation = self._slice_temporal_observation(
                motion_weight_observation, index
            )
        return sliced_observation

    def _get_motion_weight_source_observation(self, observation):
        """Return the observation variant that should be used to build motion weights."""
        if observation is None:
            return None
        return getattr(observation, "motion_weight_observation", observation)



    def _extract_reference_view_image(self, observation, view_name: str) -> torch.Tensor | None:
        """Extract a normalized [B,3,H,W] image for a named view from an observation container."""
        if observation is None or not hasattr(observation, "images"):
            return None

        for key, value in observation.images.items():
            key_lower = key.lower()
            mapped_view = None
            if key == "image" or "agent" in key_lower or "high" in key_lower or "cam_high" in key_lower or "exterior" in key_lower or "base" in key_lower:
                mapped_view = "agent"
            elif "left_wrist" in key_lower or "wrist_left" in key_lower:
                mapped_view = "wrist"
            elif "right_wrist" in key_lower or "wrist_right" in key_lower:
                if value.min() == value.max() == -1.0:
                    continue
                mapped_view = "wrist"
            elif "wrist" in key_lower or "bravo" in key_lower:
                mapped_view = "wrist"

            if mapped_view != view_name:
                continue

            img_tensor = value
            if img_tensor.ndim == 5:
                img_tensor = img_tensor[:, -1]
            if img_tensor.shape[1] != 3 and img_tensor.shape[-1] == 3:
                img_tensor = img_tensor.permute(0, 3, 1, 2)
            return (img_tensor + 1.0) / 2.0
        return None

    def _extract_temporal_view_images(self, observation, view_name: str) -> torch.Tensor | None:
        """Extract a normalized temporal image sequence [B,T,3,H,W] for a named view."""
        if observation is None or not hasattr(observation, "images"):
            return None

        for key, value in observation.images.items():
            key_lower = key.lower()
            mapped_view = None
            if key == "image" or "agent" in key_lower or "high" in key_lower or "cam_high" in key_lower or "exterior" in key_lower or "base" in key_lower:
                mapped_view = "agent"
            elif "left_wrist" in key_lower or "wrist_left" in key_lower:
                mapped_view = "wrist"
            elif "right_wrist" in key_lower or "wrist_right" in key_lower:
                if value.min() == value.max() == -1.0:
                    continue
                mapped_view = "wrist"
            elif "wrist" in key_lower or "bravo" in key_lower:
                mapped_view = "wrist"

            if mapped_view != view_name:
                continue

            img_tensor = value
            if img_tensor.ndim == 4:
                img_tensor = img_tensor[:, None]
            if img_tensor.ndim != 5:
                return None
            if img_tensor.shape[2] != 3 and img_tensor.shape[-1] == 3:
                img_tensor = img_tensor.permute(0, 1, 4, 2, 3)
            return (img_tensor + 1.0) / 2.0
        return None

    def _build_temporal_motion_prior_maps(self, observation) -> dict[str, torch.Tensor]:
        """Build soft motion priors from past/current context only, avoiding future-target leakage."""
        motion_prior_maps: dict[str, torch.Tensor] = {}
        if observation is None:
            return motion_prior_maps

        depth_sequence = getattr(observation, "depth", None)
        if depth_sequence is not None and depth_sequence.ndim == 4:
            depth_sequence = depth_sequence[:, None]

        for view_name in ("agent", "wrist"):
            image_sequence = self._extract_temporal_view_images(observation, view_name)
            if image_sequence is None or image_sequence.ndim != 5 or image_sequence.shape[1] < 2:
                continue

            current_image = image_sequence[:, -1]
            current_depth = depth_sequence[:, -1] if view_name == "agent" and depth_sequence is not None and depth_sequence.ndim == 5 else None
            aggregated_prior = None
            for frame_idx in range(image_sequence.shape[1] - 1):
                prior_map = self._build_future_motion_weight_map(
                    image_sequence[:, frame_idx],
                    current_image,
                    depth_sequence[:, frame_idx] if current_depth is not None else None,
                    current_depth,
                )
                aggregated_prior = prior_map if aggregated_prior is None else torch.maximum(aggregated_prior, prior_map)

            if aggregated_prior is None:
                continue

            denom = max(1e-6, self.future_motion_loss_max_weight - self.future_motion_loss_min_weight)
            normalized_prior = (aggregated_prior - self.future_motion_loss_min_weight) / denom
            motion_prior_maps[f"{view_name}_image"] = torch.clamp(normalized_prior, 0.0, 1.0)

        return motion_prior_maps

    def _build_future_motion_weight_map(
        self,
        current_image: torch.Tensor | None,
        future_image: torch.Tensor,
        current_depth: torch.Tensor | None,
        future_depth: torch.Tensor | None,
    ) -> torch.Tensor:
        """Create a soft motion weight map that emphasizes regions changing over time."""
        device = future_image.device
        bsize, _, height, width = future_image.shape
        motion_rgb = torch.zeros((bsize, 1, height, width), device=device, dtype=torch.float32)
        motion_depth = torch.zeros_like(motion_rgb)

        if current_image is not None:
            if current_image.shape[-2:] != future_image.shape[-2:]:
                current_image = F.interpolate(
                    current_image, size=future_image.shape[-2:], mode="bilinear", align_corners=False
                )
            motion_rgb = (future_image - current_image).abs().mean(dim=1, keepdim=True).float()
            motion_rgb = torch.relu(motion_rgb - self.future_motion_rgb_threshold)

        if current_depth is not None and future_depth is not None:
            current_depth_map = current_depth.float()
            future_depth_map = future_depth.float()
            if current_depth_map.ndim == 3:
                current_depth_map = current_depth_map.unsqueeze(1)
            if future_depth_map.ndim == 3:
                future_depth_map = future_depth_map.unsqueeze(1)
            if current_depth_map.shape[-2:] != future_image.shape[-2:]:
                current_depth_map = F.interpolate(
                    current_depth_map, size=future_image.shape[-2:], mode="bilinear", align_corners=False
                )
            if future_depth_map.shape[-2:] != future_image.shape[-2:]:
                future_depth_map = F.interpolate(
                    future_depth_map, size=future_image.shape[-2:], mode="bilinear", align_corners=False
                )
            motion_depth = (future_depth_map - current_depth_map).abs()
            motion_depth = torch.relu(motion_depth - self.future_motion_depth_threshold)

        motion = motion_rgb + self.future_motion_depth_weight * motion_depth
        if self.future_motion_blur_kernel > 1:
            pad = self.future_motion_blur_kernel // 2
            motion = F.avg_pool2d(motion, kernel_size=self.future_motion_blur_kernel, stride=1, padding=pad)

        motion_mean = motion.mean(dim=(-2, -1), keepdim=True)
        motion = motion / (motion_mean + 1e-6)
        weight_map = 1.0 + self.future_motion_loss_gain * motion
        weight_map = torch.clamp(
            weight_map,
            min=self.future_motion_loss_min_weight,
            max=self.future_motion_loss_max_weight,
        )
        return weight_map.squeeze(1)

    def _compute_gaussian_supervision_loss(
        self,
        gaussian_params: dict,
        target_observation,
        reference_observation,
        *,
        step: int | None = None,
        time_suffix: str = "",
        visualize: bool = True,
        horizon_idx: int = 0,
        return_gaussian_params: bool = False,
        enable_depth_supervision: bool = True,
        enable_render_supervision: bool = True,
    ):
        """Compute depth / render supervision for already-decoded Gaussian parameters."""
        device = next(
            (value.device for value in gaussian_params.values() if torch.is_tensor(value)),
            None,
        )
        if device is None:
            raise ValueError("gaussian_params must contain at least one tensor")

        total_loss = torch.zeros((), dtype=torch.float32, device=device)

        import torch.distributed as dist

        is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        gaussian_params = dict(gaussian_params)

        motion_weight_reference = self._get_motion_weight_source_observation(reference_observation)
        motion_weight_base_depth = getattr(motion_weight_reference, "depth", None)
        if motion_weight_base_depth is not None and motion_weight_base_depth.ndim == 5:
            motion_weight_base_depth = motion_weight_base_depth[:, -1]

        depth_map = gaussian_params.pop("depth_map", None)
        depth_delta_map = gaussian_params.pop("depth_delta_map", None)
        raw_delta_xyz = gaussian_params.pop("raw_delta_xyz", None)
        slot_trans_reg = gaussian_params.pop("slot_trans_reg", None)
        slot_usage = gaussian_params.pop("slot_usage", None)
        gaussian_params.pop("slot_probs", None)
        gaussian_params.pop("slot_logits", None)
        gaussian_params.pop("slot_trans", None)
        gaussian_params.pop("slot_rot_6d", None)
        gaussian_params.pop("slot_pivots", None)

        if step is not None and step % 400 == 0:
            raw_delta_abs_mean = float("nan")
            if raw_delta_xyz is not None:
                raw_delta_abs_mean = raw_delta_xyz.float().abs().mean().item()
            logging.info(
                f"Step {step}{time_suffix}: depth_map={depth_map is not None}, "
                f"raw_delta_abs_mean={raw_delta_abs_mean:.6f}, "
                f"has_depth_attr={hasattr(target_observation, 'depth')}, "
                f"depth_value={target_observation.depth is not None if hasattr(target_observation, 'depth') else 'N/A'}"
            )
            if slot_usage is not None:
                slot_usage_str = ", ".join(f"{value:.4f}" for value in slot_usage.float().mean(dim=0).detach().cpu().tolist())
                logging.info(
                    f"Step {step}{time_suffix}: Slot Motion Diagnostics | "
                    f"trans_reg={slot_trans_reg.item() if slot_trans_reg is not None else float('nan'):.6f}, "
                    f"usage=[{slot_usage_str}]"
                )

        if enable_depth_supervision and depth_map is not None and hasattr(target_observation, "depth") and target_observation.depth is not None:
            gt_depth = target_observation.depth
            if depth_map.shape != gt_depth.shape:
                depth_map = F.interpolate(
                    depth_map, size=gt_depth.shape[-2:], mode="bilinear", align_corners=False
                )
            depth_loss = F.l1_loss(depth_map, gt_depth)
            if torch.isfinite(depth_loss):
                total_loss = total_loss + self.depth_loss_weight * depth_loss
            if step is not None and step % 400 == 0:
                logging.info(
                    f"Step {step}{time_suffix}: Depth Loss = {depth_loss.item():.6f}, "
                    f"weight={self.depth_loss_weight}"
                )

        if self.flow_loss_weight > 0.0:
            flow_loss = self._compute_flow_supervision_loss(
                raw_delta_xyz,
                target_observation,
                step=step,
                time_suffix=time_suffix,
                horizon_idx=horizon_idx,
            )
            if torch.isfinite(flow_loss):
                total_loss = total_loss + self.flow_loss_weight * flow_loss

        if self.slot_transform_reg_weight > 0.0:
            if slot_trans_reg is not None and torch.isfinite(slot_trans_reg):
                total_loss = total_loss + self.slot_transform_reg_weight * slot_trans_reg.to(total_loss.dtype)

        for key, value in gaussian_params.items():
            if torch.is_tensor(value) and (torch.isnan(value).any() or torch.isinf(value).any()):
                gaussian_params[key] = torch.where(torch.isfinite(value), value, torch.zeros_like(value))

        target_obs = {}
        cam_params_dict = {}
        valid_views = []

        for key, value in target_observation.images.items():
            img_tensor = value
            key_lower = key.lower()
            view_name = None
            if key == "image" or "agent" in key_lower or "high" in key_lower or "cam_high" in key_lower or "exterior" in key_lower or "base" in key_lower:
                view_name = "agent"
            elif "left_wrist" in key_lower or "wrist_left" in key_lower:
                view_name = "wrist"
            elif "right_wrist" in key_lower or "wrist_right" in key_lower:
                if img_tensor.min() == img_tensor.max() == -1.0:
                    continue
                view_name = "wrist"
            elif "wrist" in key_lower or "bravo" in key_lower:
                view_name = "wrist"

            if view_name:
                if img_tensor.shape[1] != 3 and img_tensor.shape[-1] == 3:
                    img_tensor = img_tensor.permute(0, 3, 1, 2)
                img_tensor = (img_tensor + 1.0) / 2.0
                view_key = f"{view_name}_image"
                if view_key not in target_obs:
                    target_obs[view_key] = img_tensor
                    cam_params_dict[view_name] = self._get_camera_params_for_view(view_name, device, img_tensor.shape[0])
                    valid_views.append(view_name)

        if enable_render_supervision and valid_views:
            render_views = ["agent"] if "agent" in valid_views else valid_views
            motion_weight_maps = {}
            for render_view in render_views:
                view_key = f"{render_view}_image"
                current_image = self._extract_reference_view_image(motion_weight_reference, render_view)
                future_image = target_obs[view_key]
                current_depth_for_view = motion_weight_base_depth if render_view == "agent" else None
                future_depth_for_view = (
                    target_observation.depth
                    if render_view == "agent" and hasattr(target_observation, "depth")
                    else None
                )
                motion_weight_maps[render_view] = self._build_future_motion_weight_map(
                    current_image,
                    future_image,
                    current_depth_for_view,
                    future_depth_for_view,
                )

            render_loss, render_loss_dict = compute_multi_view_rendering_loss(
                gaussian_params,
                target_obs,
                cam_params_dict,
                self.gaussian_renderer,
                M_attn_dict=motion_weight_maps,
                view_names=render_views,
                step=step,
                depth_map=depth_map,
                lambda_scale=0.001,
                lambda_opacity=0.01,
                lambda_edge_smooth=0.01,
                lpips_fn=self.lpips_fn,
                lpips_weight=self.lpips_weight,
            )
            if torch.isfinite(render_loss):
                total_loss = total_loss + self.render_loss_weight * render_loss.to(total_loss.dtype)

            if "sh" in gaussian_params:
                sh_dc = gaussian_params["sh"]
                sh_dc_reg = (sh_dc ** 2).mean() * 0.01
                total_loss = total_loss + sh_dc_reg
                if step is not None and step % 400 == 0:
                    logging.info(
                        f"Step {step}{time_suffix}: SH DC Reg = {sh_dc_reg.item():.6f}, "
                        f"SH DC mean = {sh_dc.mean().item():.6f}"
                    )

            should_log = step is not None and step % 400 == 0
            if should_log:
                loss_parts = ", ".join(f"{k}={v.item():.6f}" for k, v in render_loss_dict.items())
                weight_parts = ", ".join(
                    f"{name}=mean:{weight.mean().item():.4f}/max:{weight.max().item():.4f}"
                    for name, weight in motion_weight_maps.items()
                )
                logging.info(
                    f"Step {step}{time_suffix}: Render Loss = {render_loss.item():.4f}, "
                    f"weight={self.render_loss_weight}, breakdown: {loss_parts}, motion_weights: {weight_parts}"
                )

            if visualize and should_log and is_main_process:
                with torch.no_grad():
                    try:
                        temporal_frames = {}
                        temporal_labels = getattr(reference_observation, "raw_temporal_labels", None)
                        if hasattr(reference_observation, "raw_temporal_images"):
                            temporal_frames = reference_observation.raw_temporal_images
                            for key, value in temporal_frames.items():
                                logging.info(f"[Viz] Extracted raw temporal {key} with shape {value.shape}")
                        elif hasattr(reference_observation, "images"):
                            for key, value in reference_observation.images.items():
                                if value.ndim == 5 and value.shape[1] >= 2:
                                    temporal_frames[key] = value
                                    logging.info(f"[Viz] Extracted {key} with shape {value.shape}")

                        future_label = "current" if "current" in time_suffix or "static" in time_suffix else "future"
                        if time_suffix.startswith("_tplus"):
                            future_label = time_suffix[len("_"):].replace("_pred_vlm", "")

                        self._visualize_rendering_comparison(
                            step,
                            gaussian_params,
                            target_obs,
                            cam_params_dict,
                            view_names=render_views,
                            time_suffix=time_suffix,
                            temporal_frames=temporal_frames,
                            temporal_labels=temporal_labels,
                            future_label=future_label,
                        )
                    except Exception as viz_error:
                        logging.warning(f"Step {step}{time_suffix}: Visualization failed: {viz_error}")

        if return_gaussian_params:
            if depth_map is not None:
                gaussian_params["depth_map"] = depth_map
            if depth_delta_map is not None:
                gaussian_params["depth_delta_map"] = depth_delta_map
            if raw_delta_xyz is not None:
                gaussian_params["raw_delta_xyz"] = raw_delta_xyz
            return total_loss, gaussian_params
        return total_loss

    def _build_world_decoder_state(
        self,
        z_tokens: torch.Tensor,
        *,
        step: int | None = None,
    ) -> dict | None:
        if self.world_model is None or z_tokens is None:
            return None
        decoder_state = self.world_model.prepare_decoder_state(
            z_tokens.float(),
            horizon_idx=0,
        )
        if step is not None and step % 400 == 0:
            shared_features = decoder_state.get("shared_features")
            if torch.is_tensor(shared_features):
                logging.info(
                    f"Step {step}: Shared Decoder State | "
                    f"shared_features.shape={tuple(shared_features.shape)}, "
                    f"canonical_tokens={tuple(decoder_state['tokens'].shape)}"
                )
        return decoder_state

    def _compute_current_frame_recon_loss(
        self,
        z_current: torch.Tensor,
        current_target,
        *,
        step: int | None = None,
        time_suffix: str = "_t_current_static",
        visualize: bool = False,
        static_gaussian_params: dict | None = None,
        return_gaussian_params: bool = False,
        decoder_state: dict | None = None,
    ):
        """Decode the current frame static template and compute reconstruction supervision."""
        device = z_current.device
        total_loss = torch.zeros((), dtype=torch.float32, device=device)
        if self.world_model is None or self.gaussian_renderer is None or current_target is None:
            return (total_loss, {}) if return_gaussian_params else total_loss

        gaussian_params: dict = {}
        try:
            if static_gaussian_params is not None:
                gaussian_params = dict(static_gaussian_params)
            else:
                camera_params_for_decode = self._get_camera_params_for_view("agent", device, z_current.shape[0])
                base_depth = getattr(current_target, "depth", None)
                if base_depth is not None and base_depth.ndim == 5:
                    base_depth = base_depth[:, -1]
                gaussian_params = self.world_model.decode_gaussian_prefix_template(
                    z_current.float(),
                    gaussian_adapter=self.gaussian_adapter,
                    current_observation=current_target,
                    camera_params=camera_params_for_decode,
                    base_depth=base_depth,
                    step=step,
                    shared_state=decoder_state,
                )

            return self._compute_gaussian_supervision_loss(
                gaussian_params,
                current_target,
                current_target,
                step=step,
                time_suffix=time_suffix,
                visualize=visualize,
                horizon_idx=0,
                return_gaussian_params=return_gaussian_params,
                enable_depth_supervision=True,
                enable_render_supervision=True,
            )
        except Exception as error:
            import traceback

            logging.warning(f"Step {step}{time_suffix}: Current-frame static reconstruction failed: {error}")
            if step is not None and step % 400 == 0:
                traceback.print_exc()
                logging.warning(traceback.format_exc())

        if return_gaussian_params:
            return total_loss, gaussian_params
        return total_loss

    def _compute_world_model_frame_loss(
        self,
        z_next: torch.Tensor,
        future_target,
        preprocessed_observation,
        *,
        step: int | None = None,
        time_suffix: str = "",
        visualize: bool = True,
        horizon_idx: int = 0,
        static_gaussian_params: dict | None = None,
        velocity_time_factor: float = 1.0,
        return_gaussian_params: bool = False,
        decoder_state: dict | None = None,
        return_aux: bool = False,
    ):
        """Decode one future horizon and compute depth/render supervision."""
        device = z_next.device
        total_loss = torch.zeros((), dtype=torch.float32, device=device)
        aux_outputs: dict[str, torch.Tensor] = {}
        if self.world_model is None or self.gaussian_renderer is None or future_target is None:
            if return_aux and return_gaussian_params:
                return total_loss, {}, aux_outputs
            if return_aux:
                return total_loss, aux_outputs
            return (total_loss, {}) if return_gaussian_params else total_loss

        gaussian_params: dict = {}
        try:
            camera_params_for_decode = self._get_camera_params_for_view("agent", device, z_next.shape[0])
            base_depth = getattr(preprocessed_observation, "depth", None)
            if base_depth is not None and base_depth.ndim == 5:
                base_depth = base_depth[:, -1]

            current_condition_observation = preprocessed_observation
            current_obs_steps = self._get_temporal_observation_length(preprocessed_observation)
            if current_obs_steps > 0:
                current_condition_observation = self._slice_temporal_observation(
                    preprocessed_observation, current_obs_steps - 1
                )

            gaussian_params = self.world_model.decode(
                z_next.float(),
                future_observation=future_target,
                gaussian_adapter=self.gaussian_adapter,
                camera_params=camera_params_for_decode,
                step=step,
                current_observation=preprocessed_observation,
                base_depth=base_depth,
                horizon_idx=horizon_idx,
                static_reference_params=static_gaussian_params,
                velocity_time_factor=velocity_time_factor,
                shared_state=decoder_state,
            )

            if self.use_future_depth_aux and decoder_state is not None:
                aux_future_depth = self.world_model.predict_future_depth_aux(decoder_state, horizon_idx=horizon_idx)
                if aux_future_depth is not None:
                    aux_outputs["future_depth_aux"] = aux_future_depth
                    gt_depth = getattr(future_target, "depth", None)
                    if gt_depth is not None:
                        gt_aux_depth = gt_depth
                        if gt_aux_depth.shape[-2:] != aux_future_depth.shape[-2:]:
                            gt_aux_depth = F.interpolate(
                                gt_aux_depth,
                                size=aux_future_depth.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            )
                        aux_depth_loss = F.l1_loss(aux_future_depth, gt_aux_depth)
                        if torch.isfinite(aux_depth_loss) and self.future_depth_aux_loss_weight > 0.0:
                            total_loss = total_loss + self.future_depth_aux_loss_weight * aux_depth_loss
                        if step is not None and step % 400 == 0:
                            logging.info(
                                f"Step {step}{time_suffix}: Future Depth Aux Loss = {aux_depth_loss.item():.6f}, "
                                f"weight={self.future_depth_aux_loss_weight}"
                            )

            supervision_result = self._compute_gaussian_supervision_loss(
                gaussian_params,
                future_target,
                preprocessed_observation,
                step=step,
                time_suffix=time_suffix,
                visualize=visualize,
                horizon_idx=horizon_idx,
                return_gaussian_params=return_gaussian_params,
                enable_depth_supervision=True,
                enable_render_supervision=True,
            )
            if return_gaussian_params:
                supervision_loss, returned_gaussian_params = supervision_result
                total_loss = total_loss + supervision_loss
                if return_aux:
                    return total_loss, returned_gaussian_params, aux_outputs
                return total_loss, returned_gaussian_params
            total_loss = total_loss + supervision_result
            if return_aux:
                return total_loss, aux_outputs
            return total_loss
        except Exception as error:
            import traceback

            logging.warning(f"Step {step}{time_suffix}: Decode/render failed: {error}")
            if step is not None and step % 400 == 0:
                traceback.print_exc()
                logging.warning(traceback.format_exc())

        if return_aux and return_gaussian_params:
            return total_loss, gaussian_params, aux_outputs
        if return_aux:
            return total_loss, aux_outputs
        if return_gaussian_params:
            return total_loss, gaussian_params
        return total_loss

    def _get_segment_slice(self, segment_info: dict | None, name: str) -> tuple[int, int] | None:
        if not segment_info:
            return None
        segment_ranges = segment_info.get("ranges", {})
        segment_range = segment_ranges.get(name)
        if segment_range is None:
            return None
        return int(segment_range[0]), int(segment_range[1])

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, gaussian_inputs=None,
        return_segment_lengths=False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Embed current RGB images and language tokens for prefix processing.

        When world modeling is enabled, future motion-query tokens are also appended.

        Returns:
            If return_segment_lengths=False: (embs, pad_masks, att_masks)
            If return_segment_lengths=True: (embs, pad_masks, att_masks, segment_info)
                where segment_info contains lengths/ranges for image segments,
                language, optional world, and future segments.
        """
        embs = []
        pad_masks = []
        att_masks = []
        segment_lengths = {}  # Track lengths of each segment for extracting future tokens later
        segment_ranges = {}

        def _append_segment(name: str, emb: torch.Tensor, mask: torch.Tensor, att_value: int = 0):
            start = sum(t.shape[1] for t in embs) if embs else 0
            embs.append(emb)
            pad_masks.append(mask)
            seg_len = emb.shape[1]
            att_masks.extend([att_value] * seg_len)
            segment_lengths[name] = seg_len
            segment_ranges[name] = (start, start + seg_len)

        # Process language tokens first to get text embedding for LGPD
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        # Handle temporal dimension in lang_emb if present
        # lang_emb might be [B, SeqLen, D] or [B, T, SeqLen, D]
        # For concatenation with other embeddings, we need [B, SeqLen, D] or [B, T*SeqLen, D]
        if lang_emb.ndim == 4:  # [B, T, SeqLen, D] - has temporal dimension
            B, T, S, D = lang_emb.shape
            lang_emb = lang_emb.reshape(B, T * S, D)  # [B, T*SeqLen, D]
            lang_masks = lang_masks.reshape(B, T * S)  # [B, T*SeqLen]

        text_embedding = None
        world_tokens = None
        world_mask = None

        mta_features = None
        if self.use_world_tokens_in_prefix:
            _, _, mta_features = self.gaussian_adapter(
                gaussian_inputs,
                text_embedding=text_embedding,
                return_mta_features=True,
            )

        # Process images
        total_img_tokens = 0
        for idx, (img, img_mask) in enumerate(zip(images, img_masks, strict=True)):
            has_temporal = img.ndim == 5
            current_frame_index = self._get_current_frame_index(img)
            if has_temporal:
                img_current = img[:, current_frame_index]
                if img_mask.ndim == 2:
                    img_mask_flat = img_mask[:, current_frame_index][:, None]
                else:
                    img_mask_flat = img_mask[:, None]

                def image_embed_func(img_current):
                    return self.paligemma_with_expert.embed_image(img_current)

                img_emb = self._apply_checkpoint(image_embed_func, img_current)
            else:
                def image_embed_func(img):
                    return self.paligemma_with_expert.embed_image(img)

                img_emb = self._apply_checkpoint(image_embed_func, img)
                img_mask_flat = img_mask[:, None]

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask_flat = img_mask_flat.expand(bsize, num_img_embs)
            _append_segment(f'image_{idx}', img_emb, img_mask_flat, att_value=0)
            total_img_tokens += num_img_embs
        if total_img_tokens > 0:
            first_image_start = min(segment_ranges[name][0] for name in segment_ranges if name.startswith('image_'))
            last_image_end = max(segment_ranges[name][1] for name in segment_ranges if name.startswith('image_'))
            segment_lengths['images'] = total_img_tokens
            segment_ranges['images'] = (first_image_start, last_image_end)

        # Append language tokens (already computed)
        _append_segment('language', lang_emb, lang_masks, att_value=0)

        # --- Add World Tokens (NEW) ---
        if world_tokens is not None:
            _append_segment('world', world_tokens, world_mask, att_value=0)

        if self.use_world_tokens_in_prefix and self.future_query_tokens is not None:
            B = pad_masks[0].shape[0] if pad_masks else 1
            device = pad_masks[0].device if pad_masks else next(self.parameters()).device
            future_dtype = self.future_query_tokens.dtype

            delta_q = self.future_query_tokens.expand(B, -1, -1)
            segment_lengths['debug_delta_q_shape'] = tuple(delta_q.shape)
            if mta_features is not None:
                segment_lengths['debug_mta_shapes'] = {
                    layer_idx: tuple(layer_feat.shape)
                    for layer_idx, layer_feat in mta_features.items()
                    if layer_feat is not None
                }
            else:
                segment_lengths['debug_mta_shapes'] = {}
            if (
                mta_features is not None
                and self.future_query_to_mta is not None
                and self.future_mta_encoder is not None
                and self.mta_to_future is not None
            ):
                layer_order = list(self.future_mta_layer_keys) if self.future_mta_layer_keys else sorted(mta_features.keys())
                layer_features = [mta_features.get(layer_idx) for layer_idx in layer_order]
                expected_frames = self.temporal_context_count
                has_all_layers = all(feature is not None for feature in layer_features)
                frames_match = has_all_layers and all(
                    feature.shape[1] == expected_frames for feature in layer_features
                )
                if not has_all_layers:
                    missing_layers = [
                        str(layer_idx)
                        for layer_idx, feature in zip(layer_order, layer_features, strict=True)
                        if feature is None
                    ]
                    raise RuntimeError(
                        "Missing MTA features for layers " + ", ".join(missing_layers)
                    )
                if not frames_match:
                    frame_mismatches = [
                        f"{layer_idx}:{feature.shape[1]}"
                        for layer_idx, feature in zip(layer_order, layer_features, strict=True)
                        if feature is not None and feature.shape[1] != expected_frames
                    ]
                    raise RuntimeError(
                        f"MTA feature frame count mismatch (expected {expected_frames}, got {', '.join(frame_mismatches)})"
                    )

                motion_tokens = self.future_query_to_mta(delta_q)
                motion_tokens = motion_tokens.unsqueeze(1).expand(-1, expected_frames, -1, -1)
                motion_tokens = self.future_mta_encoder(motion_tokens, layer_features)
                ta_current = motion_tokens[:, expected_frames - 1]
                future_tokens = self.mta_to_future(ta_current).to(future_dtype)
            else:
                raise RuntimeError("MTA-only future token path requires mta_features and MTA modules")

            spatial_pos = self.future_spatial_pos.reshape(1, self.future_token_count, -1)
            if hasattr(self, 'use_sinusoidal_spatial') and self.use_sinusoidal_spatial:
                sinusoidal_pos = self.future_spatial_sinusoidal.reshape(1, self.future_token_count, -1)
                spatial_pos = spatial_pos + sinusoidal_pos

            future_tokens = future_tokens + spatial_pos.to(future_tokens.dtype)
            future_mask = torch.ones(B, self.future_token_count, dtype=torch.bool, device=device)

            start = sum(t.shape[1] for t in embs) if embs else 0
            embs.append(future_tokens)
            pad_masks.append(future_mask)
            att_masks.extend([1] + [0] * (self.future_token_count - 1))
            segment_lengths['future'] = self.future_token_count
            segment_ranges['future'] = (start, start + self.future_token_count)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        if return_segment_lengths:
            return embs, pad_masks, att_masks, {"lengths": segment_lengths, "ranges": segment_ranges}
        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def _prepare_gaussian_inputs(
        self,
        observation,
        device,
        batch_size,
        is_training=None,
        preferred_camera_name: str | None = None,
        force_num_frames: int | None = None,
    ):
        """Helper to prepare inputs for 3DGS encoder from observation.

        Args:
            is_training: If True, uses training mode (more frames). If False, uses inference mode (fewer frames).
                        If None, auto-detects from model.training.
            preferred_camera_name: Optional exact camera key for VGGT encoding.
            force_num_frames: Optional explicit frame count override for cases like single-frame teacher alignment.
        """
        if is_training is None:
            is_training = self.training
        return self.gaussian_adapter.prepare_inputs(
            observation,
            device,
            batch_size,
            is_training=is_training,
            preferred_camera_name=preferred_camera_name,
            force_num_frames=force_num_frames,
        )

    def _get_camera_params_for_view(self, view_name, device, batch_size):
        """Get camera parameters for specific view (agent or wrist)."""
        # LIBERO camera intrinsics for 256×256 depth resolution
        # Must match depth2pc input size (256×256), NOT the image encoder size (224)
        W = 256.0
        H = 256.0
        fx = 221.7025
        fy = 221.7025
        cx = W / 2.0   # 128.0
        cy = H / 2.0   # 128.0
        fov_deg = 2.0 * math.degrees(math.atan(W / (2.0 * fx)))  # ~60° derived from fx
        tanfov = math.tan(0.5 * math.radians(fov_deg))
        
        # Helper for Projection Matrix (OpenGL style)
        def getProjectionMatrix(znear, zfar, fovX, fovY):
            tanHalfFovY = math.tan((fovY / 2))
            tanHalfFovX = math.tan((fovX / 2))
            
            P = torch.zeros(4, 4, device=device)
            z_sign = 1.0 

            P[0, 0] = 1 / tanHalfFovX
            P[1, 1] = 1 / tanHalfFovY
            P[3, 2] = z_sign
            P[2, 2] = z_sign * zfar / (zfar - znear)
            P[2, 3] = -(zfar * znear) / (zfar - znear)
            return P

        # Intrinsics (same for both views)
        intrinsics = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        if view_name == "agent":
            # Agent camera: identity viewmatrix (no translation)
            # depth2pc already outputs points in camera space (z = depth > 0),
            # so no Z-axis translation is needed. Adding translation would
            # push Gaussians further away and cause uniform gray rendering.
            viewmatrix = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

        elif view_name == "wrist":
            # Wrist camera: also identity (same reasoning as agent)
            viewmatrix = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

        else:
            # Default: identity
            viewmatrix = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

        # Create proper Projection Matrix
        # Note: rasterizer usually expects P @ V, i.e. Full MVP for "projmatrix" argument 
        # or just P if it does V * pos separately? 
        # The diff-gaussian-rasterization cuda code does: p_hom = projmatrix * p_orig
        # So "projmatrix" passed to settings MUST BE the full View+Projection matrix (MVP).
        
        proj_base = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=math.radians(fov_deg), fovY=math.radians(fov_deg))
        proj_base = proj_base.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # MVP = P @ V
        # Torch matmul is (..., N, M) x (..., M, P) -> (..., N, P)
        # We need standard multiplication order P * V
        projmatrix = torch.bmm(proj_base, viewmatrix)

        # Calculate camera position from viewmatrix
        # viewmatrix transforms world to camera: xyz_cam = xyz_world @ R^T + T
        # So viewmatrix = [R^T | T; 0 0 0 1]
        # Inverse: viewmatrix_inv = [R | -R^T @ T; 0 0 0 1]
        # Camera position in world: campos = -R^T @ T = viewmatrix_inv[:3, 3]
        viewmatrix_inv = torch.inverse(viewmatrix)  # [B, 4, 4]
        campos = viewmatrix_inv[:, :3, 3]  # [B, 3] - camera position in world coordinates

        # LIBERO canonical_agentview camera pose (for action coordinate transformation)
        camera_pos = [0.5386131746834771, 0.0, 0.7903500240372423]
        camera_quat = [0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]  # [w, x, y, z]

        return {
            "viewmatrix": viewmatrix,
            "projmatrix": projmatrix,
            "tanfovx": tanfov,
            "tanfovy": tanfov,
            "campos": campos,  # Now correctly computed from viewmatrix
            "intrinsics": intrinsics,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "camera_pos": camera_pos,
            "camera_quat": camera_quat,
        }
    

    def _compute_2d_maps_loss(self, pred_2d_maps, gt_2d_maps):
        """
        Compute loss between predicted and GT 2D maps from VGGT decoder.
        
        Args:
            pred_2d_maps: Dict with keys:
                - rot_maps: [B, S, H, W, 4]
                - scale_maps: [B, S, H, W, 3]
                - opacity_maps: [B, S, H, W, 1]
                - sh_maps: [B, S, H, W, K, 3]
                - depth_maps: [B, S, H, W, 1]
            gt_2d_maps: Same structure as pred_2d_maps
        
        Returns:
            loss: Scalar loss value
        """
        import torch.nn.functional as F
        
        # Use last frame (S-1) for single frame prediction
        frame_idx = -1
        
        # Extract single frame
        pred_rot = pred_2d_maps["rot_maps"][:, frame_idx]  # [B, H, W, 4]
        pred_scale = pred_2d_maps["scale_maps"][:, frame_idx]  # [B, H, W, 3]
        pred_opacity = pred_2d_maps["opacity_maps"][:, frame_idx]  # [B, H, W, 1]
        pred_sh = pred_2d_maps["sh_maps"][:, frame_idx]  # [B, H, W, K, 3]
        pred_depth = pred_2d_maps["depth_maps"][:, frame_idx]  # [B, H, W, 1]
        
        gt_rot = gt_2d_maps["rot_maps"][:, frame_idx]
        gt_scale = gt_2d_maps["scale_maps"][:, frame_idx]
        gt_opacity = gt_2d_maps["opacity_maps"][:, frame_idx]
        gt_sh = gt_2d_maps["sh_maps"][:, frame_idx]
        gt_depth = gt_2d_maps["depth_maps"][:, frame_idx]
        
        # Compute losses for each component
        # Rotation: L2 loss on quaternions (already normalized)
        loss_rot = F.mse_loss(pred_rot, gt_rot)
        
        # Scale: L2 loss
        loss_scale = F.mse_loss(pred_scale, gt_scale)
        
        # Opacity: L2 loss
        loss_opacity = F.mse_loss(pred_opacity, gt_opacity)
        
        # SH: L2 loss
        loss_sh = F.mse_loss(pred_sh, gt_sh)
        
        # Depth: L2 loss (with optional weighting for closer objects)
        loss_depth = F.mse_loss(pred_depth, gt_depth)
        
        # Weighted combination
        total_loss = (
            0.1 * loss_rot +      # Rotation is less critical
            1.0 * loss_scale +    # Scale is important
            1.0 * loss_opacity +  # Opacity is important
            1.0 * loss_sh +       # SH (color) is important
            2.0 * loss_depth      # Depth is very important for 3D structure
        )
        
        return total_loss


    def _get_flow_horizon_weight(self, horizon_idx: int) -> float:
        if self.flow_horizon_weights is None:
            return 1.0
        if horizon_idx < len(self.flow_horizon_weights):
            return self.flow_horizon_weights[horizon_idx]
        return self.flow_horizon_weights[-1]

    def _slice_flow_target(self, observation, horizon_idx: int):
        """Extract one horizon of optional flow supervision."""
        flow_3d = getattr(observation, "flow_3d", None)
        flow_valid_mask = getattr(observation, "flow_valid_mask", None)
        
        def _slice_flow(value):
            if value is None:
                return None
            if value.ndim >= 5:
                return value[:, horizon_idx]
            return value

        def _slice_mask(value):
            if value is None:
                return None
            if value.ndim >= 4:
                return value[:, horizon_idx]
            return value

        return _slice_flow(flow_3d), _slice_mask(flow_valid_mask)

    def _compute_flow_supervision_loss(
        self,
        raw_delta_xyz: torch.Tensor | None,
        target_observation,
        *,
        step: int | None = None,
        time_suffix: str = "",
        horizon_idx: int = 0,
    ) -> torch.Tensor:
        if self.flow_loss_weight <= 0.0:
            return torch.zeros((), dtype=torch.float32, device=raw_delta_xyz.device if raw_delta_xyz is not None else "cpu")
        if raw_delta_xyz is None or raw_delta_xyz.ndim != 3:
            target_device = None
            if target_observation is not None:
                if getattr(target_observation, "flow_3d", None) is not None:
                    target_device = target_observation.flow_3d.device
                elif getattr(target_observation, "state", None) is not None:
                    target_device = target_observation.state.device
            return torch.zeros((), dtype=torch.float32, device=target_device or torch.device("cpu"))
        if self.flow_first_horizon_only and horizon_idx != 0:
            return torch.zeros((), dtype=torch.float32, device=raw_delta_xyz.device)

        horizon_weight = self._get_flow_horizon_weight(horizon_idx)
        if horizon_weight <= 0.0:
            return torch.zeros((), dtype=torch.float32, device=raw_delta_xyz.device)

        flow_target, flow_valid_mask = self._slice_flow_target(target_observation, horizon_idx)
        if flow_target is None or flow_valid_mask is None:
            return torch.zeros((), dtype=torch.float32, device=raw_delta_xyz.device)

        flow_target = flow_target.to(device=raw_delta_xyz.device, dtype=torch.float32)
        flow_valid_mask = flow_valid_mask.to(device=raw_delta_xyz.device, dtype=torch.bool)

        bsize, num_points, coord_dim = raw_delta_xyz.shape
        if coord_dim != 3:
            return torch.zeros((), dtype=torch.float32, device=raw_delta_xyz.device)

        grid_size = int(math.isqrt(num_points))
        if grid_size * grid_size != num_points:
            return torch.zeros((), dtype=torch.float32, device=raw_delta_xyz.device)

        pred_flow = raw_delta_xyz.float().reshape(bsize, grid_size, grid_size, 3).permute(0, 3, 1, 2)
        target_flow = flow_target.permute(0, 3, 1, 2)
        valid_mask = flow_valid_mask.unsqueeze(1)

        if pred_flow.shape[-2:] != target_flow.shape[-2:]:
            pred_flow = F.interpolate(pred_flow, size=target_flow.shape[-2:], mode="bilinear", align_corners=False)
        if valid_mask.shape[-2:] != target_flow.shape[-2:]:
            valid_mask = F.interpolate(valid_mask.float(), size=target_flow.shape[-2:], mode="nearest") > 0.5

        valid_mask_f = valid_mask.float()
        denom = valid_mask_f.sum().clamp_min(1.0)
        if denom.item() <= 0:
            return torch.zeros((), dtype=torch.float32, device=raw_delta_xyz.device)

        if self.flow_loss_type == "mse":
            diff = F.mse_loss(pred_flow, target_flow, reduction="none")
        else:
            diff = F.smooth_l1_loss(pred_flow, target_flow, reduction="none")

        channel_weights = torch.tensor(
            self.flow_loss_channel_weights,
            device=pred_flow.device,
            dtype=pred_flow.dtype,
        ).view(1, pred_flow.shape[1], 1, 1)
        diff = diff * channel_weights
        flow_loss = (diff * valid_mask_f).sum() / (denom * pred_flow.shape[1])
        flow_loss = flow_loss * horizon_weight

        if step is not None and step % 400 == 0:
            logging.info(
                f"Step {step}{time_suffix}: Flow Loss = {flow_loss.item():.6f}, "
                f"weight={self.flow_loss_weight}, loss_type={self.flow_loss_type}, "
                f"channel_weights={self.flow_loss_channel_weights}, horizon_weight={horizon_weight:.4f}, "
                f"valid_ratio={(valid_mask_f.mean().item()):.6f}, horizon_idx={horizon_idx}"
            )

        return flow_loss.to(dtype=torch.float32)

    def _build_future_velocity_heatmap(
        self,
        raw_delta_xyz: torch.Tensor | None,
        target_hw: tuple[int, int],
    ) -> torch.Tensor | None:
        """Convert predicted xyz deltas into an image-space speed heatmap."""
        if raw_delta_xyz is None or raw_delta_xyz.ndim != 3:
            return None

        bsize, num_points, coord_dim = raw_delta_xyz.shape
        if coord_dim != 3:
            return None

        grid_size = int(math.isqrt(num_points))
        if grid_size * grid_size != num_points:
            return None

        speed_map = raw_delta_xyz.float().norm(dim=-1).reshape(bsize, 1, grid_size, grid_size)
        if speed_map.shape[-2:] != target_hw:
            speed_map = F.interpolate(speed_map, size=target_hw, mode="bilinear", align_corners=False)

        return speed_map.squeeze(1)

    def _visualize_rendering_comparison(self, step, gaussian_params, target_obs, cam_params_dict, view_names, time_suffix="", temporal_frames=None, temporal_labels=None, future_label="future"):
        """Helper to visualize Rendered vs GT images. Delegated to GaussianRenderer."""
        # Cleanly moved to gaussian_renderer.py
        visualize_rendering_comparison(
            step,
            gaussian_params,
            target_obs,
            cam_params_dict,
            self.gaussian_renderer,
            view_names,
            save_dir=self.vis_save_dir,
            time_suffix=time_suffix,
            temporal_frames=temporal_frames,
            temporal_labels=temporal_labels,
            future_label=future_label,
        )

    def _get_temporal_frames_for_viz(self, preprocessed_observation):
        temporal_frames = {}
        if hasattr(preprocessed_observation, "raw_temporal_images"):
            temporal_frames = preprocessed_observation.raw_temporal_images
            for key, value in temporal_frames.items():
                logging.info(f"[Viz] Extracted raw temporal {key} with shape {value.shape}")
        elif hasattr(preprocessed_observation, "images"):
            for key, value in preprocessed_observation.images.items():
                if value.ndim == 5 and value.shape[1] >= 2:
                    temporal_frames[key] = value
                    logging.info(f"[Viz] Extracted {key} with shape {value.shape}")
        return temporal_frames

    def _visualize_future_rollout(
        self,
        step,
        z_future_pred_tokens,
        future_observation,
        preprocessed_observation,
        static_template: dict | None = None,
        per_step_delta: torch.Tensor | None = None,
    ):
        """Visualize context + future rollout GT/render/diff in one grid.

        static_template: Prefix-derived base Gaussian template reused for all horizons.
        When None, fallback to h=0 full decode then reuse for later horizons.
        """
        from gaussiandream.models_pytorch.gaussian_renderer import visualize_future_rollout_comparison

        if self.world_model is None or self.gaussian_renderer is None or future_observation is None:
            return

        temporal_frames = self._get_temporal_frames_for_viz(preprocessed_observation)
        available_future_steps = self._get_temporal_observation_length(future_observation)
        rollout_horizon = min(self.future_prediction_horizon, available_future_steps or 1)
        if rollout_horizon <= 0:
            return

        first_future_target = (
            self._slice_temporal_observation(future_observation, 0)
            if available_future_steps > 0
            else future_observation
        )
        view_names = []
        for key in first_future_target.images.keys():
            key_lower = key.lower()
            if key == "image" or "agent" in key_lower or "high" in key_lower or "cam_high" in key_lower or "exterior" in key_lower or "base" in key_lower:
                if "agent" not in view_names:
                    view_names.append("agent")
            elif "wrist" in key_lower or "bravo" in key_lower:
                if "wrist" not in view_names:
                    view_names.append("wrist")
        if not view_names:
            view_names = ["agent"]

        target_obs_seq = []
        rendered_obs_seq = []
        motion_weight_seq = []
        pred_velocity_seq = []
        aux_future_depth_seq = []
        gaussian_depth_seq = []
        delta_xyz_seq = []
        overlay_render_seq = []
        render_views = None
        base_target_obs = None
        base_rendered_obs = None
        base_aux_depth_map = None
        base_gaussian_depth_map = None

        def _build_target_obs_and_cameras(obs):
            target_obs = {}
            cam_params_dict = {}
            valid_views = []
            if obs is None:
                return target_obs, cam_params_dict, valid_views

            for key, value in obs.images.items():
                img_tensor = value
                mapped_view_name = None
                key_lower = key.lower()
                if key == "image" or "agent" in key_lower or "high" in key_lower or "cam_high" in key_lower or "exterior" in key_lower or "base" in key_lower:
                    mapped_view_name = "agent"
                elif "left_wrist" in key_lower or "wrist_left" in key_lower:
                    mapped_view_name = "wrist"
                elif "right_wrist" in key_lower or "wrist_right" in key_lower:
                    if img_tensor.min() == img_tensor.max() == -1.0:
                        continue
                    mapped_view_name = "wrist"
                elif "wrist" in key_lower or "bravo" in key_lower:
                    mapped_view_name = "wrist"

                if mapped_view_name:
                    if img_tensor.shape[1] != 3 and img_tensor.shape[-1] == 3:
                        img_tensor = img_tensor.permute(0, 3, 1, 2)
                    img_tensor = (img_tensor + 1.0) / 2.0
                    view_key = f"{mapped_view_name}_image"
                    if view_key not in target_obs:
                        target_obs[view_key] = img_tensor
                        cam_params_dict[mapped_view_name] = self._get_camera_params_for_view(
                            mapped_view_name, z_future_pred_tokens.device, z_future_pred_tokens.shape[0]
                        )
                        valid_views.append(mapped_view_name)
            return target_obs, cam_params_dict, valid_views

        def _render_single_batch(gaussian_params, cam_params_dict, views_to_render):
            rendered_obs = {}
            params_single = {
                "xyz": gaussian_params["xyz"][:1],
                "sh": gaussian_params["sh"][:1],
                "opacity": gaussian_params["opacity"][:1],
                "scales": gaussian_params["scales"][:1],
                "rotations": gaussian_params["rotations"][:1],
            }
            for render_view in views_to_render:
                cam_params = {
                    key: value[:1] if isinstance(value, torch.Tensor) else value
                    for key, value in cam_params_dict[render_view].items()
                }
                rendered_obs[f"{render_view}_image"] = self.gaussian_renderer(params_single, cam_params).float()
            return rendered_obs

        with torch.no_grad():
            decoder_state = self._build_world_decoder_state(z_future_pred_tokens, step=step)
            base_depth = getattr(preprocessed_observation, "depth", None)
            if base_depth is not None and base_depth.ndim == 5:
                base_depth = base_depth[:, -1]
            motion_weight_reference = self._get_motion_weight_source_observation(preprocessed_observation)
            motion_weight_base_depth = getattr(motion_weight_reference, "depth", None)
            if motion_weight_base_depth is not None and motion_weight_base_depth.ndim == 5:
                motion_weight_base_depth = motion_weight_base_depth[:, -1]

            current_obs_steps = self._get_temporal_observation_length(preprocessed_observation)
            current_target = (
                self._slice_temporal_observation(preprocessed_observation, current_obs_steps - 1)
                if current_obs_steps > 0
                else preprocessed_observation
            )

            viz_static_template = static_template
            for horizon_idx in range(rollout_horizon):
                future_target = (
                    self._slice_temporal_observation(future_observation, horizon_idx)
                    if available_future_steps > 0
                    else future_observation
                )
                camera_params_for_decode = self._get_camera_params_for_view(
                    "agent", z_future_pred_tokens.device, z_future_pred_tokens.shape[0]
                )
                reused_template = None
                vtf = 1.0
                if getattr(self, "use_velocity_future_gaussians", False):
                    reused_template = viz_static_template
                    o0 = self.future_prediction_offsets[0]
                    oh = self.future_prediction_offsets[horizon_idx]
                    vtf = float(oh) / float(o0) if o0 else 1.0

                gaussian_params = self.world_model.decode(
                    z_future_pred_tokens.float(),
                    future_observation=future_target,
                    gaussian_adapter=self.gaussian_adapter,
                    camera_params=camera_params_for_decode,
                    step=step,
                    current_observation=preprocessed_observation,
                    base_depth=base_depth,
                    horizon_idx=horizon_idx,
                    static_reference_params=reused_template,
                    velocity_time_factor=vtf,
                    shared_state=decoder_state,
                )
                aux_future_depth = self.world_model.predict_future_depth_aux(decoder_state, horizon_idx=horizon_idx)
                if getattr(self, "use_velocity_future_gaussians", False) and horizon_idx == 0 and viz_static_template is None:
                    viz_static_template = {
                        k: v.detach() if torch.is_tensor(v) else v for k, v in gaussian_params.items()
                    }

                target_obs, cam_params_dict, valid_views = _build_target_obs_and_cameras(future_target)

                if valid_views:
                    render_views = ["agent"] if "agent" in valid_views else valid_views
                elif render_views is None:
                    render_views = ["agent"]

                rendered_obs = _render_single_batch(gaussian_params, cam_params_dict, render_views)
                target_obs_seq.append({key: value[:1].float() for key, value in target_obs.items()})
                rendered_obs_seq.append(rendered_obs)

                if aux_future_depth is not None:
                    aux_future_depth_seq.append(aux_future_depth[:1].float())
                else:
                    aux_future_depth_seq.append(None)

                gaussian_depth = gaussian_params.get("depth_map")
                if gaussian_depth is not None:
                    gaussian_depth_seq.append(gaussian_depth[:1].float())
                else:
                    gaussian_depth_seq.append(None)

                raw_delta_xyz = gaussian_params.get("raw_delta_xyz")
                if raw_delta_xyz is not None:
                    num_points = raw_delta_xyz.shape[1]
                    grid_size = int(round(num_points ** 0.5))
                    if grid_size * grid_size == num_points:
                        delta_xyz_map = raw_delta_xyz[:1].reshape(1, grid_size, grid_size, 3).permute(0, 3, 1, 2).contiguous()
                        delta_xyz_seq.append(delta_xyz_map.float())
                    else:
                        delta_xyz_seq.append(None)
                else:
                    delta_xyz_seq.append(None)

                overlay_render_entry = {}
                for render_view in render_views:
                    view_key = f"{render_view}_image"
                    overlay_render_entry[view_key] = rendered_obs[view_key][:1].float()
                overlay_render_seq.append(overlay_render_entry)

                motion_weight_entry = {}
                for render_view in render_views:
                    view_key = f"{render_view}_image"
                    current_image = self._extract_reference_view_image(motion_weight_reference, render_view)
                    future_image = target_obs[view_key]
                    current_depth_for_view = motion_weight_base_depth if render_view == "agent" else None
                    future_depth_for_view = (
                        future_target.depth
                        if render_view == "agent" and hasattr(future_target, "depth")
                        else None
                    )
                    motion_weight_entry[view_key] = self._build_future_motion_weight_map(
                        current_image,
                        future_image,
                        current_depth_for_view,
                        future_depth_for_view,
                    )[:1].float()
                motion_weight_seq.append(motion_weight_entry)

                pred_velocity_entry = {}
                raw_delta_xyz = gaussian_params.get("raw_delta_xyz")
                if raw_delta_xyz is not None:
                    for render_view in render_views:
                        view_key = f"{render_view}_image"
                        target_hw = target_obs[view_key].shape[-2:]
                        pred_velocity_map = build_projected_velocity_map(
                            gaussian_params,
                            cam_params_dict[render_view],
                            target_hw,
                        )
                        if pred_velocity_map is None:
                            pred_velocity_map = self._build_future_velocity_heatmap(raw_delta_xyz, target_hw)
                        if pred_velocity_map is not None:
                            pred_velocity_entry[view_key] = pred_velocity_map[:1].float()
                pred_velocity_seq.append(pred_velocity_entry)


            base_target_obs_raw, base_cam_params_dict, base_valid_views = _build_target_obs_and_cameras(current_target)
            if base_target_obs_raw:
                base_target_obs = {key: value[:1].float() for key, value in base_target_obs_raw.items()}
            base_render_views = ["agent"] if "agent" in base_valid_views else base_valid_views
            if not base_render_views:
                base_render_views = render_views or view_names
            if viz_static_template is not None and base_target_obs is not None and base_render_views:
                base_rendered_obs = _render_single_batch(viz_static_template, base_cam_params_dict, base_render_views)
            if viz_static_template is not None:
                base_aux_depth_map = viz_static_template.get("depth_map")
                if base_aux_depth_map is not None:
                    base_aux_depth_map = base_aux_depth_map[:1].float()
                base_gaussian_depth_map = viz_static_template.get("depth_map")
                if base_gaussian_depth_map is not None:
                    base_gaussian_depth_map = base_gaussian_depth_map[:1].float()

        visualize_future_rollout_comparison(
            step,
            target_obs_seq,
            rendered_obs_seq,
            render_views or view_names,
            save_dir=self.vis_save_dir,
            temporal_frames=temporal_frames,
            time_suffix="_future_rollout",
            horizon_labels=[f"t+{offset}" for offset in self.future_prediction_offsets[:rollout_horizon]],
            base_target_obs=base_target_obs,
            base_rendered_obs=base_rendered_obs,
            base_label="t/base",
            motion_weight_seq=motion_weight_seq,
            pred_velocity_seq=pred_velocity_seq,
            context_labels=getattr(preprocessed_observation, "raw_temporal_labels", self._temporal_context_labels()),
            aux_future_depth_seq=aux_future_depth_seq,
            overlay_render_seq=overlay_render_seq,
            base_aux_depth_map=base_aux_depth_map,
            gaussian_depth_seq=gaussian_depth_seq,
            base_gaussian_depth_map=base_gaussian_depth_map,
            delta_xyz_seq=delta_xyz_seq,
        )
        self._log_future_rollout_pixel_lpips(step, rendered_obs_seq)

    def export_future_rollout_gaussians(self, observation, actions=None):
        """Run offline rollout export without entering the training-loss path."""
        if not self.use_world_tokens_in_prefix or self.world_model is None or self.gaussian_renderer is None:
            raise ValueError("World-model Gaussian rollout export is not available for this model/config")

        first_image = next(iter(observation.images.values()))
        device = first_image.device
        batch_size = first_image.shape[0]

        if actions is None:
            actions = torch.zeros(
                batch_size,
                self.config.action_horizon,
                self.config.action_dim,
                device=device,
                dtype=torch.float32,
            )

        images, img_masks, lang_tokens, lang_masks, state, future_observation, preprocessed_observation = self._preprocess_observation(
            observation, train=False
        )

        gaussian_inputs = None
        if self.gaussian_adapter.use_gaussian:
            gaussian_inputs = self._prepare_gaussian_inputs(preprocessed_observation, device, batch_size, is_training=False)

        prefix_result = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            gaussian_inputs=gaussian_inputs,
            return_segment_lengths=True,
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks, segment_info = prefix_result

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state,
            torch.zeros_like(actions, dtype=torch.float32),
            torch.zeros(batch_size, device=device, dtype=torch.float32),
        )
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        (prefix_out, _), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        z_future_pred_tokens = None
        future_slice = self._get_segment_slice(segment_info, "future")
        if future_slice is not None:
            future_start, future_end = future_slice
            z_future_pred_tokens = prefix_out[:, future_start:future_end, :]

        if z_future_pred_tokens is None:
            raise ValueError("Failed to extract future rollout tokens from prefix output")

        available_future_steps = self._get_temporal_observation_length(future_observation)
        rollout_horizon = min(self.future_prediction_horizon, available_future_steps or 1)
        if rollout_horizon <= 0:
            raise ValueError("No future horizons available in observation batch")

        decoder_state = self._build_world_decoder_state(z_future_pred_tokens, step=None)
        current_target = self._slice_temporal_observation(preprocessed_observation, self.temporal_context_count - 1)
        if current_target is None:
            current_target = preprocessed_observation

        current_recon_tokens = z_future_pred_tokens
        current_decoder_state = decoder_state
        _, static_template_raw = self._compute_current_frame_recon_loss(
            current_recon_tokens,
            current_target,
            step=None,
            time_suffix="_export_static",
            visualize=False,
            return_gaussian_params=True,
            decoder_state=current_decoder_state,
        )
        if not static_template_raw or static_template_raw.get("xyz") is None:
            raise ValueError("Failed to decode base Gaussian template for export")

        static_template = {
            k: v.detach() if torch.is_tensor(v) else v for k, v in static_template_raw.items()
        }

        context_length = self._get_temporal_observation_length(preprocessed_observation)
        context_observations = []
        if context_length > 0:
            for idx in range(context_length):
                context_observations.append(self._slice_temporal_observation(preprocessed_observation, idx))
        else:
            context_observations.append(preprocessed_observation)

        context_labels = getattr(preprocessed_observation, "raw_temporal_labels", self._temporal_context_labels())
        context_labels = list(context_labels[: len(context_observations)])

        future_targets = []
        future_gaussians = []
        future_depth_aux = []
        for horizon_idx in range(rollout_horizon):
            future_target = (
                self._slice_temporal_observation(future_observation, horizon_idx)
                if available_future_steps > 0
                else future_observation
            )
            future_targets.append(future_target)
            offset0 = self.future_prediction_offsets[0] if self.future_prediction_offsets else 1
            offseth = self.future_prediction_offsets[horizon_idx] if horizon_idx < len(self.future_prediction_offsets) else (horizon_idx + 1)
            velocity_time_factor = float(offseth) / float(offset0) if offset0 else 1.0
            _, gaussian_params, aux_outputs = self._compute_world_model_frame_loss(
                z_future_pred_tokens,
                future_target,
                preprocessed_observation,
                step=None,
                time_suffix=f"_export_tplus{offseth}",
                visualize=False,
                horizon_idx=horizon_idx,
                static_gaussian_params=static_template if self.use_velocity_future_gaussians else None,
                velocity_time_factor=velocity_time_factor,
                return_gaussian_params=True,
                decoder_state=decoder_state,
                return_aux=True,
            )
            future_gaussians.append({
                k: v.detach() if torch.is_tensor(v) else v for k, v in gaussian_params.items()
            })
            aux_depth = aux_outputs.get("future_depth_aux")
            future_depth_aux.append(aux_depth.detach() if torch.is_tensor(aux_depth) else None)

        return {
            "context_observations": context_observations,
            "context_labels": context_labels,
            "current_observation": current_target,
            "base_gaussian_params": static_template,
            "future_gaussian_params_seq": future_gaussians,
            "future_targets": future_targets,
            "future_offsets": list(self.future_prediction_offsets[:rollout_horizon]),
            "future_depth_aux_seq": future_depth_aux,
            "preprocessed_observation": preprocessed_observation,
        }

    def render_gaussian_views_for_export(
        self,
        gaussian_params: dict,
        *,
        device: torch.device,
        batch_size: int,
        agent_view: bool = True,
        orbit_azimuth_deg: float | None = None,
        orbit_elevation_deg: float = 20.0,
        orbit_radius_scale: float = 2.2,
        sweep_phase: float | None = None,
        target_hw: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Render Gaussian params from agent view and optional novel view."""
        renders: dict[str, torch.Tensor] = {}
        params_single = {
            "xyz": gaussian_params["xyz"][:1],
            "sh": gaussian_params["sh"][:1],
            "opacity": gaussian_params["opacity"][:1],
            "scales": gaussian_params["scales"][:1],
            "rotations": gaussian_params["rotations"][:1],
        }
        if target_hw is None:
            render_size = int(getattr(self.gaussian_renderer, "image_size", 224))
            target_hw = (render_size, render_size)
        if agent_view:
            agent_camera = {
                key: value[:1] if isinstance(value, torch.Tensor) else value
                for key, value in self._get_camera_params_for_view("agent", device, batch_size).items()
            }
            renders["agent"] = self.gaussian_renderer(params_single, agent_camera).float()
        if orbit_azimuth_deg is not None:
            orbit_camera = build_orbit_camera_params(
                params_single["xyz"],
                target_hw=target_hw,
                azimuth_deg=orbit_azimuth_deg,
                elevation_deg=orbit_elevation_deg,
                radius_scale=orbit_radius_scale,
                device=device,
            )
            renders["orbit"] = self.gaussian_renderer(params_single, orbit_camera).float()
        if sweep_phase is not None:
            sweep_camera = build_sweep_camera_params(
                params_single["xyz"],
                target_hw=target_hw,
                lateral_phase=sweep_phase,
                elevation_deg=orbit_elevation_deg,
                radius_scale=orbit_radius_scale,
                device=device,
            )
            renders["sweep"] = self.gaussian_renderer(params_single, sweep_camera).float()
        return renders

    def _run_prefix_backbone(
        self,
        prefix_embs: torch.Tensor,
        suffix_embs: torch.Tensor,
        att_2d_masks_4d: torch.Tensor,
        position_ids: torch.Tensor,
        adarms_cond,
        *,
        output_hidden_states: bool = False,
    ):
        return self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
            output_hidden_states=output_hidden_states,
        )

    def forward(self, observation, actions, noise=None, time=None, step=None) -> Tensor:
        """Do a full training forward pass and compute the loss."""
        self._apply_training_stage_if_needed(step)
        images, img_masks, lang_tokens, lang_masks, state, future_observation, preprocessed_observation = self._preprocess_observation(
            observation, train=True
        )

        if self.lpips_fn is not None and hasattr(self.lpips_fn, "net"):
            device = actions.device
            if next(self.lpips_fn.parameters()).device != device:
                self.lpips_fn = self.lpips_fn.to(device)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        gaussian_inputs = None
        if self.gaussian_adapter.use_gaussian:
            gaussian_inputs = self._prepare_gaussian_inputs(
                preprocessed_observation,
                actions.device,
                actions.shape[0],
            )

        prefix_result = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            gaussian_inputs=gaussian_inputs,
            return_segment_lengths=self.use_world_tokens_in_prefix,
        )
        if self.use_world_tokens_in_prefix:
            prefix_embs, prefix_pad_masks, prefix_att_masks, segment_info = prefix_result
        else:
            prefix_embs, prefix_pad_masks, prefix_att_masks = prefix_result
            segment_info = {"lengths": {}, "ranges": {}}

        segment_lengths = segment_info.get("lengths", {})
        if step is not None and step % 400 == 0 and self.use_world_tokens_in_prefix:
            delta_q_shape = segment_lengths.get("debug_delta_q_shape")
            debug_mta_shapes = segment_lengths.get("debug_mta_shapes", {})
            mta_shape_str = (
                ", ".join(f"layer{layer_idx}={shape}" for layer_idx, shape in sorted(debug_mta_shapes.items()))
                if debug_mta_shapes
                else "None"
            )
            logging.info(
                f"Step {step}: Future Token Input Shapes | "
                f"delta_q.shape={delta_q_shape}, mta_features={mta_shape_str}"
            )

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        (prefix_out, suffix_out), _ = self._apply_checkpoint(
            self._run_prefix_backbone,
            prefix_embs,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
            output_hidden_states=False,
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        suffix_out = self.action_out_proj(suffix_out)

        action_warmup_active = step is not None and step < self.action_warmup_steps
        action_loss_enabled = self._action_loss_enabled and not action_warmup_active
        if action_loss_enabled:
            loss = F.mse_loss(suffix_out, u_t)
        else:
            loss = (suffix_out * 0).sum()

        if step is not None and step % 400 == 0:
            logging.info(
                f"Step {step}: Action loss schedule | enabled={action_loss_enabled}, "
                f"warmup_steps={self.action_warmup_steps}, action_branch_trainable={self._action_loss_enabled}"
            )

        z_t1_pred_tokens = None
        z_future_pred_tokens = None
        per_step_delta = None
        future_slice = self._get_segment_slice(segment_info, "future")
        if self.use_world_tokens_in_prefix and future_slice is not None:
            future_start, future_end = future_slice
            z_t1_pred_tokens = prefix_out[:, future_start:future_end, :]
            z_future_pred_tokens = z_t1_pred_tokens

            if step is not None and step % 400 == 0:
                logging.info(
                    f"Step {step}: Future Token Shapes | "
                    f"z_t1_pred_tokens.shape={tuple(z_t1_pred_tokens.shape)}, "
                    f"future_start={future_start}, future_end={future_end}, "
                    f"segment_ranges={segment_info.get('ranges', {})}"
                )
                logging.info(
                    f"Step {step}: Future Motion Token Stats | "
                    f"abs_mean={z_t1_pred_tokens.float().abs().mean().item():.6f}, "
                    f"rms={z_t1_pred_tokens.float().pow(2).mean().sqrt().item():.6f}, "
                    "mode=direct motion-aware velocity token"
                )

        if (
            self._world_supervision_enabled
            and self.use_world_tokens_in_prefix
            and z_future_pred_tokens is not None
            and future_observation is not None
        ):
            available_future_steps = self._get_temporal_observation_length(future_observation)
            rollout_horizon = min(self.future_prediction_horizon, available_future_steps or 1)
            world_model_loss = torch.zeros((), dtype=torch.float32, device=loss.device)
            horizon_loss_weights = self._get_future_horizon_loss_weights(
                step, rollout_horizon, loss.device, torch.float32
            )
            decoder_state = self._build_world_decoder_state(z_future_pred_tokens, step=step)

            current_template_loss = torch.zeros((), dtype=torch.float32, device=loss.device)
            static_template: dict | None = None
            current_target = self._slice_temporal_observation(preprocessed_observation, self.temporal_context_count - 1)
            if current_target is None:
                current_target = preprocessed_observation

            current_recon_tokens = z_future_pred_tokens
            current_recon_source = "future_tokens_shared_latent"

            if current_recon_tokens is not None:
                try:
                    current_decoder_state = decoder_state
                    current_template_loss, static_template_raw = self._compute_current_frame_recon_loss(
                        current_recon_tokens,
                        current_target,
                        step=step,
                        time_suffix="_t_current_static",
                        visualize=False,
                        return_gaussian_params=True,
                        decoder_state=current_decoder_state,
                    )
                    if static_template_raw and static_template_raw.get("xyz") is not None:
                        static_template = {
                            k: v.detach() if torch.is_tensor(v) else v for k, v in static_template_raw.items()
                        }
                    else:
                        static_template = None
                    if torch.isfinite(current_template_loss):
                        loss = loss + self.current_frame_recon_loss_weight * current_template_loss.to(loss.dtype)
                    if step is not None and step % 400 == 0:
                        logging.info(
                            f"Step {step}: Current Frame Static Recon = {current_template_loss.item():.6f}, "
                            f"weight={self.current_frame_recon_loss_weight}, source={current_recon_source}"
                        )
                except Exception as error:
                    logging.warning(f"Current-frame static reconstruction failed: {error}")
                    static_template = None

            if rollout_horizon > 0:
                delta_reg_weights = self._get_future_horizon_loss_weights(
                    step, rollout_horizon, z_future_pred_tokens.device, torch.float32
                )
                per_horizon_delta = [z_future_pred_tokens.float() for _ in range(rollout_horizon)]
                per_step_delta = torch.stack(per_horizon_delta, dim=1)
                raw_delta_reg = (
                    per_step_delta.pow(2).mean(dim=(0, 2, 3)) * delta_reg_weights
                ).sum() / delta_reg_weights.sum().clamp_min(1e-6)
                delta_reg = self.future_delta_reg_weight * raw_delta_reg
                if torch.isfinite(delta_reg):
                    loss = loss + delta_reg.to(loss.dtype)
                self._log_future_rollout_diagnostics(
                    step,
                    z_t1_pred_tokens,
                    z_future_pred_tokens,
                    per_step_delta,
                    delta_reg_weights,
                )

            if rollout_horizon > 0:
                if self.use_velocity_future_gaussians and static_template is None:
                    logging.warning("Skipping future velocity rollout because static template decode failed.")
                    return loss
                reused_template = static_template if self.use_velocity_future_gaussians else None
                for horizon_idx in range(rollout_horizon):
                    future_target = (
                        self._slice_temporal_observation(future_observation, horizon_idx)
                        if available_future_steps > 0
                        else future_observation
                    )
                    offset0 = self.future_prediction_offsets[0] if self.future_prediction_offsets else 1
                    offseth = (
                        self.future_prediction_offsets[horizon_idx]
                        if horizon_idx < len(self.future_prediction_offsets)
                        else (horizon_idx + 1)
                    )
                    velocity_time_factor = float(offseth) / float(offset0) if offset0 else 1.0
                    horizon_suffix = f"_tplus{offseth}_pred_vlm"
                    horizon_loss = self._compute_world_model_frame_loss(
                        z_future_pred_tokens,
                        future_target,
                        preprocessed_observation,
                        step=step,
                        time_suffix=horizon_suffix,
                        visualize=False,
                        horizon_idx=horizon_idx,
                        static_gaussian_params=reused_template,
                        velocity_time_factor=velocity_time_factor,
                        decoder_state=decoder_state,
                    )
                    world_model_loss = world_model_loss + horizon_loss_weights[horizon_idx] * horizon_loss

                loss = loss + (world_model_loss / horizon_loss_weights.sum().clamp_min(1e-6)).to(loss.dtype)

                import torch.distributed as dist

                should_log = step is not None and step % 200 == 0
                is_main_process = not dist.is_initialized() or dist.get_rank() == 0
                if should_log and is_main_process:
                    try:
                        self._visualize_future_rollout(
                            step,
                            z_future_pred_tokens,
                            future_observation,
                            preprocessed_observation,
                            static_template=static_template,
                            per_step_delta=per_step_delta,
                        )
                    except Exception as viz_error:
                        logging.warning(f"Step {step}: Future rollout visualization failed: {viz_error}")

        return loss

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action."""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state, _, preprocessed_observation = self._preprocess_observation(
            observation, train=False
        )

        gaussian_inputs = None
        if self.gaussian_adapter.use_gaussian:
            gaussian_inputs = self._prepare_gaussian_inputs(preprocessed_observation, device, bsize, is_training=False)

        prefix_result = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            gaussian_inputs=gaussian_inputs,
            return_segment_lengths=self.use_world_tokens_in_prefix,
        )
        if self.use_world_tokens_in_prefix:
            prefix_embs, prefix_pad_masks, prefix_att_masks, segment_info = prefix_result
        else:
            prefix_embs, prefix_pad_masks, prefix_att_masks = prefix_result
            segment_info = {"lengths": {}, "ranges": {}}
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        timestep = torch.tensor(1.0, dtype=torch.float32, device=device)
        while timestep >= -dt / 2:
            expanded_time = timestep.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            x_t = x_t + dt * v_t
            timestep += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
