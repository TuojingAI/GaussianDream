import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _UpsampleBlock(nn.Module): 
    """ConvTranspose upsample block with GroupNorm + GELU + residual."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        # Residual projection when channels change
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = F.interpolate(self.skip(x), scale_factor=2, mode="bilinear", align_corners=False)
        h = F.gelu(self.norm1(self.up(x)))
        h = F.gelu(self.norm2(self.conv(h)))
        return h + skip


class _FeatureFusionBlock(nn.Module):
    """DPT-style feature fusion block for multi-scale feature integration."""
    
    def __init__(self, features: int, has_residual: bool = True):
        super().__init__()
        self.has_residual = has_residual
        
        if has_residual:
            self.residual_conv = nn.Sequential(
                nn.Conv2d(features, features, 3, padding=1, bias=True),
                nn.GroupNorm(min(32, features), features),
                nn.GELU(),
                nn.Conv2d(features, features, 3, padding=1, bias=True),
            )
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1, bias=True),
            nn.GroupNorm(min(32, features), features),
            nn.GELU(),
            nn.Conv2d(features, features, 1, bias=True),
        )
    
    def forward(self, x, residual=None, size=None):
        """
        Args:
            x: Main feature [B, C, H, W]
            residual: Optional residual feature [B, C, H_res, W_res]
            size: Target size (H, W) for upsampling
        """
        if self.has_residual and residual is not None:
            # Upsample residual to match x if needed
            if residual.shape[2:] != x.shape[2:]:
                residual = F.interpolate(residual, size=x.shape[2:], mode='bilinear', align_corners=True)
            x = x + self.residual_conv(residual)
        
        x = self.fusion_conv(x)
        
        # Upsample to target size if specified
        if size is not None and x.shape[2:] != size:
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        
        return x


class SharedGaussianBackbone(nn.Module):
    """Shared token-to-feature backbone for both static and velocity decoding."""

    def __init__(self, token_dim: int = 2048):
        super().__init__()
        self.layer1 = _UpsampleBlock(token_dim, 512)   # 32×32 -> 64×64
        self.layer2 = _UpsampleBlock(512, 256)         # 64×64 -> 128×128
        self.layer3 = _UpsampleBlock(256, 128)         # 128×128 -> 256×256

        self.fusion1 = _FeatureFusionBlock(128, has_residual=False)
        self.fusion2 = _FeatureFusionBlock(128, has_residual=True)
        self.fusion3 = _FeatureFusionBlock(128, has_residual=True)

        self.proj_feat2 = nn.Conv2d(256, 128, 1)
        self.proj_feat1 = nn.Conv2d(512, 128, 1)

    def forward(self, token_grid: torch.Tensor) -> torch.Tensor:
        feat1 = self.layer1(token_grid)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)

        fused = self.fusion1(feat3)
        fused = self.fusion2(fused, residual=self.proj_feat2(feat2))
        fused = self.fusion3(fused, residual=self.proj_feat1(feat1))
        return fused


class GeometryHead(nn.Module):
    """Predict geometry-owned Gaussian parameters from shared geometry features."""

    def __init__(self, predict_depth: bool = True, feature_dim: int = 128):
        super().__init__()
        self.predict_depth = predict_depth
        self.feature_dim = feature_dim

        # rot(4) + scale(3) + opacity(1)
        self.param_head = nn.Conv2d(feature_dim, 8, 3, padding=1)

        if predict_depth:
            self.depth_refine = nn.Sequential(
                nn.Conv2d(feature_dim, 64, 3, padding=1),
                nn.GroupNorm(min(32, 64), 64),
                nn.GELU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.GroupNorm(min(32, 64), 64),
                nn.GELU(),
                nn.Conv2d(64, 1, 3, padding=1),
            )
            for module in self.depth_refine.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.xavier_uniform_(module.weight, gain=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        nn.init.xavier_uniform_(self.param_head.weight, gain=0.01)
        nn.init.zeros_(self.param_head.bias)

    def forward(self, shared_features: torch.Tensor) -> dict[str, torch.Tensor]:
        result = {"geom_params": self.param_head(shared_features)}
        if self.predict_depth:
            result["depth"] = self.depth_refine(shared_features)
        return result


class AppearanceHead(nn.Module):
    """Predict appearance-only SH coefficients from geometry context and RGB."""

    def __init__(
        self,
        use_image_fusion: bool = True,
        img_dim: int = 3,
        feature_dim: int = 128,
        image_fusion_alpha: float = 1.0,
        stop_grad_geometry: bool = True,
    ):
        super().__init__()
        self.use_image_fusion = bool(use_image_fusion)
        self.feature_dim = feature_dim
        self.image_fusion_alpha = float(image_fusion_alpha)
        self.stop_grad_geometry = bool(stop_grad_geometry)

        if use_image_fusion:
            self.img_merger = nn.Sequential(
                nn.Conv2d(img_dim, feature_dim, 7, padding=3),
                nn.GELU(),
            )

        self.sh_head = nn.Conv2d(feature_dim, 9, 3, padding=1)
        nn.init.xavier_uniform_(self.sh_head.weight, gain=0.01)
        nn.init.zeros_(self.sh_head.bias)

    def set_image_fusion_enabled(self, enabled: bool) -> None:
        self.use_image_fusion = bool(enabled)

    def forward(self, shared_features: torch.Tensor, images: torch.Tensor | None = None) -> torch.Tensor:
        geom_context = shared_features.detach() if self.stop_grad_geometry else shared_features
        appearance_features = geom_context
        if self.use_image_fusion and images is not None:
            if images.shape[2:] != geom_context.shape[2:]:
                images = F.interpolate(images, size=geom_context.shape[2:], mode="bilinear", align_corners=True)
            appearance_features = appearance_features + self.image_fusion_alpha * self.img_merger(images)
        return self.sh_head(appearance_features)


class StaticGaussianHead(nn.Module):
    """Decode geometry and appearance with separate heads.

    Geometry-owned outputs are predicted only from ``shared_features``. Appearance
    uses a separate SH branch so RGB fusion cannot backprop directly into geometry
    features when ``stop_grad_geometry_for_appearance`` is enabled.
    """

    def __init__(
        self,
        use_image_fusion: bool = True,
        img_dim: int = 3,
        predict_depth: bool = True,
        image_fusion_alpha: float = 1.0,
        stop_grad_geometry_for_appearance: bool = True,
    ):
        super().__init__()
        self.use_image_fusion = bool(use_image_fusion)
        self.predict_depth = bool(predict_depth)

        self.geometry_head = GeometryHead(predict_depth=predict_depth)
        self.appearance_head = AppearanceHead(
            use_image_fusion=use_image_fusion,
            img_dim=img_dim,
            image_fusion_alpha=image_fusion_alpha,
            stop_grad_geometry=stop_grad_geometry_for_appearance,
        )

        self.register_buffer(
            "sh_mask",
            torch.tensor(
                [1.0, 1.0, 1.0, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025],
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def set_image_fusion_enabled(self, enabled: bool) -> None:
        self.use_image_fusion = bool(enabled)
        self.appearance_head.set_image_fusion_enabled(enabled)

    def forward(self, shared_features: torch.Tensor, images: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        geometry_output = self.geometry_head(shared_features)
        sh_logits = self.appearance_head(shared_features, images=images)
        result = {
            "gaussian_params": torch.cat([geometry_output["geom_params"], sh_logits], dim=1),
        }
        if self.predict_depth:
            result["depth"] = geometry_output["depth"]
        return result


class FutureDepthHead(nn.Module):
    """Predict coarse future depth directly from shared geometry features."""

    def __init__(self, future_prediction_horizon: int, downsample_factor: int = 2, model_dim: int = 128):
        super().__init__()
        self.downsample_factor = max(1, int(downsample_factor))
        self.model_dim = model_dim
        self.horizon_proj = nn.Embedding(max(1, int(future_prediction_horizon)), model_dim)
        self.input_proj = nn.Sequential(
            nn.Conv2d(128, model_dim, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(min(32, model_dim), model_dim),
            nn.GELU(),
        )
        self.refine_block = nn.Sequential(
            nn.Conv2d(model_dim, model_dim, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(min(32, model_dim), model_dim),
            nn.GELU(),
            nn.Conv2d(model_dim, model_dim, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(min(32, model_dim), model_dim),
            nn.GELU(),
        )
        self.scale_proj = nn.Linear(model_dim, model_dim)
        self.shift_proj = nn.Linear(model_dim, model_dim)
        self.depth_head = nn.Sequential(
            nn.Conv2d(model_dim, model_dim, 3, padding=1, bias=True),
            nn.GroupNorm(min(32, model_dim), model_dim),
            nn.GELU(),
            nn.Conv2d(model_dim, 1, 3, padding=1),
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)
        nn.init.zeros_(self.shift_proj.weight)
        nn.init.zeros_(self.shift_proj.bias)

    def forward(self, shared_features: torch.Tensor, horizon_idx: int = 0) -> torch.Tensor:
        batch_size = shared_features.shape[0]
        feat = self.input_proj(shared_features)
        horizon_idx = max(0, min(int(horizon_idx), self.horizon_proj.num_embeddings - 1))
        horizon_ids = torch.full((batch_size,), horizon_idx, device=shared_features.device, dtype=torch.long)
        horizon_embed = self.horizon_proj(horizon_ids)
        horizon_scale = self.scale_proj(horizon_embed).view(batch_size, self.model_dim, 1, 1)
        horizon_shift = self.shift_proj(horizon_embed).view(batch_size, self.model_dim, 1, 1)
        refined = self.refine_block(feat)
        feat = feat + refined * (1.0 + horizon_scale) + horizon_shift
        depth_logits = self.depth_head(feat)
        if self.downsample_factor > 1:
            depth_logits = F.avg_pool2d(
                depth_logits,
                kernel_size=self.downsample_factor,
                stride=self.downsample_factor,
                ceil_mode=False,
            )
        return 8.0 * torch.sigmoid(depth_logits)


class VelocityGaussianHead(nn.Module):
    """Decode shared features into a per-Gaussian shared velocity field."""

    def __init__(
        self,
        future_prediction_horizon: int,
        model_dim: int = 128,
        velocity_scale: float = 1.0,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.velocity_scale = float(velocity_scale)
        self.horizon_proj = nn.Embedding(max(1, int(future_prediction_horizon)), model_dim)
        self.input_proj = nn.Sequential(
            nn.Conv2d(128, model_dim, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(min(32, model_dim), model_dim),
            nn.GELU(),
        )
        self.res_block = nn.Sequential(
            nn.Conv2d(model_dim, model_dim, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(min(32, model_dim), model_dim),
            nn.GELU(),
            nn.Conv2d(model_dim, model_dim, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(min(32, model_dim), model_dim),
        )
        self.velocity_head = nn.Sequential(
            nn.Conv2d(model_dim, model_dim, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(min(32, model_dim), model_dim),
            nn.GELU(),
            nn.Conv2d(model_dim, 3, kernel_size=1, bias=True),
        )
        nn.init.xavier_uniform_(self.velocity_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.velocity_head[-1].bias)

    def forward(
        self,
        shared_features: torch.Tensor,
        horizon_idx: int,
    ) -> dict[str, torch.Tensor]:
        batch_size, _, height, width = shared_features.shape
        feat = self.input_proj(shared_features)
        horizon_idx = max(0, min(int(horizon_idx), self.horizon_proj.num_embeddings - 1))
        horizon_ids = torch.full((batch_size,), horizon_idx, device=shared_features.device, dtype=torch.long)
        horizon_bias = self.horizon_proj(horizon_ids).view(batch_size, self.model_dim, 1, 1)
        feat = feat + horizon_bias
        feat = feat + self.res_block(feat)
        nu_xyz_map = torch.tanh(self.velocity_head(feat)) * self.velocity_scale
        velocity_reg = nu_xyz_map.pow(2).mean()
        velocity_abs_mean = nu_xyz_map.abs().mean()
        velocity_max = nu_xyz_map.abs().max()
        return {
            "nu_xyz_map": nu_xyz_map.reshape(batch_size, 3, height, width),
            "velocity_reg": velocity_reg,
            "velocity_abs_mean": velocity_abs_mean,
            "velocity_max": velocity_max,
        }


class GaussianDecoder(nn.Module):
    """
    Decoder that converts shared future-token features into 3D Gaussian parameters.

    The shared token block is normalized onto the canonical future-token grid once,
    passed through one shared spatial backbone, then split into two heads:
    - static head: predicts current/base Gaussian parameters + absolute depth
    - velocity head: predicts a shared per-Gaussian nu_xyz velocity field

    Future rollouts reuse the detached static template for scale / opacity / SH /
    rotation while applying horizon-scaled xyz motion from the velocity head.
    """

    def __init__(
        self,
        token_dim: int,
        input_num_tokens: int = 256,
        future_input_num_tokens: int | None = None,
        predict_depth: bool = True,
        use_incremental_depth: bool = True,
        future_prediction_horizon: int = 1,
        use_velocity_future_gaussians: bool = False,
        velocity_world_model_scale: float = 2.0,
        use_action_conditioning: bool = False,
        num_motion_slots: int = 8,
        slot_assignment_temperature: float = 1.0,
        slot_translation_scale: float | None = None,
        slot_rotation_scale: float = 1.0,
        use_future_depth_aux: bool = False,
        future_depth_aux_downsample: int = 2,
        use_future_motion_gate: bool = False,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.input_num_tokens = input_num_tokens
        self.static_input_num_tokens = input_num_tokens
        self.future_input_num_tokens = future_input_num_tokens or input_num_tokens
        self.predict_depth = predict_depth
        self.use_incremental_depth = use_incremental_depth
        self.future_prediction_horizon = max(1, int(future_prediction_horizon))
        self.use_velocity_future_gaussians = use_velocity_future_gaussians
        self.velocity_world_model_scale = float(velocity_world_model_scale)
        self.use_action_conditioning = use_action_conditioning
        self.num_motion_slots = max(1, int(num_motion_slots))
        self.slot_assignment_temperature = float(slot_assignment_temperature)
        self.slot_translation_scale = float(slot_translation_scale if slot_translation_scale is not None else velocity_world_model_scale)
        self.slot_rotation_scale = float(slot_rotation_scale)
        self.use_future_depth_aux = bool(use_future_depth_aux)
        self.future_depth_aux_downsample = max(1, int(future_depth_aux_downsample))
        self.use_future_motion_gate = bool(use_future_motion_gate)

        # Horizon embedding helps the decoder distinguish t+1 vs t+H.
        self.horizon_embed = nn.Embedding(self.future_prediction_horizon, token_dim)

        self.static_grid_size = int(self.static_input_num_tokens ** 0.5)
        self.future_grid_size = int(self.future_input_num_tokens ** 0.5)

        # One shared decoder backbone consumes the canonical 32×32 future-token grid.
        # Static/base reconstruction and future velocity rollout both branch from this shared feature map.
        self.canonical_grid_size = self.future_grid_size
        self.shared_backbone = SharedGaussianBackbone(token_dim=token_dim)
        self.static_head = StaticGaussianHead(
            use_image_fusion=True,
            img_dim=3,
            predict_depth=predict_depth,
        )
        self.future_depth_head = FutureDepthHead(
            future_prediction_horizon=self.future_prediction_horizon,
            downsample_factor=self.future_depth_aux_downsample,
        ) if self.use_future_depth_aux else None

        if use_velocity_future_gaussians:
            self.velocity_head = VelocityGaussianHead(
                self.future_prediction_horizon,
                velocity_scale=self.slot_translation_scale,
            )
        else:
            self.velocity_head = None

    def set_image_fusion_enabled(self, enabled: bool) -> None:
        self.static_head.set_image_fusion_enabled(enabled)

    def _normalize_tokens_to_canonical_grid(self, z: torch.Tensor) -> torch.Tensor:
        """Normalize 256/768/1024-token inputs onto the canonical future-token grid."""
        if z.ndim != 3:
            raise ValueError(f"Expected token tensor [B, N, D], got {tuple(z.shape)}")

        if z.shape[1] == self.static_input_num_tokens * max(1, self.future_prediction_horizon):
            z = z[:, -self.static_input_num_tokens :, :]
        elif z.shape[1] == self.static_input_num_tokens * 3:
            z = z[:, -self.static_input_num_tokens :, :]

        src_tokens = z.shape[1]
        src_grid = int(math.isqrt(src_tokens))
        if src_grid * src_grid != src_tokens:
            raise ValueError(f"Cannot reshape token count {src_tokens} into a square grid")

        if src_grid == self.canonical_grid_size:
            return z

        token_grid = z.transpose(1, 2).reshape(z.shape[0], z.shape[2], src_grid, src_grid)
        token_grid = F.interpolate(
            token_grid,
            size=(self.canonical_grid_size, self.canonical_grid_size),
            mode="bilinear",
            align_corners=False,
        )
        return token_grid.reshape(z.shape[0], z.shape[2], -1).transpose(1, 2)

    def _prepare_shared_token_features(
        self,
        z: torch.Tensor,
        *,
        horizon_idx: int = 0,
    ) -> dict[str, torch.Tensor | int]:
        """Normalize token count and run the shared decoder backbone once.

        Horizon-specific rollout variation now lives in VelocityGaussianHead.
        The shared backbone stays horizon-agnostic so one decoder state can be reused.
        """
        horizon_idx = max(0, min(int(horizon_idx), self.future_prediction_horizon - 1))

        z_canonical = self._normalize_tokens_to_canonical_grid(z)
        token_grid = z_canonical.transpose(1, 2).reshape(
            z_canonical.shape[0], z_canonical.shape[2], self.canonical_grid_size, self.canonical_grid_size
        )
        shared_features = self.shared_backbone(token_grid)
        return {
            "tokens": z_canonical,
            "token_grid": token_grid,
            "shared_features": shared_features,
            "horizon_idx": horizon_idx,
        }

    def prepare_decoder_state(
        self,
        z: torch.Tensor,
        *,
        horizon_idx: int = 0,
    ) -> dict[str, torch.Tensor | int]:
        return self._prepare_shared_token_features(
            z,
            horizon_idx=horizon_idx,
        )

    def predict_future_depth_aux(
        self,
        shared_state: dict[str, torch.Tensor | int],
        *,
        horizon_idx: int = 0,
    ) -> torch.Tensor | None:
        if self.future_depth_head is None:
            return None
        shared_features = shared_state.get("shared_features")
        if not isinstance(shared_features, torch.Tensor):
            return None
        return self.future_depth_head(shared_features, horizon_idx=horizon_idx)

    def _decode_static_from_shared(
        self,
        shared_state: dict[str, torch.Tensor | int],
        *,
        gaussian_adapter=None,
        camera_params=None,
        current_observation=None,
        future_observation=None,
        base_depth: torch.Tensor | None = None,
    ) -> dict:
        shared_features = shared_state["shared_features"]
        assert isinstance(shared_features, torch.Tensor)

        current_frame_img = None
        if future_observation is None and current_observation is not None and gaussian_adapter is not None:
            vggt_inputs = gaussian_adapter.prepare_inputs(
                current_observation, shared_features.device, shared_features.shape[0], is_training=True
            )
            if vggt_inputs is not None:
                current_frame_img = vggt_inputs[:, -1]

        decoder_output = self.static_head(shared_features, images=current_frame_img)
        raw = decoder_output["gaussian_params"]
        rot_raw, scale_raw, opa_raw, sh_raw = raw.split([4, 3, 1, 9], dim=1)

        depth_delta_map = None
        if self.predict_depth and "depth" in decoder_output:
            depth_raw = decoder_output["depth"]
            final_depth = 8.0 * torch.sigmoid(depth_raw.squeeze(1))
        elif base_depth is not None:
            final_depth = base_depth.squeeze(1) if base_depth.ndim == 4 else base_depth
        else:
            raise ValueError("Decoder requires predicted depth or base_depth")

        H_dec, W_dec = final_depth.shape[-2:]
        B = final_depth.shape[0]

        rot_maps = rot_raw.permute(0, 2, 3, 1)
        rot_maps = rot_maps / (rot_maps.norm(dim=-1, keepdim=True) + 1e-8)

        scale_maps = F.softplus(scale_raw.permute(0, 2, 3, 1), beta=1) * 0.01
        opacity_maps = torch.sigmoid(opa_raw.permute(0, 2, 3, 1))

        sh_maps = sh_raw.permute(0, 2, 3, 1)
        sh_mask = self.static_head.sh_mask.view(1, 1, 1, 9)
        sh_maps = sh_maps * sh_mask

        if camera_params is not None and "fx" in camera_params:
            xyz_base = self.depth2pc(
                final_depth,
                fx=camera_params["fx"], fy=camera_params["fy"],
                cx=camera_params["cx"], cy=camera_params["cy"],
                downsample_factor=1,
            )
        else:
            xyz_base = self.depth2pc(
                final_depth,
                fx=221.7025, fy=221.7025,
                cx=128.0, cy=128.0,
                downsample_factor=1,
            )

        N = H_dec * W_dec
        xyz = xyz_base

        rot_flat = rot_maps.reshape(B, N, 4)
        scale_flat = torch.clamp(scale_maps.reshape(B, N, 3), min=1e-7, max=10.0)
        opacity_flat = opacity_maps.reshape(B, N, 1)
        sh_flat = sh_maps.reshape(B, N, 9)

        xyz = torch.clamp(xyz, min=-100.0, max=100.0)
        xyz = torch.where(torch.isnan(xyz) | torch.isinf(xyz), torch.zeros_like(xyz), xyz)

        return {
            "xyz": xyz,
            "scales": scale_flat,
            "opacity": opacity_flat,
            "sh": sh_flat,
            "rotations": rot_flat,
            "depth_map": final_depth.unsqueeze(1),
            "depth_delta_map": None if depth_delta_map is None else depth_delta_map.unsqueeze(1),
        }

    @staticmethod
    def depth2pc(
        depth: torch.Tensor,
        fx: float, fy: float, cx: float, cy: float,
        downsample_factor: int = 1,
    ) -> torch.Tensor:
        """
        Unproject depth map to 3D points using real camera intrinsics.

        Args:
            depth: [B, H, W] depth values
            fx, fy, cx, cy: camera intrinsic parameters
            downsample_factor: if depth was downsampled, scale intrinsics accordingly
        Returns:
            xyz: [B, H*W, 3] camera-space 3D points
        """
        B, H, W = depth.shape
        device, dtype = depth.device, depth.dtype

        fx_s = fx / downsample_factor
        fy_s = fy / downsample_factor
        cx_s = cx / downsample_factor
        cy_s = cy / downsample_factor

        u = torch.arange(0.5, W + 0.5, device=device, dtype=dtype)
        v = torch.arange(0.5, H + 0.5, device=device, dtype=dtype)
        v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")

        x = ((u_grid[None] - cx_s) * depth) / fx_s
        y = ((v_grid[None] - cy_s) * depth) / fy_s
        z = depth

        xyz = torch.stack([x, y, z], dim=-1)
        return xyz.reshape(B, H * W, 3)

    # ------------------------------------------------------------------
    # Main decode entry point
    # ------------------------------------------------------------------
    def decode(
        self,
        z: torch.Tensor,
        future_observation=None,
        gaussian_adapter=None,
        camera_params=None,
        step=None,
        current_observation=None,
        base_depth: torch.Tensor | None = None,
        horizon_idx: int = 0,
        static_reference_params: dict | None = None,
        velocity_time_factor: float = 1.0,
        shared_state: dict[str, torch.Tensor | int] | None = None,
    ):
        """Decode latent tokens → Gaussian parameters."""
        return self._decode_independent(
            z,
            gaussian_adapter=gaussian_adapter,
            camera_params=camera_params,
            current_observation=current_observation,
            future_observation=future_observation,
            step=step,
            base_depth=base_depth,
            horizon_idx=horizon_idx,
            static_reference_params=static_reference_params,
            velocity_time_factor=velocity_time_factor,
            shared_state=shared_state,
        )

    def decode_gaussian_prefix_template(
        self,
        z: torch.Tensor,
        gaussian_adapter,
        current_observation,
        camera_params=None,
        base_depth: torch.Tensor | None = None,
        step=None,
        shared_state: dict[str, torch.Tensor | int] | None = None,
    ):
        """Decode the current/base Gaussian template from shared future-token features."""
        return self._decode_independent(
            z,
            gaussian_adapter=gaussian_adapter,
            camera_params=camera_params,
            current_observation=current_observation,
            future_observation=None,
            step=step,
            base_depth=base_depth,
            horizon_idx=0,
            static_reference_params=None,
            velocity_time_factor=1.0,
            skip_horizon_embedding=True,
            shared_state=shared_state,
        )

    def _decode_velocity_from_static(
        self,
        shared_state: dict[str, torch.Tensor | int],
        static_reference_params: dict,
        velocity_time_factor: float,
        step: int | None,
        horizon_idx: int = 0,
        base_depth: torch.Tensor | None = None,
        current_observation=None,
    ) -> dict:
        """Reuse the base Gaussian template and predict future dynamics via a shared nu_xyz field."""
        shared_features = shared_state["shared_features"]
        assert isinstance(shared_features, torch.Tensor)

        if self.velocity_head is None:
            raise ValueError("Velocity head is not initialized")

        motion_outputs = self.velocity_head(
            shared_features,
            horizon_idx=horizon_idx,
        )

        xyz0 = static_reference_params["xyz"]
        B, Npts, _ = xyz0.shape
        H = W = int(math.sqrt(Npts))
        if H * W != Npts:
            raise ValueError(f"static xyz N={Npts} is not a square grid")

        nu_xyz_map = motion_outputs["nu_xyz_map"].to(dtype=xyz0.dtype)
        motion_gate = None
        if self.use_future_motion_gate:
            future_motion_prior = getattr(current_observation, "future_motion_prior", None)
            if isinstance(future_motion_prior, dict):
                motion_prior = future_motion_prior.get("agent_image")
                if motion_prior is not None:
                    if motion_prior.ndim == 3:
                        motion_prior = motion_prior.unsqueeze(1)
                    motion_prior = F.interpolate(
                        motion_prior.to(device=nu_xyz_map.device, dtype=nu_xyz_map.dtype),
                        size=nu_xyz_map.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).clamp_(0.0, 1.0)
                    gate_floor = 0.1
                    motion_gate = gate_floor + (1.0 - gate_floor) * motion_prior
                    nu_xyz_map = nu_xyz_map * motion_gate
        nu_xyz = nu_xyz_map.permute(0, 2, 3, 1).reshape(B, Npts, 3)
        nu_xyz = torch.where(torch.isnan(nu_xyz) | torch.isinf(nu_xyz), torch.zeros_like(nu_xyz), nu_xyz)
        raw_delta = nu_xyz * float(velocity_time_factor)
        xyz = xyz0 + raw_delta

        xyz = torch.clamp(xyz, min=-100.0, max=100.0)
        xyz = torch.where(torch.isnan(xyz) | torch.isinf(xyz), torch.zeros_like(xyz), xyz)
        raw_delta = torch.where(torch.isnan(raw_delta) | torch.isinf(raw_delta), torch.zeros_like(raw_delta), raw_delta)

        z_cam = xyz[..., 2].reshape(B, H, W)
        depth_map = z_cam.unsqueeze(1).clamp(min=0.0, max=8.0)
        depth_delta_map = None
        if base_depth is not None:
            base_depth_map = base_depth.squeeze(1) if base_depth.ndim == 4 else base_depth
            if base_depth_map.shape[-2:] != depth_map.shape[-2:]:
                base_depth_map = F.interpolate(
                    base_depth_map.unsqueeze(1), size=depth_map.shape[-2:], mode="bilinear", align_corners=False
                ).squeeze(1)
            depth_delta_map = depth_map - base_depth_map.unsqueeze(1)

        scales = static_reference_params["scales"]
        opacity = static_reference_params["opacity"]
        sh = static_reference_params["sh"]
        rotations = static_reference_params["rotations"]

        if step is not None and step % 400 == 0:
            import logging

            static_scale_mean = static_reference_params["scales"].float().mean().item()
            static_scale_max = static_reference_params["scales"].float().max().item()
            gate_mean = motion_gate.mean().item() if motion_gate is not None else 1.0
            gate_min = motion_gate.min().item() if motion_gate is not None else 1.0
            gate_max = motion_gate.max().item() if motion_gate is not None else 1.0
            logging.info(
                f"[VelocityDecoder][h={horizon_idx}][t+~{horizon_idx + 1}] shared_nu: "
                f"velocity_scale={self.slot_translation_scale}, "
                f"time_factor={velocity_time_factor:.4f}, |delta|_mean={raw_delta.abs().mean().item():.6f}, "
                f"|delta|_max={raw_delta.abs().max().item():.6f}, |nu|_mean={motion_outputs['velocity_abs_mean'].item():.6f}, "
                f"|nu|_max={motion_outputs['velocity_max'].item():.6f}, "
                f"motion_gate_mean={gate_mean:.6f}, "
                f"motion_gate_min={gate_min:.6f}, "
                f"motion_gate_max={gate_max:.6f}, "
                f"static_gaussian_scale_mean={static_scale_mean:.6f}, "
                f"static_gaussian_scale_max={static_scale_max:.6f}"
            )

        return {
            "xyz": xyz,
            "scales": scales,
            "opacity": opacity,
            "sh": sh,
            "rotations": rotations,
            "depth_map": depth_map,
            "depth_delta_map": depth_delta_map,
            "raw_delta_xyz": raw_delta,
            "nu_xyz": nu_xyz,
            "slot_probs": None,
            "slot_usage": None,
            "slot_trans_reg": motion_outputs["velocity_reg"],
            "slot_trans": None,
            "slot_rot_6d": None,
            "slot_pivots": None,
        }

    def decode_dynamic_gaussians_from_static(
        self,
        shared_state: dict[str, torch.Tensor | int],
        static_reference_params: dict,
        velocity_time_factor: float,
        step: int | None,
        horizon_idx: int = 0,
        base_depth: torch.Tensor | None = None,
        current_observation=None,
    ) -> dict:
        """Decode shared motion-query features into a constant-velocity dynamic Gaussian update."""
        return self._decode_velocity_from_static(
            shared_state,
            static_reference_params,
            velocity_time_factor,
            step,
            horizon_idx=horizon_idx,
            base_depth=base_depth,
            current_observation=current_observation,
        )

    def _decode_independent(
        self,
        z,
        gaussian_adapter=None,
        camera_params=None,
        current_observation=None,
        future_observation=None,
        step=None,
        base_depth: torch.Tensor | None = None,
        horizon_idx: int = 0,
        static_reference_params: dict | None = None,
        velocity_time_factor: float = 1.0,
        skip_horizon_embedding: bool = False,
        shared_state: dict[str, torch.Tensor | int] | None = None,
    ):
        """Decode VLM tokens into Gaussian parameters using the shared backbone state."""
        if shared_state is None:
            shared_state = self._prepare_shared_token_features(
                z,
                horizon_idx=horizon_idx,
            )

        if (
            self.use_velocity_future_gaussians
            and static_reference_params is not None
            and self.velocity_head is not None
        ):
            return self.decode_dynamic_gaussians_from_static(
                shared_state,
                static_reference_params,
                velocity_time_factor,
                step,
                horizon_idx=horizon_idx,
                base_depth=base_depth,
                current_observation=current_observation,
            )

        gaussian_params = self._decode_static_from_shared(
            shared_state,
            gaussian_adapter=gaussian_adapter,
            camera_params=camera_params,
            current_observation=current_observation,
            future_observation=future_observation,
            base_depth=base_depth,
        )

        if step is not None and step % 100 == 0:
            import logging

            depth_map = gaussian_params["depth_map"]
            scale_flat = gaussian_params["scales"]
            sh_flat = gaussian_params["sh"]
            logging.info(
                f"[IndependentDecoder] Step {step}: "
                f"depth=[{depth_map.min():.3f}, {depth_map.max():.3f}], "
                f"scales=[{scale_flat.min():.3f}, {scale_flat.max():.3f}], "
                f"sh=[{sh_flat.min():.3f}, {sh_flat.max():.3f}], N={scale_flat.shape[1]}"
            )

        return gaussian_params
