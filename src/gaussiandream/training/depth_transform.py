"""
Transform to load depth data from parquet files with depth annotations.
"""
from pathlib import Path
import numpy as np
import torch


class LoadDepthTransform:
    """
    Transform that loads depth data from parquet files.

    Expects the dataset to have 'depth' and 'wrist_depth' columns containing
    serialized float32 depth maps of shape (256, 256).
    """

    def __init__(
        self,
        use_depth: bool = True,
        depth_key: str = "depth",
        wrist_depth_key: str = "wrist_depth",
    ):
        """
        Args:
            use_depth: Whether to load depth data
            depth_key: Key name for the main camera depth in the parquet file
            wrist_depth_key: Key name for the wrist camera depth in the parquet file
        """
        self.use_depth = use_depth
        self.depth_key = depth_key
        self.wrist_depth_key = wrist_depth_key

    def _decode_depth_map(self, depth_data, depth_shape: tuple[int, int] | None = None) -> np.ndarray:
        if isinstance(depth_data, bytes):
            depth_array = np.frombuffer(depth_data, dtype=np.float32)
            if depth_shape is None:
                side = int(round(np.sqrt(depth_array.size)))
                if side * side != depth_array.size:
                    raise ValueError(
                        f"Cannot infer square depth shape from {depth_array.size} float32 values"
                    )
                depth_shape = (side, side)
            return depth_array.reshape(depth_shape)
        return np.array(depth_data, dtype=np.float32)

    def __call__(self, sample: dict) -> dict:
        """
        Load depth data from the sample if available.

        Args:
            sample: Dictionary containing the data sample

        Returns:
            Modified sample with depth data added
        """
        import logging

        if not self.use_depth:
            return sample

        # Try to load main camera depth (4 frames to match image temporal dimension)
        if self.depth_key in sample:
            depth_bytes = sample[self.depth_key]

            if isinstance(depth_bytes, (list, tuple)):
                # Multiple frames (expected: 4 frames [t-2, t-1, t, t+1])
                depth_frames = []
                for depth_data in depth_bytes:
                    depth_frames.append(self._decode_depth_map(depth_data))

                # Stack frames: [T, H, W] -> [T, 1, H, W] (add channel dimension)
                depth_tensor = torch.from_numpy(np.stack([f.copy() for f in depth_frames], axis=0))
                depth_tensor = depth_tensor.unsqueeze(1)  # [T, H, W] -> [T, 1, H, W]
            else:
                # Single frame (fallback)
                depth_map = self._decode_depth_map(depth_bytes)

                # [H, W] -> [1, 1, H, W]
                depth_tensor = torch.from_numpy(depth_map.copy()).unsqueeze(0).unsqueeze(0)

            sample["observation/depth"] = depth_tensor

        # Try to load wrist camera depth (optional, 4 frames to match image temporal dimension)
        if self.wrist_depth_key in sample:
            wrist_depth_bytes = sample[self.wrist_depth_key]
            if isinstance(wrist_depth_bytes, (list, tuple)):
                # Multiple frames (expected: 4 frames [t-2, t-1, t, t+1])
                depth_frames = []
                for depth_data in wrist_depth_bytes:
                    depth_frames.append(self._decode_depth_map(depth_data))

                # Stack frames: [T, H, W] -> [T, 1, H, W]
                wrist_depth_tensor = torch.from_numpy(np.stack([f.copy() for f in depth_frames], axis=0))
                wrist_depth_tensor = wrist_depth_tensor.unsqueeze(1)
            else:
                # Single frame (fallback)
                depth_map = self._decode_depth_map(wrist_depth_bytes)

                wrist_depth_tensor = torch.from_numpy(depth_map.copy()).unsqueeze(0).unsqueeze(0)

            sample["observation/wrist_depth"] = wrist_depth_tensor

        return sample


class LoadFlowTransform:
    """Load precomputed pseudo scene flow sidecars for LIBERO episodes."""

    def __init__(self, flow_root: str | None, future_horizon: int = 1):
        self.flow_root = Path(flow_root) if flow_root else None
        self.future_horizon = max(1, int(future_horizon))
        self._episode_cache: dict[Path, dict[str, np.ndarray]] = {}

    def _coerce_scalar_int(self, value) -> int:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            if value.size != 1:
                raise ValueError(f"Expected scalar metadata, got array with shape {value.shape}")
            value = value.reshape(-1)[0]
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"Expected scalar metadata, got sequence of length {len(value)}")
            value = value[0]
        return int(value)

    def _sidecar_path(self, episode_index: int) -> Path:
        if self.flow_root is None:
            raise ValueError("flow_root is not configured")
        chunk_name = f"chunk-{episode_index // 1000:03d}"
        return self.flow_root / "data" / chunk_name / f"episode_{episode_index:06d}.npz"

    def _load_episode(self, sidecar_path: Path) -> dict[str, np.ndarray]:
        cached = self._episode_cache.get(sidecar_path)
        if cached is not None:
            return cached

        if not sidecar_path.exists():
            raise FileNotFoundError(f"Missing flow sidecar: {sidecar_path}")

        with np.load(sidecar_path, allow_pickle=False) as data:
            cached = {
                "flow_2d": data["flow_2d"].astype(np.float32, copy=False),
                "flow_3d": data["flow_3d"].astype(np.float32, copy=False),
                "valid_mask": data["valid_mask"].astype(np.bool_, copy=False),
            }

        self._episode_cache[sidecar_path] = cached
        if len(self._episode_cache) > 8:
            oldest_key = next(iter(self._episode_cache))
            self._episode_cache.pop(oldest_key, None)
        return cached

    def _bilinear_sample_field(self, field: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        height, width = field.shape[:2]
        x0 = np.floor(xs).astype(np.int32)
        y0 = np.floor(ys).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1

        wx = xs - x0.astype(np.float32)
        wy = ys - y0.astype(np.float32)

        x0_clip = np.clip(x0, 0, width - 1)
        x1_clip = np.clip(x1, 0, width - 1)
        y0_clip = np.clip(y0, 0, height - 1)
        y1_clip = np.clip(y1, 0, height - 1)

        wa = (1.0 - wx) * (1.0 - wy)
        wb = (1.0 - wx) * wy
        wc = wx * (1.0 - wy)
        wd = wx * wy

        return (
            wa[..., None] * field[y0_clip, x0_clip]
            + wb[..., None] * field[y1_clip, x0_clip]
            + wc[..., None] * field[y0_clip, x1_clip]
            + wd[..., None] * field[y1_clip, x1_clip]
        ).astype(np.float32)

    def _sample_mask(self, mask: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        height, width = mask.shape
        in_bounds = (xs >= 0.0) & (xs <= (width - 1)) & (ys >= 0.0) & (ys <= (height - 1))
        xi = np.rint(xs).astype(np.int32)
        yi = np.rint(ys).astype(np.int32)
        xi = np.clip(xi, 0, width - 1)
        yi = np.clip(yi, 0, height - 1)
        sampled = np.zeros_like(mask, dtype=np.bool_)
        sampled[in_bounds] = mask[yi[in_bounds], xi[in_bounds]]
        return sampled

    def _compose_anchor_flow_targets(
        self,
        flow_2d: np.ndarray,
        flow_3d: np.ndarray,
        valid_mask: np.ndarray,
        frame_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = flow_3d.shape[1:3]
        flow_targets = np.zeros((self.future_horizon, height, width, 3), dtype=np.float32)
        mask_targets = np.zeros((self.future_horizon, height, width), dtype=np.bool_)

        max_pairs = flow_3d.shape[0]
        if frame_index >= max_pairs:
            return flow_targets, mask_targets

        # Compose long-range anchor flow targets by tracking each anchor pixel through
        # consecutive 2D flows and summing the sampled 3D displacements along that path.
        # This converts pairwise t->t+1 sidecars into anchor-aligned t->t+h supervision.
        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
            indexing="xy",
        )
        track_x = grid_x.copy()
        track_y = grid_y.copy()
        accumulated_flow = np.zeros((height, width, 3), dtype=np.float32)
        trajectory_valid = np.ones((height, width), dtype=np.bool_)

        max_horizon = min(self.future_horizon, max_pairs - frame_index)
        for horizon_idx in range(max_horizon):
            step_index = frame_index + horizon_idx

            sampled_flow_2d = self._bilinear_sample_field(flow_2d[step_index], track_x, track_y)
            sampled_flow_3d = self._bilinear_sample_field(flow_3d[step_index], track_x, track_y)
            sampled_mask = self._sample_mask(valid_mask[step_index], track_x, track_y)

            sampled_flow_2d = np.where(sampled_mask[..., None], sampled_flow_2d, 0.0)
            sampled_flow_3d = np.where(sampled_mask[..., None], sampled_flow_3d, 0.0)

            accumulated_flow = accumulated_flow + sampled_flow_3d
            trajectory_valid = trajectory_valid & sampled_mask

            flow_targets[horizon_idx] = np.where(trajectory_valid[..., None], accumulated_flow, 0.0)
            mask_targets[horizon_idx] = trajectory_valid

            track_x = track_x + sampled_flow_2d[..., 0]
            track_y = track_y + sampled_flow_2d[..., 1]

        return flow_targets, mask_targets

    def __call__(self, sample: dict) -> dict:
        if self.flow_root is None:
            return sample

        if "episode_index" not in sample or "frame_index" not in sample:
            return sample

        episode_index = self._coerce_scalar_int(sample["episode_index"])
        frame_index = self._coerce_scalar_int(sample["frame_index"])
        episode_data = self._load_episode(self._sidecar_path(episode_index))

        flow_2d = episode_data["flow_2d"]
        flow_3d = episode_data["flow_3d"]
        valid_mask = episode_data["valid_mask"]
        if frame_index < 0 or frame_index >= flow_3d.shape[0]:
            height, width = flow_3d.shape[1:3]
            sample["observation/flow_3d"] = np.zeros((self.future_horizon, height, width, 3), dtype=np.float32)
            sample["observation/flow_valid_mask"] = np.zeros((self.future_horizon, height, width), dtype=np.bool_)
            return sample

        flow_targets, mask_targets = self._compose_anchor_flow_targets(flow_2d, flow_3d, valid_mask, frame_index)

        sample["observation/flow_3d"] = flow_targets
        sample["observation/flow_valid_mask"] = mask_targets
        return sample
