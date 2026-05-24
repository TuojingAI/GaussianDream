#!/usr/bin/env python3
"""Generate pseudo scene flow sidecars for RoboCasa legacy episode parquet files.

Outputs per episode:
- flow_2d: [T-1, H, W, 2]
- flow_3d: [T-1, H, W, 3]
- valid_mask: [T-1, H, W]

This mirrors `generate_libero_pseudo_scene_flow.py`, but reads RoboCasa's legacy
episode parquet layout produced by `convert_robocasa_h50_to_legacy.py`.
"""

from __future__ import annotations

import argparse
import io
import math
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

try:
    import torch
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
except ImportError:  # pragma: no cover
    torch = None
    Raft_Large_Weights = None
    raft_large = None


SCHEMA_VERSION = 2
IMAGE_KEY = "observation.images.robot0_agentview_left_image"
DEPTH_KEY = "observation.depth"


class FlowBackend(Protocol):
    def __call__(self, image_t: np.ndarray, image_t1: np.ndarray) -> np.ndarray: ...


def decode_image(cell) -> np.ndarray:
    if isinstance(cell, dict):
        image_bytes = cell["bytes"]
    else:
        image_bytes = cell
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.asarray(image)


def infer_depth_shape(depth_bytes: bytes) -> tuple[int, int]:
    num_values = len(depth_bytes) // np.dtype(np.float32).itemsize
    side = int(round(math.sqrt(num_values)))
    if side * side != num_values:
        raise ValueError(f"Cannot infer square depth shape from {num_values} float32 values")
    return side, side


def decode_depth(depth_bytes: bytes, depth_shape: tuple[int, int] | None = None) -> np.ndarray:
    if depth_shape is None:
        depth_shape = infer_depth_shape(depth_bytes)
    return np.frombuffer(depth_bytes, dtype=np.float32).reshape(depth_shape).astype(np.float32)


def compute_intrinsics(image_height: int, image_width: int, fovy_deg: float) -> np.ndarray:
    fovy_rad = math.radians(fovy_deg)
    fy = (image_height / 2.0) / math.tan(fovy_rad / 2.0)
    fx = fy
    cx = image_width / 2.0
    cy = image_height / 2.0
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def compute_farneback_flow(image_t: np.ndarray, image_t1: np.ndarray) -> np.ndarray:
    gray_t = cv2.cvtColor(image_t, cv2.COLOR_RGB2GRAY)
    gray_t1 = cv2.cvtColor(image_t1, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray_t,
        gray_t1,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    return flow.astype(np.float32)


class RAFTBackend:
    def __init__(self, device: str) -> None:
        if torch is None or Raft_Large_Weights is None or raft_large is None:
            raise ImportError("RAFT backend requires torch and torchvision optical flow support.")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.weights = Raft_Large_Weights.DEFAULT
        self.transforms = self.weights.transforms()
        self.model = raft_large(weights=self.weights, progress=True).to(self.device).eval()

    @torch.inference_mode()
    def __call__(self, image_t: np.ndarray, image_t1: np.ndarray) -> np.ndarray:
        tensor_t = torch.from_numpy(np.ascontiguousarray(image_t).copy()).permute(2, 0, 1).unsqueeze(0)
        tensor_t1 = torch.from_numpy(np.ascontiguousarray(image_t1).copy()).permute(2, 0, 1).unsqueeze(0)
        tensor_t, tensor_t1 = self.transforms(tensor_t, tensor_t1)
        flow_predictions = self.model(tensor_t.to(self.device), tensor_t1.to(self.device))
        return flow_predictions[-1][0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)


def create_flow_backend(name: str, device: str) -> FlowBackend:
    if name == "farneback":
        return compute_farneback_flow
    if name == "raft":
        return RAFTBackend(device)
    raise ValueError(f"Unsupported backend: {name}")


def bilinear_sample_scalar(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = image.shape
    x0 = np.floor(xs).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(ys).astype(np.int32)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)

    wa = (x1 - xs) * (y1 - ys)
    wb = (x1 - xs) * (ys - y0)
    wc = (xs - x0) * (y1 - ys)
    wd = (xs - x0) * (ys - y0)

    return (
        wa * image[y0, x0]
        + wb * image[y1, x0]
        + wc * image[y0, x1]
        + wd * image[y1, x1]
    ).astype(np.float32)


def backproject_camera(uv_x: np.ndarray, uv_y: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    x = (uv_x - cx) * depth / fx
    y = (uv_y - cy) * depth / fy
    z = depth
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def build_valid_mask(
    depth_t: np.ndarray,
    depth_t1_sampled: np.ndarray,
    warped_x: np.ndarray,
    warped_y: np.ndarray,
    min_depth: float,
    max_depth: float | None,
) -> np.ndarray:
    h, w = depth_t.shape
    in_bounds = (warped_x >= 0.0) & (warped_x <= (w - 1)) & (warped_y >= 0.0) & (warped_y <= (h - 1))
    valid = in_bounds & np.isfinite(depth_t) & np.isfinite(depth_t1_sampled) & (depth_t > min_depth) & (depth_t1_sampled > min_depth)
    if max_depth is not None:
        valid &= depth_t < max_depth
        valid &= depth_t1_sampled < max_depth
    return valid


def process_episode(
    parquet_path: Path,
    output_path: Path,
    intrinsics: np.ndarray,
    max_frames: int | None,
    min_depth: float,
    max_depth: float | None,
    flow_backend: FlowBackend,
    backend_name: str,
) -> dict:
    table = pq.read_table(parquet_path)
    data = table.to_pydict()

    num_rows = len(data["timestamp"])
    if max_frames is not None:
        num_rows = min(num_rows, max_frames)
    if num_rows < 2:
        raise ValueError(f"Need at least 2 frames, got {num_rows}")

    depth_shape = infer_depth_shape(data[DEPTH_KEY][0])
    flow_2d_list = []
    flow_3d_list = []
    valid_mask_list = []

    grid_y, grid_x = np.meshgrid(
        np.arange(depth_shape[0], dtype=np.float32),
        np.arange(depth_shape[1], dtype=np.float32),
        indexing="ij",
    )

    episode_task_index = int(data["task_index"][0]) if "task_index" in data and data["task_index"] else None
    episode_index = int(data["episode_index"][0]) if "episode_index" in data and data["episode_index"] else None

    for i in range(num_rows - 1):
        image_t = decode_image(data[IMAGE_KEY][i])
        image_t1 = decode_image(data[IMAGE_KEY][i + 1])
        depth_t = decode_depth(data[DEPTH_KEY][i], depth_shape)
        depth_t1 = decode_depth(data[DEPTH_KEY][i + 1], depth_shape)

        flow_2d = flow_backend(image_t, image_t1)
        warped_x = grid_x + flow_2d[..., 0]
        warped_y = grid_y + flow_2d[..., 1]

        sampled_depth_t1 = bilinear_sample_scalar(depth_t1, warped_x, warped_y)
        valid_mask = build_valid_mask(depth_t, sampled_depth_t1, warped_x, warped_y, min_depth, max_depth)

        points_t = backproject_camera(grid_x, grid_y, depth_t, intrinsics)
        points_t1 = backproject_camera(warped_x, warped_y, sampled_depth_t1, intrinsics)
        flow_3d = (points_t1 - points_t).astype(np.float32)
        flow_3d[~valid_mask] = 0.0
        flow_2d[~valid_mask] = 0.0

        flow_2d_list.append(flow_2d)
        flow_3d_list.append(flow_3d)
        valid_mask_list.append(valid_mask)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        flow_2d=np.stack(flow_2d_list, axis=0).astype(np.float32),
        flow_3d=np.stack(flow_3d_list, axis=0).astype(np.float32),
        valid_mask=np.stack(valid_mask_list, axis=0).astype(np.bool_),
        intrinsics=intrinsics.astype(np.float32),
        camera_name=np.array("robot0_agentview_left"),
        depth_mode=np.array("metric"),
        rotate_180=np.array(False),
        episode_index=np.array(-1 if episode_index is None else episode_index),
        task_index=np.array(-1 if episode_task_index is None else episode_task_index),
        source_parquet=np.array(str(parquet_path)),
        backend=np.array(backend_name),
        schema_version=np.array(SCHEMA_VERSION, dtype=np.int32),
    )

    valid_ratio = float(np.mean(np.stack(valid_mask_list, axis=0)))
    return {
        "output_path": str(output_path),
        "num_pairs": len(flow_2d_list),
        "valid_ratio": valid_ratio,
        "episode_index": episode_index,
        "task_index": episode_task_index,
        "backend": backend_name,
    }


def apply_shard(paths: list[Path], shard_index: int, num_shards: int) -> list[Path]:
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    return paths[shard_index::num_shards]


def resolve_episode_paths(
    dataset_root: Path,
    parquet_path: str | None,
    chunk: str | None,
    limit: int | None,
    *,
    shard_index: int = 0,
    num_shards: int = 1,
) -> list[Path]:
    if parquet_path is not None:
        return [Path(parquet_path)]

    data_dir = dataset_root / "data"
    if chunk is not None:
        paths = sorted((data_dir / chunk).glob("episode_*.parquet"))
    else:
        paths = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
    if limit is not None:
        paths = paths[:limit]
    return apply_shard(paths, shard_index, num_shards)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pseudo scene flow for RoboCasa legacy parquet episodes")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for .npz sidecars")
    parser.add_argument("--parquet-path", type=str, default=None, help="Optional single parquet episode path")
    parser.add_argument("--chunk", type=str, default=None, help="Optional chunk name, e.g. chunk-000")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of episodes")
    parser.add_argument("--shard-index", type=int, default=0, help="Shard index for parallel generation (0-based)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards for parallel generation")
    parser.add_argument("--backend", type=str, choices=["raft", "farneback"], default="raft")
    parser.add_argument("--device", type=str, default="auto", help="Device for RAFT inference: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--fovy-deg", type=float, default=60.0, help="Vertical field of view for agentview-left")
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--image-width", type=int, default=128)
    parser.add_argument("--min-depth", type=float, default=0.01)
    parser.add_argument("--max-depth", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None, help="Optional max frames per episode for debugging")
    parser.add_argument("--skip-existing", action="store_true", help="Skip episodes whose sidecar already exists")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    episode_paths = resolve_episode_paths(
        dataset_root,
        args.parquet_path,
        args.chunk,
        args.limit,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    print(
        f"Shard {args.shard_index}/{args.num_shards}: processing {len(episode_paths)} episode(s)"
        + (f" from {args.chunk}" if args.chunk is not None else "")
    )

    intrinsics = compute_intrinsics(args.image_height, args.image_width, args.fovy_deg)
    flow_backend = create_flow_backend(args.backend, args.device)

    for parquet_path in episode_paths:
        rel = parquet_path.relative_to(dataset_root)
        output_path = output_dir / rel.with_suffix(".npz")
        if args.skip_existing and output_path.exists():
            print(f"Skipped {parquet_path.name}: existing sidecar -> {output_path}")
            continue
        summary = process_episode(
            parquet_path=parquet_path,
            output_path=output_path,
            intrinsics=intrinsics,
            max_frames=args.max_frames,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            flow_backend=flow_backend,
            backend_name=args.backend,
        )
        print(
            f"Processed {parquet_path.name}: backend={summary['backend']} pairs={summary['num_pairs']} "
            f"valid_ratio={summary['valid_ratio']:.4f} -> {output_path}"
        )


if __name__ == "__main__":
    main()
