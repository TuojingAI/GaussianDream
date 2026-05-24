#!/usr/bin/env python3
"""Generate pseudo scene flow sidecars for LIBERO parquet episodes.

Outputs per episode:
- flow_2d: [T-1, H, W, 2]
- flow_3d: [T-1, H, W, 3]
- valid_mask: [T-1, H, W]
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

try:
    import torch
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
except ImportError:  # pragma: no cover - handled when RAFT backend is requested.
    torch = None
    Raft_Large_Weights = None
    raft_large = None


SCHEMA_VERSION = 2


class FlowBackend(Protocol):
    def __call__(self, image_t: np.ndarray, image_t1: np.ndarray) -> np.ndarray: ...


def decode_image(cell) -> np.ndarray:
    if isinstance(cell, dict):
        image_bytes = cell["bytes"]
    else:
        image_bytes = cell
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.asarray(image)


def decode_depth(depth_bytes, depth_shape, depth_dtype) -> np.ndarray:
    dtype = np.dtype(depth_dtype)
    return np.frombuffer(depth_bytes, dtype=dtype).reshape(depth_shape).astype(np.float32)


def maybe_rotate_180(arr: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return arr
    return np.ascontiguousarray(arr[::-1, ::-1])


def load_camera_payload(camera_params_path: Path) -> dict:
    with open(camera_params_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_camera_entry(payload: dict, episode_task_index: int | None, camera_name: str) -> dict:
    if "entries" in payload and episode_task_index is not None:
        for entry in payload["entries"]:
            if entry.get("task_id") == episode_task_index:
                cameras = entry.get("cameras", {})
                if camera_name in cameras and "error" not in cameras[camera_name]:
                    return cameras[camera_name]
    alias = {"agentview": "agent", "robot0_eye_in_hand": "wrist"}.get(camera_name, camera_name)
    if alias in payload:
        return payload[alias]
    cameras = payload.get("cameras", {})
    if camera_name in cameras:
        return cameras[camera_name]
    raise KeyError(f"Camera parameters for {camera_name} not found")


def infer_depth_mode(depth_map: np.ndarray) -> str:
    finite = depth_map[np.isfinite(depth_map)]
    if finite.size == 0:
        return "metric"
    if finite.min() >= 0.0 and finite.max() <= 1.0:
        return "mujoco_normalized"
    return "metric"


def depth_to_metric(depth_map: np.ndarray, depth_mode: str, camera_entry: dict | None) -> np.ndarray:
    if depth_mode == "metric":
        return depth_map.astype(np.float32)
    if depth_mode == "mujoco_normalized":
        if camera_entry is None:
            raise ValueError("camera_entry required for mujoco_normalized depth conversion")
        near = float(camera_entry["near"])
        far = float(camera_entry["far"])
        clipped = np.clip(depth_map, 0.0, 1.0)
        return near / (1.0 - clipped * (1.0 - near / far))
    raise ValueError(f"Unsupported depth mode: {depth_mode}")


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
            raise ImportError(
                "RAFT backend requires torch and torchvision with optical flow models available."
            )
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
    camera_entry: dict,
    rotate_180: bool,
    depth_mode: str,
    min_depth: float,
    max_depth: float | None,
    max_frames: int | None,
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

    depth_shape = tuple(data["depth_shape"][0])
    depth_dtype = data["depth_dtype"][0]
    intrinsics = np.asarray(camera_entry["intrinsics"], dtype=np.float32)

    flow_2d_list = []
    flow_3d_list = []
    valid_mask_list = []

    grid_y, grid_x = np.meshgrid(np.arange(depth_shape[0], dtype=np.float32), np.arange(depth_shape[1], dtype=np.float32), indexing="ij")

    episode_task_index = int(data["task_index"][0]) if "task_index" in data and len(data["task_index"]) else None
    episode_index = int(data["episode_index"][0]) if "episode_index" in data and len(data["episode_index"]) else None

    auto_mode = depth_mode
    if depth_mode == "auto":
        first_depth = decode_depth(data["depth"][0], depth_shape, depth_dtype)
        auto_mode = infer_depth_mode(first_depth)

    for i in range(num_rows - 1):
        image_t = maybe_rotate_180(decode_image(data["image"][i]), rotate_180)
        image_t1 = maybe_rotate_180(decode_image(data["image"][i + 1]), rotate_180)
        depth_t = maybe_rotate_180(decode_depth(data["depth"][i], depth_shape, depth_dtype), rotate_180)
        depth_t1 = maybe_rotate_180(decode_depth(data["depth"][i + 1], depth_shape, depth_dtype), rotate_180)

        depth_t = depth_to_metric(depth_t, auto_mode, camera_entry)
        depth_t1 = depth_to_metric(depth_t1, auto_mode, camera_entry)

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
        camera_name=np.array(camera_entry.get("camera_name", "agentview")),
        depth_mode=np.array(auto_mode),
        rotate_180=np.array(bool(rotate_180)),
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
        "depth_mode": auto_mode,
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
    parser = argparse.ArgumentParser(description="Generate pseudo scene flow for LIBERO parquet episodes")
    parser.add_argument("--dataset-root", type=str, default=None, help="Root of depth-augmented LIBERO dataset")
    parser.add_argument("--parquet-path", type=str, default=None, help="Optional single parquet episode path")
    parser.add_argument("--chunk", type=str, default=None, help="Optional chunk name, e.g. chunk-000")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of episodes")
    parser.add_argument("--shard-index", type=int, default=0, help="Shard index for parallel generation (0-based)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards for parallel generation")
    parser.add_argument("--camera-params", type=str, required=True, help="Path to extracted camera params JSON")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for .npz sidecars")
    parser.add_argument("--camera-name", type=str, default="agentview")
    parser.add_argument("--backend", type=str, choices=["raft", "farneback"], default="raft")
    parser.add_argument("--device", type=str, default="auto", help="Device for RAFT inference: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--depth-mode", type=str, choices=["metric", "mujoco_normalized", "auto"], default="metric")
    parser.add_argument("--rotate-180", action="store_true", help="Rotate RGB and depth by 180 degrees before processing")
    parser.add_argument("--min-depth", type=float, default=0.01)
    parser.add_argument("--max-depth", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None, help="Optional max frames per episode for debugging")
    parser.add_argument("--skip-existing", action="store_true", help="Skip episodes whose sidecar already exists")
    args = parser.parse_args()

    dataset_root_value = args.dataset_root or os.environ.get("LIBERO_DATA_WITH_DEPTH_ROOT")
    if dataset_root_value is None:
        raise ValueError("Pass --dataset-root or set LIBERO_DATA_WITH_DEPTH_ROOT.")
    dataset_root = Path(dataset_root_value)
    output_dir = Path(args.output_dir)
    camera_payload = load_camera_payload(Path(args.camera_params))
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

    flow_backend = create_flow_backend(args.backend, args.device)

    for parquet_path in episode_paths:
        table = pq.read_table(parquet_path, columns=["task_index"])
        task_values = table["task_index"].to_pylist()
        episode_task_index = int(task_values[0]) if task_values else None
        camera_entry = get_camera_entry(camera_payload, episode_task_index, args.camera_name)

        rel = parquet_path.relative_to(dataset_root)
        output_path = output_dir / rel.with_suffix(".npz")
        if args.skip_existing and output_path.exists():
            print(f"Skipped {parquet_path.name}: existing sidecar -> {output_path}")
            continue
        summary = process_episode(
            parquet_path=parquet_path,
            output_path=output_path,
            camera_entry=camera_entry,
            rotate_180=args.rotate_180,
            depth_mode=args.depth_mode,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            max_frames=args.max_frames,
            flow_backend=flow_backend,
            backend_name=args.backend,
        )
        print(
            f"Processed {parquet_path.name}: backend={summary['backend']} pairs={summary['num_pairs']} "
            f"valid_ratio={summary['valid_ratio']:.4f} depth_mode={summary['depth_mode']} -> {output_path}"
        )


if __name__ == "__main__":
    main()
