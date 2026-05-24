#!/usr/bin/env python3
"""Generate pseudo depth maps for a LeRobot-formatted RoboCasa dataset.

This mirrors the existing LIBERO depth generation flow, but adapts to RoboCasa H50's
LeRobot v3 layout:
- tabular data in `data/chunk-*/file-*.parquet`
- episode metadata in `meta/episodes/chunk-*/file-*.parquet`
- RGB videos in `videos/<camera_key>/chunk-*/file-*.mp4`

The script writes a new dataset root with:
- copied metadata
- symlinked videos
- parquet files augmented with `observation.depth` and `observation.wrist_depth`

Example:
  python scripts/generate_depth_for_robocasa.py \
      --dataset-root <ROBOCASA_H50_ROOT> \
      --output-root <ROBOCASA_H50_WITH_DEPTH_ROOT> \
      --device cuda
"""

from __future__ import annotations

import argparse
import av
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_depth_for_libero import load_depth_anything_v2


MAIN_CAMERA_KEY = "observation.images.robot0_agentview_left_image"
WRIST_CAMERA_KEY = "observation.images.robot0_eye_in_hand_image"


def _preprocess_image_array(image: np.ndarray, target_size: int = 518) -> tuple[torch.Tensor, tuple[int, int]]:
    pil_image = Image.fromarray(image)
    width, height = pil_image.size
    scale = target_size / max(height, width)
    new_h, new_w = int(height * scale), int(width * scale)
    new_h = max(14, (new_h // 14) * 14)
    new_w = max(14, (new_w // 14) * 14)
    pil_image = pil_image.resize((new_w, new_h), Image.BILINEAR)

    image_np = np.array(pil_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    return image_tensor, (height, width)


@torch.no_grad()
def predict_depth_batch(model, frames: list[np.ndarray], device: str) -> list[np.ndarray]:
    tensors = []
    original_sizes = []
    for frame in frames:
        image_tensor, original_size = _preprocess_image_array(frame)
        tensors.append(image_tensor)
        original_sizes.append(original_size)

    batch = torch.stack(tensors, dim=0).to(device)
    depth = model(batch)  # [B, H, W]

    outputs: list[np.ndarray] = []
    for i, original_size in enumerate(original_sizes):
        resized = F.interpolate(
            depth[i : i + 1].unsqueeze(0),
            size=original_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        outputs.append(resized.cpu().numpy().astype(np.float32))
    return outputs


class SequentialVideoReader:
    def __init__(self) -> None:
        self._video_path: Path | None = None
        self._container: av.container.InputContainer | None = None
        self._frame_iter = None
        self._current_frame = 0

    def close(self) -> None:
        if self._container is not None:
            self._container.close()
        self._container = None
        self._frame_iter = None
        self._video_path = None
        self._current_frame = 0

    def _open(self, video_path: Path) -> None:
        self.close()
        self._container = av.open(str(video_path))
        self._frame_iter = self._container.decode(video=0)
        self._video_path = video_path
        self._current_frame = 0

    def _skip_to(self, start_frame: int) -> None:
        if self._frame_iter is None:
            raise RuntimeError("Video iterator is not initialized")
        while self._current_frame < start_frame:
            try:
                next(self._frame_iter)
            except StopIteration as exc:
                raise RuntimeError(
                    f"Unexpected EOF while seeking {self._video_path} to frame {start_frame}"
                ) from exc
            self._current_frame += 1

    def read_range(self, video_path: Path, start_frame: int, length: int) -> list[np.ndarray]:
        if self._video_path != video_path:
            self._open(video_path)

        if self._frame_iter is None:
            raise RuntimeError("Video iterator is not initialized")

        if self._current_frame > start_frame:
            self._open(video_path)

        if self._current_frame < start_frame:
            self._skip_to(start_frame)

        frames: list[np.ndarray] = []
        for _ in range(length):
            try:
                frame = next(self._frame_iter)
            except StopIteration as exc:
                raise RuntimeError(
                    f"Unexpected EOF while reading {video_path} from frame {start_frame} for length {length}"
                ) from exc
            self._current_frame += 1
            frames.append(frame.to_ndarray(format="rgb24"))
        return frames


def _video_file_path(dataset_root: Path, camera_key: str, chunk_index: int, file_index: int) -> Path:
    return dataset_root / "videos" / camera_key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"


def _augment_info_json(input_root: Path, output_root: Path) -> None:
    info_path = input_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.depth"] = {
        "dtype": "binary",
        "shape": [128, 128],
        "names": ["height", "width"],
    }
    info["features"]["observation.wrist_depth"] = {
        "dtype": "binary",
        "shape": [128, 128],
        "names": ["height", "width"],
    }
    (output_root / "meta" / "info.json").write_text(json.dumps(info, indent=4))


def _ensure_tasks_jsonl(meta_dir: Path) -> None:
    tasks_jsonl = meta_dir / "tasks.jsonl"
    if tasks_jsonl.exists():
        return

    tasks_parquet = meta_dir / "tasks.parquet"
    if not tasks_parquet.exists():
        return

    tasks_df = pd.read_parquet(tasks_parquet)
    with tasks_jsonl.open("w") as f:
        for task_text, task_index in zip(tasks_df.index.tolist(), tasks_df["task_index"].tolist(), strict=True):
            f.write(json.dumps({"task_index": int(task_index), "task": str(task_text)}) + "\n")


def _prepare_output_root(dataset_root: Path, output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output root already exists: {output_root}")
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dataset_root / "README.md", output_root / "README.md")
    shutil.copytree(dataset_root / "meta", output_root / "meta")
    os.symlink(dataset_root / "videos", output_root / "videos", target_is_directory=True)
    (output_root / "data").mkdir(parents=True, exist_ok=True)
    _augment_info_json(dataset_root, output_root)
    _ensure_tasks_jsonl(output_root / "meta")


def _process_data_file(
    dataset_root: Path,
    output_root: Path,
    data_file: Path,
    episodes_file: Path,
    model,
    device: str,
    batch_size: int,
    max_episodes: int | None,
    fps: float,
) -> None:
    data_df = pd.read_parquet(data_file)
    episodes_df = pd.read_parquet(episodes_file).sort_values("episode_index").reset_index(drop=True)
    if max_episodes is not None:
        episodes_df = episodes_df.head(max_episodes)

    left_reader = SequentialVideoReader()
    wrist_reader = SequentialVideoReader()

    output_data_file = output_root / "data" / data_file.relative_to(dataset_root / "data")
    output_data_file.parent.mkdir(parents=True, exist_ok=True)

    writer: pq.ParquetWriter | None = None

    try:
        for _, episode in tqdm(
            episodes_df.iterrows(),
            total=len(episodes_df),
            desc=f"Processing {data_file.relative_to(dataset_root)}",
        ):
            start = int(episode["dataset_from_index"])
            end = int(episode["dataset_to_index"])
            length = int(episode["length"])
            if end - start != length:
                raise ValueError(
                    f"Episode {episode['episode_index']} has inconsistent slice: {start=} {end=} {length=}"
                )

            main_video_path = _video_file_path(
                dataset_root,
                MAIN_CAMERA_KEY,
                int(episode[f"videos/{MAIN_CAMERA_KEY}/chunk_index"]),
                int(episode[f"videos/{MAIN_CAMERA_KEY}/file_index"]),
            )
            wrist_video_path = _video_file_path(
                dataset_root,
                WRIST_CAMERA_KEY,
                int(episode[f"videos/{WRIST_CAMERA_KEY}/chunk_index"]),
                int(episode[f"videos/{WRIST_CAMERA_KEY}/file_index"]),
            )

            start_frame_main = int(round(float(episode[f"videos/{MAIN_CAMERA_KEY}/from_timestamp"]) * fps))
            start_frame_wrist = int(round(float(episode[f"videos/{WRIST_CAMERA_KEY}/from_timestamp"]) * fps))

            main_frames = left_reader.read_range(main_video_path, start_frame_main, length)
            wrist_frames = wrist_reader.read_range(wrist_video_path, start_frame_wrist, length)

            main_depth_bytes: list[bytes] = []
            wrist_depth_bytes: list[bytes] = []

            for i in range(0, length, batch_size):
                main_batch = main_frames[i : i + batch_size]
                wrist_batch = wrist_frames[i : i + batch_size]
                main_depths = predict_depth_batch(model, main_batch, device)
                wrist_depths = predict_depth_batch(model, wrist_batch, device)
                main_depth_bytes.extend(depth.tobytes() for depth in main_depths)
                wrist_depth_bytes.extend(depth.tobytes() for depth in wrist_depths)

            episode_df = data_df.iloc[start:end].copy()
            episode_df["observation.depth"] = main_depth_bytes
            episode_df["observation.wrist_depth"] = wrist_depth_bytes

            table = pa.Table.from_pandas(episode_df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_data_file, table.schema)
            writer.write_table(table)
    finally:
        left_reader.close()
        wrist_reader.close()
        if writer is not None:
            writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pseudo depth maps for RoboCasa LeRobot datasets.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-size", type=str, default="small", choices=["small", "base", "large"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    _prepare_output_root(dataset_root, output_root, args.overwrite)

    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    fps = float(info["fps"])

    print("Loading Depth Anything V2...")
    model = load_depth_anything_v2(args.model_size, args.device)

    data_files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    for data_file in data_files:
        rel = data_file.relative_to(dataset_root / "data")
        episodes_file = dataset_root / "meta" / "episodes" / rel
        if not episodes_file.exists():
            raise FileNotFoundError(f"Missing episode metadata file for {data_file}: {episodes_file}")
        _process_data_file(
            dataset_root,
            output_root,
            data_file,
            episodes_file,
            model,
            args.device,
            args.batch_size,
            args.max_episodes,
            fps,
        )

    print(f"\nDepth-augmented dataset written to: {output_root}")


if __name__ == "__main__":
    main()
