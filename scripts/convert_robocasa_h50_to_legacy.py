#!/usr/bin/env python3
"""Convert a LeRobot v3 RoboCasa H50 dataset into the legacy episode-parquet layout.

The current GaussianDream workspace pins an older lerobot stack that expects:
- meta/tasks.jsonl
- meta/episodes.jsonl
- data/chunk-xxx/episode_xxxxxx.parquet

RoboCasa H50 ships in LeRobot v3 chunk/file layout with RGB stored in AV1 videos. This
script rewrites the dataset into the older layout while preserving:
- task_index / task text
- RGB observations as embedded image dicts: {"bytes": ..., "path": ...}
- action / state / timestamps / indices
- optional depth sidecars if the source dataset already contains them

Example:
  python scripts/convert_robocasa_h50_to_legacy.py \
      --dataset-root <ROBOCASA_H50_WITH_DEPTH_ROOT> \
      --output-root <ROBOCASA_H50_LEGACY_ROOT> \
      --overwrite
"""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import shutil
import sys

import av
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MAIN_CAMERA_KEY = "observation.images.robot0_agentview_left_image"
WRIST_CAMERA_KEY = "observation.images.robot0_eye_in_hand_image"
RIGHT_CAMERA_KEY = "observation.images.robot0_agentview_right_image"
LEGACY_CHUNK_SIZE = 1000
LEGACY_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
LEGACY_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


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


def _encode_png_dict(frame: np.ndarray, frame_index: int) -> dict[str, bytes | str]:
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="PNG")
    return {
        "bytes": buffer.getvalue(),
        "path": f"frame_{frame_index:06d}.png",
    }


def _load_task_records(dataset_root: Path) -> list[dict[str, int | str]]:
    tasks_df = pd.read_parquet(dataset_root / "meta" / "tasks.parquet")
    records = []
    for task_text, task_index in zip(tasks_df.index.tolist(), tasks_df["task_index"].tolist(), strict=True):
        records.append({"task_index": int(task_index), "task": str(task_text)})
    records.sort(key=lambda item: item["task_index"])
    return records


def _iter_episode_rows(dataset_root: Path) -> list[dict]:
    rows: list[dict] = []
    for episodes_file in sorted((dataset_root / "meta" / "episodes").glob("chunk-*/*.parquet")):
        df = pd.read_parquet(episodes_file)
        for row in df.to_dict(orient="records"):
            rows.append(row)
    rows.sort(key=lambda item: int(item["episode_index"]))
    return rows


def _select_feature(input_info: dict, key: str) -> dict:
    feature = dict(input_info["features"][key])
    feature["shape"] = list(feature["shape"])
    return feature


def _build_legacy_info(
    input_info: dict,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    include_depth: bool,
) -> dict:
    features = {
        MAIN_CAMERA_KEY: {
            "dtype": "image",
            "shape": list(input_info["features"][MAIN_CAMERA_KEY]["shape"]),
            "names": list(input_info["features"][MAIN_CAMERA_KEY]["names"]),
        },
        WRIST_CAMERA_KEY: {
            "dtype": "image",
            "shape": list(input_info["features"][WRIST_CAMERA_KEY]["shape"]),
            "names": list(input_info["features"][WRIST_CAMERA_KEY]["names"]),
        },
        "observation.state": _select_feature(input_info, "observation.state"),
        "action": _select_feature(input_info, "action"),
        "timestamp": _select_feature(input_info, "timestamp"),
        "frame_index": _select_feature(input_info, "frame_index"),
        "episode_index": _select_feature(input_info, "episode_index"),
        "index": _select_feature(input_info, "index"),
        "task_index": _select_feature(input_info, "task_index"),
    }
    if "next.done" in input_info["features"]:
        features["next.done"] = _select_feature(input_info, "next.done")
    if include_depth:
        features["observation.depth"] = {
            "dtype": "binary",
            "shape": [128, 128],
            "names": ["height", "width"],
        }
        features["observation.wrist_depth"] = {
            "dtype": "binary",
            "shape": [128, 128],
            "names": ["height", "width"],
        }

    total_chunks = max(1, math.ceil(total_episodes / LEGACY_CHUNK_SIZE))
    return {
        "codebase_version": "v2.0",
        "robot_type": input_info["robot_type"],
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": 0,
        "total_chunks": total_chunks,
        "chunks_size": LEGACY_CHUNK_SIZE,
        "fps": input_info["fps"],
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": LEGACY_DATA_PATH,
        "video_path": LEGACY_VIDEO_PATH,
        "features": features,
    }


def _build_legacy_stats(input_stats: dict, include_depth: bool) -> dict:
    legacy_stats = {}
    key_mapping = {
        MAIN_CAMERA_KEY: MAIN_CAMERA_KEY,
        WRIST_CAMERA_KEY: WRIST_CAMERA_KEY,
        "observation.state": "observation.state",
        "action": "action",
        "timestamp": "timestamp",
        "frame_index": "frame_index",
        "episode_index": "episode_index",
        "index": "index",
        "task_index": "task_index",
        "next.done": "next.done",
    }
    for target_key, source_key in key_mapping.items():
        if source_key not in input_stats:
            continue
        source_stats = input_stats[source_key]
        legacy_stats[target_key] = {
            stat_name: source_stats[stat_name]
            for stat_name in ("mean", "std", "max", "min")
            if stat_name in source_stats
        }

    if include_depth:
        zero_stats = {
            "mean": 0.0,
            "std": 1.0,
            "max": 0.0,
            "min": 0.0,
        }
        legacy_stats["observation.depth"] = dict(zero_stats)
        legacy_stats["observation.wrist_depth"] = dict(zero_stats)

    return legacy_stats


def _prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output root already exists: {output_root}")
        shutil.rmtree(output_root)
    (output_root / "meta").mkdir(parents=True, exist_ok=True)
    (output_root / "data").mkdir(parents=True, exist_ok=True)


def _write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=4))


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _convert_episode_dataframe(
    episode_df: pd.DataFrame,
    main_frames: list[np.ndarray],
    wrist_frames: list[np.ndarray],
) -> pd.DataFrame:
    if len(episode_df) != len(main_frames) or len(episode_df) != len(wrist_frames):
        raise ValueError(
            f"Episode length mismatch: table={len(episode_df)} main_frames={len(main_frames)} wrist_frames={len(wrist_frames)}"
        )

    out_df = pd.DataFrame(
        {
            MAIN_CAMERA_KEY: [
                _encode_png_dict(frame, int(frame_index))
                for frame, frame_index in zip(main_frames, episode_df["frame_index"].tolist(), strict=True)
            ],
            WRIST_CAMERA_KEY: [
                _encode_png_dict(frame, int(frame_index))
                for frame, frame_index in zip(wrist_frames, episode_df["frame_index"].tolist(), strict=True)
            ],
            "observation.state": episode_df["observation.state"].tolist(),
            "action": episode_df["action"].tolist(),
            "timestamp": episode_df["timestamp"].tolist(),
            "frame_index": episode_df["frame_index"].tolist(),
            "episode_index": episode_df["episode_index"].tolist(),
            "index": episode_df["index"].tolist(),
            "task_index": episode_df["task_index"].tolist(),
        }
    )
    if "next.done" in episode_df.columns:
        out_df["next.done"] = episode_df["next.done"].tolist()
    if "observation.depth" in episode_df.columns:
        out_df["observation.depth"] = episode_df["observation.depth"].tolist()
    if "observation.wrist_depth" in episode_df.columns:
        out_df["observation.wrist_depth"] = episode_df["observation.wrist_depth"].tolist()
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert RoboCasa H50 from LeRobot v3 layout to legacy episode-parquet layout.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()

    input_info = json.loads((dataset_root / "meta" / "info.json").read_text())
    input_stats = json.loads((dataset_root / "meta" / "stats.json").read_text())
    task_records = _load_task_records(dataset_root)
    episode_rows = _iter_episode_rows(dataset_root)
    if args.max_episodes is not None:
        episode_rows = episode_rows[: args.max_episodes]

    if not episode_rows:
        raise ValueError("No episodes found to convert")

    include_depth = (
        "observation.depth" in input_info["features"] and "observation.wrist_depth" in input_info["features"]
    )
    total_frames = sum(int(row["length"]) for row in episode_rows)

    _prepare_output_root(output_root, args.overwrite)
    shutil.copy2(dataset_root / "README.md", output_root / "README.md")
    _write_json(
        _build_legacy_info(
            input_info=input_info,
            total_episodes=len(episode_rows),
            total_frames=total_frames,
            total_tasks=len(task_records),
            include_depth=include_depth,
        ),
        output_root / "meta" / "info.json",
    )
    _write_json(_build_legacy_stats(input_stats, include_depth), output_root / "meta" / "stats.json")
    _write_jsonl(task_records, output_root / "meta" / "tasks.jsonl")

    left_reader = SequentialVideoReader()
    wrist_reader = SequentialVideoReader()
    episodes_jsonl: list[dict] = []
    cached_data_file: Path | None = None
    cached_data_df: pd.DataFrame | None = None

    try:
        for episode in tqdm(episode_rows, desc="Converting episodes"):
            episode_index = int(episode["episode_index"])
            data_chunk_index = int(episode["data/chunk_index"])
            data_file_index = int(episode["data/file_index"])
            data_file = dataset_root / "data" / f"chunk-{data_chunk_index:03d}" / f"file-{data_file_index:03d}.parquet"
            if cached_data_file != data_file:
                cached_data_df = pd.read_parquet(data_file)
                cached_data_file = data_file
            if cached_data_df is None:
                raise RuntimeError("Failed to load source data parquet")

            start = int(episode["dataset_from_index"])
            end = int(episode["dataset_to_index"])
            length = int(episode["length"])
            if end - start != length:
                raise ValueError(
                    f"Episode {episode_index} has inconsistent slice: start={start} end={end} length={length}"
                )

            episode_df = cached_data_df.iloc[start:end].copy().reset_index(drop=True)

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
            start_frame_main = int(round(float(episode[f"videos/{MAIN_CAMERA_KEY}/from_timestamp"]) * float(input_info["fps"])))
            start_frame_wrist = int(round(float(episode[f"videos/{WRIST_CAMERA_KEY}/from_timestamp"]) * float(input_info["fps"])))

            main_frames = left_reader.read_range(main_video_path, start_frame_main, length)
            wrist_frames = wrist_reader.read_range(wrist_video_path, start_frame_wrist, length)
            out_df = _convert_episode_dataframe(episode_df, main_frames, wrist_frames)

            out_path = output_root / LEGACY_DATA_PATH.format(
                episode_chunk=episode_index // LEGACY_CHUNK_SIZE,
                episode_index=episode_index,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_df.to_parquet(out_path, index=False)

            tasks = [str(task) for task in np.asarray(episode["tasks"], dtype=object).tolist()]
            episodes_jsonl.append(
                {
                    "episode_index": episode_index,
                    "tasks": tasks,
                    "length": length,
                }
            )
    finally:
        left_reader.close()
        wrist_reader.close()

    _write_jsonl(episodes_jsonl, output_root / "meta" / "episodes.jsonl")
    print(f"Legacy dataset written to: {output_root}")


if __name__ == "__main__":
    main()
