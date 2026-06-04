"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""

import dataclasses
import json
from pathlib import Path
import random
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME as LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# Keep the converter self-contained; raw downloads are handled outside this script.
import numpy as np
import torch
import tqdm
import tyro


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def normalize_instructions(raw_instructions) -> list[str]:
    if raw_instructions is None:
        return []
    if isinstance(raw_instructions, str):
        candidates = [raw_instructions]
    elif isinstance(raw_instructions, np.ndarray):
        candidates = raw_instructions.tolist()
    elif isinstance(raw_instructions, (list, tuple)):
        candidates = list(raw_instructions)
    else:
        candidates = [str(raw_instructions)]

    normalized = []
    for item in candidates:
        text = str(item).strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def align_feature_dim(tensor: torch.Tensor | None, target_dim: int, *, fill_value: float = 0.0) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.ndim != 2:
        return tensor

    current_dim = tensor.shape[-1]
    if current_dim == target_dim:
        return tensor
    if current_dim > target_dim:
        return tensor[:, :target_dim]

    pad = torch.full(
        (tensor.shape[0], target_dim - current_dim),
        fill_value,
        dtype=tensor.dtype,
    )
    return torch.cat([tensor, pad], dim=-1)


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    root: Path | None = None,
) -> LeRobotDataset:
    motors = [
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
    ]
    cameras = [
        "cam_high",
        "cam_left_wrist",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    root = root or LEROBOT_HOME
    if (root / repo_id).exists():
        shutil.rmtree(root / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
        root=root,
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        # ignore depth channel, not currently handled
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118


def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            # load all images in RAM
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2

            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                # Ensure data is treated as a numpy array of bytes (uint8)
                data_bytes = np.frombuffer(data, dtype=np.uint8)
                imgs_array.append(cv2.cvtColor(cv2.imdecode(data_bytes, 1), cv2.COLOR_BGR2RGB))
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[
    dict[str, np.ndarray],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    list[str],
]:
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/observations/qpos"][:])
        action = torch.from_numpy(ep["/action"][:])

        velocity = None
        if "/observations/qvel" in ep:
            velocity = torch.from_numpy(ep["/observations/qvel"][:])

        effort = None
        if "/observations/effort" in ep:
            effort = torch.from_numpy(ep["/observations/effort"][:])

        imgs_per_cam = load_raw_images_per_camera(
            ep,
            [
                "cam_high",
                "cam_left_wrist",
            ],
        )

        instructions = []
        if "instructions_json" in ep.attrs:
            instructions = normalize_instructions(json.loads(ep.attrs["instructions_json"]))
        elif "primary_instruction" in ep.attrs:
            instructions = normalize_instructions(ep.attrs["primary_instruction"])

        if not instructions:
            instructions_path = ep_path.parent / "instructions.json"
            if instructions_path.exists():
                with open(instructions_path, "r", encoding="utf-8") as handle:
                    instructions = normalize_instructions(json.load(handle).get("instructions"))

    target_dim = state.shape[-1]
    velocity = align_feature_dim(velocity, target_dim, fill_value=0.0)
    effort = align_feature_dim(effort, target_dim, fill_value=0.0)

    return imgs_per_cam, state, action, velocity, effort, instructions


def select_episode_tasks(
    instructions: list[str],
    fallback_task: str | None,
    selection: Literal["first", "random", "all"],
    rng: random.Random,
) -> list[str]:
    candidates = normalize_instructions(instructions)
    if not candidates:
        candidates = normalize_instructions(fallback_task)

    if not candidates:
        raise ValueError("Episode does not contain instructions and no fallback --task was provided.")

    if selection == "first":
        return [candidates[0]]
    if selection == "random":
        return [rng.choice(candidates)]
    if selection == "all":
        return candidates
    raise ValueError(f"Unsupported instruction selection mode: {selection}")


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str | None,
    episodes: list[int] | None = None,
    *,
    instruction_selection: Literal["first", "random", "all"] = "all",
    instruction_seed: int = 0,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))
    rng = random.Random(instruction_seed)

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        imgs_per_cam, state, action, velocity, effort, instructions = load_raw_episode_data(ep_path)
        episode_tasks = select_episode_tasks(instructions, task, instruction_selection, rng)
        num_frames = state.shape[0]

        for episode_task in episode_tasks:
            for i in range(num_frames):
                frame = {
                    "observation.state": state[i],
                    "action": action[i],
                }

                for camera, img_array in imgs_per_cam.items():
                    frame[f"observation.images.{camera}"] = img_array[i]

                if velocity is not None:
                    frame["observation.velocity"] = velocity[i]
                if effort is not None:
                    frame["observation.effort"] = effort[i]

                frame["task"] = episode_task

                dataset.add_frame(frame)

            dataset.save_episode()

    return dataset


def port_aloha(
    raw_dir: Path,
    repo_id: str,
    task: str | None = None,
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = True,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    local_dir: Path | None = None,
    root_dir: Path | None = None,
    instruction_selection: Literal["first", "random", "all"] = "all",
    instruction_seed: int = 0,
):
    root = root_dir or local_dir or LEROBOT_HOME

    if root_dir is not None and (root / repo_id).exists():
        print(f"Removing existing repo dir: {root / repo_id}")
        shutil.rmtree(root / repo_id)
    elif local_dir and local_dir.exists():
        print(f"Removing existing local dir: {local_dir}")
        shutil.rmtree(local_dir)
    elif (root / repo_id).exists():
        print(f"Removing existing repo dir: {root / repo_id}")
        shutil.rmtree(root / repo_id)

    # Check the local data directory directly; this converter does not download raw data.
    if not raw_dir.exists():
        raise ValueError(f"Raw data directory does not exist: {raw_dir}")

    # Support the nested episode layout emitted by process_data_pickle2hdf5.py.
    hdf5_files = sorted(raw_dir.glob("*/episode_*.hdf5"))
    if not hdf5_files:
        # Fall back to the legacy flat episode layout.
        hdf5_files = sorted(raw_dir.glob("episode_*.hdf5"))
    
    if not hdf5_files:
        raise ValueError(f"No HDF5 files found in {raw_dir} (checked '*/episode_*.hdf5' and 'episode_*.hdf5')")
    
    print(f"Found {len(hdf5_files)} episodes.")

    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        mode=mode,
        has_effort=has_effort(hdf5_files),
        has_velocity=has_velocity(hdf5_files),
        dataset_config=dataset_config,
        root=root,
    )
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task=task,
        episodes=episodes,
        instruction_selection=instruction_selection,
        instruction_seed=instruction_seed,
    )
    # dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_aloha)
