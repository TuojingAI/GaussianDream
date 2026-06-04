import argparse
import json
import os
from pathlib import Path
import pickle
import re

import cv2
import h5py
import numpy as np
from tqdm import tqdm

DOWN_INST = [
    "Move the robotic arm, and actuate the end-effector to press the elevator's DOWN call button.",
    "Arm to elevator; press DOWN button.",
    "Move arm, actuate gripper on DOWN call.",
    "Navigate to elevator, push DOWN.",
    "Go to DOWN button, actuate end-effector.",
    "Manipulator: approach, press DOWN.",
    "Execute press sequence on DOWN call.",
    "Arm moves, press elevator down.",
    "Target DOWN button; execute press.",
    "Press DOWN button with manipulator.",
    "Extend the manipulator assembly, then initiate the gripper to engage the DOWN button.",
]
UP_INST = [
    "Move the robotic arm, and actuate the end-effector to press the elevator's UP call button.",
    "Arm to elevator; press UP button.",
    "Move arm, actuate gripper on UP call.",
    "Navigate to elevator, push UP.",
    "Go to UP button, actuate end-effector.",
    "Manipulator: approach, press UP.",
    "Execute press sequence on UP call.",
    "Arm moves, press elevator up.",
    "Target UP button; execute press.",
    "Press UP button with manipulator.",
    "Extend the manipulator assembly, then initiate the gripper to engage the UP button.",
]

SMOOTH_WINDOW_DEFAULT = 5


def _expand_path(path: str) -> str:
    return os.path.expanduser(os.path.expandvars(path))


def natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def smooth_qpos(qpos: np.ndarray, window: int = SMOOTH_WINDOW_DEFAULT) -> np.ndarray:
    if window is None or window <= 1:
        return qpos.astype(np.float32)

    kernel = np.ones(window, dtype=np.float32) / float(window)
    _, dim = qpos.shape
    smoothed = np.empty_like(qpos, dtype=np.float32)
    for joint_idx in range(dim):
        smoothed[:, joint_idx] = np.convolve(qpos[:, joint_idx], kernel, mode="same")
    return smoothed


def denoise_qvel(qvel_data: np.ndarray) -> np.ndarray:
    if len(qvel_data) == 0:
        return qvel_data

    denoised = qvel_data.copy()
    smooth_threshold = 0.2
    spike_threshold = 0.3

    num_steps, num_joints = qvel_data.shape
    for joint_idx in range(num_joints):
        if num_steps > 1:
            vel_0 = abs(qvel_data[0, joint_idx])
            vel_1 = abs(qvel_data[1, joint_idx])
            if vel_0 < smooth_threshold and vel_1 < smooth_threshold:
                denoised[0, joint_idx] = 0.0

        for step in range(1, num_steps - 1):
            prev_vel = abs(qvel_data[step - 1, joint_idx])
            cur_vel = abs(qvel_data[step, joint_idx])
            next_vel = abs(qvel_data[step + 1, joint_idx])

            condition_smooth = (
                prev_vel < smooth_threshold and cur_vel < smooth_threshold and next_vel < smooth_threshold
            )
            condition_spike = prev_vel < 1e-3 and next_vel < 1e-3 and cur_vel < spike_threshold
            if condition_smooth or condition_spike:
                denoised[step, joint_idx] = 0.0

    return denoised


def find_action_start_index(qvel_data: np.ndarray) -> int:
    for step in range(len(qvel_data)):
        if np.any(np.abs(qvel_data[step]) > 1e-6):
            return step
    return 0


def find_action_end_index(qvel_data: np.ndarray) -> int:
    for step in range(len(qvel_data) - 1, -1, -1):
        if np.any(np.abs(qvel_data[step]) > 1e-6):
            return step
    return len(qvel_data) - 1


def trim_trajectory_deadzone(
    traj_data: dict,
    *,
    trim_mode: str,
) -> tuple[dict, dict[str, int]]:
    if "/observations/qvel" not in traj_data:
        return traj_data, {"trim_start": 0, "trim_end": 0, "frames_before_trim": -1, "frames_after_trim": -1}

    qvel_data = np.asarray(traj_data["/observations/qvel"], dtype=np.float32)
    if qvel_data.ndim != 2 or len(qvel_data) == 0:
        return traj_data, {"trim_start": 0, "trim_end": 0, "frames_before_trim": -1, "frames_after_trim": -1}

    denoised_qvel = denoise_qvel(qvel_data)
    original_length = len(denoised_qvel)

    start_idx = 0 if trim_mode == "end" else find_action_start_index(denoised_qvel)
    end_idx = find_action_end_index(denoised_qvel)

    if end_idx < start_idx:
        start_idx = 0
        end_idx = original_length - 1

    trimmed = {}
    for key, value in traj_data.items():
        if key == "instructions":
            trimmed[key] = value
        else:
            trimmed[key] = value[start_idx : end_idx + 1]

    return trimmed, {
        "trim_start": int(start_idx),
        "trim_end": int(original_length - end_idx - 1),
        "frames_before_trim": int(original_length),
        "frames_after_trim": int(end_idx - start_idx + 1),
    }


def images_encoding(imgs: list[np.ndarray]) -> tuple[list[bytes], int]:
    encoded = []
    max_len = 0
    for image in imgs:
        success, encoded_image = cv2.imencode(".jpg", image)
        if not success:
            raise ValueError("Failed to encode image as JPEG.")
        jpeg_data = encoded_image.tobytes()
        encoded.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    return encoded, max_len


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


def resolve_instructions(traj_data: dict, pkl_path: Path, fallback_instruction: str | None) -> list[str]:
    instructions = normalize_instructions(traj_data.get("instructions"))
    if instructions:
        return instructions

    if fallback_instruction:
        return [fallback_instruction]

    lower_path = str(pkl_path).lower()
    if "up" in lower_path:
        print(f"Warning: no instructions in {pkl_path}, fallback to UP instructions.")
        return UP_INST
    if "down" in lower_path:
        print(f"Warning: no instructions in {pkl_path}, fallback to DOWN instructions.")
        return DOWN_INST
    raise ValueError(f"Cannot infer instruction for {pkl_path}.")


def find_pickle_files(load_dir: str, recursive: bool = True) -> list[Path]:
    root = Path(load_dir)
    pattern = "**/*.pkl" if recursive else "*.pkl"
    files = [path for path in root.glob(pattern) if path.is_file()]
    return sorted(files, key=lambda path: natural_key(str(path.relative_to(root))))


def build_episode_payload(
    traj_data: dict,
    smooth_window: int,
    *,
    include_velocity: bool = False,
    include_effort: bool = False,
) -> dict[str, np.ndarray | list[np.ndarray]]:
    qpos_all = np.asarray(traj_data["/observations/qpos"], dtype=np.float32)
    qpos_all = smooth_qpos(qpos_all, window=smooth_window)

    if len(qpos_all) < 2:
        raise ValueError("Episode must contain at least 2 frames.")

    states = qpos_all[:-1]
    actions = qpos_all[1:]

    payload: dict[str, np.ndarray | list[np.ndarray]] = {
        "qpos": states.astype(np.float32),
        "action": actions.astype(np.float32),
        "left_arm_dim": np.full((len(actions),), 6, dtype=np.int32),
        "right_arm_dim": np.full((len(actions),), 6, dtype=np.int32),
    }

    if include_velocity and "/observations/qvel" in traj_data:
        payload["qvel"] = np.asarray(traj_data["/observations/qvel"], dtype=np.float32)[:-1]
    if include_effort and "/observations/effort" in traj_data:
        payload["effort"] = np.asarray(traj_data["/observations/effort"], dtype=np.float32)[:-1]

    head_images = []
    left_images = []
    for head_image in traj_data["/observations/images/cam_second"][:-1]:
        head_images.append(cv2.resize(head_image[:, :, 0:3], (640, 480)))
    for wrist_image in traj_data["/observations/images/cam_main"][:-1]:
        left_images.append(cv2.resize(wrist_image, (640, 480)))

    payload["cam_high"] = head_images
    payload["cam_left_wrist"] = left_images
    return payload


def write_episode(
    episode_dir: Path,
    episode_idx: int,
    payload: dict[str, np.ndarray | list[np.ndarray]],
    instructions: list[str],
    source_path: Path,
    load_root: Path,
    trim_info: dict[str, int],
) -> None:
    episode_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "instructions": instructions,
        "primary_instruction": instructions[0],
        "source_path": str(source_path),
        "source_relpath": str(source_path.relative_to(load_root)),
        "num_frames_raw": int(len(payload["action"]) + 1),
        "num_frames_saved": int(len(payload["action"])),
        **trim_info,
    }
    with open(episode_dir / "instructions.json", "w", encoding="utf-8") as handle:
        json.dump({"instructions": instructions}, handle, indent=2, ensure_ascii=False)
    with open(episode_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    hdf5_path = episode_dir / f"episode_{episode_idx}.hdf5"
    with h5py.File(hdf5_path, "w") as handle:
        handle.attrs["source_relpath"] = metadata["source_relpath"]
        handle.attrs["primary_instruction"] = instructions[0]
        handle.attrs["instructions_json"] = json.dumps(instructions, ensure_ascii=False)
        handle.attrs["trim_start"] = metadata["trim_start"]
        handle.attrs["trim_end"] = metadata["trim_end"]

        handle.create_dataset("action", data=payload["action"])
        observations = handle.create_group("observations")
        observations.create_dataset("qpos", data=payload["qpos"])
        observations.create_dataset("left_arm_dim", data=payload["left_arm_dim"])
        observations.create_dataset("right_arm_dim", data=payload["right_arm_dim"])

        if "qvel" in payload:
            observations.create_dataset("qvel", data=payload["qvel"])
        if "effort" in payload:
            observations.create_dataset("effort", data=payload["effort"])

        images = observations.create_group("images")
        for camera_name in ("cam_high", "cam_left_wrist"):
            encoded, max_len = images_encoding(payload[camera_name])
            images.create_dataset(camera_name, data=encoded, dtype=f"S{max_len}")


def data_transform(
    path: str,
    save_path: str,
    instruction: str | None = None,
    *,
    recursive: bool = True,
    max_episodes: int | None = None,
    smooth_window: int = SMOOTH_WINDOW_DEFAULT,
    include_velocity: bool = False,
    include_effort: bool = False,
    trim_deadzone: bool = False,
    trim_mode: str = "end",
) -> int:
    load_root = Path(path)
    save_root = Path(save_path)
    save_root.mkdir(parents=True, exist_ok=True)

    pkl_files = find_pickle_files(path, recursive=recursive)
    if max_episodes is not None:
        pkl_files = pkl_files[:max_episodes]

    for episode_idx, pkl_path in tqdm(enumerate(pkl_files), total=len(pkl_files)):
        with open(pkl_path, "rb") as handle:
            traj_data = pickle.load(handle)

        trim_info = {"trim_start": 0, "trim_end": 0, "frames_before_trim": -1, "frames_after_trim": -1}
        if trim_deadzone:
            traj_data, trim_info = trim_trajectory_deadzone(traj_data, trim_mode=trim_mode)

        instructions = resolve_instructions(traj_data, pkl_path, instruction)
        payload = build_episode_payload(
            traj_data,
            smooth_window=smooth_window,
            include_velocity=include_velocity,
            include_effort=include_effort,
        )
        episode_dir = save_root / f"episode_{episode_idx}"
        write_episode(episode_dir, episode_idx, payload, instructions, pkl_path, load_root, trim_info)

    return len(pkl_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert pickle episodes into nested HDF5 episodes.")
    parser.add_argument(
        "--task_name",
        type=str,
        default="my_real_robot_task",
        help="Fallback instruction if a pickle does not contain instructions.",
    )
    parser.add_argument(
        "--load_dir",
        type=str,
        default="./data/raw_real_robot",
        help="Directory containing pickle files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for processed HDF5 data. Defaults to ./data/processed_hdf5/<task_name>.",
    )
    parser.add_argument("--setting", type=str, default="pi0")
    parser.add_argument(
        "--expert_data_num",
        type=int,
        default=1000,
        help="Maximum number of episodes to process.",
    )
    parser.add_argument(
        "--non_recursive",
        action="store_true",
        help="Only scan the top-level directory for pickle files.",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=SMOOTH_WINDOW_DEFAULT,
        help="Sliding window size used to smooth qpos before generating states/actions.",
    )
    parser.add_argument(
        "--include_velocity",
        action="store_true",
        help="Also store qvel in the output HDF5. Disabled by default to match the old pipeline.",
    )
    parser.add_argument(
        "--include_effort",
        action="store_true",
        help="Also store effort in the output HDF5. Disabled by default to match the old pipeline.",
    )
    parser.add_argument(
        "--trim_deadzone",
        action="store_true",
        help="Trim static segments using qvel before converting to HDF5.",
    )
    parser.add_argument(
        "--trim_mode",
        choices=["end", "both"],
        default="end",
        help="Trim only the tail or trim both the start and tail. Default trims the tail only.",
    )

    args = parser.parse_args()

    target_dir = _expand_path(args.output_dir or f"./data/processed_hdf5/{args.task_name}")
    load_dir = _expand_path(args.load_dir)
    data_transform(
        load_dir,
        target_dir,
        instruction=args.task_name,
        recursive=not args.non_recursive,
        max_episodes=args.expert_data_num,
        smooth_window=args.smooth_window,
        include_velocity=args.include_velocity,
        include_effort=args.include_effort,
        trim_deadzone=args.trim_deadzone,
        trim_mode=args.trim_mode,
    )
