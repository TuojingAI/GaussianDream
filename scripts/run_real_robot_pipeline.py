import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.aloha_real.convert_aloha_data_to_lerobot import port_aloha
from scripts.process_data_pickle2hdf5 import data_transform


def _expand_path(path: str) -> str:
    return os.path.expanduser(os.path.expandvars(path))


def count_hdf5_episodes(processed_hdf5_dir: str) -> int:
    root = Path(processed_hdf5_dir)
    return len(list(root.glob("episode_*/episode_*.hdf5")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GaussianDream ALOHA data preprocessing pipeline.")
    parser.add_argument(
        "--raw-dir",
        type=str,
        required=True,
        help="Root directory of the collected pickle dataset.",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="gaussiandream_aloha",
        help="Used as the default processed HDF5 directory name.",
    )
    parser.add_argument(
        "--processed-hdf5-dir",
        type=str,
        default=None,
        help="Directory for intermediate HDF5 episodes.",
    )
    parser.add_argument(
        "--lerobot-root",
        type=str,
        default="${HF_LEROBOT_HOME:-./data/lerobot_real}",
        help="Parent directory that will contain the converted LeRobot dataset.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="local/gaussiandream_aloha",
        help="LeRobot repo id used by training config.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="gaussiandream_aloha_jax",
        help="Training config name to print in the follow-up commands.",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="gaussiandream_aloha_jax_run",
        help="Training experiment name to print in the follow-up commands.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of pickle episodes to convert.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Sliding window used to smooth qpos before action generation.",
    )
    parser.add_argument(
        "--instruction-selection",
        choices=["first", "random", "all"],
        default="all",
        help="How to turn an episode-level instruction list into LeRobot episodes.",
    )
    parser.add_argument(
        "--instruction-seed",
        type=int,
        default=0,
        help="Seed used when instruction selection is random.",
    )
    parser.add_argument(
        "--fallback-task",
        type=str,
        default=None,
        help="Fallback instruction if a pickle has no instructions field.",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only scan the first level for pickle files.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the converted LeRobot dataset to the hub after conversion.",
    )
    parser.add_argument(
        "--include-velocity",
        action="store_true",
        help="Keep qvel in intermediate HDF5. Disabled by default to match the old script.",
    )
    parser.add_argument(
        "--include-effort",
        action="store_true",
        help="Keep effort in intermediate HDF5. Disabled by default to match the old script.",
    )
    parser.add_argument(
        "--skip-hdf5",
        action="store_true",
        help="Reuse an existing processed HDF5 directory and skip pickle-to-HDF5 conversion.",
    )
    parser.add_argument(
        "--skip-lerobot",
        action="store_true",
        help="Only run pickle-to-HDF5 conversion and skip LeRobot conversion.",
    )
    parser.add_argument(
        "--trim-deadzone",
        action="store_true",
        help="Trim static frames using qvel before converting to HDF5.",
    )
    parser.add_argument(
        "--trim-mode",
        choices=["end", "both"],
        default="end",
        help="Trim only the tail or trim both the start and tail. Default trims the tail only.",
    )

    args = parser.parse_args()

    lerobot_root = _expand_path(args.lerobot_root)
    processed_hdf5_dir = _expand_path(args.processed_hdf5_dir or f"./data/processed_hdf5/{args.task_name}")

    if args.skip_hdf5:
        episode_count = count_hdf5_episodes(processed_hdf5_dir)
        print(f"[1/2] Skipping pickle->HDF5, reusing {processed_hdf5_dir} ({episode_count} episodes)")
    else:
        raw_dir = _expand_path(args.raw_dir)
        print(f"[1/2] Converting pickle episodes under {raw_dir}")
        episode_count = data_transform(
            raw_dir,
            processed_hdf5_dir,
            instruction=args.fallback_task,
            recursive=not args.non_recursive,
            max_episodes=args.max_episodes,
            smooth_window=args.smooth_window,
            include_velocity=args.include_velocity,
            include_effort=args.include_effort,
            trim_deadzone=args.trim_deadzone,
            trim_mode=args.trim_mode,
        )
        print(f"Converted {episode_count} episodes into {processed_hdf5_dir}")

    if args.skip_lerobot:
        print(f"[2/2] Skipping HDF5->LeRobot conversion, processed HDF5 stays at {processed_hdf5_dir}")
    else:
        print(f"[2/2] Converting HDF5 episodes into LeRobot format at {lerobot_root}")
        local_dataset_dir = Path(lerobot_root) / args.repo_id
        port_aloha(
            raw_dir=Path(processed_hdf5_dir),
            repo_id=args.repo_id,
            task=args.fallback_task,
            push_to_hub=args.push_to_hub,
            mode="image",
            local_dir=local_dataset_dir,
            instruction_selection=args.instruction_selection,
            instruction_seed=args.instruction_seed,
        )

    print("\nNext commands:")
    print(f"export HF_LEROBOT_HOME={lerobot_root}")
    print(f"uv run --active scripts/compute_norm_stats.py --config-name {args.config_name}")
    print(
        "CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=offline "
        f"nohup uv run --active scripts/train.py {args.config_name} "
        f"--exp-name={args.exp_name} --overwrite > train_{args.task_name}.log 2>&1 &"
    )


if __name__ == "__main__":
    main()
