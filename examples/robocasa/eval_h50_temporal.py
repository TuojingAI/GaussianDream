import collections
import csv
import dataclasses
import datetime as dt
import json
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import imageio
import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy


OFFICIAL_LAYOUT_AND_STYLE_IDS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (2, 2),
    (4, 4),
    (6, 9),
    (7, 10),
)

H50_FAMILY_ORDER: tuple[str, ...] = (
    "CloseDoubleDoor",
    "CloseDrawer",
    "CloseSingleDoor",
    "CoffeePressButton",
    "CoffeeServeMug",
    "CoffeeSetupMug",
    "OpenDoubleDoor",
    "OpenDrawer",
    "OpenSingleDoor",
    "PnPCabToCounter",
    "PnPCounterToCab",
    "PnPCounterToMicrowave",
    "PnPCounterToSink",
    "PnPCounterToStove",
    "PnPMicrowaveToCounter",
    "PnPSinkToCounter",
    "PnPStoveToCounter",
    "TurnOffMicrowave",
    "TurnOffSinkFaucet",
    "TurnOffStove",
    "TurnOnMicrowave",
    "TurnOnSinkFaucet",
    "TurnOnStove",
    "TurnSinkSpout",
)

OFFICIAL_CAMERA_NAMES: list[str] = [
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]


def _configure_headless_rendering() -> None:
    """Prefer GLFW and transparently relaunch under Xvfb on headless Linux."""
    os.environ.setdefault("MUJOCO_GL", "glx")

    if not sys.platform.startswith("linux"):
        return
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return
    if os.environ.get("_GAUSSIANDREAM_ROBOCASA_XVFB") == "1" or os.environ.get("_OPENPI_ROBOCASA_XVFB") == "1":
        return

    xvfb_run = shutil.which("xvfb-run")
    if xvfb_run is None:
        return

    os.environ["_GAUSSIANDREAM_ROBOCASA_XVFB"] = "1"
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    print("No DISPLAY detected; relaunching RoboCasa under xvfb-run for offscreen rendering...", flush=True)
    os.execvp(
        xvfb_run,
        [
            xvfb_run,
            "-a",
            "-s",
            "-screen 0 1280x1024x24 +extension GLX +render -noreset",
            sys.executable,
            *sys.argv,
        ],
    )


import tyro


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8003
    families: str = "all"
    episodes_per_family: int = 50
    base_seed: int = 20260417
    manifest_path: str = "examples/robocasa/h50_eval_manifest.json"
    regenerate_manifest: bool = False
    manifest_only: bool = False
    strict_manifest: bool = True
    output_dir: str = "robocasa_eval_results"
    video_dir: str | None = None
    video_fps: int = 10
    camera_height: int = 128
    camera_width: int = 128
    replan_steps: int = 5
    temporal_step_offsets: str = "10,5,0"
    invert_action_6: bool = False
    obj_instance_split: str | None = "B"
    generative_textures: str | None = None
    randomize_cameras: bool = False
    layout_and_style_ids: str = "1:1,2:2,4:4,6:9,7:10"
    log_level: str = "INFO"


def _parse_layout_and_style_ids(spec: str) -> tuple[tuple[int, int], ...]:
    spec = spec.strip()
    if not spec:
        return OFFICIAL_LAYOUT_AND_STYLE_IDS

    pairs: list[tuple[int, int]] = []
    for chunk in spec.split(","):
        left, right = chunk.strip().split(":", maxsplit=1)
        pairs.append((int(left), int(right)))
    return tuple(pairs)


def _parse_families(spec: str) -> list[str]:
    if spec.strip().lower() == "all":
        return list(H50_FAMILY_ORDER)

    requested = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(H50_FAMILY_ORDER))
    if unknown:
        raise ValueError(f"Unknown families: {unknown}. Choose from: {list(H50_FAMILY_ORDER)}")
    return requested


def _parse_temporal_step_offsets(spec: str) -> tuple[int, ...]:
    offsets = tuple(int(item.strip()) for item in spec.split(",") if item.strip())
    if not offsets:
        raise ValueError("--temporal-step-offsets must contain at least one integer.")
    if any(offset < 0 for offset in offsets):
        raise ValueError(f"--temporal-step-offsets must be non-negative, got {offsets}")
    if offsets[-1] != 0:
        raise ValueError(
            f"--temporal-step-offsets must end with 0 for the current frame, got {offsets}"
        )
    if any(left < right for left, right in zip(offsets, offsets[1:], strict=False)):
        raise ValueError(
            f"--temporal-step-offsets must be ordered oldest->current, e.g. 2,1,0; got {offsets}"
        )
    return offsets


def _benchmark_seed(base_seed: int, family_index: int, episode_index: int) -> int:
    return base_seed + family_index * 1000 + episode_index


def _build_robocasa_state(obs: dict[str, Any]) -> np.ndarray:
    """Match the 16-dim state ordering declared in `${ROBOCASA_H50_ROOT}/meta/info.json`."""
    required_keys = [
        "robot0_base_to_eef_pos",
        "robot0_base_to_eef_quat",
        "robot0_gripper_qpos",
        "robot0_base_pos",
        "robot0_base_quat",
    ]
    missing_keys = [key for key in required_keys if key not in obs]
    if missing_keys:
        raise KeyError(f"Missing RoboCasa state keys: {missing_keys}")

    return np.concatenate(
        [
            obs["robot0_base_to_eef_pos"],
            obs["robot0_base_to_eef_quat"],
            obs["robot0_gripper_qpos"],
            obs["robot0_base_pos"],
            obs["robot0_base_quat"],
        ]
    ).astype(np.float32)


def _flip_image_for_dataset(image: np.ndarray) -> np.ndarray:
    """Match the upside-down correction used when RoboCasa extracts dataset observations."""
    return np.ascontiguousarray(image[::-1])


def _stack_temporal_frames(
    past_frames: collections.deque[np.ndarray],
    current_frame: np.ndarray,
    step_offsets: tuple[int, ...],
) -> np.ndarray:
    """Build a (T, H, W, C) stack ordered from oldest context to current frame."""
    seq = list(past_frames) + [current_frame]
    out: list[np.ndarray] = []
    for offset in step_offsets:
        index = len(seq) - 1 - offset
        out.append(seq[index] if index >= 0 else seq[-1])
    return np.stack(out, axis=0)


def _build_policy_request(
    obs: dict[str, Any],
    prompt: str,
    *,
    agentview_history: collections.deque[np.ndarray],
    wrist_history: collections.deque[np.ndarray],
    temporal_step_offsets: tuple[int, ...],
) -> tuple[dict[str, Any], np.ndarray]:
    """Convert raw RoboCasa env observations into the dataset-aligned GaussianDream request format."""
    agentview_img = _flip_image_for_dataset(obs["robot0_agentview_left_image"])
    wrist_img = _flip_image_for_dataset(obs["robot0_eye_in_hand_image"])
    request = {
        "observation/image": _stack_temporal_frames(
            agentview_history, agentview_img, temporal_step_offsets
        ),
        "observation/wrist_image": _stack_temporal_frames(
            wrist_history, wrist_img, temporal_step_offsets
        ),
        "observation/state": _build_robocasa_state(obs),
        "prompt": prompt,
    }
    return request, agentview_img


def _apply_eval_action_overrides(action: np.ndarray, args: Args) -> np.ndarray:
    """Apply temporary eval-time overrides for debugging action semantics."""
    if args.invert_action_6:
        action[6] *= -1.0
    return action


def _family_horizon(family: str) -> int:
    from robocasa.utils.dataset_registry import SINGLE_STAGE_TASK_DATASETS

    try:
        return int(SINGLE_STAGE_TASK_DATASETS[family]["horizon"])
    except KeyError as exc:
        raise KeyError(f"Missing horizon for family {family}") from exc


def _make_env(
    family: str,
    *,
    seed: int,
    camera_height: int,
    camera_width: int,
    obj_instance_split: str | None,
    generative_textures: str | None,
    randomize_cameras: bool,
    layout_and_style_ids: tuple[tuple[int, int], ...],
):
    import robocasa  # noqa: F401  # Registers RoboCasa environments with robosuite.
    from robocasa.utils.env_utils import create_env as create_robocasa_env

    return create_robocasa_env(
        env_name=family,
        robots="PandaOmron",
        camera_names=OFFICIAL_CAMERA_NAMES,
        camera_heights=camera_height,
        camera_widths=camera_width,
        seed=seed,
        render_onscreen=False,
        obj_instance_split=obj_instance_split,
        generative_textures=generative_textures,
        randomize_cameras=randomize_cameras,
        layout_and_style_ids=layout_and_style_ids,
    )


def _episode_signature(ep_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": ep_meta.get("lang", ""),
        "layout_id": ep_meta.get("layout_id"),
        "style_id": ep_meta.get("style_id"),
        "task_refs": ep_meta.get("task_refs"),
    }


def _generate_manifest(args: Args, families: list[str]) -> dict[str, Any]:
    layout_and_style_ids = _parse_layout_and_style_ids(args.layout_and_style_ids)
    episodes: list[dict[str, Any]] = []

    logging.info("Generating deterministic H50 benchmark manifest at %s", args.manifest_path)
    for family_index, family in enumerate(families):
        max_steps = _family_horizon(family)
        for episode_index in range(args.episodes_per_family):
            seed = _benchmark_seed(args.base_seed, family_index, episode_index)
            env = _make_env(
                family,
                seed=seed,
                camera_height=args.camera_height,
                camera_width=args.camera_width,
                obj_instance_split=args.obj_instance_split,
                generative_textures=args.generative_textures,
                randomize_cameras=args.randomize_cameras,
                layout_and_style_ids=layout_and_style_ids,
            )
            try:
                env.reset()
                ep_meta = env.get_ep_meta()
            finally:
                env.close()

            episodes.append(
                {
                    "family": family,
                    "env_name": family,
                    "family_episode_index": episode_index,
                    "seed": seed,
                    "max_steps": max_steps,
                    "signature": _episode_signature(ep_meta),
                }
            )

    manifest = {
        "version": 1,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "benchmark": "robocasa_h50_24",
        "episodes_per_family": args.episodes_per_family,
        "base_seed": args.base_seed,
        "camera_height": args.camera_height,
        "camera_width": args.camera_width,
        "obj_instance_split": args.obj_instance_split,
        "generative_textures": args.generative_textures,
        "randomize_cameras": args.randomize_cameras,
        "layout_and_style_ids": [list(pair) for pair in layout_and_style_ids],
        "families": families,
        "episodes": episodes,
    }

    manifest_path = Path(args.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logging.info("Saved benchmark manifest with %d episodes to %s", len(episodes), manifest_path)
    return manifest


def _load_or_create_manifest(args: Args, families: list[str]) -> dict[str, Any]:
    manifest_path = Path(args.manifest_path)
    if args.regenerate_manifest or not manifest_path.exists():
        return _generate_manifest(args, families)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_families = manifest.get("families", [])
    missing = [family for family in families if family not in manifest_families]
    if missing:
        raise ValueError(
            f"Manifest {manifest_path} does not cover families {missing}. "
            f"Regenerate it or choose a compatible --families value."
        )
    return manifest


def _validate_manifest_entry(record: dict[str, Any], ep_meta: dict[str, Any], *, strict: bool) -> None:
    expected = record.get("signature", {})
    actual = _episode_signature(ep_meta)
    if actual == expected:
        return

    message = (
        f"Manifest mismatch for family={record['family']} seed={record['seed']}: "
        f"expected {expected}, got {actual}"
    )
    if strict:
        raise RuntimeError(message)
    logging.warning(message)


def _save_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


def _write_episode_outputs(run_dir: Path, episodes: list[dict[str, Any]]) -> None:
    jsonl_path = run_dir / "episodes.jsonl"
    csv_path = run_dir / "episodes.csv"

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in episodes:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")

    if not episodes:
        return

    fieldnames = [
        "family",
        "env_name",
        "family_episode_index",
        "seed",
        "prompt",
        "layout_id",
        "style_id",
        "task_refs",
        "max_steps",
        "steps_taken",
        "success",
        "final_reward",
        "video_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in episodes:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _summarize_results(episodes: list[dict[str, Any]], families: list[str]) -> dict[str, Any]:
    per_family: dict[str, dict[str, Any]] = {}
    total_successes = 0
    total_episodes = len(episodes)

    for family in families:
        rows = [row for row in episodes if row["family"] == family]
        successes = sum(int(row["success"]) for row in rows)
        rate = (successes / len(rows)) if rows else 0.0
        per_family[family] = {
            "episodes": len(rows),
            "successes": successes,
            "success_rate": rate,
        }
        total_successes += successes

    macro_success_rate = (
        sum(item["success_rate"] for item in per_family.values()) / len(per_family) if per_family else 0.0
    )
    micro_success_rate = (total_successes / total_episodes) if total_episodes else 0.0

    return {
        "families_evaluated": families,
        "num_families": len(families),
        "num_episodes": total_episodes,
        "successes": total_successes,
        "micro_success_rate": micro_success_rate,
        "macro_success_rate": macro_success_rate,
        "per_family": per_family,
    }


def main(args: Args) -> None:
    _configure_headless_rendering()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    if args.replan_steps <= 0:
        raise ValueError("--replan-steps must be positive")
    temporal_step_offsets = _parse_temporal_step_offsets(args.temporal_step_offsets)
    if args.invert_action_6:
        logging.warning("Eval override enabled: action[6] will be multiplied by -1 before env.step().")
    logging.info(
        "Temporal eval context enabled with step offsets %s (oldest -> current).",
        temporal_step_offsets,
    )

    families = _parse_families(args.families)
    manifest = _load_or_create_manifest(args, families)
    selected_records = [record for record in manifest["episodes"] if record["family"] in families]

    logging.info(
        "Loaded benchmark with %d records across %d families from %s",
        len(selected_records),
        len(families),
        args.manifest_path,
    )

    if args.manifest_only:
        return

    layout_and_style_ids = tuple(tuple(pair) for pair in manifest["layout_and_style_ids"])
    run_dir = Path(args.output_dir) / dt.datetime.now().strftime("h50_eval_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Writing evaluation artifacts to %s", run_dir)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    episode_results: list[dict[str, Any]] = []
    running_family_counts: dict[str, list[int]] = {family: [0, 0] for family in families}

    for global_index, record in enumerate(selected_records, start=1):
        family = record["family"]
        env = _make_env(
            family,
            seed=int(record["seed"]),
            camera_height=manifest["camera_height"],
            camera_width=manifest["camera_width"],
            obj_instance_split=manifest["obj_instance_split"],
            generative_textures=manifest["generative_textures"],
            randomize_cameras=bool(manifest["randomize_cameras"]),
            layout_and_style_ids=layout_and_style_ids,
        )

        frames: list[np.ndarray] = []
        success = False
        steps_taken = 0
        final_reward = 0.0
        prompt = ""
        ep_meta: dict[str, Any] = {}
        video_path = ""
        hist_len = max(temporal_step_offsets)
        agentview_history: collections.deque[np.ndarray] = collections.deque(maxlen=hist_len)
        wrist_history: collections.deque[np.ndarray] = collections.deque(maxlen=hist_len)

        try:
            obs = env.reset()
            ep_meta = env.get_ep_meta()
            _validate_manifest_entry(record, ep_meta, strict=args.strict_manifest)
            prompt = str(ep_meta.get("lang", ""))
            if not prompt:
                raise ValueError(f"Missing language prompt for family {family} seed={record['seed']}")

            while steps_taken < int(record["max_steps"]):
                request, agentview_img = _build_policy_request(
                    obs,
                    prompt,
                    agentview_history=agentview_history,
                    wrist_history=wrist_history,
                    temporal_step_offsets=temporal_step_offsets,
                )
                response = client.infer(request)
                action_chunk = np.asarray(response["actions"], dtype=np.float32)
                if action_chunk.ndim != 2 or action_chunk.shape[1] < 12:
                    raise ValueError(f"Expected an action chunk shaped [T, >=12], got {action_chunk.shape}")

                chunk_len = min(args.replan_steps, len(action_chunk), int(record["max_steps"]) - steps_taken)
                for chunk_idx in range(chunk_len):
                    current_agentview_img = _flip_image_for_dataset(obs["robot0_agentview_left_image"])
                    current_wrist_img = _flip_image_for_dataset(obs["robot0_eye_in_hand_image"])
                    if args.video_dir is not None:
                        frames.append(current_agentview_img)

                    action = np.asarray(action_chunk[chunk_idx, :12], dtype=np.float32).copy()
                    action = _apply_eval_action_overrides(action, args)
                    tail_action = action[7:12]
                    if (
                        steps_taken < 5
                        or np.any(np.abs(tail_action[:4]) > 1e-6)
                        or abs(float(tail_action[4]) + 1.0) > 1e-6
                    ):
                        logging.info(
                            "Executed RoboCasa action tail at step=%d chunk_idx=%d: %s",
                            steps_taken,
                            chunk_idx,
                            np.array2string(tail_action, precision=4),
                        )

                    prev_base_pos = np.asarray(obs["robot0_base_pos"], dtype=np.float32).copy()
                    obs, reward, done, _ = env.step(action)
                    if hist_len > 0:
                        agentview_history.append(current_agentview_img)
                        wrist_history.append(current_wrist_img)
                    next_base_pos = np.asarray(obs["robot0_base_pos"], dtype=np.float32).copy()
                    base_delta = next_base_pos - prev_base_pos
                    if steps_taken < 5 or np.any(np.abs(base_delta) > 1e-6):
                        logging.info(
                            "RoboCasa base_pos delta at step=%d chunk_idx=%d: %s -> %s (delta=%s)",
                            steps_taken,
                            chunk_idx,
                            np.array2string(prev_base_pos, precision=5),
                            np.array2string(next_base_pos, precision=5),
                            np.array2string(base_delta, precision=6),
                        )
                    final_reward = float(reward)
                    steps_taken += 1
                    success = bool(env._check_success())
                    if success or done:
                        break
                if success or done:
                    break
        finally:
            if args.video_dir is not None:
                video_root = Path(args.video_dir)
                video_path = str(
                    video_root
                    / family
                    / f"seed_{int(record['seed'])}_episode_{int(record['family_episode_index']):03d}_{'success' if success else 'failure'}.mp4"
                )
                _save_video(frames, Path(video_path), args.video_fps)
            env.close()

        running_family_counts[family][0] += int(success)
        running_family_counts[family][1] += 1
        family_successes, family_total = running_family_counts[family]
        logging.info(
            "[%d/%d] %s episode=%03d seed=%d success=%s steps=%d running_family_sr=%d/%d=%.3f prompt=%s",
            global_index,
            len(selected_records),
            family,
            int(record["family_episode_index"]),
            int(record["seed"]),
            success,
            steps_taken,
            family_successes,
            family_total,
            family_successes / family_total,
            prompt,
        )

        episode_results.append(
            {
                "family": family,
                "env_name": record["env_name"],
                "family_episode_index": int(record["family_episode_index"]),
                "seed": int(record["seed"]),
                "prompt": prompt,
                "layout_id": ep_meta.get("layout_id"),
                "style_id": ep_meta.get("style_id"),
                "task_refs": json.dumps(ep_meta.get("task_refs", {}), ensure_ascii=True, sort_keys=True),
                "max_steps": int(record["max_steps"]),
                "steps_taken": steps_taken,
                "success": bool(success),
                "final_reward": final_reward,
                "video_path": video_path,
            }
        )

    summary = _summarize_results(episode_results, families)
    summary["manifest_path"] = str(Path(args.manifest_path).resolve())
    summary["run_dir"] = str(run_dir.resolve())
    summary["host"] = args.host
    summary["port"] = args.port

    _write_episode_outputs(run_dir, episode_results)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "manifest_snapshot.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logging.info("Finished evaluation: micro_success_rate=%.3f macro_success_rate=%.3f", summary["micro_success_rate"], summary["macro_success_rate"])
    for family in families:
        family_summary = summary["per_family"][family]
        logging.info(
            "%s: %d/%d = %.3f",
            family,
            family_summary["successes"],
            family_summary["episodes"],
            family_summary["success_rate"],
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
