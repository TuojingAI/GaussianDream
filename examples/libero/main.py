from __future__ import annotations

import os
import sys

# CRITICAL: Set rendering backend BEFORE any imports that might initialize OpenGL/GLFW
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"  # Use EGL for GPU-accelerated rendering

# Disable GLFW (prevents X11 initialization issues in headless environments)
os.environ["PYOPENGL_PLATFORM"] = "egl"

import collections
import dataclasses
import hashlib
import logging
import math
import pathlib
import pickle

import imageio
import numpy as np
from gaussiandream_client import image_tools
from gaussiandream_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8045
    resize_size: int = 224
    replan_steps: int = 5  # Test with 1 for closed-loop control

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_10"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    task_id: list[int] | None = None  # Specific task ID(s) to evaluate (None = evaluate all tasks)
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data_502/gaussian_vla_exp501_45000_libero_10/videos"  # Path to save videos
    # Gaussian_vla_exp315_12000_libero_10 这个实际上是goal
    save_videos: bool = True  # Whether to save rollout videos
    seed: int = 7  # Random Seed (for reproducibility)
    debug_log_path: str | None = None  # Path to save debug logs (None = disabled)
    # Match training `temporal_context_offsets=(-10, -5, 0)`: steps before current at control (env) rate (~10Hz).
    # History shorter than max offset pads with the current frame (same as server-side repeat, but real frames when available).
    agent_temporal_step_offsets: tuple[int, int, int] = (10, 5, 0)


def _stack_agent_temporal_frames(
    past_frames: collections.deque,
    current_frame: np.ndarray,
    step_offsets: tuple[int, ...],
) -> np.ndarray:
    """Build (T, H, W, C) uint8 stack: one slot per offset in order (oldest context first).

    `step_offsets` are non-negative integers = control steps before the current frame
    (10, 5, 0 corresponds to training t-10, t-5, t). If a requested index is before the
    start of `past_frames + [current]`, use the newest available frame (typically current).
    """
    seq = list(past_frames) + [current_frame]
    l = len(seq)
    out: list[np.ndarray] = []
    for off in step_offsets:
        i = l - 1 - off
        if i < 0:
            out.append(seq[-1])
        else:
            out.append(seq[i])
    return np.stack(out, axis=0)


def _configure_logging(video_out_path: str) -> None:
    data_dir = pathlib.Path(video_out_path).parent
    data_dir.mkdir(parents=True, exist_ok=True)

    log_file = data_dir / "eval.log"
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    logging.info(f"Logging to: {log_file}")


def eval_libero(args: Args) -> None:
    from libero.libero import benchmark

    _configure_logging(args.video_out_path)

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Setup debug logging directory
    debug_log_dir = None
    if args.debug_log_path:
        debug_log_dir = pathlib.Path(args.debug_log_path)
        debug_log_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Debug logging enabled, saving to: {debug_log_dir}")

    # Start evaluation
    total_episodes, total_successes = 0, 0

    # Determine which tasks to evaluate
    if args.task_id is not None:
        task_ids = args.task_id if isinstance(args.task_id, list) else [args.task_id]
        logging.info(f"Evaluating tasks: {task_ids}")
    else:
        task_ids = range(num_tasks_in_suite)
        logging.info(f"Evaluating all {num_tasks_in_suite} tasks")

    for task_id in tqdm.tqdm(task_ids):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()
            hist_len = max(args.agent_temporal_step_offsets)
            agent_img_history: collections.deque = collections.deque(maxlen=hist_len)

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            debug_records = []  # per-step debug records for this episode
            done = False

            logging.info(f"Starting episode {task_episodes + 1}...")
            logging.info(
                f"Simulator warmup: {args.num_steps_wait} env steps for objects to settle "
                "(dummy actions only; no policy calls yet)."
            )
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    if t == args.num_steps_wait:
                        logging.info(f"Warmup finished at env timestep {t}. Starting policy-controlled rollout.")

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    agent_img_stack = _stack_agent_temporal_frames(
                        agent_img_history, img, args.agent_temporal_step_offsets
                    )

                    # Save preprocessed image for replay video
                    if args.save_videos:
                        replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": agent_img_stack,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        logging.debug(
                            f"Querying policy server for a new action chunk (env timestep {t}). "
                            "First call can take a long time; if this line stays for many minutes, "
                            "check the policy server process and GPU in another terminal."
                        )
                        infer_result = client.infer(element)
                        action_chunk = infer_result["actions"]
                        logging.debug(
                            f"Received action chunk of length {len(action_chunk)} "
                            f"(using {args.replan_steps} steps per replan)."
                        )
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                        # Debug logging: record input and output
                        if debug_log_dir:
                            debug_records.append({
                                "timestep": t,
                                "state": element["observation/state"].copy(),
                                "image": img.copy(),
                                "agent_image_stack": agent_img_stack.copy(),
                                "wrist_image": wrist_img.copy(),
                                "image_hash": hashlib.md5(img.tobytes()).hexdigest(),
                                "agent_image_stack_hash": hashlib.md5(agent_img_stack.tobytes()).hexdigest(),
                                "wrist_image_hash": hashlib.md5(wrist_img.tobytes()).hexdigest(),
                                "prompt": element["prompt"],
                                "action_chunk": [a.tolist() if hasattr(a, 'tolist') else a for a in action_chunk],
                            })

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    agent_img_history.append(img)
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save debug records for this episode
            if debug_log_dir and debug_records:
                pkl_path = debug_log_dir / f"task_{task_id}_ep_{episode_idx}.pkl"
                with open(pkl_path, "wb") as f:
                    pickle.dump({
                        "task_id": task_id,
                        "episode_idx": episode_idx,
                        "task_description": str(task_description),
                        "success": done,
                        "total_steps": t,
                        "records": debug_records,
                    }, f)

            # Save a replay video of the episode
            if args.save_videos:
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)

    # Set seed - handle both old gym API (seed method) and new API (reset with seed)
    try:
        env.seed(seed)  # Old gym API
    except (TypeError, AttributeError):
        # New gym/gymnasium API - seed is passed to reset()
        pass  # Will be handled in reset() call

    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    tyro.cli(eval_libero)
