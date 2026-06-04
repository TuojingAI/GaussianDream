import dataclasses
import pathlib
import logging
from collections import deque

import imageio
import numpy as np
from gaussiandream_client import websocket_client_policy as _websocket_client_policy
import robocasa  # noqa: F401  # Registers RoboCasa environments with robosuite.make
import robosuite
from robosuite.controllers import load_composite_controller_config
import tyro


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8010
    env_name: str = "PnPCounterToCab"
    prompt: str | None = None
    num_episodes: int = 20
    max_steps: int = 500
    replan_steps: int = 1
    camera_height: int = 128
    camera_width: int = 128
    save_videos: bool = True
    video_dir: str = "robocasa_eval_videos"


DEFAULT_PROMPTS = {
    "PnPCounterToCab": "pick and place from counter to cabinet",
    "PnPCounterToSink": "pick and place from counter to sink",
    "PnPMicrowaveToCounter": "pick and place from microwave to counter",
    "PnPStoveToCounter": "pick and place from stove to counter",
    "OpenSingleDoor": "open cabinet or microwave door",
    "CloseDrawer": "close drawer",
    "TurnOnMicrowave": "turn on microwave",
    "TurnOnSinkFaucet": "turn on sink faucet",
    "TurnOnStove": "turn on stove",
    "ArrangeVegetables": "arrange vegetables on a cutting board",
    "MicrowaveThawing": "place frozen food in microwave for thawing",
    "RestockPantry": "restock cans in pantry",
    "PreSoakPan": "prepare pan for washing",
    "PrepareCoffee": "make coffee",
}


def _get_prompt(args: Args) -> str:
    return args.prompt or DEFAULT_PROMPTS.get(args.env_name, args.env_name)


def _get_state(obs: dict) -> np.ndarray:
    """Match the 16-D mobile-manipulation state used by the RoboCasa training set."""
    state_keys = [
        "robot0_base_to_eef_pos",
        "robot0_base_to_eef_quat",
        "robot0_gripper_qpos",
        "robot0_base_pos",
        "robot0_base_quat",
    ]
    return np.concatenate([np.asarray(obs[key], dtype=np.float32).reshape(-1) for key in state_keys])


def _is_success(env) -> bool:
    if hasattr(env, "is_success"):
        succ = env.is_success()
        if isinstance(succ, dict):
            return bool(succ.get("task", False))
        return bool(succ)
    return bool(env._check_success())


def main(args: Args):
    logging.basicConfig(level=logging.INFO)

    # Initialize Policy Client
    logging.info(f"Connecting to policy server at {args.host}:{args.port}")
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Configure Robocasa Environment
    # Note: Ensure these configurations match what the model expects (camera names, resolution, etc.)
    config = {
        "env_name": args.env_name,
        "robots": "PandaOmron",
        "controller_configs": load_composite_controller_config(robot="PandaOmron"),
    }

    logging.info(f"Creating environment: {args.env_name}")
    env = robosuite.make(
        **config,
        has_renderer=False,  # Set to True if you want to see the simulation window
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["robot0_agentview_left", "robot0_eye_in_hand"],
        camera_heights=args.camera_height,
        camera_widths=args.camera_width,
        reward_shaping=False,
        control_freq=20,
    )
    prompt = _get_prompt(args)
    logging.info("Using prompt: %s", prompt)

    success_count = 0
    if args.save_videos:
        pathlib.Path(args.video_dir).mkdir(parents=True, exist_ok=True)

    for episode_idx in range(args.num_episodes):
        obs = env.reset()
        action_plan: deque[np.ndarray] = deque()
        frames: list[np.ndarray] = []
        episode_success = False
        last_reward = 0.0

        logging.info("Episode %d/%d started", episode_idx + 1, args.num_episodes)

        for step in range(args.max_steps):
            agentview_img = obs["robot0_agentview_left_image"]
            wrist_img = obs["robot0_eye_in_hand_image"]

            if args.save_videos:
                frames.append(np.flipud(agentview_img))

            request = {
                "observation/image": agentview_img,
                "observation/wrist_image": wrist_img,
                "observation/state": _get_state(obs),
                "prompt": prompt,
            }

            if not action_plan:
                response = client.infer(request)
                action_chunk = np.asarray(response["actions"])
                steps_to_use = min(len(action_chunk), max(1, args.replan_steps))
                action_plan.extend(action_chunk[:steps_to_use])

            action = np.asarray(action_plan.popleft()).copy()
            obs, reward, done, info = env.step(action)
            last_reward = float(reward)

            if step % 10 == 0:
                logging.info("Episode %d step %d: reward=%.4f", episode_idx + 1, step, reward)

            if _is_success(env):
                episode_success = True
                success_count += 1
                logging.info("Episode %d succeeded at step %d", episode_idx + 1, step)
                break

            if done:
                logging.info("Episode %d terminated at step %d", episode_idx + 1, step)
                break

        if args.save_videos and frames:
            video_path = pathlib.Path(args.video_dir) / f"{args.env_name}_ep{episode_idx:03d}.mp4"
            try:
                imageio.mimsave(video_path, frames, fps=10)
                logging.info("Saved rollout video to %s", video_path)
            except Exception as exc:
                logging.error("Failed to save video %s: %s", video_path, exc)

        running_rate = success_count / float(episode_idx + 1)
        logging.info(
            "Episode %d result | success=%s reward=%.4f running_success_rate=%d/%d=%.3f",
            episode_idx + 1,
            episode_success,
            last_reward,
            success_count,
            episode_idx + 1,
            running_rate,
        )

    final_rate = success_count / float(max(1, args.num_episodes))
    logging.info(
        "Final success rate for %s: %d/%d = %.3f",
        args.env_name,
        success_count,
        args.num_episodes,
        final_rate,
    )

    env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
