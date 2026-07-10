# ALOHA Hardware

This directory contains the hardware runtime and dataset conversion helpers for ALOHA-style setups.
The runtime talks to a GaussianDream policy server through `gaussiandream-client`, while the robot-side
control stack still depends on the ROS workspace in `third_party/aloha`.

## What is included

- `examples/aloha_real/main.py`: hardware rollout client.
- `examples/aloha_real/env.py`: wraps hardware observations/actions into the policy-server API.
- `examples/aloha_real/real_env.py`: low-level ALOHA robot interface.
- `examples/aloha_real/convert_aloha_data_to_lerobot.py`: convert processed HDF5 episodes into LeRobot format.
- `scripts/process_data_pickle2hdf5.py`: convert your raw pickle logs into nested HDF5 episodes.
- `scripts/run_real_robot_pipeline.py`: one-command preprocessing helper for raw pickle -> HDF5 -> LeRobot.

## Prerequisites

- An ALOHA-compatible robot setup.
- ROS Noetic and the Interbotix/ALOHA stack.
- RealSense cameras configured in `third_party/aloha/aloha_scripts/realsense_publisher.py`.
- A GaussianDream environment for serving the policy.

Clone the ROS package into `third_party/aloha` before using this flow:

```bash
git clone https://github.com/Physical-Intelligence/aloha.git third_party/aloha
```

If you need stricter reproducibility for hardware bring-up, pin a tested ALOHA commit in your local setup notes as well.

## Client environment

Create a dedicated Python 3.10 environment for the hardware client:

```bash
uv venv --python 3.10 examples/aloha_real/.venv
source examples/aloha_real/.venv/bin/activate
uv pip sync examples/aloha_real/requirements.txt
uv pip install -e gaussiandream-client
```

## Run the hardware client

Terminal 1, start ROS:

```bash
roslaunch aloha ros_nodes.launch
```

Terminal 2, run the ALOHA client:

```bash
source examples/aloha_real/.venv/bin/activate
python -m examples.aloha_real.main --host 127.0.0.1 --port 8000
```

Terminal 3, serve a policy from the main GaussianDream environment:

```bash
uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config gaussiandream_aloha \
  --policy.dir <ALOHA_CKPT_DIR>/gaussiandream_aloha/<exp>/<step>
```

If you want to test the stock ALOHA default instead, you can still run:

```bash
uv run scripts/serve_policy.py --env ALOHA --default_prompt "take the toast out of the toaster"
```

## Preprocess an ALOHA dataset

If your collected data is raw pickle logs, the shortest path is:

```bash
python scripts/run_real_robot_pipeline.py \
  --raw-dir <RAW_PICKLE_DIR> \
  --lerobot-root <LEROBOT_ROOT> \
  --repo-id local/gaussiandream_aloha \
  --config-name gaussiandream_aloha \
  --exp-name gaussiandream_aloha_run
```

This helper does two things:

1. runs `scripts/process_data_pickle2hdf5.py` to create nested HDF5 episodes;
2. runs `examples/aloha_real/convert_aloha_data_to_lerobot.py` to write a LeRobot dataset.

The script prints the follow-up `HF_LEROBOT_HOME`, norm-stats, and training commands at the end.

## Manual conversion flow

If you want the steps separately:

```bash
python scripts/process_data_pickle2hdf5.py \
  --load_dir <RAW_PICKLE_DIR> \
  --output_dir <PROCESSED_HDF5_DIR> \
  --task_name gaussiandream_aloha

export HF_LEROBOT_HOME=<LEROBOT_ROOT>
python examples/aloha_real/convert_aloha_data_to_lerobot.py \
  --raw-dir <PROCESSED_HDF5_DIR> \
  --repo-id local/gaussiandream_aloha \
  --local-dir <LEROBOT_ROOT>/local/gaussiandream_aloha \
  --mode image
```

Useful options:

- `scripts/process_data_pickle2hdf5.py --trim_deadzone --trim_mode end`
- `scripts/run_real_robot_pipeline.py --instruction-selection first`
- `scripts/run_real_robot_pipeline.py --skip-hdf5`

## Hardware training configs

The repo now includes these configs in `src/gaussiandream/training/config.py`:

- `gaussiandream_aloha_demo`
- `gaussiandream_aloha_jax`
- `gaussiandream_aloha_torch`
- `gaussiandream_aloha`

Legacy aliases from the old project are still registered for compatibility, but the recommended names are the
`gaussiandream_*` ones above.

Recommended env vars before training:

```bash
export HF_LEROBOT_HOME=<LEROBOT_ROOT>
export GAUSSIANDREAM_ALOHA_CKPT_DIR=<ALOHA_CKPT_ROOT>
export GAUSSIANDREAM_ALOHA_CKPT_DIR_TORCH=<ALOHA_CKPT_ROOT_TORCH>
export GAUSSIANDREAM_PRETRAINED_DIR=<PRETRAINED_MODEL_DIR>
```

Compute normalization statistics:

```bash
uv run scripts/compute_norm_stats.py --config-name gaussiandream_aloha
```

Train the recommended PyTorch hardware config:

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/train_pytorch.py \
  gaussiandream_aloha \
  --exp-name gaussiandream_aloha_run
```

Train the JAX config instead:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  gaussiandream_aloha_jax \
  --exp-name gaussiandream_aloha_jax_run
```

## Notes on the custom ALOHA data config

`LeRobotCustomDataConfig` assumes:

- one base camera stored as `observation.images.cam_high`
- one wrist camera stored as `observation.images.cam_left_wrist`
- state stored as `observation.state`
- actions stored as `action`
- task text stored as `task`

It mirrors the single wrist camera into the right-wrist input slot and converts the first 6 action dimensions
from absolute joint targets to delta actions, while keeping the gripper dimension absolute.

If your dataset keys differ, update `LeRobotCustomDataConfig` in `src/gaussiandream/training/config.py`.

## Docker

A ROS-based Docker setup is included:

```bash
docker compose -f examples/aloha_real/compose.yml up --build
```

This path depends on `scripts/docker/serve_policy.Dockerfile` and the vendored `third_party/aloha` ROS package.
