# LIBERO Training and Evaluation

This directory contains the GaussianDream LIBERO evaluation clients and robustness scripts. LIBERO remains the benchmark name; GaussianDream uses the OpenPI-derived policy server internally for checkpoint loading and inference.

## Setup

From the repository root:

```bash
# tested commit
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/libero
git -C third_party/libero checkout f78abd6
uv sync
```

For the LIBERO client environment:

```bash
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match
uv pip install -e gaussiandream-client
uv pip install -e third_party/libero
export PYTHONPATH=$PWD/src:$PWD/third_party/libero:$PYTHONPATH
```

When updating `requirements.txt`, keep the `--extra-index-url https://download.pytorch.org/whl/cu113` flag used by LIBERO.

## Policy server

Run the server from the GaussianDream root environment:

```bash
uv run scripts/serve_policy.py --env LIBERO
```

To evaluate a custom GaussianDream checkpoint:

```bash
uv run scripts/serve_policy.py \
  --port 8010 \
  policy:checkpoint \
  --policy.config gaussiandream_libero \
  --policy.dir <CHECKPOINT_DIR>/<config>/<exp>/<step>
```

Legacy config names such as `pi05_libero` are still supported for existing checkpoints.

## LIBERO training

Set the dataset and pretrained-weight paths in the main GaussianDream environment:

```bash
export LIBERO_DATA_WITH_DEPTH_ROOT=<LIBERO_DATA_WITH_DEPTH_ROOT>
export LIBERO_FLOW_ROOT=<LIBERO_FLOW_ROOT>
export GAUSSIANDREAM_PRETRAINED_DIR=<PRETRAINED_MODEL_DIR>
```

Compute normalization statistics once before the first run:

```bash
uv run scripts/compute_norm_stats.py --config-name gaussiandream_libero
```

Train with the PyTorch entrypoint:

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/train_pytorch.py \
  gaussiandream_libero \
  --exp-name gaussiandream_libero_run
```

For multi-GPU training on one node:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train_pytorch.py \
  gaussiandream_libero \
  --exp-name gaussiandream_libero_run
```

Checkpoints are written under:

```bash
checkpoints/gaussiandream_libero/gaussiandream_libero_run/
```

## LIBERO rollout

Run this in the LIBERO client environment:

```bash
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PWD/src:$PWD/third_party/libero:$PYTHONPATH
python examples/libero/main.py --args.task-suite-name libero_10
```

Use `MUJOCO_GL=glx` if EGL is not available:

```bash
MUJOCO_GL=glx python examples/libero/main.py --args.task-suite-name libero_10
```

## Docker

```bash
sudo xhost +local:docker
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

For a custom checkpoint:

```bash
export SERVER_ARGS="policy:checkpoint --policy.config gaussiandream_libero --policy.dir <CHECKPOINT_DIR>/<config>/<exp>/<step> --port 8010"
export CLIENT_ARGS="--args.task-suite-name libero_10"
docker compose -f examples/libero/compose.yml up --build
```

## Robustness evaluation scripts

These scripts are separate LIBERO robustness entry points and should be run from the repository root with the LIBERO client environment active:

- `robust_eval_light.py`: lighting perturbations.
- `robust_eval_texture.py`: object texture/material perturbations.
- `robust_eval_init_pose.py`: initial object pose perturbations.
- `robust_eval_init_force.py`: initial external force perturbations.
- `robust_eval_mid_force.py`: mid-rollout external force perturbations.

Examples:

```bash
python examples/libero/robust_eval_light.py --light-position-jitter-m 0.2
python examples/libero/robust_eval_texture.py --texture-variation swap
python examples/libero/robust_eval_init_pose.py --initial-perturb-xy-m 0.03 --initial-perturb-yaw-deg 15
python examples/libero/robust_eval_init_force.py --initial-force-xy-n 2.5 --initial-force-duration-steps 5
python examples/libero/robust_eval_mid_force.py --mid-force-xy-n 1.5 --mid-force-after-steps 40
```

`examples/libero/main.py` uses the nested `--args.` prefix for its dataclass arguments; the `robust_eval_*.py` scripts expose their arguments directly.

## Spatial analysis

```bash
python examples/libero/main.py \
  --args.use-3d-guard \
  --args.no-active-3d-takeover \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 50 \
  --args.csv-filename spatial_alignment_analysis.csv

python examples/libero/analyze_spatial_hypothesis.py \
  --csv data/libero_spatial_vis_3d_aware/videos/spatial_alignment_analysis.csv
```

## Expected output

LIBERO evaluation reports per-task and aggregate success rates. The upstream OpenPI checkpoint `gs://openpi-assets/checkpoints/pi05_libero/` remains usable as a baseline; GaussianDream checkpoints should use the matching GaussianDream config aliases documented above.
