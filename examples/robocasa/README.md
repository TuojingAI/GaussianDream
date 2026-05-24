# RoboCasa Evaluation

This directory contains the RoboCasa rollout clients for GaussianDream policy servers. RoboCasa and robosuite run in a separate client environment, while the policy server runs from the main GaussianDream environment.

## Setup

Prerequisites:

- Python 3.10
- Conda
- RoboCasa kitchen assets
- a GaussianDream checkpoint trained with a RoboCasa-compatible config

Create the RoboCasa client environment:

```bash
conda create -n robocasa python=3.10
conda activate robocasa
cd <GAUSSIANDREAM_ROOT>
```

Install client-side dependencies:

```bash
pip install -r examples/robocasa/requirements.in
pip install -e third_party/robocasa
pip install -e openpi-client
```

Download RoboCasa kitchen assets:

```bash
python third_party/robocasa/robocasa/scripts/download_kitchen_assets.py
```

If RoboCasa cannot find MuJoCo libraries, export the MuJoCo package path from your conda environment:

```bash
export LD_LIBRARY_PATH=<CONDA_ENV>/lib/python3.10/site-packages/mujoco:$LD_LIBRARY_PATH
export MUJOCO_GL=egl
```

Quick check:

```bash
python -c "import mujoco, robocasa, robosuite, openpi_client, tyro; print('all ok')"
```

## Environment sanity check

```bash
python examples/robocasa/check_env.py
```

## Single-task rollout evaluation

Terminal 1, start the GaussianDream policy server from the main environment:

```bash
cd <GAUSSIANDREAM_ROOT>
uv run scripts/serve_policy.py \
  --port 8010 \
  policy:checkpoint \
  --policy.config gaussiandream_robocasa \
  --policy.dir <CHECKPOINT_DIR>/<config>/<exp>/<step>
```

Terminal 2, run the RoboCasa client from the `robocasa` conda environment:

```bash
conda activate robocasa
cd <GAUSSIANDREAM_ROOT>
export MUJOCO_GL=egl

python examples/robocasa/main.py \
  --host 127.0.0.1 \
  --port 8010 \
  --env-name PnPCounterToCab \
  --prompt "pick and place from counter to cabinet" \
  --num-episodes 20 \
  --max-steps 500 \
  --replan-steps 1
```

To save videos, add:

```bash
--save-videos --video-dir <OUTPUT_DIR>/robocasa_videos
```

## RoboCasa H50 temporal evaluation

For the H50 multi-family temporal evaluation, run:

```bash
python examples/robocasa/eval_h50_temporal.py \
  --host 127.0.0.1 \
  --port 8010 \
  --families all \
  --episodes-per-family 50 \
  --temporal-step-offsets 10,5,0 \
  --output-dir <OUTPUT_DIR>/robocasa_h50
```

Parse a run directory with:

```bash
python examples/robocasa/parse_eval.py <OUTPUT_DIR>/robocasa_h50/<run_dir>
```

## Expected outputs

During evaluation you should see lines like:

```text
Episode 1 result | success=False reward=0.0000 running_success_rate=0/1=0.000
Episode 2 result | success=True reward=0.0000 running_success_rate=1/2=0.500
Final success rate for PnPCounterToCab: 7/20 = 0.350
```

## Docker

A Docker-based client flow is available for local debugging:

```bash
docker compose -f examples/robocasa/compose.yml build
docker compose -f examples/robocasa/compose.yml run --rm robocasa
```

For local checkpoint evaluation, the conda client plus GaussianDream policy server split is the recommended path.

## Common issues

- `ModuleNotFoundError: No module named 'tyro'`: install `examples/robocasa/requirements.in` into the `robocasa` conda environment.
- `libmujoco.so.3.6.0: cannot open shared object file`: export `LD_LIBRARY_PATH` to the MuJoCo package directory.
- `Environment PnPCounterToCab not found`: make sure `third_party/robocasa` is installed in the current environment.
- MP4 writing fails: install `imageio[ffmpeg]` or run without `--save-videos`.
