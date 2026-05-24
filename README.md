## :book: GaussianDream: A Feed-Forward 3D Gaussian World Model for Robotic Manipulation

<p align="center">
  <small>🔥 We would appreciate it if you could star GaussianDream ⭐ and share it. Thanks! 🔥</small>
</p>

> Zijian Zhang<sup>2,3,1,&#42;</sup>, Yuqing Jiang<sup>2,3,1,&#42;</sup>, Qian Cheng<sup>4</sup>, Si Liu<sup>5</sup>, Ding Zhao<sup>6</sup>, Ping Luo<sup>7</sup>, Weitao Zhou<sup>4</sup>, Haibao Yu<sup>7,1,#</sup> <br>
> <sup>1</sup> Tuojing Intelligence, <sup>2</sup> University of Chinese Academy of Sciences, <sup>3</sup> Institute of Automation, Chinese Academy of Sciences <br>
> <sup>4</sup> Tsinghua University, <sup>5</sup> Beihang University, <sup>6</sup> Carnegie Mellon University, <sup>7</sup> The University of Hong Kong <br>
> <sup>&#42;</sup> Equal contribution, <sup>#</sup> Corresponding author

<p align="center">

[![Code](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/TuojingAI/GaussianDream)&nbsp;
[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-red)](#release-status)&nbsp;
[![Release](https://img.shields.io/badge/Release-Coming%20Soon-blue)](#release-status)

</p>

## Introduction

GaussianDream is a feed-forward 3D Gaussian world model for robotic manipulation. The core implementation lives under the `gaussiandream` Python package, while legacy checkpoint/config identifiers and external asset paths are kept compatible with upstream OpenPI releases.

<div align="center">
<img src="assets/illustration/comparation_3.png" />
</div>

## Framework

<div align="center">
<img src="assets/illustration/framework_final_v.drawio.png" />
</div>

## Installation

```bash
git clone https://github.com/TuojingAI/GaussianDream.git
cd GaussianDream
git submodule update --init --recursive
uv sync
```

The package metadata and implementation import namespace are both `gaussiandream`.

Optional cache paths:

```bash
export GAUSSIANDREAM_DATA_HOME=<CACHE_DIR>
export CHECKPOINT_DIR=<CHECKPOINT_DIR>
export DATA_ROOT=<DATA_ROOT>
```

`OPENPI_DATA_HOME` is still supported as a fallback for users with existing caches.

## Evaluation tracks

GaussianDream includes three evaluation paths:

- real-robot / runtime clients built around the shared policy server and `openpi-client`
- LIBERO simulation evaluation in `examples/libero/`
- RoboCasa simulation evaluation in `examples/robocasa/`

Start a policy server from a checkpoint:

```bash
uv run scripts/serve_policy.py \
  --port 8010 \
  policy:checkpoint \
  --policy.config gaussiandream_libero \
  --policy.dir <CHECKPOINT_DIR>/<config>/<exp>/<step>
```

For RoboCasa checkpoints, use `--policy.config gaussiandream_robocasa` and a matching checkpoint directory.

### LIBERO

See `examples/libero/README.md` for setup and evaluation commands.

Typical flow:

```bash
uv run scripts/serve_policy.py --env LIBERO
python examples/libero/main.py --args.task-suite-name libero_10
```

### RoboCasa

See `examples/robocasa/README.md` for conda setup, RoboCasa assets, and rollout commands.

Typical flow:

```bash
uv run scripts/serve_policy.py \
  --port 8010 \
  policy:checkpoint \
  --policy.config gaussiandream_robocasa \
  --policy.dir <CHECKPOINT_DIR>/<config>/<exp>/<step>

python examples/robocasa/main.py \
  --host 127.0.0.1 \
  --port 8010 \
  --env-name PnPCounterToCab \
  --prompt "pick and place from counter to cabinet"
```

For RoboCasa H50 temporal evaluation, use `examples/robocasa/eval_h50_temporal.py` after installing the RoboCasa client environment.

## Experiments

<div align="center">
<img src="assets/illustration/q_vis.drawio.png" />
<br>
<img src="assets/illustration/piper_setup_2.drawio.png" />
</div>

## Release status

Paper, checkpoints, datasets, and complete reproduction instructions are coming soon. Large artifacts such as datasets, checkpoints, rendered videos, logs, and experiment outputs are intentionally not tracked in git.

## Contact

If you have any question, please email zijianzhang821@gmail.com.

## Sincere Acknowledgement

Appreciate the following works for their great contributions:

- VGGT: Inspires our 3D-aware visual representation design.
- OpenPI and π0: Serve as the foundation for our policy backbone and codebase.
- 3D Gaussian Splatting: Inspires our 3D Gaussian world modeling design.
- LIBERO and RoboCasa: Serve as the benchmarks for simulation training and evaluation.

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{zhang2026gaussiandream,
  title={GaussianDream: A Feed-Forward 3D Gaussian World Model for Robotic Manipulation},
  author={Zhang, Zijian and Jiang, Yuqing and Cheng, Qian and Liu, Si and Zhao, Ding and Luo, Ping and Zhou, Weitao and Yu, Haibao},
  year={2026}
}
```
