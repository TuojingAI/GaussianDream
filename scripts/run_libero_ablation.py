"""Launch predefined LIBERO GS-VLA ablation experiments.

This script clones the registered ``pi05_libero`` training config, applies a
named ablation preset, prints the effective differences, and can directly call
the existing PyTorch training loop.

Example usages
--------------
Dry-run and print the effective paper-budget config:
  uv run scripts/run_libero_ablation.py --preset no_flow --dry-run

Launch a single-process smoke run with an auto-generated experiment name:
  CUDA_VISIBLE_DEVICES=0 uv run scripts/run_libero_ablation.py --preset no_renderer --budget smoke_2k

Override the experiment name and checkpoint directory:
  CUDA_VISIBLE_DEVICES=0 uv run scripts/run_libero_ablation.py \
      --preset no_flow \
      --exp-name gaussiandream_no_flow \
      --checkpoint-base-dir <CHECKPOINT_BASE_DIR>
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import pathlib
import sys
from typing import Any

import tyro


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import gaussiandream.training.config as _config


TRAIN_PYTORCH_PATH = PROJECT_ROOT / "scripts" / "train_pytorch.py"


BUDGETS: dict[str, dict[str, Any]] = {
    # Match the practical full GS-VLA training budget used for the LIBERO comparison:
    # 15k representation/world-model steps + 15k action-focused steps.
    "paper_30k": {
        "num_train_steps": 30_000,
        "stage1_steps": 15_000,
        "save_interval": 3_000,
    },
    # Keep whatever the base config currently says. Useful when reproducing an
    # older run whose TrainConfig already encodes the intended schedule.
    "config": {},
    # Cheap wiring check; disables wandb by default so failed smoke tests do not
    # create noisy runs.
    "smoke_2k": {
        "num_train_steps": 2_000,
        "stage1_steps": 1_000,
        "save_interval": 1_000,
        "wandb_enabled": False,
    },
}


def _load_train_pytorch_module():
    spec = importlib.util.spec_from_file_location("train_pytorch_entry", TRAIN_PYTORCH_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load training entry from {TRAIN_PYTORCH_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _set_attr(obj: Any, name: str, value: Any) -> None:
    object.__setattr__(obj, name, value)


def _apply_updates(obj: Any, updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        _set_attr(obj, key, value)


def _summarize_config(cfg: _config.TrainConfig) -> dict[str, Any]:
    model = cfg.model
    return {
        "name": cfg.name,
        "exp_name": cfg.exp_name,
        "checkpoint_dir": str(cfg.checkpoint_dir),
        "batch_size": cfg.batch_size,
        "num_train_steps": cfg.num_train_steps,
        "save_interval": cfg.save_interval,
        "wandb_enabled": cfg.wandb_enabled,
        "stage1_steps": cfg.stage1_steps,
        "stage1_render_weight": getattr(cfg, "stage1_render_weight", None),
        "stage2_render_weight": getattr(cfg, "stage2_render_weight", None),
        "stage2_world_loss_multiplier": cfg.stage2_world_loss_multiplier,
        "stage2_keep_world_model_trainable": getattr(cfg, "stage2_keep_world_model_trainable", None),
        "stage2_shared_backbone_lr_scale": getattr(cfg, "stage2_shared_backbone_lr_scale", None),
        "model": {
            "use_gaussian": getattr(model, "use_gaussian", None),
            "use_world_model": getattr(model, "use_world_model", None),
            "use_current_gaussian_tokens": getattr(model, "use_current_gaussian_tokens", None),
            "disable_future_tokens": getattr(model, "disable_future_tokens", None),
            "use_velocity_future_gaussians": getattr(model, "use_velocity_future_gaussians", None),
            "use_future_depth_aux": getattr(model, "use_future_depth_aux", None),
            "current_frame_recon_loss_weight": getattr(model, "current_frame_recon_loss_weight", None),
            "render_loss_weight": getattr(model, "render_loss_weight", None),
            "depth_loss_weight": getattr(model, "depth_loss_weight", None),
            "future_depth_aux_loss_weight": getattr(model, "future_depth_aux_loss_weight", None),
            "flow_loss_weight": getattr(model, "flow_loss_weight", None),
            "flow_first_horizon_only": getattr(model, "flow_first_horizon_only", None),
            "use_lpips": getattr(model, "use_lpips", None),
            "future_motion_depth_weight": getattr(model, "future_motion_depth_weight", None),
            "temporal_context_offsets": getattr(model, "temporal_context_offsets", None),
            "future_prediction_offsets": getattr(model, "future_prediction_offsets", None),
            "action_warmup_steps": getattr(model, "action_warmup_steps", None),
        },
    }


ABLATION_PRESETS: dict[str, dict[str, Any]] = {
    "full": {
        "description": "Full GS-VLA baseline matching the current pi05_libero config.",
        "train": {},
        "model": {},
    },
    "base_vla": {
        "description": "Pure VLA baseline without Gaussian geometry or world modeling.",
        "train": {
            "stage1_steps": 0,
            "stage2_world_loss_multiplier": 0.0,
            "stage2_keep_world_model_trainable": False,
        },
        "model": {
            "use_gaussian": False,
            "use_world_model": False,
            "use_velocity_future_gaussians": False,
            "use_future_depth_aux": False,
            "render_loss_weight": 0.0,
            "depth_loss_weight": 0.0,
            "future_depth_aux_loss_weight": 0.0,
            "flow_loss_weight": 0.0,
            "current_frame_recon_loss_weight": 0.0,
        },
    },
    "no_structured_future_rollout": {
        "description": (
            "Disable static-template + velocity rollout while keeping future query tokens and depth/render "
            "world supervision. Flow is also disabled because there is no explicit delta field to supervise."
        ),
        "train": {},
        "model": {
            "use_velocity_future_gaussians": False,
            "flow_loss_weight": 0.0,
        },
    },
    "no_flow": {
        "description": "Disable 3D flow / motion supervision while keeping the GS world model.",
        "train": {},
        "model": {
            "flow_loss_weight": 0.0,
        },
    },
    "no_future_depth_aux": {
        "description": "Disable the auxiliary future depth head and its supervision.",
        "train": {},
        "model": {
            "use_future_depth_aux": False,
            "future_depth_aux_loss_weight": 0.0,
        },
    },
    "no_renderer": {
        "description": "Disable all rendering-based supervision while keeping depth / flow branches.",
        "train": {
            "stage1_render_weight": 0.0,
            "stage2_render_weight": 0.0,
        },
        "model": {
            "render_loss_weight": 0.0,
            "current_frame_recon_loss_weight": 0.0,
            "use_lpips": False,
        },
    },
    "no_depth": {
        "description": "Disable current/future depth supervision and depth-based motion weighting; keep renderer / flow active.",
        "train": {},
        "model": {
            "depth_loss_weight": 0.0,
            "use_future_depth_aux": False,
            "future_depth_aux_loss_weight": 0.0,
            "future_motion_depth_weight": 0.0,
        },
    },
    "depth_only_proxy": {
        "description": (
            "Depth-only weaker geometry proxy: keep future tokens and depth-style supervision, "
            "remove render/flow and the structured velocity rollout."
        ),
        "train": {
            "stage1_render_weight": 0.0,
            "stage2_render_weight": 0.0,
        },
        "model": {
            "use_velocity_future_gaussians": False,
            "render_loss_weight": 0.0,
            "current_frame_recon_loss_weight": 0.0,
            "flow_loss_weight": 0.0,
            "use_lpips": False,
        },
    },
    "current_gaussian_only": {
        "description": (
            "Keep current-scene Gaussian/VGGT tokens in the policy prefix and supervise only current-frame "
            "Gaussian reconstruction; remove predictive future tokens, structured rollout, flow, and future depth aux."
        ),
        "train": {},
        "model": {
            "use_gaussian": True,
            "use_world_model": True,
            "use_current_gaussian_tokens": True,
            "disable_future_tokens": True,
            "use_velocity_future_gaussians": False,
            "use_future_depth_aux": False,
            "future_depth_aux_loss_weight": 0.0,
            "flow_loss_weight": 0.0,
            "future_motion_depth_weight": 0.0,
        },
    },
    "no_future_tokens": {
        "description": (
            "Remove predictive future/world query tokens from the policy prefix. "
            "World-model supervision is disabled because there is no future-token latent to decode."
        ),
        "train": {
            "stage1_steps": 0,
            "stage2_world_loss_multiplier": 0.0,
            "stage2_keep_world_model_trainable": False,
        },
        "model": {
            "use_world_model": False,
            "use_velocity_future_gaussians": False,
            "use_future_depth_aux": False,
            "render_loss_weight": 0.0,
            "depth_loss_weight": 0.0,
            "future_depth_aux_loss_weight": 0.0,
            "flow_loss_weight": 0.0,
            "current_frame_recon_loss_weight": 0.0,
            "use_lpips": False,
        },
    },
}


@dataclasses.dataclass
class Args:
    preset: str | None = None
    base_config: str = "pi05_libero"
    budget: str = "paper_30k"
    exp_name: str | None = None
    checkpoint_base_dir: str | None = None
    assets_base_dir: str | None = None
    pytorch_weight_path: str | None = None
    batch_size: int | None = None
    num_train_steps: int | None = None
    stage1_steps: int | None = None
    save_interval: int | None = None
    log_interval: int | None = None
    action_warmup_steps: int | None = None
    wandb_enabled: bool | None = None
    overwrite: bool = False
    resume: bool = False
    dry_run: bool = False
    print_config: bool = True
    list_presets: bool = False


def _build_config(args: Args) -> tuple[_config.TrainConfig, dict[str, Any]]:
    if args.preset not in ABLATION_PRESETS:
        raise ValueError(
            f"Unknown preset {args.preset!r}. Available presets: {', '.join(sorted(ABLATION_PRESETS))}"
        )

    preset = ABLATION_PRESETS[args.preset]
    cfg = copy.deepcopy(_config.get_config(args.base_config))

    if args.exp_name is None:
        exp_name = f"{args.base_config}_{args.preset}"
    else:
        exp_name = args.exp_name

    _set_attr(cfg, "exp_name", exp_name)
    _set_attr(cfg, "overwrite", args.overwrite)
    _set_attr(cfg, "resume", args.resume)

    if args.budget not in BUDGETS:
        raise ValueError(f"Unknown budget {args.budget!r}. Available budgets: {', '.join(sorted(BUDGETS))}")
    _apply_updates(cfg, BUDGETS[args.budget])
    _apply_updates(cfg, preset["train"])
    _apply_updates(cfg.model, preset["model"])

    if args.checkpoint_base_dir is not None:
        _set_attr(cfg, "checkpoint_base_dir", args.checkpoint_base_dir)
    if args.assets_base_dir is not None:
        _set_attr(cfg, "assets_base_dir", args.assets_base_dir)
    if args.pytorch_weight_path is not None:
        _set_attr(cfg, "pytorch_weight_path", args.pytorch_weight_path)
    if args.batch_size is not None:
        _set_attr(cfg, "batch_size", args.batch_size)
    if args.num_train_steps is not None:
        _set_attr(cfg, "num_train_steps", args.num_train_steps)
    if args.stage1_steps is not None:
        _set_attr(cfg, "stage1_steps", args.stage1_steps)
    if args.save_interval is not None:
        _set_attr(cfg, "save_interval", args.save_interval)
    if args.log_interval is not None:
        _set_attr(cfg, "log_interval", args.log_interval)
    if args.action_warmup_steps is not None:
        _set_attr(cfg.model, "action_warmup_steps", args.action_warmup_steps)
    if args.wandb_enabled is not None:
        _set_attr(cfg, "wandb_enabled", args.wandb_enabled)
    return cfg, preset


def _print_presets() -> None:
    print("Available LIBERO ablation presets:")
    for name in sorted(ABLATION_PRESETS):
        print(f"- {name}: {ABLATION_PRESETS[name]['description']}")
    print("\nAvailable budgets:")
    for name in sorted(BUDGETS):
        print(f"- {name}: {BUDGETS[name] if BUDGETS[name] else 'use base config schedule'}")


def _install_ablation_runtime_patches() -> None:
    """Keep loss-disable ablations from doing disabled supervision work.

    The base model source is shared with ongoing experiments, so the ablation
    launcher applies a narrow runtime patch instead of changing global training
    behavior. In particular, ``render_loss_weight=0`` should mean "do not call
    the renderer", not just "call it and multiply by zero".
    """

    from gaussiandream.models_pytorch.policy import PolicyModel

    if getattr(PolicyModel, "_libero_ablation_runtime_patched", False):
        return

    original_compute = PolicyModel._compute_gaussian_supervision_loss

    def patched_compute(self, *args, **kwargs):
        if float(getattr(self, "render_loss_weight", 0.0)) <= 0.0:
            kwargs["enable_render_supervision"] = False
        if float(getattr(self, "depth_loss_weight", 0.0)) <= 0.0:
            kwargs["enable_depth_supervision"] = False
        return original_compute(self, *args, **kwargs)

    PolicyModel._compute_gaussian_supervision_loss = patched_compute
    PolicyModel._libero_ablation_runtime_patched = True


def main(args: Args) -> None:
    if args.list_presets:
        _print_presets()
        return

    if args.preset is None:
        raise ValueError("Please provide --preset, or use --list-presets to see available presets.")

    cfg, preset = _build_config(args)
    summary = _summarize_config(cfg)

    print(f"[run_libero_ablation] preset={args.preset}")
    print(f"[run_libero_ablation] budget={args.budget}")
    print(f"[run_libero_ablation] description={preset['description']}")
    if args.print_config:
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))

    if args.dry_run:
        print("[run_libero_ablation] dry-run enabled, not launching training.")
        return

    train_module = _load_train_pytorch_module()
    _install_ablation_runtime_patches()
    train_module.init_logging()
    train_module.train_loop(cfg)


if __name__ == "__main__":
    main(tyro.cli(Args))
