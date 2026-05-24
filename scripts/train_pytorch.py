"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PolicyModel` model and your existing config/data
pipeline from `src/gaussiandream/training/config.py` and `src/gaussiandream/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""

import dataclasses
import gc
import logging
import math
import os
import platform
import shutil
import sys
import time

# Ensure current project's src directory is in Python path (before other paths)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_path = os.path.join(_project_root, "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

import gaussiandream.models.pi0_config
import gaussiandream.models_pytorch.policy
import gaussiandream.shared.normalize as _normalize
import gaussiandream.training.config as _config
import gaussiandream.training.data_loader as _data


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        # Increase timeout to 30 minutes (1800 seconds) to handle slow operations like visualization
        import datetime
        timeout = datetime.timedelta(seconds=1800)
        torch.distributed.init_process_group(
            backend=backend, 
            init_method="env://",
            timeout=timeout
        )

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    if not is_main:
        return

    # Only save if it's time to save or if it's the final step
    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1:
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state using safetensors (handle shared tensors)
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # Save optimizer state using PyTorch format
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # save norm stats
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # Log checkpoint to wandb
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def compute_lora_grad_stats(model: torch.nn.Module) -> dict[str, float | int]:
    """Compute grad/param norms for trainable LoRA parameters."""
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    grad_sq_sum = 0.0
    param_sq_sum = 0.0
    lora_param_count = 0
    lora_with_grad_count = 0

    for name, param in model.named_parameters():
        if "lora" not in name.lower():
            continue
        if not param.requires_grad:
            continue

        lora_param_count += 1
        param_sq_sum += float(param.detach().float().pow(2).sum().item())

        if param.grad is not None:
            lora_with_grad_count += 1
            grad_sq_sum += float(param.grad.detach().float().pow(2).sum().item())

    return {
        "lora_grad_norm": math.sqrt(grad_sq_sum) if grad_sq_sum > 0 else 0.0,
        "lora_param_norm": math.sqrt(param_sq_sum) if param_sq_sum > 0 else 0.0,
        "lora_param_count": lora_param_count,
        "lora_with_grad_count": lora_with_grad_count,
    }




def _collect_param_groups(raw_model: torch.nn.Module):
    grouped_params = {
        "default": [],
        "wm_shared_backbone": [],
        "wm_static_head": [],
        "wm_velocity_head": [],
    }

    for name, param in raw_model.named_parameters():
        if name.startswith("world_model.shared_backbone."):
            grouped_params["wm_shared_backbone"].append(param)
        elif name.startswith("world_model.static_head."):
            grouped_params["wm_static_head"].append(param)
        elif name.startswith("world_model.velocity_head."):
            grouped_params["wm_velocity_head"].append(param)
        else:
            grouped_params["default"].append(param)

    return grouped_params


def _create_optimizer(raw_model: torch.nn.Module, config: _config.TrainConfig, peak_lr: float):
    grouped_params = _collect_param_groups(raw_model)
    param_groups = []
    for group_name in ["default", "wm_shared_backbone", "wm_static_head", "wm_velocity_head"]:
        params = grouped_params[group_name]
        if not params:
            continue
        param_groups.append({
            "name": group_name,
            "params": params,
            "lr": peak_lr,
            "lr_scale": 1.0,
        })

    optim = torch.optim.AdamW(
        param_groups,
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )
    return optim


def _set_optimizer_lr_scales(optimizer: torch.optim.Optimizer, lr_scales: dict[str, float]):
    for pg in optimizer.param_groups:
        pg["lr_scale"] = float(lr_scales.get(pg.get("name", "default"), 1.0))


def _optimizer_group_lr_summary(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {pg.get("name", f"group_{idx}"): float(pg["lr"]) for idx, pg in enumerate(optimizer.param_groups)}


def _stage_lr_scales(config: _config.TrainConfig, stage: int) -> dict[str, float]:
    if stage == 1:
        return {
            "default": 1.0,
            "wm_shared_backbone": 1.0,
            "wm_static_head": 1.0,
            "wm_velocity_head": 1.0,
        }
    if stage == 2:
        world_lr_scale = max(0.0, float(getattr(config, "stage2_shared_backbone_lr_scale", 0.0)))
        return {
            "default": 1.0,
            "wm_shared_backbone": world_lr_scale,
            "wm_static_head": world_lr_scale,
            "wm_velocity_head": world_lr_scale,
        }
    raise ValueError(f"Unsupported stage: {stage}")


def _apply_stage_state(
    raw_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: _config.TrainConfig,
    stage: int,
    render_weight: float,
    *,
    is_main: bool,
    label: str,
):
    raw_model.apply_world_model_stage(stage, render_weight)
    _set_optimizer_lr_scales(optimizer, _stage_lr_scales(config, stage))
    raw_model._stage_applied = stage
    raw_model._active_training_stage = stage

    if is_main:
        trainability = raw_model.get_stage_trainability_summary() if hasattr(raw_model, "get_stage_trainability_summary") else {}
        lr_scales = {pg.get("name", f"group_{idx}"): float(pg.get("lr_scale", 1.0)) for idx, pg in enumerate(optimizer.param_groups)}
        static_head = getattr(getattr(raw_model, "world_model", None), "static_head", None)
        image_fusion_enabled = getattr(static_head, "use_image_fusion", "n/a") if static_head is not None else "n/a"
        logging.info(
            "%s | render_weight=%.4f | image_fusion=%s | action_loss=%s | world_losses=%s | lr_scales=%s | trainable=%s",
            label,
            render_weight,
            image_fusion_enabled,
            getattr(raw_model, "_action_loss_enabled", "n/a"),
            getattr(raw_model, "_world_loss_enabled", "n/a"),
            lr_scales,
            trainability,
        )


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    # Initialize checkpoint directory and wandb
    resuming = False
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        exp_checkpoint_dir = config.checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    # Initialize wandb (only on main process)
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # Staged training changes which branches participate in backward across steps,
    # which is incompatible with DDP static_graph mode.
    # Stage schedule:
    # - Stage 1: [0, stage1_steps)
    #   Representation/world-model training only, image fusion on, action off.
    # - Stage 2: [stage1_steps, ...)
    #   Action-only training, image fusion stays on, world-model params frozen and world losses off.
    stage1_steps = getattr(config, "stage1_steps", 0)
    stage2_steps = getattr(config, "stage2_steps", 0)
    stage3_steps = getattr(config, "stage3_steps", 0)
    stage1_render_weight = getattr(config, "stage1_render_weight", 0.0)
    stage2_render_weight = getattr(config, "stage2_render_weight", 0.2)
    stage3_render_weight = getattr(config, "stage3_render_weight", 0.1)
    stage4_render_weight = getattr(config, "stage4_render_weight", stage3_render_weight)
    staged_training_enabled = stage1_steps > 0

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    loader, data_config = build_datasets(config)

    # Log sample images to wandb on first batch
    if is_main and config.wandb_enabled and not resuming:
        # Create a separate data loader for sample batch to avoid consuming the main loader
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        # Convert observation and actions to torch tensors
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        # Create sample images for wandb
        images_to_log = []
        # Get batch size from the first image tensor
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            # Concatenate all camera views horizontally for this batch item
            imgs_list = []
            for img in sample_batch["image"].values():
                # img is [B, ..., H, W, C] (due to model.py Observation format)
                # img[i] gets i-th batch element
                single_img = img[i]
                
                # Handle Temporal Dimension [T, H, W, C]
                if single_img.ndim == 4: 
                    single_img = single_img[0] # Take first frame
                
                # Check dims. If [C, H, W] -> permute to [H, W, C].
                # If [H, W, C], keep it. 
                # Observation guarantees H,W,C, but let's be robust/explicit.
                # Heuristic: C is usually 3.
                if single_img.shape[0] == 3 and single_img.shape[-1] != 3:
                     single_img = single_img.permute(1, 2, 0)
                     
                # Shift from [-1, 1] to [0, 1] for visualization
                single_img = (single_img + 1.0) / 2.0
                single_img = torch.clamp(single_img, 0, 1)
                     
                imgs_list.append(single_img)

            img_concatenated = torch.cat(imgs_list, axis=1)
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)

        # Clear sample batch from memory aggressively
        del sample_batch, observation, actions, images_to_log, img_concatenated
        del sample_data_loader  # Also delete the sample data loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    # Build model
    if not isinstance(config.model, gaussiandream.models.pi0_config.Pi0Config):
        # Convert dataclass to Pi0Config if needed
        model_cfg = gaussiandream.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
            state_norm_stats=data_config.norm_stats.get("state") if data_config.norm_stats is not None else None,
            state_use_quantile_norm=data_config.use_quantile_norm,
        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)
        object.__setattr__(model_cfg, "state_norm_stats", data_config.norm_stats.get("state") if data_config.norm_stats is not None else None)
        object.__setattr__(model_cfg, "state_use_quantile_norm", data_config.use_quantile_norm)

    # Mirror stage-schedule controls from the TrainConfig onto the model config so the
    # PyTorch model can apply the same runtime schedule inside forward/resume paths.
    for attr in (
        "stage1_steps",
        "stage2_steps",
        "stage3_steps",
        "stage1_render_weight",
        "stage2_render_weight",
        "stage3_render_weight",
        "stage4_render_weight",
        "stage1_freeze_velocity_head",
        "stage2_freeze_static_head",
        "stage2_shared_backbone_lr_scale",
        "stage2_world_loss_multiplier",
        "stage2_keep_world_model_trainable",
        "stage4_freeze_world_model",
        "stage4_disable_world_model_losses",
    ):
        object.__setattr__(model_cfg, attr, getattr(config, attr))

    model = gaussiandream.models_pytorch.policy.PolicyModel(model_cfg).to(device)

    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    # Always enable expandable_segments to avoid fragmentation (helps with OOM)
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        logging.info("Enabled memory optimizations (expandable_segments)")
    
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    ddp_static_graph = world_size >= 8 and not staged_training_enabled
    if use_ddp and is_main:
        if ddp_static_graph:
            logging.info("DDP static_graph enabled (8+ GPUs and no staged training detected)")
        elif world_size >= 8 and staged_training_enabled:
            logging.info("DDP static_graph disabled because staged training changes the backward graph across steps")

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,  # Disable for memory efficiency
            gradient_as_bucket_view=True,  # Enable for memory efficiency
            static_graph=ddp_static_graph,
        )

    # Load weights from weight_loader if specified (for fine-tuning)
    if config.pytorch_weight_path is not None:
        pytorch_weight_path = os.path.expanduser(os.path.expandvars(config.pytorch_weight_path))
        logging.info(f"Loading weights from: {pytorch_weight_path}")

        if os.path.isdir(pytorch_weight_path):
            model_path = os.path.join(pytorch_weight_path, "model.safetensors")
        else:
            model_path = pytorch_weight_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"PyTorch weight file not found: {model_path}")
        safetensors.torch.load_model(
            (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model), model_path, strict=False
        )
        logging.info(f"Loaded PyTorch weights from {model_path}")

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # Get the underlying model (unwrap DDP if needed)
    raw_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    # Create optimizer with fixed parameter groups so stage-specific LR scaling remains stable across resumes
    optim = _create_optimizer(raw_model, config, peak_lr)

    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    start_time = time.time()
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    if staged_training_enabled and is_main:
        stage2_world_loss_multiplier = getattr(config, "stage2_world_loss_multiplier", 0.0)
        stage2_keep_world_model_trainable = getattr(config, "stage2_keep_world_model_trainable", False)
        logging.info("=== Staged Training Enabled ===")
        logging.info(
            f"Stage 1 (Representation/world-model): steps 0-{stage1_steps}, render_weight={stage1_render_weight}, image_fusion=on, action=off, world_losses=on"
        )
        logging.info(
            f"Stage 2 (Action-focused): steps {stage1_steps}-{config.num_train_steps}, "
            f"render_weight={stage2_render_weight}, world_loss_multiplier={stage2_world_loss_multiplier}, "
            f"world_model_trainable={stage2_keep_world_model_trainable}, image_fusion=on, action=on"
        )

    def _apply_stage_for_step(step: int):
        if not staged_training_enabled:
            return
        if step < stage1_steps:
            if not hasattr(raw_model, '_stage_applied') or raw_model._stage_applied != 1:
                _apply_stage_state(
                    raw_model,
                    optim,
                    config,
                    1,
                    stage1_render_weight,
                    is_main=is_main,
                    label="=== Stage 1 Active: Representation/world-model training (image_fusion=on, action=off, world_losses=on) ===",
                )
        else:
            if not hasattr(raw_model, '_stage_applied') or raw_model._stage_applied != 2:
                _apply_stage_state(
                    raw_model,
                    optim,
                    config,
                    2,
                    stage2_render_weight,
                    is_main=is_main,
                    label="=== Stage 2 Active: Action-focused training (image_fusion=on, action=on, weak world supervision) ===",
                )

    _apply_stage_for_step(global_step)

    while global_step < config.num_train_steps:
        # Set epoch for distributed training
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for observation, actions in loader:
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            # Staged training: apply correct stage based on current global_step (checked every step)
            _apply_stage_for_step(global_step)
            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(device)  # noqa: PLW2901

            # Update LR
            base_lr = lr_schedule(global_step)
            for pg in optim.param_groups:
                pg["lr"] = base_lr * float(pg.get("lr_scale", 1.0))

            # Forward pass
            losses = model(observation, actions, step=global_step)
            # Ensure losses is a tensor and handle different return types
            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)

            loss = losses.mean()

            # Backward pass
            loss.backward()

            if is_main and global_step % 100 == 0:
                lora_stats = compute_lora_grad_stats(model)
                logging.info(
                    "Step %d: LoRA grad_norm=%.6f param_norm=%.6f params_with_grad=%d/%d",
                    global_step,
                    lora_stats["lora_grad_norm"],
                    lora_stats["lora_param_norm"],
                    lora_stats["lora_with_grad_count"],
                    lora_stats["lora_param_count"],
                )

            # Log memory usage after backward pass
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # NaN gradient sanitization: replace NaN/Inf gradients with 0
            # This allows action loss gradients (typically fine) to still update
            # while zeroing out problematic render loss gradients
            # (e.g. from degenerate Gaussians, numerical overflow in SH evaluation).
            nan_grad_count = 0
            for param in model.parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    nan_grad_count += 1
                    param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)

            if torch.isfinite(grad_norm):
                optim.step()
            else:
                # This should rarely happen now since we sanitized NaN grads above
                logging.warning(f"Step {global_step}: grad_norm is {grad_norm} after sanitization, skipping optimizer step")

            if nan_grad_count > 0 and global_step % 100 == 0:
                logging.warning(f"Step {global_step}: sanitized NaN grads in {nan_grad_count} params, grad_norm={grad_norm:.4f}")

            optim.zero_grad(set_to_none=True)

            # Clear gradients more aggressively
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            # Collect stats
            if is_main:
                infos.append(
                    {
                        "loss": loss.item(),
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    }
                )

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )

                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []  # Reset stats collection

            global_step += 1
            # Save checkpoint using the new mechanism
            save_checkpoint(model, optim, global_step, config, is_main, data_config)

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
