#!/bin/bash

# Scene-based inference script with multi-GPU support
# Usage: 
#   Single GPU: ./scene_inference.sh
#   Multi-GPU:  ./scene_inference.sh --multi_gpu --device="0,1"
#   Debug mode: ./scene_inference.sh --debug

# Default parameters
CONFIG_PATH='configs/nuscenes/vggt3dgs_inference.yaml'
CHECKPOINT_PATH="${VGGT3DGS_CHECKPOINT_PATH:-<CHECKPOINT_PATH>}"
OUTPUT_DIR="${VGGT3DGS_OUTPUT_DIR:-work_dirs/vggt3dgs_1207_12hz}"
DEVICE='0,1,2,3,4,5,6,7'
# MAX_SCENES="2"
MULTI_GPU_FLAG=""
DEBUG_MODE=false
NOVEL_VIEW_DISTANCES="1.0,2.0,3.0"
EVAL_RESOLUTION="280x518" 
# 测评需要分两步（测gt_view和测novel_view是分开进行的）
# 以下设置是novel_view的参数
#   此时不进行SAVE_RENDERS（因为这个参数是保存left1m/left2m/left3m用的）
#   此时也不进行eval_frame的显式设置（非显示情况下，会渲染并测试所有的novel_view图像）
# SAVE_RENDERS=false

# 以下测试是gt_view的参数
#   此时将eval_frame显式设置为0，只渲染gt_view
#   此时开启SAVE_RENDERS=true，渲染left和right的偏移视角图像，用于后续感知loss计算
EVAL_FRAME="0,1,2,3,4,5" #  -1 for pre, 0 for "cur", 1 for "next"
SAVE_RENDERS=true

# Frame skip parameter (default: skip every 6 frames)
FRAME_SKIP="6"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --device=*)
            DEVICE="${1#*=}"
            shift
            ;;
        --max_scenes)
            MAX_SCENES="$2"
            shift 2
            ;;
        --max_scenes=*)
            MAX_SCENES="${1#*=}"
            shift
            ;;
        --multi_gpu)
            MULTI_GPU_FLAG="--multi_gpu"
            shift
            ;;
        --debug)
            DEBUG_MODE=true
            MAX_SCENES="3"
            shift
            ;;
        --save_renders)
            SAVE_RENDERS=true
            shift
            ;;
        --no_renders)
            SAVE_RENDERS=false
            shift
            ;;
        --novel_distances)
            NOVEL_VIEW_DISTANCES="$2"
            shift 2
            ;;
        --novel_distances=*)
            NOVEL_VIEW_DISTANCES="${1#*=}"
            shift
            ;;
        --eval_resolution)
            EVAL_RESOLUTION="$2"
            shift 2
            ;;
        --eval_resolution=*)
            EVAL_RESOLUTION="${1#*=}"
            shift
            ;;
        --eval_frame)
            EVAL_FRAME="$2"
            shift 2
            ;;
        --eval_frame=*)
            EVAL_FRAME="${1#*=}"
            shift
            ;;
        --frame_skip)
            FRAME_SKIP="$2"
            shift 2
            ;;
        --frame_skip=*)
            FRAME_SKIP="${1#*=}"
            shift
            ;;
        -h|--help)
            echo "Scene-based inference script with multi-GPU support"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config PATH          Configuration file path (default: configs/nuscenes/vggt3dgs.yaml)"
            echo "  --checkpoint PATH      Model checkpoint path"
            echo "  --output_dir PATH      Output directory for results (default: ./work_dirs/test1/scene_inference_results)"
            echo "  --device DEVICE        Device(s) to use:"
            echo "                           Single GPU: 'cuda:0' or '0'"
            echo "                           Multi-GPU: '0,1' or '0,1,2,3'"
            echo "  --max_scenes N         Maximum number of scenes to process (default: all scenes)"
            echo "  --multi_gpu            Enable multi-GPU inference (auto-enabled if multiple devices specified)"
            echo "  --debug                Debug mode: process only 3 scenes with batch_size=2"
            echo "  --save_renders         Enable saving rendered images and novel views (default: enabled)"
            echo "  --no_renders           Disable saving rendered images and novel views"
            echo "  --novel_distances DIST Novel view translation distances in meters (default: '0.5,1.0,2.0,3.0')"
            echo "                         Format: comma-separated list, e.g., '1.0,2.0' or '0.5,1.0,2.0,3.0'"
            echo "  --eval_resolution RES  Evaluation resolution mode:"
            echo "                           'original': Use original 280x518 resolution (default)"
            echo "                           'upsampled': Upsample to 900x1600 for evaluation"
            echo "  --eval_frame FRAME     Evaluation frame selection:"
            echo "                           -1: previous frame"
            echo "                            0: current frame (default: 0)"
            echo "                            1: next frame"
            echo "  --frame_skip N         Skip interval for frames (e.g., 6 to evaluate frames 0, 6, 12, 18, ...)"
            echo "                         Default: no skip (evaluate all frames)"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Single GPU inference"
            echo "  $0 --device='cuda:1'"
            echo ""
            echo "  # Multi-GPU inference with 2 GPUs"
            echo "  $0 --device='0,1' --multi_gpu"
            echo ""
            echo "  # Debug mode"
            echo "  $0 --debug --device='0,1'"
            echo ""
            echo "  # Process specific number of scenes"
            echo "  $0 --device='0,1' --multi_gpu --max_scenes=10"
            echo ""
            echo "  # Custom novel view distances"
            echo "  $0 --device='0,1' --novel_distances='1.0,2.0'"
            echo ""
            echo "  # Disable rendering (metrics only)"
            echo "  $0 --device='0,1' --no_renders"
            echo ""
            echo "  # Use upsampled resolution for evaluation"
            echo "  $0 --device='0,1' --eval_resolution='upsampled'"
            echo ""
            echo "  # Use original resolution for evaluation (default)"
            echo "  $0 --device='0,1' --eval_resolution='original'"
            echo ""
            echo "  # Evaluate on previous frame"
            echo "  $0 --device='0,1' --eval_frame=0"
            echo ""
            echo "  # Evaluate every 6th frame (0, 6, 12, 18, ...)"
            echo "  $0 --device='0,1' --frame_skip=6"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Auto-enable multi-GPU if multiple devices specified
if [[ "$DEVICE" == *","* ]]; then
    MULTI_GPU_FLAG="--multi_gpu"
fi

# Print configuration
echo "=================================================="
echo "Scene-based Inference Configuration"
echo "=================================================="
echo "Config file:     $CONFIG_PATH"
echo "Checkpoint:      $CHECKPOINT_PATH"
echo "Output dir:      $OUTPUT_DIR"
echo "Device(s):       $DEVICE"
echo "Multi-GPU:       $([ -n "$MULTI_GPU_FLAG" ] && echo "Enabled" || echo "Disabled")"
echo "Max scenes:      $([ -n "$MAX_SCENES" ] && echo "$MAX_SCENES" || echo "All scenes")"
echo "Debug mode:      $([ "$DEBUG_MODE" = true ] && echo "Enabled" || echo "Disabled")"
echo "Save renders:    $([ "$SAVE_RENDERS" = true ] && echo "Enabled" || echo "Disabled")"
echo "Novel view dist: $NOVEL_VIEW_DISTANCES meters"
echo "Eval resolution: $EVAL_RESOLUTION"
echo "Eval frame:      $EVAL_FRAME (-1: prev, 0: cur, 1: next)"
echo "Frame skip:      $([ -n "$FRAME_SKIP" ] && echo "$FRAME_SKIP" || echo "None (evaluate all frames)")"
echo "=================================================="
echo ""

# Build command arguments
CMD_ARGS=(
    --cfg_path="$CONFIG_PATH"
    --restore_ckpt="$CHECKPOINT_PATH"
    --output_dir="$OUTPUT_DIR"
    --device="$DEVICE"
)

# Add optional arguments
if [ -n "$MAX_SCENES" ]; then
    CMD_ARGS+=(--max_scenes="$MAX_SCENES")
fi

if [ -n "$MULTI_GPU_FLAG" ]; then
    CMD_ARGS+=($MULTI_GPU_FLAG)
fi

# Add rendering control arguments
if [ "$SAVE_RENDERS" = false ]; then
    CMD_ARGS+=(--no_renders)
fi

if [ -n "$NOVEL_VIEW_DISTANCES" ]; then
    CMD_ARGS+=(--novel_distances="$NOVEL_VIEW_DISTANCES")
fi

# Add evaluation resolution argument
CMD_ARGS+=(--eval_resolution="$EVAL_RESOLUTION")

# Add evaluation frame argument
CMD_ARGS+=(--eval_frame="$EVAL_FRAME")

# Add frame skip argument if specified
if [ -n "$FRAME_SKIP" ]; then
    CMD_ARGS+=(--frame_skip="$FRAME_SKIP")
fi

# Print command being executed
echo "Executing command:"
echo "python -m scripts.vggt3dgs_inference ${CMD_ARGS[@]}"
echo ""

# Run the inference
echo "Starting scene-based inference..."
echo ""

python scripts/vggt3dgs_inference.py "${CMD_ARGS[@]}"

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "✅ Scene-based inference completed successfully!"
    echo "📁 Results saved to: $OUTPUT_DIR"
else
    echo "❌ Scene-based inference failed with exit code: $exit_code"
fi

echo "=================================================="