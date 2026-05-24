#!/usr/bin/env python3
"""
Generate depth maps for LIBERO dataset using Depth Anything V2.

Usage:
    python scripts/generate_depth_for_libero.py \
        --data_dir <LIBERO_DATA_ROOT> \
        --output_dir <LIBERO_DATA_WITH_DEPTH_ROOT> \
        --model_size small \
        --batch_size 8
"""

import argparse
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
import pyarrow as pa
from PIL import Image
import io


class LiberoDataset(Dataset):
    """Dataset for loading LIBERO parquet files."""

    def __init__(self, data_dir, chunk_name="chunk-000"):
        self.data_dir = Path(data_dir)
        self.chunk_dir = self.data_dir / chunk_name

        # Get all parquet files
        self.parquet_files = sorted(list(self.chunk_dir.glob("episode_*.parquet")))
        print(f"Found {len(self.parquet_files)} episodes in {chunk_name}")

        # Load all episodes
        self.episodes = []
        for pq_file in tqdm(self.parquet_files, desc="Loading episodes"):
            table = pq.read_table(pq_file)
            self.episodes.append(table.to_pydict())

        # Count total frames
        self.total_frames = sum(len(ep['image']) for ep in self.episodes)
        print(f"Total frames: {self.total_frames}")

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        # Find which episode and frame
        episode_idx = 0
        frame_idx = idx

        for ep_idx, episode in enumerate(self.episodes):
            ep_len = len(episode['image'])
            if frame_idx < ep_len:
                episode_idx = ep_idx
                break
            frame_idx -= ep_len

        episode = self.episodes[episode_idx]

        # Load images
        image_bytes = episode['image'][frame_idx]
        wrist_image_bytes = episode['wrist_image'][frame_idx]

        # Decode images
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        wrist_image = Image.open(io.BytesIO(wrist_image_bytes)).convert('RGB')

        return {
            'image': image,
            'wrist_image': wrist_image,
            'episode_idx': episode_idx,
            'frame_idx': frame_idx,
            'parquet_file': self.parquet_files[episode_idx],
        }


def load_depth_anything_v2(model_size='small', device='cuda'):
    """
    Load Depth Anything V2 model.

    Args:
        model_size: 'small', 'base', or 'large'
        device: 'cuda' or 'cpu'

    Returns:
        model: Depth Anything V2 model
        transform: Image preprocessing transform
    """
    size_map = {'small': 'vits', 'base': 'vitb', 'large': 'vitl'}
    encoder = size_map[model_size]

    try:
        from depth_anything_v2.dpt import DepthAnythingV2

        # Model configurations
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        model = DepthAnythingV2(**model_configs[encoder])

        checkpoint_urls = {
            'vits': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth',
            'vitb': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth',
            'vitl': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth',
        }

        checkpoint_path = f'/tmp/depth_anything_v2_{encoder}.pth'
        if not os.path.exists(checkpoint_path):
            print(f"Downloading {encoder} model...")
            os.system(f"wget {checkpoint_urls[encoder]} -O {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model = model.to(device).eval()
        print(f"Loaded Depth Anything V2 ({model_size}) on {device} via depth_anything_v2 package")
        return model
    except ImportError:
        pass

    try:
        from transformers import DepthAnythingForDepthEstimation
    except ImportError as exc:
        raise ImportError(
            "Neither depth_anything_v2 nor transformers DepthAnythingForDepthEstimation is available."
        ) from exc

    repo_map = {
        'small': 'depth-anything/Depth-Anything-V2-Small-hf',
        'base': 'depth-anything/Depth-Anything-V2-Base-hf',
        'large': 'depth-anything/Depth-Anything-V2-Large-hf',
    }
    repo_id = repo_map[model_size]

    class _HFDepthAnythingWrapper(torch.nn.Module):
        def __init__(self, hf_model: torch.nn.Module) -> None:
            super().__init__()
            self.hf_model = hf_model

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            return self.hf_model(pixel_values=pixel_values).predicted_depth

    try:
        hf_model = DepthAnythingForDepthEstimation.from_pretrained(repo_id, local_files_only=True)
    except Exception as exc:
        raise FileNotFoundError(
            f"Missing local Hugging Face cache for {repo_id}. "
            f"Please download it first or install depth_anything_v2 weights."
        ) from exc

    model = _HFDepthAnythingWrapper(hf_model).to(device).eval()
    print(f"Loaded Depth Anything V2 ({model_size}) on {device} via local Hugging Face cache")
    return model


def preprocess_image(image, target_size=518):
    """
    Preprocess image for Depth Anything V2.

    Args:
        image: PIL Image
        target_size: Target size (default 518 for Depth Anything V2)

    Returns:
        tensor: [1, 3, H, W] normalized tensor
    """
    # Resize to target size (keep aspect ratio)
    w, h = image.size
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Make dimensions divisible by 14 (patch size)
    new_h = (new_h // 14) * 14
    new_w = (new_w // 14) * 14

    image = image.resize((new_w, new_h), Image.BILINEAR)

    # Convert to tensor and normalize
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)

    # Normalize (ImageNet stats)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    image_tensor = (image_tensor - mean) / std

    return image_tensor, (h, w)


@torch.no_grad()
def predict_depth(model, image, device='cuda', original_size=None):
    """
    Predict depth map for an image.

    Args:
        model: Depth Anything V2 model
        image: PIL Image
        device: 'cuda' or 'cpu'
        original_size: (H, W) to resize depth back to original size

    Returns:
        depth: [H, W] depth map (numpy array)
    """
    # Preprocess
    image_tensor, orig_size = preprocess_image(image)
    image_tensor = image_tensor.to(device)

    # Predict
    depth = model(image_tensor)  # [1, H, W]

    # Resize to original size if specified
    if original_size is not None:
        depth = F.interpolate(
            depth.unsqueeze(0),
            size=original_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
    else:
        # Resize to input image size
        depth = F.interpolate(
            depth.unsqueeze(0),
            size=orig_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

    depth = depth.squeeze(0).cpu().numpy()  # [H, W]

    return depth


def process_chunk(data_dir, chunk_name, output_dir, model, device='cuda', batch_size=8):
    """
    Process one chunk of LIBERO data and add depth maps.

    Args:
        data_dir: Path to LIBERO data directory
        chunk_name: Name of chunk (e.g., 'chunk-000')
        output_dir: Output directory for processed data
        model: Depth Anything V2 model
        device: 'cuda' or 'cpu'
        batch_size: Batch size for processing
    """
    chunk_dir = Path(data_dir) / chunk_name
    output_chunk_dir = Path(output_dir) / chunk_name
    output_chunk_dir.mkdir(parents=True, exist_ok=True)

    # Get all parquet files
    parquet_files = sorted(list(chunk_dir.glob("episode_*.parquet")))

    print(f"\nProcessing {chunk_name}: {len(parquet_files)} episodes")

    for pq_file in tqdm(parquet_files, desc=f"Processing {chunk_name}"):
        # Load episode
        table = pq.read_table(pq_file)
        episode = table.to_pydict()

        num_frames = len(episode['image'])

        # Process images in batches
        depth_maps = []
        wrist_depth_maps = []

        for i in range(0, num_frames, batch_size):
            batch_end = min(i + batch_size, num_frames)

            # Process main camera
            for j in range(i, batch_end):
                image_bytes = episode['image'][j]
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                depth = predict_depth(model, image, device, original_size=(256, 256))
                depth_maps.append(depth)

            # Process wrist camera
            for j in range(i, batch_end):
                wrist_bytes = episode['wrist_image'][j]
                wrist_image = Image.open(io.BytesIO(wrist_bytes)).convert('RGB')
                wrist_depth = predict_depth(model, wrist_image, device, original_size=(256, 256))
                wrist_depth_maps.append(wrist_depth)

        # Add depth to episode data
        episode['depth'] = depth_maps  # List of [H, W] arrays
        episode['wrist_depth'] = wrist_depth_maps

        # Convert to PyArrow table
        # Need to serialize depth arrays
        depth_serialized = [d.tobytes() for d in depth_maps]
        wrist_depth_serialized = [d.tobytes() for d in wrist_depth_maps]

        episode['depth'] = depth_serialized
        episode['wrist_depth'] = wrist_depth_serialized

        # Add metadata
        episode['depth_shape'] = [(256, 256)] * num_frames
        episode['depth_dtype'] = ['float32'] * num_frames

        # Save to new parquet file
        output_file = output_chunk_dir / pq_file.name
        table = pa.table(episode)
        pq.write_table(table, output_file)

    print(f"Finished processing {chunk_name}")


def main():
    parser = argparse.ArgumentParser(description="Generate depth maps for LIBERO dataset")
    parser.add_argument('--data_dir', type=str, default=os.environ.get('LIBERO_DATA_ROOT'),
                       help='Path to LIBERO data directory')
    parser.add_argument('--output_dir', type=str, default=os.environ.get('LIBERO_DATA_WITH_DEPTH_ROOT'),
                       help='Output directory for data with depth')
    parser.add_argument('--model_size', type=str, default='small', choices=['small', 'base', 'large'],
                       help='Depth Anything V2 model size')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--chunks', type=str, nargs='+', default=['chunk-000', 'chunk-001'],
                       help='Chunks to process')

    args = parser.parse_args()

    if args.data_dir is None:
        raise ValueError("Pass --data_dir or set LIBERO_DATA_ROOT.")
    if args.output_dir is None:
        raise ValueError("Pass --output_dir or set LIBERO_DATA_WITH_DEPTH_ROOT.")

    # Load model
    print("Loading Depth Anything V2...")
    model = load_depth_anything_v2(args.model_size, args.device)

    # Process each chunk
    for chunk_name in args.chunks:
        process_chunk(
            args.data_dir,
            chunk_name,
            args.output_dir,
            model,
            args.device,
            args.batch_size
        )

    print("\n✅ All chunks processed successfully!")
    print(f"Output saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
