#!/usr/bin/env python3
"""
Scene-based inference script for VGGT3DGS
This script demonstrates inference using scene-by-scene iteration
"""

import yaml
import argparse
import os
import sys
import torch
import json
from pathlib import Path
import pathlib
import pandas as pd
import time
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import threading
from PIL import Image
import torch.nn.functional as F
# from gsplat.rendering import rasterization


project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from dataset.inference_scene_data_module import VGGT3DGS_SceneDataModule
# from models.vggt3dgs_model_module import VGGT3DGS_LITModelModule
from models.df3dgs_model_module import DF3DGS_LITModelModule
from models.gaussian_util import render, focal2fov, getProjectionMatrix

uniad_path = '/workspace/data_storage/ruiyanghao/projects/uniadv2/UniAD'
sys.path.append(uniad_path)
from uniad_api import run_uniad

class SceneSampleDataset(Dataset):
    """Dataset wrapper that supports both pre-loaded samples and lazy loading"""
    
    def __init__(self, samples_or_indices, dataset=None, scene_idx=None):
        if dataset is not None and scene_idx is not None:
            # Lazy loading mode: samples_or_indices is list of sample indices
            self.lazy_mode = True
            self.sample_indices = samples_or_indices
            self.dataset = dataset
            self.scene_idx = scene_idx
        else:
            # Pre-loaded mode: samples_or_indices is list of actual samples
            self.lazy_mode = False
            self.samples = samples_or_indices
    
    def __len__(self):
        if self.lazy_mode:
            return len(self.sample_indices) //6
        else:
            return len(self.samples)
    
    def __getitem__(self, idx):
        if self.lazy_mode:
            # Load sample on-demand
            sample_idx = self.sample_indices[idx] 
            return self.dataset.get_scene_sample(self.scene_idx, sample_idx)
        else:
            return self.samples[idx]


def custom_collate_fn(batch):
    """Custom collate function to handle complex data structures"""
    if len(batch) == 0:
        return {}
    
    # Initialize the result dict
    collated = {}
    
    # Get all keys from the first sample
    sample_keys = batch[0].keys()
    
    for key in sample_keys:
        values = [sample[key] for sample in batch]
        
        # Handle different data types
        if isinstance(values[0], torch.Tensor):
            # Stack tensors along batch dimension
            collated[key] = torch.stack(values, dim=0)
        elif isinstance(values[0], (list, tuple)):
            # For lists/tuples, keep as list (don't collate)
            collated[key] = values
        else:
            # For other types (strings, numbers), keep as list
            collated[key] = values
    
    return collated


def load_model_from_checkpoint(checkpoint_path, model_cfg, device):
    """Load model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Ensure batch_size is in model config
    if 'batch_size' not in model_cfg:
        model_cfg['batch_size'] = 1  # Set default batch_size for inference
    
    # Initialize model
    model = DF3DGS_LITModelModule(
        cfg=model_cfg,
        save_dir='./temp_log',
        logger=None
    )
    model.model.load_official_weights(device=device)
    # Load checkpoint
    # if checkpoint_path.endswith('.ckpt'):
    #     # PyTorch Lightning checkpoint
    #     checkpoint = torch.load(checkpoint_path, map_location=device)
    #     model.load_state_dict(checkpoint['state_dict'])
    #     print("Loaded PyTorch Lightning checkpoint")
    # else:
    #     # Regular PyTorch checkpoint
    #     checkpoint = torch.load(checkpoint_path, map_location=device)
    #     if 'model_state_dict' in checkpoint:
    #         model.load_state_dict(checkpoint['model_state_dict'])
    #     else:
    #         model.load_state_dict(checkpoint)
    #     print("Loaded PyTorch checkpoint")
    
    model.to(device)
    model.eval()
    return model


def run_inference(model_cfg=None, model=None, checkpoint_path=None, 
                  scene_dataloader=None, devices='cuda:0', 
                  save_results=True, output_dir=None, novel_distances=[1.0, 2.0],
                  eval_resolution='280x518',eval_frame=None):
    """
    Unified scene-based inference function - handles single/multi-GPU automatically
    
    Args:
        model_cfg: Model configuration (for multi-GPU or when model=None)
        model: Pre-loaded model (for single GPU, optional)  
        checkpoint_path: Path to model checkpoint
        scene_dataloader: Scene data loader
        devices: Device(s) - str for single GPU, list for multi-GPU
        save_results: Whether to save results
        output_dir: Output directory
        novel_distances: List of distances for novel view generation
        eval_resolution: Resolution mode - 'original' or 'upsampled'
        eval_frame: Frame index for evaluation
    """
    # Normalize devices to list
    if isinstance(devices, str):
        device_list = [devices]
    else:
        device_list = devices
    
    # Determine mode and print info
    is_multi_gpu = len(device_list) > 1
    if is_multi_gpu:
        print(f"\nStarting multi-GPU scene-based inference")
        print(f"Devices: {device_list}")
    else:
        print(f"\nStarting single-GPU scene-based inference on {device_list[0]}")
    
    print(f"Number of scenes to process: {len(scene_dataloader)}")
    print(f'eval_frame: {eval_frame}')
    
    # Single GPU path
    if not is_multi_gpu:
        device = device_list[0]
        
        # Load model if not provided
        if model is None:
            if model_cfg is None or checkpoint_path is None:
                raise ValueError("model_cfg and checkpoint_path required when model is None")
            model = load_model_from_checkpoint(checkpoint_path, model_cfg, device)
        
        return _run_single_gpu_inference(model, scene_dataloader, device, save_results, output_dir, novel_distances, eval_resolution,eval_frame)
    
    # Multi-GPU path  
    else:
        if model_cfg is None or checkpoint_path is None:
            raise ValueError("model_cfg and checkpoint_path required for multi-GPU")
            
        return _run_multi_gpu_inference(model_cfg, scene_dataloader, device_list, checkpoint_path, save_results, output_dir, novel_distances, eval_resolution,eval_frame)


def save_rendered_image(tensor_img, save_path, upsample_to=None):
    """Save a tensor image to file with optional upsampling"""
    # Convert from tensor [C, H, W] to numpy [H, W, C] and scale to [0, 255]
    if tensor_img.dim() == 4:
        tensor_img = tensor_img.squeeze(0)
    
    # Apply upsampling if specified
    if upsample_to is not None:
        target_height, target_width = upsample_to
        # Ensure tensor is on GPU for upsampling, then move back to CPU
        device = tensor_img.device
        if tensor_img.device.type == 'cpu':
            tensor_img = tensor_img.cuda()
        
        # Add batch dimension for interpolation
        tensor_img = tensor_img.unsqueeze(0)
        # Upsample using bilinear interpolation
        tensor_img = F.interpolate(tensor_img, size=(target_height, target_width), 
                                 mode='bilinear', align_corners=False)
        # Remove batch dimension
        tensor_img = tensor_img.squeeze(0)
        
        # Move back to original device
        if device.type == 'cpu':
            tensor_img = tensor_img.cpu()
    
    img_np = tensor_img.detach().cpu().numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save as PNG
    Image.fromarray(img_np).save(save_path)


def create_lateral_translation_matrices(translation_distances=[1.0, 2.0]):
    """Create transformation matrices for lateral (left/right) ego vehicle translation"""
    transforms = {}
    
    for dist in translation_distances:
        # Left translation (negative Y in ego coordinate)
        left_transform = torch.eye(4)
        left_transform[1, 3] = dist  # Negative Y for left
        transforms[f'left_{dist}m'] = left_transform
        
        # Right translation (positive Y in ego coordinate)
        right_transform = torch.eye(4)
        right_transform[1, 3] = -dist   # Positive Y for right
        transforms[f'right_{dist}m'] = right_transform
    
    return transforms


def render_novel_views(model, recontrast_data, render_data,novel_render_frames, device, scene_name, sample_idx, save_dir, actual_sample_idx, translation_distances=[1.0, 2.0], eval_resolution='280x518'):
    """Render novel views with lateral ego translation"""
    # Get transformation matrices for translation
    translation_transforms = create_lateral_translation_matrices(translation_distances)
    
    saved_paths = []
    
    # Original 3DGS data for single sample
    xyz_i = recontrast_data['xyz'][sample_idx:sample_idx+1]  # Keep batch dimension
    rot_i = recontrast_data['rot_maps'][sample_idx:sample_idx+1]
    scale_i = recontrast_data['scale_maps'][sample_idx:sample_idx+1]
    opacity_i = recontrast_data['opacity_maps'][sample_idx:sample_idx+1]
    sh_i = recontrast_data['sh_maps'][sample_idx:sample_idx+1]
    pts_vaild = recontrast_data['pts_valid'][sample_idx:sample_idx+1]
    xyz_i = xyz_i[pts_vaild]
    rot_i = rot_i[pts_vaild]
    scale_i = scale_i[pts_vaild]
    opacity_i = opacity_i[pts_vaild]
    sh_i = sh_i[pts_vaild]
    # Get camera parameters

    num_cams = getattr(model, 'num_cams', 6)

    for transform_name, transform_matrix in translation_transforms.items():
        transform_matrix = transform_matrix.to(device)
        for frame_id in novel_render_frames:
            for cam_id in range(num_cams):
                # Get original camera extrinsics and intrinsics
                original_e2c_extr = render_data[('e2c_extr', frame_id, cam_id)][sample_idx]
                K_i = render_data[('K', frame_id, cam_id)][sample_idx]

                # Apply lateral translation to camera pose
                # Transform ego to camera: new_e2c = e2c @ inv(transform)
                novel_e2c_extr = torch.matmul(original_e2c_extr, torch.linalg.inv(transform_matrix))

                model_width = getattr(model, 'width', 640)
                model_height = getattr(model, 'height', 352)
                # Render with new camera pose

                FovX = torch.tensor(focal2fov(K_i[0, 0], model_width)).to('cuda:0')
                FovY = torch.tensor(focal2fov(K_i[1, 1], model_height)).to('cuda:0')
                projection_matrix = getProjectionMatrix(znear=1.5, zfar=100, K=K_i, h=model_height, w=model_width).transpose(0, 1).to('cuda:0')
                world_view_transform = novel_e2c_extr.transpose(0, 1) 
                # full_proj_transform: (E^T K^T) = (K E)^T
                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                camera_center = world_view_transform.inverse()[3, :3] 

                render_rgb = render(novel_FovX=FovX,
                                        novel_FovY=FovY,
                                        novel_height=model_height,
                                        novel_width=model_width,
                                        novel_world_view_transform=world_view_transform,
                                        novel_full_proj_transform=full_proj_transform,
                                        novel_camera_center=camera_center,
                                        pts_xyz=xyz_i,#.contiguous(), 
                                        pts_rgb=None, 
                                        rotations=rot_i,#.contiguous(), 
                                        scales=scale_i,#.contiguous(), 
                                        opacity=opacity_i,#.contiguous(), 
                                        shs=sh_i,#.contiguous(), 
                                        sh_degree=4,
                                        bg_color=[1.0, 1.0, 1.0]) 

                global_sample_idx = actual_sample_idx + frame_id
                save_path = os.path.join(save_dir, scene_name, 
                                        f'sample_{global_sample_idx:04d}', transform_name, f'{eval_resolution}_cam_{cam_id}.png')
                # Save novel view with appropriate resolution
                resize_height, resize_width = eval_resolution.split('x')
                resize_height = int(resize_height)
                resize_width = int(resize_width)
                if resize_height !=model_height or resize_width != model_width:
                    save_rendered_image(render_rgb, save_path, upsample_to=(resize_height, resize_width))
                else:  # eval_resolution == 'original'
                    save_rendered_image(render_rgb, save_path)
                saved_paths.append(save_path)
        
    return saved_paths


def _process_scene_batch(model, scene_batch, device, gpu_id=0, save_renders=True, output_dir=None, novel_distances=[1.0, 2.0], eval_resolution='280x518',eval_frame=None):
    """Process a single scene batch and return results"""
    scene_start_time = time.time()

    scene_name = scene_batch['scene_name']
    scene_token = scene_batch['scene_token']
    
    frame_skip = 6
    # Support both lazy loading and pre-loaded modes
    if 'samples' in scene_batch:
        scene_samples = scene_batch['samples']
        scene_length = len(scene_samples)
        
        if frame_skip is not None and frame_skip > 1:
            # Calculate which indices to keep: 0, frame_skip, 2*frame_skip, ...
            original_sample_indices = list(range(0, scene_length, frame_skip))
            filtered_samples = [scene_samples[i] for i in original_sample_indices]
            scene_dataset = SceneSampleDataset(filtered_samples)
            actual_samples = len(filtered_samples)
            
            print(f"GPU {gpu_id}: Using frame skip of {frame_skip}, processing {actual_samples} out of {scene_length} samples")
        else:
            scene_dataset = SceneSampleDataset(scene_samples)
            actual_samples = scene_length
            original_sample_indices = list(range(scene_length))
    else:
        scene_length = scene_batch['scene_length']
        all_indices = scene_batch['sample_indices']
        
        if frame_skip is not None and frame_skip > 1:
            # Calculate which indices to keep: 0, frame_skip, 2*frame_skip, ...
            positions_to_keep = list(range(0, len(all_indices), frame_skip))
            original_sample_indices = [all_indices[pos] for pos in positions_to_keep]
            
            print(f"GPU {gpu_id}: Using frame skip of {frame_skip}, processing {len(original_sample_indices)} out of {scene_length} samples")
            
            scene_dataset = SceneSampleDataset(
                original_sample_indices,
                dataset=scene_batch['dataset'], 
                scene_idx=scene_batch['scene_idx']
            )
            actual_samples = len(original_sample_indices)
        else:
            original_sample_indices = all_indices
            scene_dataset = SceneSampleDataset(
                all_indices,
                dataset=scene_batch['dataset'], 
                scene_idx=scene_batch['scene_idx']
            )
            actual_samples = len(all_indices)
    
    print(f"GPU {gpu_id}: Processing Scene: {scene_name} ({actual_samples} samples)")
    
    
    # Create DataLoader
    scene_loader = DataLoader(
        scene_dataset, batch_size=1, shuffle=False,
        pin_memory=False, num_workers=0, drop_last=False,
        collate_fn=custom_collate_fn
    )
    
    scene_psnr_list, scene_ssim_list, scene_lpips_list = [], [], []
    batch_count = 0
    
    for batch in scene_loader:
        batch_count += 1

        if original_sample_indices is not None:
            if batch_count - 1 < len(original_sample_indices):
                actual_sample_idx = original_sample_indices[batch_count - 1]
            else:
                actual_sample_idx = batch_count - 1
        else:
            actual_sample_idx = batch_count - 1

        # Move batch to device
        batch_gpu = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        print(batch_gpu['frame_context'][0])

        if model.model.novel_view_mode=='SF':
            recontrast_frames = [0]
            if eval_frame is None:
                novel_render_frames = batch_gpu['frame_context'][0]
            else:
                novel_render_frames = eval_frame
        elif model.model.novel_view_mode=='MF':
            if len(batch_gpu['frame_context'][0])<2:
                continue
            MF_frame = batch_gpu['frame_context'][0][-1]
            recontrast_frames = [0, MF_frame]
            if eval_frame is None:
                novel_render_frames = batch_gpu['frame_context'][0][:-1]
            else:
                novel_render_frames = eval_frame

        print(f"scene_name: {scene_name}, batch_count: {batch_count}, novel_render_frames: {novel_render_frames}")

        output = model.predict_step(batch_gpu, recontrast_frames, novel_render_frames)
        model_width = getattr(model, 'width', 640)
        model_height = getattr(model, 'height', 352)
        if isinstance(output, tuple):
            batch_recontrast_data, batch_render_data, batch_splating_data = output
            
            # Calculate metrics for this batch
            batch_psnr, batch_ssim, batch_lpips = [], [], []
            num_cams = getattr(model, 'num_cams', 6)
            
            sample_psnr, sample_ssim, sample_lpips = [], [], []
            
            for frame_id in novel_render_frames:
                global_sample_idx = actual_sample_idx + frame_id
                for cam_id in range(num_cams):
                    pred_key = ('gaussian_color', frame_id, cam_id)
                    gt_key = ('groudtruth', frame_id, cam_id)
                    
                    if pred_key in batch_splating_data and gt_key in batch_splating_data:
                        pred = batch_splating_data[pred_key][0:1]
                        gt = batch_splating_data[gt_key][0:1]

                        resize_height, resize_width = eval_resolution.split('x')
                        resize_height = int(resize_height)
                        resize_width = int(resize_width)
                        if (resize_height == model_height) and (resize_width == model_width):
                            # Original mode: Use original model resolution (280x518)
                            if frame_id == 0 and cam_id == 0:
                                print(f"GPU {gpu_id}: Original mode - Using model resolution: pred={pred.shape}, gt={gt.shape}")
                                print(f"GPU {gpu_id}: Original mode - pred range: [{pred.min():.3f}, {pred.max():.3f}], gt range: [{gt.min():.3f}, {gt.max():.3f}]")
                            
                            # Ensure both pred and gt are in [0,1] range
                            pred_eval = pred.clamp(0, 1)
                            gt_eval = gt.clamp(0, 1)
                        else:
                            if ('color_org', frame_id) in batch_gpu:
                                # Use original high-resolution GT and upsample to 900x1600
                                gt_original = batch_gpu[('color_org', frame_id)][:, cam_id, ...][0:1]
                                if frame_id == 0 and cam_id == 0:
                                    print(f"GPU {gpu_id}: Upsampled mode - Original GT shape: {gt_original.shape}")
                                    print(f"GPU {gpu_id}: Upsampled mode - Original GT range: [{gt_original.min():.3f}, {gt_original.max():.3f}]")
                                
                                # Ensure GT is in [0,1] range and on correct device
                                gt_original = gt_original.clamp(0, 1).to(pred.device)
                                gt_eval = F.interpolate(gt_original, size=(resize_height, resize_width), mode='bilinear', align_corners=False)
                            else:
                                # Fallback: use downsampled GT if original not available
                                if frame_id == 0 and cam_id == 0:
                                    print(f"GPU {gpu_id}: Upsampled mode - Warning: Using downsampled GT: {gt.shape}")
                                gt_eval = F.interpolate(gt, size=(resize_height, resize_width), mode='bilinear', align_corners=False)
                                gt_eval = gt_eval.clamp(0, 1)
                            
                            # Upsample predicted image to 900x1600 and ensure in [0,1] range
                            pred_eval = F.interpolate(pred, size=(resize_height, resize_width), mode='bilinear', align_corners=False)
                            pred_eval = pred_eval.clamp(0, 1)

                        # Calculate metrics on evaluation resolution
                        psnr_val = model.compute_psnr(gt_eval, pred_eval).mean().item()
                        ssim_val = model.compute_ssim(gt_eval, pred_eval).mean().item()
                        lpips_val = model.compute_lpips(gt_eval, pred_eval).mean().item()
                        
                        # Debug metrics calculation
                        if frame_id == 0 and cam_id == 0:
                            print(f"GPU {gpu_id}: Metrics - PSNR: {psnr_val:.3f}, SSIM: {ssim_val:.3f}, LPIPS: {lpips_val:.3f}")
                        
                        sample_psnr.append(psnr_val)
                        sample_ssim.append(ssim_val)
                        sample_lpips.append(lpips_val)
                        
                        # Save rendered images if requested

                        # Save rendered image at evaluation resolution
                        frame_dir = 'gt_views'

                        pred_save_path = os.path.join(output_dir, scene_name,
                                                    f'sample_{global_sample_idx:04d}', frame_dir,  f'{eval_resolution}_cam_{cam_id}_pred.png')
                        save_rendered_image(pred_eval.squeeze(0), pred_save_path)
                        
                        # Save ground truth image at evaluation resolution
                        gt_save_path = os.path.join(output_dir, scene_name,
                                                    f'sample_{global_sample_idx:04d}', frame_dir,  f'{eval_resolution}_cam_{cam_id}_gt.png')
                        save_rendered_image(gt_eval.squeeze(0), gt_save_path)
            
            # Generate and save novel views for all samples
            if save_renders:
                try:
                    novel_view_paths = render_novel_views(
                        model, batch_recontrast_data, batch_render_data, novel_render_frames,
                        device, scene_name, 0, output_dir, actual_sample_idx, novel_distances, eval_resolution
                    )
                    print(f"GPU {gpu_id}: Saved novel views for sample {actual_sample_idx}: {len(novel_view_paths)} images")
                except Exception as e:
                    print(f"GPU {gpu_id}: Error generating novel views for sample {actual_sample_idx}: {e}")
            
            if sample_psnr:
                batch_psnr.append(np.mean(sample_psnr))
                batch_ssim.append(np.mean(sample_ssim))
                batch_lpips.append(np.mean(sample_lpips))
            
            scene_psnr_list.extend(batch_psnr)
            scene_ssim_list.extend(batch_ssim)
            scene_lpips_list.extend(batch_lpips)
                

    
    # Calculate processing time
    scene_processing_time = time.time() - scene_start_time
    
    # Return scene results
    return {
        'scene_idx': scene_batch.get('scene_idx', 0),
        'scene_name': scene_name,
        'scene_token': scene_token,
        'scene_length': scene_length,
        'processed_samples': len(scene_psnr_list),
        'processing_time': scene_processing_time,
        'avg_sample_time': scene_processing_time / max(1, len(scene_psnr_list)),
        'gpu_id': gpu_id,
        'metrics': {
            'psnr': np.mean(scene_psnr_list) if scene_psnr_list else 0.0,
            'ssim': np.mean(scene_ssim_list) if scene_ssim_list else 0.0,
            'lpips': np.mean(scene_lpips_list) if scene_lpips_list else 0.0,
            'psnr_std': np.std(scene_psnr_list) if scene_psnr_list else 0.0,
            'ssim_std': np.std(scene_ssim_list) if scene_ssim_list else 0.0,
            'lpips_std': np.std(scene_lpips_list) if scene_lpips_list else 0.0
        },
        'sample_metrics': {
            'psnr_list': scene_psnr_list,
            'ssim_list': scene_ssim_list,
            'lpips_list': scene_lpips_list
        }
    }


def _run_single_gpu_inference(model, scene_dataloader, device, save_results=True, output_dir=None, novel_distances=[1.0, 2.0], eval_resolution='280x518',eval_frame=None):
    """Run inference on all scenes - simplified using unified scene processing"""
    print(f"\nStarting scene-based inference on device: {device}")
    print(f"Number of scenes: {len(scene_dataloader)}")
    print(f'eval_frame: {eval_frame}')
    all_scene_results = []
    overall_psnr, overall_ssim, overall_lpips = [], [], []
    
    with torch.no_grad():
        for scene_idx, scene_batch in enumerate(scene_dataloader):
            scene_batch['scene_idx'] = scene_idx  # Ensure scene_idx is set
            result = _process_scene_batch(model, scene_batch, device, gpu_id=0, 
                                        save_renders=save_results, output_dir=output_dir, novel_distances=novel_distances, eval_resolution=eval_resolution,eval_frame=eval_frame)
            
            all_scene_results.append(result)
            overall_psnr.extend(result['sample_metrics']['psnr_list'])
            overall_ssim.extend(result['sample_metrics']['ssim_list'])
            overall_lpips.extend(result['sample_metrics']['lpips_list'])
    
    # Print final results
    final_psnr = np.mean(overall_psnr) if overall_psnr else 0.0
    final_ssim = np.mean(overall_ssim) if overall_ssim else 0.0
    final_lpips = np.mean(overall_lpips) if overall_lpips else 0.0
    
    print(f"\n{'='*60}")
    print(f"SINGLE-GPU INFERENCE COMPLETED")
    print(f"{'='*60}")
    print(f"Scenes: {len(all_scene_results)}, Samples: {sum(r['processed_samples'] for r in all_scene_results)}")
    print(f"Overall PSNR: {final_psnr:.4f}, SSIM: {final_ssim:.4f}, LPIPS: {final_lpips:.4f}")
    
    # Save results
    if save_results and output_dir:
        final_results = {
            'overall_metrics': {'psnr': final_psnr, 'ssim': final_ssim, 'lpips': final_lpips,
                               'psnr_std': np.std(overall_psnr), 'ssim_std': np.std(overall_ssim), 'lpips_std': np.std(overall_lpips)},
            'scene_results': all_scene_results
        }
        save_inference_results(final_results, output_dir)
    
    return all_scene_results


def run_single_gpu_inference(gpu_id, scene_indices, all_scenes, config, checkpoint_path, save_results, output_dir, results_queue, novel_distances=[1.0, 2.0], eval_resolution='280x518',eval_frame=None):
    """
    Run inference on a single GPU for assigned scenes - simplified using unified processing
    
    Args:
        gpu_id: GPU device ID
        scene_indices: List of scene indices to process
        all_scenes: List of all scene data
        config: Model configuration
        checkpoint_path: Path to model checkpoint
        batch_size: Batch size for processing
        output_dir: Output directory
        results_queue: Queue for returning results
        novel_distances: List of distances for novel view generation
        eval_resolution: Resolution mode ('original' or 'upsampled')
    """
    try:
        device = f'cuda:{gpu_id}'
        print(f"\nGPU {gpu_id}: Starting inference on device {device}")
        print(f"GPU {gpu_id}: Processing {len(scene_indices)} scenes: {scene_indices}")
        
        # Load model on this GPU
        model = load_model_from_checkpoint(checkpoint_path, config['model_cfg'], device)
        
        gpu_results = []
        gpu_psnr_list, gpu_ssim_list, gpu_lpips_list = [], [], []
        
        with torch.no_grad():
            for scene_idx in scene_indices:
                scene_batch = all_scenes[scene_idx]
                scene_batch['scene_idx'] = scene_idx  # Ensure scene_idx is set
                
                # Process scene using unified function
                result = _process_scene_batch(model, scene_batch, device, gpu_id,
                                            save_renders=save_results, output_dir=output_dir, novel_distances=novel_distances, eval_resolution=eval_resolution, eval_frame=eval_frame)
                
                gpu_results.append(result)
                gpu_psnr_list.extend(result['sample_metrics']['psnr_list'])
                gpu_ssim_list.extend(result['sample_metrics']['ssim_list'])
                gpu_lpips_list.extend(result['sample_metrics']['lpips_list'])
        
        # Put results in queue
        gpu_result = {
            'gpu_id': gpu_id,
            'scene_results': gpu_results,
            'gpu_metrics': {
                'psnr_list': gpu_psnr_list,
                'ssim_list': gpu_ssim_list,
                'lpips_list': gpu_lpips_list
            }
        }
        results_queue.put(gpu_result)
        print(f"GPU {gpu_id}: Completed all assigned scenes")
        
    except Exception as e:
        print(f"GPU {gpu_id}: Error in GPU worker: {e}")
        import traceback
        print(f"GPU {gpu_id}: Error details: {traceback.format_exc()}")
        results_queue.put({'gpu_id': gpu_id, 'error': str(e)})


def _run_multi_gpu_inference(model_cfg, scene_dataloader, devices, checkpoint_path, save_results=True, output_dir=None, novel_distances=[1.0, 2.0], eval_resolution='280x518',eval_frame=None):
    """Run inference on multiple GPUs with scene distribution"""
    print(f"\nStarting multi-GPU scene-based inference")
    print(f"Devices: {devices}")
    print(f"Number of scenes to process: {len(scene_dataloader)}")
    
    # Collect all scenes
    all_scenes = list(scene_dataloader)
    total_scenes = len(all_scenes)
    
    # Distribute scenes across GPUs
    scenes_per_gpu = total_scenes // len(devices)
    remainder = total_scenes % len(devices)
    
    scene_distribution = []
    start_idx = 0
    
    for i, device_id in enumerate(devices):
        # Add one extra scene to first 'remainder' GPUs
        num_scenes = scenes_per_gpu + (1 if i < remainder else 0)
        end_idx = start_idx + num_scenes
        scene_indices = list(range(start_idx, end_idx))
        scene_distribution.append((device_id, scene_indices))
        start_idx = end_idx
        
        print(f"GPU {device_id}: assigned {len(scene_indices)} scenes (indices {scene_indices})")
    
    # Start inference on all GPUs
    total_start_time = time.time()
    
    # Use threading for multi-GPU execution
    results_queue = mp.Queue()
    threads = []
    
    for device_id, scene_indices in scene_distribution:
        if len(scene_indices) > 0:  # Only start thread if there are scenes to process
            thread = threading.Thread(
                target=run_single_gpu_inference,
                args=(device_id, scene_indices, all_scenes, {'model_cfg': model_cfg}, checkpoint_path, save_results, output_dir, results_queue, novel_distances, eval_resolution, eval_frame)
            )
            thread.start()
            threads.append(thread)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Collect results from all GPUs
    all_results = []
    overall_psnr = []
    overall_ssim = []
    overall_lpips = []
    
    while not results_queue.empty():
        gpu_result = results_queue.get()
        if 'error' not in gpu_result:
            all_results.extend(gpu_result['scene_results'])
            overall_psnr.extend(gpu_result['gpu_metrics']['psnr_list'])
            overall_ssim.extend(gpu_result['gpu_metrics']['ssim_list'])
            overall_lpips.extend(gpu_result['gpu_metrics']['lpips_list'])
        else:
            print(f"GPU {gpu_result['gpu_id']} encountered error: {gpu_result['error']}")
    
    # Sort results by scene index
    all_results.sort(key=lambda x: x['scene_idx'])
    
    total_time = time.time() - total_start_time
    total_samples = sum(result['processed_samples'] for result in all_results)
    
    # Calculate overall metrics
    final_psnr = np.mean(overall_psnr) if overall_psnr else 0.0
    final_ssim = np.mean(overall_ssim) if overall_ssim else 0.0
    final_lpips = np.mean(overall_lpips) if overall_lpips else 0.0
    
    print(f"\n{'='*60}")
    print(f"MULTI-GPU INFERENCE COMPLETED")
    print(f"{'='*60}")
    print(f"Total scenes processed: {len(all_results)}")
    print(f"Total samples processed: {total_samples}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per scene: {total_time / max(1, len(all_results)):.2f}s")
    print(f"Average time per sample: {total_time / max(1, total_samples):.3f}s")
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Overall PSNR: {final_psnr:.4f} (±{np.std(overall_psnr):.4f})")
    print(f"Overall SSIM: {final_ssim:.4f} (±{np.std(overall_ssim):.4f})")
    print(f"Overall LPIPS: {final_lpips:.4f} (±{np.std(overall_lpips):.4f})")
    
    # Save results
    if save_results and output_dir:
        final_results = {
            'overall_metrics': {
                'psnr': final_psnr,
                'ssim': final_ssim,
                'lpips': final_lpips,
                'psnr_std': np.std(overall_psnr) if overall_psnr else 0.0,
                'ssim_std': np.std(overall_ssim) if overall_ssim else 0.0,
                'lpips_std': np.std(overall_lpips) if overall_lpips else 0.0
            },
            'scene_results': all_results,
            'multi_gpu_info': {
                'devices': devices,
                'scene_distribution': {f'gpu_{dev}': indices for dev, indices in scene_distribution}
            }
        }
        save_inference_results(final_results, output_dir)
    
    return all_results


def save_inference_results(results, output_dir):
    """Save inference results to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    
    if 'overall_metrics' in results:
        # New format with overall metrics
        scene_results = results['scene_results']
        overall_metrics = results['overall_metrics']
        
        # Create summary
        summary = {
            'overall_metrics': overall_metrics,
            'total_scenes': len(scene_results),
            'total_samples': sum(r['processed_samples'] for r in scene_results),
            'total_time': sum(r['processing_time'] for r in scene_results),
            'scenes': [
                {
                    'scene_idx': r['scene_idx'],
                    'scene_name': r['scene_name'],
                    'scene_token': r['scene_token'],
                    'processed_samples': r['processed_samples'],
                    'processing_time': r['processing_time'],
                    'metrics': r['metrics']
                }
                for r in scene_results
            ]
        }
        
        # Save summary
        summary_file = os.path.join(output_dir, 'inference_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results
        detailed_file = os.path.join(output_dir, 'inference_detailed.json')
        with open(detailed_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save per-scene evaluation results
        for scene_result in scene_results:
            scene_name = scene_result['scene_name']
            scene_eval_file = os.path.join(output_dir, f'scene_{scene_name}_evaluation.json')
            scene_eval_data = {
                'scene_name': scene_name,
                'scene_token': scene_result['scene_token'],
                'metrics': scene_result['metrics'],
                'sample_metrics': scene_result['sample_metrics'],
                'processing_info': {
                    'processed_samples': scene_result['processed_samples'],
                    'processing_time': scene_result['processing_time'],
                    'avg_sample_time': scene_result['avg_sample_time']
                }
            }
            
            with open(scene_eval_file, 'w') as f:
                json.dump(scene_eval_data, f, indent=2)
        
        print(f"\nResults saved:")
        print(f"  Summary: {summary_file}")
        print(f"  Detailed: {detailed_file}")
        print(f"  Per-scene evaluations: {output_dir}/scene_*_evaluation.json")
        
    else:
        # Legacy format
        summary = {
            'total_scenes': len(results),
            'total_samples': sum(r['processed_samples'] for r in results),
            'total_time': sum(r['processing_time'] for r in results),
            'scenes': [
                {
                    'scene_idx': r['scene_idx'],
                    'scene_name': r['scene_name'],
                    'scene_token': r['scene_token'],
                    'processed_samples': r['processed_samples'],
                    'processing_time': r['processing_time'],
                    'scene_stats': r.get('scene_stats', {})
                }
                for r in results
            ]
        }
        
        # Save summary
        summary_file = os.path.join(output_dir, 'inference_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results
        detailed_file = os.path.join(output_dir, 'inference_detailed.json')
        with open(detailed_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved:")
        print(f"  Summary: {summary_file}")
        print(f"  Detailed: {detailed_file}")


def main():
    parser = argparse.ArgumentParser(description='Scene-based inference for VGGT3DGS')
    parser.add_argument('--cfg_path', type=str, required=True, help='Configuration file path')
    parser.add_argument('--restore_ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for results')
    parser.add_argument('--max_scenes', type=int, default=None, help='Maximum number of scenes to process (default: all scenes)')
    parser.add_argument('--device', type=str, default=None, help='Device to use. For single GPU: cuda:0. For multi-GPU: 0,1 (comma-separated)')

    parser.add_argument('--multi_gpu', action='store_true', help='Enable multi-GPU inference')
    parser.add_argument('--no_renders', action='store_true', help='Disable saving rendered images and novel views')
    parser.add_argument('--novel_distances', type=str, default='0.5,1.0,2.0,3.0', 
                       help='Novel view translation distances in meters (comma-separated, e.g., "0.5,1.0,2.0,3.0")')
    parser.add_argument('--eval_resolution', type=str, default='original',# choices=['original', 'upsampled'],
                       help='Evaluation resolution mode: "original" for 280x518, "upsampled" for 900x1600')
    parser.add_argument('--eval_frame', type=str, default=None, help='frame_ids for inference (default: None for all frame)')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.cfg_path}")
    with open(args.cfg_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Set batch_size in model config for inference
    config['model_cfg']['batch_size'] = 1
    config['data_cfg']['batch_size'] = 1

    
    # Parse devices
    if args.device:
        if ',' in args.device:
            # Multi-GPU specified: "0,1" -> [0, 1]
            devices = [int(d.strip()) for d in args.device.split(',')]
            args.multi_gpu = True
        else:
            # Single GPU specified: "cuda:0" or "0"
            if args.device.startswith('cuda:'):
                device = args.device
                devices = [int(args.device.split(':')[1])]
            else:
                device = f"cuda:{args.device}"
                devices = [int(args.device)]
    elif config.get('devices'):
        devices = config['devices']
        device = f"cuda:{devices[0]}"
        if len(devices) > 1:
            args.multi_gpu = True
    else:
        devices = [0] if torch.cuda.is_available() else ['cpu']
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    print(f"Batch size: {config['data_cfg']['batch_size']}")
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(config['save_dir'], 'scene_inference_results')
    
    print(f"Output directory: {args.output_dir}")
    
    # Parse save renders flag
    save_renders = not args.no_renders
    
    # Parse novel view distances
    try:
        novel_distances = [float(d.strip()) for d in args.novel_distances.split(',')]
    except ValueError:
        raise ValueError(f"Invalid novel_distances format: {args.novel_distances}. Use comma-separated floats like '0.5,1.0,2.0,3.0'")
    
    print(f"Save renders: {save_renders}")
    print(f"Novel view distances: {novel_distances}")
    print(f"Evaluation resolution: {args.eval_resolution}")

    try:
        eval_frame = [int(d.strip()) for d in args.eval_frame.split(',')]
    except ValueError:
        print(f"Invalid eval_frame format: {args.eval_frame}. Use 'eval_frame=None' ")
        eval_frame = None

    print(f"Evaluation frame: {eval_frame}")
    # Initialize scene-based data module
    print("Initializing scene-based data module...")
    data_module = VGGT3DGS_SceneDataModule(cfg=config['data_cfg'])
    data_module.setup(stage='test')
    
    # Get scene dataloader
    scene_dataloader = data_module.test_scene_dataloader()
    total_scenes = len(scene_dataloader)
    
    if args.max_scenes:
        print(f"Limiting to {args.max_scenes} scenes (out of {total_scenes})")
        # Limit scenes if requested
        scene_list = []
        for i, scene_batch in enumerate(scene_dataloader):
            if i >= args.max_scenes:
                break
            scene_list.append(scene_batch)
        scene_dataloader = scene_list
    else:
        print(f"Processing all {total_scenes} scenes")
    
    # Run unified inference (automatically detects single/multi-GPU)
    if args.multi_gpu and len(devices) > 1:
        device_input = devices  # List for multi-GPU
    else:
        device_input = device   # String for single GPU
    
    results = run_inference(
        model_cfg=config['model_cfg'],
        checkpoint_path=args.restore_ckpt,
        scene_dataloader=scene_dataloader,
        devices=device_input,
        save_results=save_renders,
        output_dir=args.output_dir,
        novel_distances=novel_distances,
        eval_resolution=args.eval_resolution,
        eval_frame=eval_frame,
    )
    
    print(f"\nScene-based inference completed successfully!")
    print(f"Results saved to: {args.output_dir}")

    # eval_result_csv_root = "/workspace/data_storage/ruiyanghao/projects/uniadv2/UniAD/eval_res"

    # eval_sets = [ 
    #     ("normal_gen", 0),
    #     ("left", 1), ("left", 2), ("left", 3),
    #     ("right", 1), ("right", 2), ("right", 3)
    # ]

    # all_metrics = {}

    # for eval_set in eval_sets:
    #     print("Eval Setting:", eval_set)
    #     metrics = run_uniad(
    #         generated_path=args.output_dir,
    #         translation_mode=eval_set[0],
    #         translation_offset=eval_set[1]
    #     )
    #     print(metrics)

    #     key = f"{eval_set[0]}_{eval_set[1]}"
    #     all_metrics[key] = metrics

    # df = pd.DataFrame.from_dict(all_metrics, orient="index")

    # mean_metrics = df.mean().round(3)

    # df.loc["mean"] = mean_metrics

    # print("\n==== Summary of All Metrics ====")
    # print(df)

    # try:
    #     dir_name = pathlib.Path(args.output_dir).name
    #     csv_filename = f"{dir_name}_evaluation_summary.csv" 
    # except Exception:
    #     csv_filename = "evaluation_summary.csv"

    # df.to_csv(os.path.join(eval_result_csv_root, csv_filename))
    
    # print(f"评估结果已保存到: {os.path.join(eval_result_csv_root, csv_filename)}")

if __name__ == "__main__":
    main()