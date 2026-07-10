from collections import defaultdict
import PIL.Image as Image
import torch
import gc
import torch.nn.functional as F
import torch.optim as optim
import os
import shutil
import numpy as np
import pytorch_lightning as pl
from einops import rearrange, reduce
from torch import Tensor
from lpips import LPIPS
from jaxtyping import Float, UInt8
from pytorch_lightning.utilities import rank_zero_only
from skimage.metrics import structural_similarity
from kornia.losses import SSIMLoss
from optimi import StableAdamW
from math import log2, log
import psutil


import sys
from gsplat.rendering import rasterization
import cv2
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import numpy as np
import os
from pathlib import Path
# from models.vggt3dgs_model2 import VGGT3DGSModel
from models.vggt4dgs_model import VGGT4DGSModel

from models.loss_util import compute_masked_loss, compute_edg_smooth_loss

EPSILON = 1e-6
def print_memory(msg):
    # torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"[MEM] {msg}: Alloc={allocated:.2f}GB, Reserved={reserved:.2f}GB")




class VGGT4DGS_LITModelModule(pl.LightningModule):
    def __init__(self, cfg, save_dir='.', logger=None):
        super().__init__()
        self.read_config(cfg)

        # Set default values for ego transformation configuration if not in config
        if not hasattr(self, 'use_gt_ego_trans'):
            self.use_gt_ego_trans = False
        if not hasattr(self, 'use_icp_refinement'):
            self.use_icp_refinement = False
        if not hasattr(self, 'translate_3dgs'):
            self.translate_3dgs = False
        if not hasattr(self, 'use_icp_v2'):
            self.use_icp_v2 = False
        if not hasattr(self, 'use_icp_separate'):
            self.use_icp_separate = False
        if not hasattr(self, 'icp_subsample_rate'):
            self.icp_subsample_rate = 10
        if not hasattr(self, 'icp_mask'):
            self.icp_mask = 10
        self.save_visualizations = False # True

        self.save_dir = save_dir
        self.model = VGGT4DGSModel(sh_degree=self.sh_degree,height=self.height,width=self.width,min_depth=self.min_depth, max_depth=self.max_depth) 
        self.lpips = LPIPS(net="vgg")
        self.ssim_fn = SSIMLoss(window_size=11,reduction='none')
        self.l1_fn = torch.nn.L1Loss(reduction='none')
        self.lpips.eval()

        self.save_hyperparameters('cfg','save_dir')
    


    def read_config(self, cfg):
        for k, v in cfg.items():
            setattr(self, k, v)

    @rank_zero_only
    def print_fn(self, log_msg):
        print(log_msg)

    @rank_zero_only
    def log_system_status(self,):
        """记录系统状态用于诊断"""
        # 内存信息
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
    
        process = psutil.Process(os.getpid())

        print(f"内存使用: {memory.used/1024/1024/1024:.2f}GB / {memory.total/1024/1024/1024:.2f}GB ({memory.percent}%)")
        print(f"交换空间: {swap.used/1024/1024/1024:.2f}GB / {swap.total/1024/1024/1024:.2f}GB")
        print(f"文件描述符： {process.num_fds()}")
        print_memory('GPU')
        with open('/proc/sys/fs/file-nr', 'r') as f:
            allocated, free, max_fds = map(int, f.read().split())
            fd_usage = allocated / max_fds
            print(f"系统FD使用率: {allocated} {free} {max_fds} {fd_usage:.1%}")

        print(f"进程内存: {process.memory_info().rss/1024/1024:.2f} MB")

    def _log_weights_and_grads(self, inputs):
        current_step = self.global_step
        max_weight = -np.inf
        max_grad = -np.inf
        max_weight_name = ""
        max_grad_name = ""
        nan_params = []
        inf_params = []

        for name ,val in inputs.items():
            if isinstance(val,torch.Tensor):
                if torch.isnan(val).any():
                    nan_params.append(name)        
                if torch.isinf(val).any():
                    inf_params.append(name)     

        # 检测并报告NaN
        if len(nan_params)>0 or len(inf_params)>0:
            self.print_fn(f'nan_prams: {nan_params}')
            self.print_fn(f'inf_prams: {inf_params}')
            sys.exit(-1)
        
        # 2. 检查优化器状态
        for optimizer in self.trainer.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                for j, param in enumerate(param_group['params']):
                    state = optimizer.state[param]
                    if 'exp_avg' in state and torch.isnan(state['exp_avg']).any():
                        nan_params.append(f'exp_avg:param_group={i}, param={j}')
                    if 'exp_avg_sq' in state and torch.isnan(state['exp_avg_sq']).any():
                        nan_params.append(f'exp_avg_sq:param_group={i}, param={j}')

                    if 'exp_avg' in state and torch.isinf(state['exp_avg']).any():
                        inf_params.append(f'exp_avg:param_group={i}, param={j}')
                    if 'exp_avg_sq' in state and torch.isinf(state['exp_avg_sq']).any():
                        inf_params.append(f'exp_avg_sq:param_group={i}, param={j}')

        # 检测并报告NaN
        if len(nan_params)>0 or len(inf_params)>0:
            self.print_fn(f'nan_prams: {nan_params}')
            self.print_fn(f'inf_prams: {inf_params}')
            
            sys.exit(-1)

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
                
            # 检测NaN值
            if torch.isnan(param.data).any() or torch.isnan(param.grad).any():
                nan_params.append(name)
            if torch.isinf(param.data).any() or torch.isinf(param.grad).any():
                inf_params.append(name)
            
            # 查找最大权重值
            param_max = param.data.abs().max().item()
            if param_max > max_weight:
                max_weight = param_max
                max_weight_name = name
                
            # 查找最大梯度值
            grad_max = param.grad.data.abs().max().item()
            if grad_max > max_grad:
                max_grad = grad_max
                max_grad_name = name
        
        # 检测并报告NaN
        if len(nan_params)>0 or len(inf_params)>0:
            self.print_fn(f'nan_prams: {nan_params}')
            self.print_fn(f'inf_prams: {inf_params}')
            sys.exit(-1)
    
    def on_before_optimizer_step(self, optimizer):
        # warmup in first 100 steps
        if self.current_epoch == 0 and self.global_step < self.warmup_step:
            # Warmup: 从0.1倍学习率线性增长到原始学习率
            warmup_ratio = 0.1 + 0.9 * (self.global_step / self.warmup_step)
            base_lr = self.learning_rate
            if self.auto_scale_lr:
                num_devices = self.trainer.num_devices
                scale_devices = max(1, log2(num_devices))  # 至少按1计算
                base_lr = self.learning_rate * scale_devices
            
            warmup_lr = base_lr * warmup_ratio
            
            for param_group in self.trainer.optimizers[0].param_groups:
                param_group['lr'] = warmup_lr
                
        """在优化器步骤前检查梯度"""
        valid_gradients = True
        
        for name, param in self.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    self.print_fn(f"NaN 梯度: {name}")
                    valid_gradients = False
                if torch.isinf(param.grad).any():
                    self.print_fn(f"Inf 梯度: {name}")
                    valid_gradients = False
        
        if not valid_gradients:
            self.print_fn("检测到无效梯度，跳过本次更新")
            # 清零梯度并跳过步骤
            optimizer.zero_grad()
            return False
        return True
                        
    def configure_optimizers(self):
        for name, parameters in self.model.named_parameters():
            if parameters.requires_grad:
                self.print_fn(f'Training {name}')

        if self.auto_scale_lr:
            num_devices = self.trainer.num_devices
            scale_devices = max(1, log2(num_devices))  # 至少按1计算
            base_lr = self.learning_rate * scale_devices
        else:
            base_lr = self.learning_rate

        # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=base_lr,momentum=self.momentum,weight_decay=self.weight_decay)
        # optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=base_lr,betas=(0.9,0.98),eps=1e-7,weight_decay=self.weight_decay)
        optimizer = StableAdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=base_lr,betas=(0.9,0.98),weight_decay=self.weight_decay)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=self.scheduler_step_size,gamma=self.scheduler_gamma)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=self.lr_restart_epoch,  # 每5个epoch重启
                    T_mult=self.lr_restart_mult,
                    eta_min=base_lr*self.lr_min_factor*0.1  # 调整最小学习率以适应较小的学习率
                )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }
    

    def split_input_views(self, all_valid_frames, gtview_frame_ids=None):

        if gtview_frame_ids is None:
            gtview_frame_ids = np.random.choice(all_valid_frames,size=2,replace=False).tolist()

        nvview_frame_ids = set(all_valid_frames) - set(gtview_frame_ids)
        
        return sorted(gtview_frame_ids), sorted(nvview_frame_ids)


    
    def training_step(self, batch_input, batch_idx):
        self.stage =  'train'

        # self.log_system_status()
        self._log_weights_and_grads(batch_input)

        all_valid_frames = batch_input['frame_context'][0].cpu().tolist()
        gtview_frame_ids, nvview_frame_ids = self.split_input_views(all_valid_frames)

        render_frame_ids = gtview_frame_ids + np.random.choice(nvview_frame_ids,size=1).tolist()
        batch_data = self.predict_step(batch_input,recontrast_frames=gtview_frame_ids,render_frames=render_frame_ids)

        loss_depth_smooth = self.compute_smooth_loss(batch_data)
        # loss_norm = self.compute_norm_loss(batch_recontrast_data)
        loss_gaussian = self.compute_gaussian_loss(batch_data)

        if batch_idx%self.save_image_duration==0:
            return_imgs = True
        else:
            return_imgs = False
        loss_project = self.compute_project_loss(batch_data, return_imgs=return_imgs)

        self.log(f'{self.stage}/gs', loss_gaussian.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{self.stage}/proj', loss_project.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'{self.stage}/depth', loss_depth_smooth.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        # self.log(f'{self.stage}/norm', loss_norm.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        loss_all = loss_gaussian + loss_project + loss_depth_smooth # + loss_norm
        psnr, ssim, lpips = self.compute_reconstruction_metrics(batch_data)

        for frame_id in batch_data['render_frames']:
            depth_list = []
            for cam_id in range(self.num_cams):
                depth_list.append(batch_data[('gaussian_depth', frame_id, cam_id)].detach())
            depth_list = torch.cat(depth_list,dim=1)
            min_depth = torch.amin(depth_list).cpu().item()
            mean_depth = torch.mean(depth_list).cpu().item()
            max_depth = torch.amax(depth_list).cpu().item()
            print_log = f" {frame_id}"
            print_log += f" | min_depth: {min_depth:.4f} | mean_depth: {mean_depth:.4f} | max_depth: {max_depth:.4f}"
            self.print_fn(print_log)

        print_log = ''
        print_log += f"[{self.stage}] {batch_idx} | loss_all: {loss_all.item():.4f} | loss_gaussian: {loss_gaussian.item():.4f} " 
        print_log += f"| loss_proj_smooth: {loss_project.item():.4f} | loss_depth_smooth: {loss_depth_smooth.item():.4f} " #| loss_norm: {loss_norm.item():.4f} "
        print_log += f"| psnr: {psnr:.4f} | ssim: {ssim:.4f} | lpips: {lpips:.4f}"
        self.print_fn(print_log)

        if return_imgs and (self.stage=='train'):
            splating_file = f'{self.stage}_{batch_idx}_splating'
            self._save_splating_images(splating_file,batch_data)
            project_file = f'{self.stage}_{batch_idx}_project'
            self._save_projection_images(project_file,batch_data)

        del batch_input, batch_data, psnr, ssim, lpips

        return loss_all

    def debug_step(self, batch_input, batch_idx):
        self.stage =  'train'

        # self._log_weights_and_grads(batch_input)
        self.log_system_status()
        all_valid_frames = batch_input['frame_context'][0].cpu().tolist()
        gtview_frame_ids, nvview_frame_ids = self.split_input_views(all_valid_frames)
        render_frame_ids = gtview_frame_ids + np.random.choice(nvview_frame_ids,size=1).tolist()
        batch_data = self.predict_step(batch_input,recontrast_frames=gtview_frame_ids,render_frames=render_frame_ids)

        loss_depth_smooth = self.compute_smooth_loss(batch_data)
        # loss_norm = self.compute_norm_loss(batch_recontrast_data)
        loss_gaussian = self.compute_gaussian_loss(batch_data)

        if batch_idx%self.save_image_duration==0:
            return_imgs = True
        else:
            return_imgs = False
        loss_project = self.compute_project_loss(batch_data, return_imgs=return_imgs)

        self.log(f'{self.stage}/gs', loss_gaussian.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{self.stage}/proj', loss_project.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'{self.stage}/depth', loss_depth_smooth.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        # self.log(f'{self.stage}/norm', loss_norm.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        loss_all = loss_gaussian + loss_project + loss_depth_smooth # + loss_norm
        psnr, ssim, lpips = self.compute_reconstruction_metrics(batch_data)

        for frame_id in batch_data['render_frames']:
            depth_list = []
            for cam_id in range(self.num_cams):
                depth_list.append(batch_data[('gaussian_depth', frame_id, cam_id)].detach())
            depth_list = torch.cat(depth_list,dim=1)
            min_depth = torch.amin(depth_list).cpu().item()
            mean_depth = torch.mean(depth_list).cpu().item()
            max_depth = torch.amax(depth_list).cpu().item()
            print_log = f" {frame_id}"
            print_log += f" | min_depth: {min_depth:.4f} | mean_depth: {mean_depth:.4f} | max_depth: {max_depth:.4f}"
            self.print_fn(print_log)

        print_log = ''
        print_log += f"[{self.stage}] {batch_idx} | loss_all: {loss_all.item():.4f} | loss_gaussian: {loss_gaussian.item():.4f} " 
        print_log += f"| loss_proj_smooth: {loss_project.item():.4f} | loss_depth_smooth: {loss_depth_smooth.item():.4f} " #| loss_norm: {loss_norm.item():.4f} "
        print_log += f"| psnr: {psnr:.4f} | ssim: {ssim:.4f} | lpips: {lpips:.4f}"
        self.print_fn(print_log)

        if return_imgs and (self.stage=='train'):
            splating_file = f'{self.stage}_{batch_idx}_splating'
            self._save_splating_images(splating_file,batch_data)
            project_file = f'{self.stage}_{batch_idx}_project'
            self._save_projection_images(project_file,batch_data)

        del batch_input, batch_data, psnr, ssim, lpips

        return loss_all
    
    def predict_step(self, batch_data, recontrast_frames=[], render_frames=[]):
        batch_data['recontrast_frames'] = recontrast_frames
        batch_data['novel_frames'] = list(set(render_frames) - set(recontrast_frames))
        batch_data['render_frames'] = render_frames

        self.refine_camera_position(batch_data)

        self.get_recontrast_data_gtview(batch_data)
        if len(batch_data['novel_frames'])>0:
            self.get_recontrast_data_nvview(batch_data)

        self.get_splating_imgs(batch_data)

        return batch_data
    
    def validation_step(self, batch_input, batch_idx):
        self.stage = 'val'

        # self._log_weights_and_grads(batch_input)

        all_valid_frames = batch_input['frame_context'][0].cpu().tolist()
        gtview_frame_ids, nvview_frame_ids = self.split_input_views(all_valid_frames,[0,6])
        render_frame_ids = all_valid_frames #gtview_frame_ids #+ nvview_frame_ids
        batch_data = self.predict_step(batch_input,recontrast_frames=gtview_frame_ids,render_frames=render_frame_ids)

        loss_depth_smooth = self.compute_smooth_loss(batch_data)
        # loss_norm = self.compute_norm_loss(batch_recontrast_data)
        loss_gaussian = self.compute_gaussian_loss(batch_data)

        if batch_idx%self.save_image_duration==0:
            return_imgs = True
        else:
            return_imgs = False
        loss_project = self.compute_project_loss(batch_data, return_imgs=return_imgs)

        self.log(f'{self.stage}/gs', loss_gaussian.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{self.stage}/proj', loss_project.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'{self.stage}/depth', loss_depth_smooth.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        # self.log(f'{self.stage}/norm', loss_norm.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        loss_all = loss_gaussian + loss_project + loss_depth_smooth # + loss_norm
        psnr, ssim, lpips = self.compute_reconstruction_metrics(batch_data)

        for frame_id in batch_data['render_frames']:
            depth_list = []
            for cam_id in range(self.num_cams):
                depth_list.append(batch_data[('gaussian_depth', frame_id, cam_id)].detach())
            depth_list = torch.cat(depth_list,dim=1)
            min_depth = torch.amin(depth_list).cpu().item()
            mean_depth = torch.mean(depth_list).cpu().item()
            max_depth = torch.amax(depth_list).cpu().item()
            print_log = f" {frame_id}"
            print_log += f" | min_depth: {min_depth:.4f} | mean_depth: {mean_depth:.4f} | max_depth: {max_depth:.4f}"
            self.print_fn(print_log)

        print_log = ''
        print_log += f"[{self.stage}] {batch_idx} | loss_all: {loss_all.item():.4f} | loss_gaussian: {loss_gaussian.item():.4f} " 
        print_log += f"| loss_proj_smooth: {loss_project.item():.4f} | loss_depth_smooth: {loss_depth_smooth.item():.4f} " #| loss_norm: {loss_norm.item():.4f} "
        print_log += f"| psnr: {psnr:.4f} | ssim: {ssim:.4f} | lpips: {lpips:.4f}"
        self.print_fn(print_log)

        if return_imgs and (self.stage=='train'):
            splating_file = f'{self.stage}_{batch_idx}_splating'
            self._save_splating_images(splating_file,batch_data)
            project_file = f'{self.stage}_{batch_idx}_project'
            self._save_projection_images(project_file,batch_data)

        del batch_input, batch_data, psnr, ssim, lpips

        return loss_all

    def test_step(self, batch_input, batch_idx):
        self.stage = 'test'

        # self._log_weights_and_grads(batch_input)

        all_valid_frames = batch_input['frame_context'][0].cpu().tolist()
        gtview_frame_ids, nvview_frame_ids = self.split_input_views(all_valid_frames,[0,1])

        batch_data = self.predict_step(batch_input,recontrast_frames=gtview_frame_ids,render_frames=gtview_frame_ids)

        loss_depth_smooth = self.compute_smooth_loss(batch_data)
        # loss_norm = self.compute_norm_loss(batch_recontrast_data)
        loss_gaussian = self.compute_gaussian_loss(batch_data)

        if batch_idx%self.save_image_duration==0:
            return_imgs = True
        else:
            return_imgs = False
        loss_project = self.compute_project_loss(batch_data, return_imgs=return_imgs)

        self.log(f'{self.stage}/gs', loss_gaussian.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{self.stage}/proj', loss_project.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'{self.stage}/depth', loss_depth_smooth.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        # self.log(f'{self.stage}/norm', loss_norm.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        loss_all = loss_gaussian + loss_project + loss_depth_smooth # + loss_norm
        psnr, ssim, lpips = self.compute_reconstruction_metrics(batch_data)

        for frame_id in batch_data['render_frames']:
            depth_list = []
            for cam_id in range(self.num_cams):
                depth_list.append(batch_data[('gaussian_depth', frame_id, cam_id)].detach())
            depth_list = torch.cat(depth_list,dim=1)
            min_depth = torch.amin(depth_list).cpu().item()
            mean_depth = torch.mean(depth_list).cpu().item()
            max_depth = torch.amax(depth_list).cpu().item()
            print_log = f" {frame_id}"
            print_log += f" | min_depth: {min_depth:.4f} | mean_depth: {mean_depth:.4f} | max_depth: {max_depth:.4f}"
            self.print_fn(print_log)

        print_log = ''
        print_log += f"[{self.stage}] {batch_idx} | loss_all: {loss_all.item():.4f} | loss_gaussian: {loss_gaussian.item():.4f} " 
        print_log += f"| loss_proj_smooth: {loss_project.item():.4f} | loss_depth_smooth: {loss_depth_smooth.item():.4f} " #| loss_norm: {loss_norm.item():.4f} "
        print_log += f"| psnr: {psnr:.4f} | ssim: {ssim:.4f} | lpips: {lpips:.4f}"
        self.print_fn(print_log)

        if return_imgs and (self.stage=='train'):
            splating_file = f'{self.stage}_{batch_idx}_splating'
            self._save_splating_images(splating_file,batch_data)
            project_file = f'{self.stage}_{batch_idx}_project'
            self._save_projection_images(project_file,batch_data)

        del batch_input, batch_data, psnr, ssim, lpips

        return loss_all
    

    def refine_camera_position(self, batch_data):
        all_valid_frames = batch_data['frame_context'][0].cpu().tolist()
        all_need_frames = batch_data['recontrast_frames'] + batch_data['novel_frames']
        
        for frame_id in all_valid_frames:
            if frame_id in all_need_frames:
                cam_T_cam = batch_data[('cam_T_cam', 0, frame_id)]
                init_c2e_extr = batch_data['c2e_extr']
                with torch.amp.autocast("cuda", enabled=False):
                    refine_c2e_extr = torch.matmul(init_c2e_extr, torch.linalg.inv(cam_T_cam))
                    refine_e2c_extr = torch.linalg.inv(refine_c2e_extr)

                batch_data[('c2e_extr', frame_id)] = refine_c2e_extr
                batch_data[('e2c_extr', frame_id)] = refine_e2c_extr
            else:
                del batch_data[('cam_T_cam', 0, frame_id)]
                del batch_data[('color_aug', frame_id)]
                del batch_data[('color_org', frame_id)]
        
        del batch_data['c2e_extr']

    def get_recontrast_data_gtview(self, batch_data):
        """
        This function computes recontrast data for each viewpoint.
        """

        context_images = []
        context_e2c_extrs = []
        context_ks = []

        for frame_id in batch_data['recontrast_frames']:
            context_ks.append(batch_data['K'])
            context_images.append(batch_data[(f'color_aug', frame_id)])
            context_e2c_extrs.append(batch_data[('e2c_extr', frame_id)])

        context_images = torch.cat(context_images,dim=1)
        context_e2c_extrs = torch.cat(context_e2c_extrs,dim=1)
        context_ks = torch.cat(context_ks,dim=1)
 
        dpt_feat, xyz_maps, rot_maps, scale_maps, opacity_maps, sh_maps, xyz_conf = self.model(context_images,(context_ks,context_e2c_extrs))

        batch_data['context_dpt_feat'] = dpt_feat
        batch_data['context_pts'] = xyz_maps
        for index,frame_id in enumerate(batch_data['recontrast_frames']):
            for cam_id in range(self.num_cams):
                batch_data[('xyz',frame_id,cam_id)] = xyz_maps[:,index,cam_id]
                batch_data[('rot',frame_id,cam_id)] = rot_maps[:,index,cam_id]
                batch_data[('scale',frame_id,cam_id)] = scale_maps[:,index,cam_id]
                batch_data[('opacity',frame_id,cam_id)] = opacity_maps[:,index,cam_id]
                batch_data[('sh',frame_id,cam_id)] = sh_maps[:,index,cam_id]
                batch_data[('xyz_conf',frame_id,cam_id)] = xyz_conf[:,index,cam_id]

    def get_recontrast_data_nvview(self, batch_data):
        context_dpt_feat = batch_data['context_dpt_feat']
        context_pts = batch_data['context_pts']
        novel_frames = batch_data['novel_frames']

        context_frames_tensor = torch.tensor(batch_data['recontrast_frames'],dtype=context_dpt_feat.dtype,device=self.device)
        batch_size = context_dpt_feat.shape[0]

        novel_ks = []
        novel_e2c_extrs = []
        for frame_id in novel_frames:
            novel_ks.append(batch_data['K'])
            novel_e2c_extrs.append(batch_data[('e2c_extr', frame_id)])

        novel_e2c_extrs = torch.stack(novel_e2c_extrs,dim=1)
        novel_ks = torch.stack(novel_ks,dim=1)

        novel_frames_tensor = torch.tensor(novel_frames,dtype=context_dpt_feat.dtype,device=self.device)
        relative_frames = context_frames_tensor.unsqueeze(0) - novel_frames_tensor.unsqueeze(1) 
        relative_frames = relative_frames / self.frame_sfreq
        relative_frames = relative_frames.unsqueeze(0).expand(batch_size,-1,-1)

        xyz_maps, rot_maps, scale_maps, opacity_maps, sh_maps, xyz_conf = \
            self.model.refine_dpt_feat_with_render_infos(context_dpt_feat, context_pts,novel_ks,novel_e2c_extrs,relative_frames)
        
        for index,frame_id in enumerate(novel_frames):
            for cam_id in range(self.num_cams):
                batch_data[('xyz',frame_id,cam_id)] = xyz_maps[:,index,cam_id]
                batch_data[('rot',frame_id,cam_id)] = rot_maps[:,index,cam_id]
                batch_data[('scale',frame_id,cam_id)] = scale_maps[:,index,cam_id]
                batch_data[('opacity',frame_id,cam_id)] = opacity_maps[:,index,cam_id]
                batch_data[('sh',frame_id,cam_id)] = sh_maps[:,index,cam_id]
                batch_data[('xyz_conf',frame_id,cam_id)] = xyz_conf[:,index,cam_id]
        
        del batch_data['context_dpt_feat']
        del batch_data['context_pts']

    def aug_render_scale(self, batch_data):
        render_prob = np.random.rand()

        if render_prob<self.render_scale_prob and self.stage=='train':
            render_scale = self.render_scale_min + np.random.rand()*(self.render_scale_max-self.render_scale_min)
            render_height, render_width = int(self.height*render_scale), int(self.width*render_scale)

            batch_data['K'][...,:2] *=  render_scale
            batch_data['K'][..., 0, 2] = (batch_data['K'][..., 0, 2] - (self.width / 2)) * render_scale + (render_width / 2)  # cx
            batch_data['K'][..., 1, 2] = (batch_data['K'][..., 1, 2] - (self.height / 2)) * render_scale + (render_height / 2)  # cy

            batch_data['redner_height'] = render_height
            batch_data['redner_width'] = render_width
            for frame_id in batch_data['render_frames']:
                for cam_id in range(self.num_cams):
                    batch_data[('groudtruth', frame_id, cam_id)] = \
                            F.interpolate(batch_data[('color_org', frame_id)][:,cam_id,...], 
                                        size=(render_height,render_width),mode = 'bilinear', align_corners=False)
                    batch_data[('xyz_conf', frame_id, cam_id)] = \
                            F.interpolate(batch_data[('xyz_conf', frame_id, cam_id)], 
                                        size=(render_height,render_width),mode = 'bilinear', align_corners=False)

                del batch_data[('color_org', frame_id)]

            mask = rearrange(batch_data['mask'],'b c i h w-> (b c) i h w')
            mask= F.interpolate(mask, size=(render_height,render_width),mode = 'bilinear', align_corners=False)
            batch_data['mask'] = rearrange(mask,'(b c) i h w-> b c i h w',c=self.num_cams)
        else:
            batch_data['redner_height'] = self.height
            batch_data['redner_width'] = self.width
            for frame_id in batch_data['render_frames']:
                for cam_id in range(self.num_cams):
                    batch_data[('groudtruth', frame_id, cam_id)] = batch_data[('color_aug', frame_id)][:,cam_id,...]

                del batch_data[('color_org', frame_id)]

        return 

    def get_splating_imgs(self, batch_data):
        all_valid_frames = batch_data['render_frames']
        if len(all_valid_frames)==0:
            return
        
        self.aug_render_scale(batch_data)

        bs = batch_data[('xyz',all_valid_frames[0],0)].shape[0]
        render_width = batch_data['redner_width']
        render_height = batch_data['redner_height']

        for frame_id in all_valid_frames:
            for cam_id in range(self.num_cams):
                gaussian_color = []
                gaussian_depth = []
                for batch_id in range(bs):
                    xyz_i = batch_data[('xyz',frame_id,cam_id)][batch_id]
                    rot_i = batch_data[('rot',frame_id,cam_id)][batch_id]
                    scale_i = batch_data[('scale',frame_id,cam_id)][batch_id]
                    opacity_i = batch_data[('opacity',frame_id,cam_id)][batch_id]
                    sh_i = batch_data[('sh',frame_id,cam_id)][batch_id]

                    e2c_extr_i = batch_data[('e2c_extr', frame_id)][batch_id:batch_id+1,cam_id]
                    K_i = batch_data['K'][batch_id:batch_id+1,cam_id,:3,:3]

                    with torch.amp.autocast("cuda", enabled=False):
                        render_colors_i, render_alphas_i, meta_i = rasterization(
                            xyz_i.to(torch.float32),  # [N, 3]
                            rot_i.to(torch.float32),  # [N, 4]
                            scale_i.to(torch.float32),  # [N, 3]
                            opacity_i.squeeze(-1).to(torch.float32),  # [N]
                            sh_i.to(torch.float32),  # [N, K, 3]
                            e2c_extr_i.to(torch.float32),  # [1, 4, 4]
                            K_i.to(torch.float32),  # [1, 3, 3]
                            render_width,
                            render_height,
                            sh_degree=self.sh_degree,
                            render_mode="RGB+D",
                            # sparse_grad=True,
                            # this is to speedup large-scale rendering by skipping far-away Gaussians.
                            # radius_clip=3,
                        )

                    render_rgb_i, render_depth_i = render_colors_i[0,:,:,:3].permute(2,0,1), render_colors_i[0,:,:,3]

                    gaussian_color.append(render_rgb_i)
                    gaussian_depth.append(render_depth_i)
                    
                    # 显式清理中间变量
                    del xyz_i, rot_i, scale_i, opacity_i, sh_i, e2c_extr_i, K_i
                    del render_colors_i, render_alphas_i, meta_i, render_rgb_i, render_depth_i

                gaussian_color = torch.stack(gaussian_color,dim=0)
                gaussian_depth = torch.stack(gaussian_depth,dim=0).unsqueeze(1)

                batch_data[('gaussian_color', frame_id, cam_id)] = gaussian_color
                batch_data[('gaussian_depth', frame_id, cam_id)] = gaussian_depth

                del gaussian_color, gaussian_depth

    
    def get_virtual_image(self, src_img, src_mask, tar_depth, tar_invK, src_K, T):
        """
        This function warps source image to target image using backprojection and reprojection process. 
        """
        # do reconstruction for target from source   
        # pix_coords = self.project(tar_depth, T, tar_invK, src_K)
        batch_size, chanel, height, width = tar_depth.shape
        img_points = np.meshgrid(range(width), range(height), indexing='xy')
        img_points = torch.from_numpy(np.stack(img_points, 0)).float()
        img_points = torch.stack([img_points[0].view(-1), img_points[1].view(-1)], 0).repeat(batch_size, 1, 1)
        img_points = img_points.to(tar_depth.device)
        
        to_homo = torch.ones([batch_size, 1, width*height],device=tar_depth.device)
        homo_points = torch.cat([img_points, to_homo], 1)

        depth = rearrange(tar_depth, 'b c h w -> b c (h w)')
        points3D = torch.matmul(tar_invK[:, :3, :3], homo_points)
        points3D = depth*points3D
        points3D =  torch.cat([points3D, to_homo], 1)

        points2D = (src_K @ T)[:,:3, :] @ points3D

        # normalize projected points for grid sample function
        norm_points2D = points2D[:, :2, :]/(points2D[:, 2:, :].clamp(min=EPSILON))
        bs = norm_points2D.shape[0]

        norm_points2D = norm_points2D.view(bs, 2, height, width)
        
        norm_points2D = norm_points2D.permute(0, 2, 3, 1)

        norm_points2D[..., 0 ] /= width - 1
        norm_points2D[..., 1 ] /= height - 1
        pix_coords = (norm_points2D-0.5)*2

        img_warped = F.grid_sample(src_img, pix_coords, mode='bilinear', padding_mode='zeros', align_corners=True)
        mask_warped = F.grid_sample(src_mask, pix_coords, mode='nearest', padding_mode='zeros', align_corners=True)

        # nan handling
        inf_img_regions = torch.isnan(img_warped)
        img_warped[inf_img_regions] = 2.0
        inf_mask_regions = torch.isnan(mask_warped)
        mask_warped[inf_mask_regions] = 0

        pix_coords = pix_coords.permute(0, 3, 1, 2)
        invalid_mask = torch.logical_or(pix_coords > 1, 
                                        pix_coords < -1).sum(dim=1, keepdim=True) > 0
        return img_warped, (~invalid_mask).float() * mask_warped
    
    def get_norm_image_single(self, src_img, src_mask, warp_img, warp_mask):
        """
        obtain normalized warped images using the mean and the variance from the overlapped regions of the target frame.
        """
        warp_mask = warp_mask.detach()

        with torch.no_grad():
            mask = (src_mask * warp_mask).bool()
            if mask.size(1) != 3:
                mask = mask.repeat(1,3,1,1)

            mask_sum = mask.sum(dim=(-3,-2,-1))
            # skip when there is no overlap
            if torch.any(mask_sum == 0):
                return warp_img

            s_mean, s_std = self.get_mean_std(src_img, mask)
            w_mean, w_std = self.get_mean_std(warp_img, mask)

        norm_warp = (warp_img - w_mean) / (w_std.clamp(min=EPSILON)) * s_std + s_mean
        return norm_warp * warp_mask.float()   

    def get_mean_std(self, feature, mask):
        """
        This function returns mean and standard deviation of the overlapped features. 
        """
        _, c, h, w = mask.size()
        mean = (feature * mask).sum(dim=(1,2,3), keepdim=True) / (mask.sum(dim=(1,2,3), keepdim=True).clamp(min=EPSILON))
        var = ((feature - mean) ** 2).sum(dim=(1,2,3), keepdim=True) / (c*h*w)
        return mean, torch.sqrt(var.clamp(min=EPSILON*EPSILON))     
    
    @rank_zero_only
    def _save_projection_images(self, save_name, batch_data, bs_id=0):
        if os.path.exists(os.path.join(self.save_dir, save_name)):
            shutil.rmtree(os.path.join(self.save_dir, save_name))
        os.makedirs(os.path.join(self.save_dir, save_name))
        
        for src_frame_id, tag_frame_id in batch_data['project_index_pairs']:
            for cam_id in range(self.num_cams): 
                pred = batch_data[('project_pred', src_frame_id, tag_frame_id, cam_id)].detach()
                gt = batch_data[('project_gt', tag_frame_id, cam_id)].detach()
                mask = batch_data[('project_mask', src_frame_id, tag_frame_id,cam_id)].detach()
                self.save_image(pred[bs_id], os.path.join(self.save_dir, save_name, f"{src_frame_id}_{tag_frame_id}_{cam_id}_pred.png"))
                self.save_image(gt[bs_id], os.path.join(self.save_dir, save_name, f"{src_frame_id}_{tag_frame_id}_{cam_id}_gt.png"))
                self.save_image(mask[bs_id], os.path.join(self.save_dir, save_name, f"{src_frame_id}_{tag_frame_id}_{cam_id}_mask.png"))

    @rank_zero_only
    def _save_splating_images(self, save_name, batch_data, bs_id=0):
        if os.path.exists(os.path.join(self.save_dir, save_name)):
            shutil.rmtree(os.path.join(self.save_dir, save_name))
        os.makedirs(os.path.join(self.save_dir, save_name))
        for frame_id in batch_data['render_frames']:
            for cam_id in range(self.num_cams): 
                pred_rgb = batch_data[('gaussian_color', frame_id, cam_id)].detach()
                pred_depth = batch_data[('gaussian_depth', frame_id, cam_id)].detach()
                pred_depth = torch.log(pred_depth)
                max_depth = log(self.max_depth)
                min_depth = log(self.min_depth)
                pred_depth = 1 - (pred_depth - min_depth) / (max_depth - min_depth)
                gt = batch_data[('groudtruth', frame_id, cam_id)]
                render_params =  ''  
                self.save_image(pred_rgb[bs_id], os.path.join(self.save_dir, save_name, f"{frame_id}_{render_params}_{cam_id}_prgb.png"))
                self.save_image(pred_depth[bs_id], os.path.join(self.save_dir, save_name, f"{frame_id}_{render_params}_{cam_id}_pdepth.png"))
                self.save_image(gt[bs_id], os.path.join(self.save_dir, save_name, f"{frame_id}_{render_params}_{cam_id}_gt.png")) 

    @rank_zero_only
    def save_image(self, image, path):
        """Save an image. Assumed to be in range 0-1."""

        # Create the parent directory if it doesn't already exist.
        # os.makedirs(os.path.dirname(path),exist_ok=True)

        image = image.detach().cpu().to(torch.float).numpy().clip(min=0, max=1)
        image_uint8 = (image * 255).astype(np.uint8)

        if len(image_uint8.shape)==2:
            pass
        elif (len(image_uint8.shape)==3) and (image_uint8.shape[0] == 1):
            image_uint8 = image_uint8[0]
        else:
            assert len(image_uint8.shape)==3, image_uint8.shape
            assert image_uint8.shape[0] == 3, image_uint8.shape
            image_uint8 = image_uint8.transpose(1, 2, 0)
        # Save the image.
        Image.fromarray(image_uint8).save(path)
        torch.cuda.empty_cache()
    
    def compute_gaussian_loss(self, batch_data):
        """
        This function computes gaussian loss.
        """
        # self occlusion mask * overlap region mask

        gaussian_loss = 0.0 
        count = 0.0

        for frame_id in batch_data['render_frames']:
            for cam_id in range(self.num_cams): 
                pred = batch_data[('gaussian_color', frame_id, cam_id)]
                gt = batch_data[('groudtruth', frame_id, cam_id)]  
                # mask = batch_data[('gaussian_mask', frame_id, cam_id)]

                lpips_loss = self.lpips(pred, gt, normalize=True)
                # lpips_loss = 0.0
                # l2_loss = ((pred - gt)**2)
                l1_loss = self.l1_fn(pred, gt)
                ssim_loss = self.ssim_fn(pred, gt)
                sum_loss = 0.6 * l1_loss + 0.2 * ssim_loss + 0.2 * lpips_loss
                gaussian_loss += sum_loss.mean()
                count += 1
        return self.lambda_gaussian * gaussian_loss / count

    def compute_project_loss(self, batch_data, return_imgs=False):
        project_loss = 0.0 
        count = 0.0

        batch_data['project_index_pairs'] = []
        for tag_frame_id in batch_data['render_frames']:

            for cam_id in range(self.num_cams):
                tag_colors = batch_data[('groudtruth', tag_frame_id, cam_id)]
                tag_depth = batch_data[('gaussian_depth', tag_frame_id, cam_id)]
                tag_mask = batch_data['mask'][:,cam_id]
                tag_depth_conf = batch_data[('xyz_conf', tag_frame_id, cam_id)]

                ref_T_tag = batch_data[('cam_T_cam', 0, tag_frame_id)][:,cam_id,:]
                tag_K = batch_data['K'][:,cam_id]
                with torch.amp.autocast("cuda", enabled=False):
                    tag_inv_K = torch.linalg.inv(tag_K)

                # outputs[('src_mask', recontrast_frame_ids, cam_id)] = src_mask
                for src_frame_id in batch_data['render_frames']:

                    if tag_frame_id == src_frame_id:
                        continue
                    src_colors = batch_data[('groudtruth', src_frame_id, cam_id)]

                    src_mask = batch_data['mask'][:,cam_id]

                    src_K = batch_data['K'][:,cam_id]

                    ref_T_src = batch_data[('cam_T_cam', 0, src_frame_id)][:,cam_id,:]
                    with torch.amp.autocast("cuda", enabled=False):
                        inv_ref_T_tag = torch.linalg.inv(ref_T_tag)
                    tag_T_src = torch.matmul(ref_T_src, inv_ref_T_tag)

                    warped_img, warped_mask = self.get_virtual_image(
                                    src_colors, 
                                    src_mask, 
                                    tag_depth, 
                                    tag_inv_K, 
                                    src_K, 
                                    tag_T_src
                                )

                    warped_img = self.get_norm_image_single(
                        tag_colors, 
                        tag_mask,
                        warped_img, 
                        warped_mask
                    )

                    warped_mask = warped_mask * tag_depth_conf
                    batch_data['project_index_pairs'].append((src_frame_id,tag_frame_id))
                    if return_imgs:
                        batch_data[('project_gt',tag_frame_id,cam_id)] = tag_colors.detach().cpu()
                        batch_data[('project_pred',src_frame_id,tag_frame_id,cam_id)] = warped_img.detach().cpu()
                        batch_data[('project_mask',src_frame_id,tag_frame_id,cam_id)] = warped_mask.detach().cpu()

                    l1_loss = self.l1_fn(warped_img, tag_colors)
                    ssim_loss = self.ssim_fn(warped_img, tag_colors)
                    lpips_loss = self.lpips(warped_img, tag_colors, normalize=True)
                    sum_loss = 0.6 * l1_loss + 0.2 * ssim_loss + 0.2 * lpips_loss

                    # 关键：置信度应与重投影误差负相关
                    # 希望：高误差区域置信度低，低误差区域置信度高
                    conf_target = torch.exp(-5 * sum_loss).mean(dim=1, keepdim=True)
                    conf_loss = F.mse_loss(tag_depth_conf, conf_target.detach())

                    min_supervision_ratio = 0.7  # 至少70%像素用于监督
                    coverage_loss = F.relu(min_supervision_ratio - tag_depth_conf.mean())

                    sum_loss = compute_masked_loss(sum_loss, warped_mask.detach(), eps=EPSILON)
                    # self.print_fn(f'sum_loss{sum_loss.item():.5f},converage_loss{coverage_loss.item():.5f},conf_loss{conf_loss.item():.5f}')
                    project_loss += sum_loss + coverage_loss + conf_loss
                    
                    # 显式清理中间变量
                    del src_colors,src_mask,warped_img,warped_mask,l1_loss,ssim_loss,sum_loss
                    del ref_T_src,inv_ref_T_tag,tag_T_src
                count += 1
                
                # 显式清理循环外变量
                del tag_colors,tag_depth,tag_mask,ref_T_tag,tag_K,tag_inv_K

        return self.lambda_project * project_loss / count

    def compute_smooth_loss(self, batch_data):
        """
        This function computes edge-aware smoothness loss for the disparity map.
        """
        depth_loss = 0.0 
        count = 0.0
        for frame_id in batch_data['render_frames']:
            for cam_id in range(self.num_cams):
                ref_colors = batch_data[('groudtruth', frame_id, cam_id)]

                ref_depth = batch_data[('gaussian_depth', frame_id, cam_id)]

                mean_depth = ref_depth.mean(2, True).mean(3, True)
                norm_depth = ref_depth / (mean_depth.clamp(min=EPSILON))
                edge_loss = compute_edg_smooth_loss(ref_colors, norm_depth)
                depth_loss += self.lambda_edge * edge_loss

                count += 1

        return  depth_loss / count
    def compute_norm_loss(self, batch_data):
        # 不透明度稀疏性正则化: 鼓励不透明度趋于0或1
        # opacity_loss = self.lambda_opacity * (0.5 - torch.mean(torch.abs(batch_data['opacity_maps']-0.5)))
        opacity_loss = self.lambda_opacity * torch.mean(torch.abs(batch_data['opacity']))
        scale_loss = self.lambda_scale * torch.mean(torch.norm(batch_data['scale'], dim=-1)) 
        # 总正则化损失
        total_reg_loss = scale_loss + opacity_loss

        return total_reg_loss

    @torch.no_grad()
    def compute_reconstruction_metrics(self, batch_data):
        """
        This function computes reconstruction metrics.
        """
        psnr = 0.0
        ssim = 0.0
        lpips = 0.0

        novel_count =0
        # frame_id = self.render_frame_ids
        for frame_id in batch_data['render_frames']:
            for cam_id in range(self.num_cams): 
                pred = batch_data[('gaussian_color', frame_id, cam_id)].detach()
                gt = batch_data[('groudtruth', frame_id, cam_id)]    
                psnr += self.compute_psnr(gt, pred).mean()
                ssim += self.compute_ssim(gt, pred).mean()
                lpips += self.compute_lpips(gt, pred).mean()
                novel_count += 1

        psnr /= novel_count
        ssim /= novel_count
        lpips /= novel_count

        self.log(f"{self.stage}/psnr", psnr.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{self.stage}/ssim", ssim.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{self.stage}/lpips", lpips.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return psnr, ssim, lpips
    
    @torch.no_grad()
    def compute_psnr(
        self,
        ground_truth: Float[Tensor, "batch channel height width"],
        predicted: Float[Tensor, "batch channel height width"],
    ) -> Float[Tensor, " batch"]:
        ground_truth = ground_truth.clip(min=0, max=1)
        predicted = predicted.clip(min=0, max=1)
        mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
        return -10 * mse.log10()
    
    @torch.no_grad()
    def compute_lpips(
        self,
        ground_truth: Float[Tensor, "batch channel height width"],
        predicted: Float[Tensor, "batch channel height width"],
    ) -> Float[Tensor, " batch"]:
        value = self.lpips.forward(ground_truth, predicted, normalize=True)
        return value[:, 0, 0, 0]
    
    @torch.no_grad()
    def compute_ssim(
        self,
        ground_truth: Float[Tensor, "batch channel height width"],
        predicted: Float[Tensor, "batch channel height width"],
    ) -> Float[Tensor, " batch"]:
        ssim = [
            structural_similarity(
                gt.detach().cpu().numpy(),
                hat.detach().cpu().numpy(),
                win_size=11,
                gaussian_weights=True,
                channel_axis=0,
                data_range=1.0,
            )
            for gt, hat in zip(ground_truth, predicted)
        ]
        return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)
    


if __name__=='__main__':

    import PIL.Image as pil
    import yaml
    import torch
    import numpy as np


    import math

    config_file = 'configs/nuscenes/vggt4dgs.yaml'
    with open(config_file) as f:
        main_cfg = yaml.load(f, Loader=yaml.FullLoader)
    main_cfg['data_cfg']['batch_size'] = 1
    main_cfg['model_cfg']['batch_size'] = main_cfg['data_cfg']['batch_size']

    pl.seed_everything(main_cfg['seed'], workers=True)

    litmodel = VGGT4DGS_LITModelModule(main_cfg['model_cfg']).to('cuda:0')

    restore_ckpt = os.environ.get('VGGT4DGS_RESTORE_CKPT', '')
    if not restore_ckpt:
        raise ValueError('Set VGGT4DGS_RESTORE_CKPT before running this debug entry point.')
    stage1_ckpt = torch.load(restore_ckpt,map_location=f"cuda:0")

    # stage2_ckpt = os.environ.get('VGGT4DGS_STAGE2_CKPT')
    # stage2_ckpt = torch.load(stage2_ckpt,map_location=f"cuda:0")

    # for param_key in list(stage1_ckpt['state_dict'].keys()):
    #     if 'model.dpt_feat_refine' in param_key:
    #         print(param_key)
    #         stage1_ckpt['state_dict'][param_key] = stage2_ckpt['state_dict'][param_key]
    #         # del load_ckpt['state_dict'][param_key]

    litmodel.load_state_dict(stage1_ckpt['state_dict'],strict=False) # 



    # for param_key in list(load_ckpt['state_dict'].keys()):
    #     if 'model.depth_feat2map' in param_key:
    #         print(param_key)
    #         del load_ckpt['state_dict'][param_key]



    # del litmodel.model.dpt_feat_proj2d
    # del litmodel.model.dpt_feat_refine

    # new_sh_degree = litmodel.sh_degree  # 确保使用相同的sh_degree
    # new_d_sh = (new_sh_degree + 1) ** 2
    # # 创建新的 sh_mask（修改衰减系数：0.25 -> 0.5）
    # new_sh_mask = torch.ones((new_d_sh,), dtype=torch.float32)
    # for degree in range(1, new_sh_degree + 1):
    #     new_sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.5**degree  # 衰减更平缓
    # litmodel.model.sh_mask = new_sh_mask.to(litmodel.device)
    # torch.save({"state_dict":litmodel.state_dict()},'restore_stage1-v3.ckpt')

    batch_input = torch.load('batch_input.ckpt',map_location=f"cuda:0")
    # print(batch_input['mask'].shape)
    for key, val in batch_input.items():
        if isinstance(val, torch.Tensor):
            repeat_list = [2]+[1]*(val.dim()-1)
            batch_input[key] = val.repeat(repeat_list)
    # with torch.amp.autocast("cuda", enabled=True):
    output = litmodel.debug_step(batch_input,0)
    print(output)
    
