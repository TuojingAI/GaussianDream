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
from math import log2, log
import sys
from gsplat.rendering import rasterization

# from models.vggt3dgs_model2 import VGGT3DGSModel
from models.vggt3dgs_model import VGGT3DGSModel
# from models.vggt4dgs_bevmodel import VGGT3DGSBEVModel

from models.gaussian_util import render, focal2fov, getProjectionMatrix,  depth2pc, rotate_sh

from models.loss_util import compute_masked_loss, compute_edg_smooth_loss

from models.geometry_util import Projection

def print_memory(msg):
    # torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"[MEM] {msg}: Alloc={allocated:.2f}GB, Reserved={reserved:.2f}GB")

class VGGT3DGS_LITModelModule(pl.LightningModule):
    def __init__(self, cfg, save_dir='.', logger=None):
        super().__init__()
        self.read_config(cfg)

        self.save_dir = save_dir
        self.model = VGGT3DGSModel(sh_degree=self.sh_degree,min_depth=self.min_depth, max_depth=self.max_depth) 
        self.lpips = LPIPS(net="vgg")
        self.ssim_fn = SSIMLoss(window_size=11,reduction='none')
        self.l1_fn = torch.nn.L1Loss(reduction='none')
        self.lpips.eval()
        self.project = Projection(self.batch_size, self.height, self.width)
        self.init_novel_view_mode()
        self.save_hyperparameters('cfg','save_dir')
    
    def init_novel_view_mode(self, recontrast_ids=0, render_ids=0):
        self.recontrast_frame_ids = recontrast_ids
        render_frame_id = render_ids

        render_scale = 1.0
        render_shift_x = 0.0
        render_shift_y = 0.0
        return {'render_frame': render_frame_id,'render_scale': render_scale,
                'render_shift_x': render_shift_x,'render_shift_y': render_shift_y}
    
    def aug_novel_view_mode(self, recontrast_ids=0, aug_frame_ids=[0]):
        self.recontrast_frame_ids = recontrast_ids
        render_prob = np.random.rand()
        if render_prob<0.25:
            render_scale = self.render_scale_min + np.random.rand()*(self.render_scale_max-self.render_scale_min)
        else:
            render_scale = 1.0
        
        if render_prob>0.75:
            render_shift_x = np.random.randn()*1
            render_shift_y = np.random.randn()*0.5
        else:
            render_shift_x = 0.0
            render_shift_y = 0.0

            # self.render_cam_mode = np.random.choice(['origin','scale','shift'])
        frame_prob = np.random.rand()
        if frame_prob<0.6:
            render_frame_id = np.random.choice(aug_frame_ids).item()
        else:
            render_frame_id = recontrast_ids

        return {'render_frame': render_frame_id,'render_scale': render_scale,
                'render_shift_x': render_shift_x,'render_shift_y': render_shift_y}

    def read_config(self, cfg):    
        for k, v in cfg.items():
            setattr(self, k, v)
    @rank_zero_only
    def print_fn(self,log_msg):
        print(log_msg)

    def training_step(self, batch_input, batch_idx):
        self.stage = stage =  'train'

        self._log_weights_and_grads(batch_input)

        novel_frame_ids = self.render_frames

        batch_recontrast_data = self.get_recontrast_data(batch_input)

        loss_norm = self.compute_norm_loss(batch_recontrast_data)

        batch_render_data = {}
        for render_idx in range(self.render_nums):
            render_params = self.aug_novel_view_mode(0,novel_frame_ids)
            batch_render_data[render_idx] = self.get_render_data(inputs=batch_input,**render_params)
            self.print_fn(f'{render_idx}: {render_params}')
        batch_render_project_data = self.render_project_imgs(batch_input,batch_recontrast_data,novel_frame_ids)

        loss_project = self.compute_project_loss(batch_render_project_data,novel_frame_ids)

        loss_depth = self.compute_depth_loss(batch_render_project_data)

        batch_splating_data  = self.render_splating_imgs(batch_recontrast_data,batch_render_data)

        # print_memory(f"After render_gaussian_imgs {batch_idx}")
        loss_gaussian = self.compute_gaussian_loss(batch_splating_data,range(self.render_nums))
        self.log(f'{stage}/gs', loss_gaussian.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}/proj', loss_project.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}/depth', loss_depth.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'{stage}/norm', loss_norm.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        loss_all = loss_gaussian + loss_project + loss_depth + loss_norm
        psnr, ssim, lpips = self.compute_reconstruction_metrics(batch_splating_data,stage,range(self.render_nums))


        if batch_idx%self.save_image_duration==0:
            # save_name = f'{stage}_{batch_idx}'
            # if os.path.exists(os.path.join(self.save_dir, save_name)):
            #     shutil.rmtree(os.path.join(self.save_dir, save_name))
            # glb_file = os.path.join(self.save_dir, save_name , 'wordpoints.glb')
            # self._save_wordpoints_glb(glb_file, batch_input, batch_recontrast_data, batch_render_data)
            reprojection_file = f'{stage}_{batch_idx}_reprojection'
            self._save_reprojection_images(reprojection_file, batch_render_project_data,novel_frame_ids)
            splating_file = f'{stage}_{batch_idx}_splating'
            self._save_splating_images(splating_file,batch_splating_data)

        del batch_input, batch_recontrast_data, batch_render_data, batch_render_project_data, batch_splating_data, psnr, ssim, lpips
        # print_memory(f"After compute_reconstruction_metrics {batch_idx}")
        # gc.collect()
        # torch.cuda.empty_cache()
        return loss_all

    def predict_step(self, batch_input, render_frames=None):
        self.stage = 'predict'
        self.init_novel_view_mode(render_ids=render_frames)
        batch_recontrast_data = self.get_recontrast_data(batch_input)

        if render_frames is None:
            novel_frame_ids = [0] + self.render_frames
        else:
            novel_frame_ids = render_frames

        batch_render_data = {}
        for render_idx, render_frame in enumerate(novel_frame_ids):
            render_params = self.init_novel_view_mode(0,render_frame)
            batch_render_data[render_frame] = self.get_render_data(inputs=batch_input,**render_params)

        # batch_render_project_data = self.render_project_imgs(batch_input,batch_recontrast_data)

        batch_splating_data  = self.render_splating_imgs(batch_recontrast_data,batch_render_data)

        # psnr, ssim, lpips = self.compute_reconstruction_metrics(batch_splating_data,self.stage)

        # print(psnr, ssim, lpips)

        return batch_recontrast_data, batch_render_data, batch_splating_data
    
    def validation_step(self, batch_input, batch_idx):
        self.stage = stage = 'val'

        novel_frame_ids = [0] + self.render_frames

        batch_recontrast_data = self.get_recontrast_data(batch_input)

        batch_render_data = {}

        for render_idx, render_frame in enumerate(novel_frame_ids):
            render_params = self.init_novel_view_mode(0,render_frame)
            batch_render_data[render_idx] = self.get_render_data(inputs=batch_input,**render_params)

        loss_norm = self.compute_norm_loss(batch_recontrast_data)
        batch_render_project_data = self.render_project_imgs(batch_input,batch_recontrast_data,novel_frame_ids)

        loss_project = self.compute_project_loss(batch_render_project_data,novel_frame_ids)

        loss_depth = self.compute_depth_loss(batch_render_project_data)

        batch_splating_data  = self.render_splating_imgs(batch_recontrast_data,batch_render_data)

        # print_memory(f"After render_gaussian_imgs {batch_idx}")
        loss_gaussian = self.compute_gaussian_loss(batch_splating_data,range(len(novel_frame_ids)))
        self.log(f'{stage}/gs', loss_gaussian.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}/proj', loss_project.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}/depth', loss_depth.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'{stage}/norm', loss_norm.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        loss_all = loss_gaussian + loss_project + loss_depth + loss_norm
        psnr, ssim, lpips = self.compute_reconstruction_metrics(batch_splating_data,stage,range(len(novel_frame_ids)))
        # print(loss_gaussian,loss_project,loss_depth,loss_norm)
        # print(psnr, ssim, lpips)
        del batch_input,batch_recontrast_data, batch_render_data, batch_render_project_data, batch_splating_data, psnr, ssim, lpips
        # print_memory(f"After compute_reconstruction_metrics {batch_idx}")
        # gc.collect()
        # torch.cuda.empty_cache()
        return loss_all

    def test_step(self, batch_input, batch_idx):
        self.stage = stage = 'test'

        novel_frame_ids = [0] + self.render_frames

        batch_recontrast_data = self.get_recontrast_data(batch_input)

        batch_render_data = {}
        for render_idx, render_frame in enumerate(novel_frame_ids):
            render_params = self.init_novel_view_mode(0,render_frame)
            batch_render_data[render_idx] = self.get_render_data(inputs=batch_input,**render_params)

        loss_norm = self.compute_norm_loss(batch_recontrast_data)
        batch_render_project_data = self.render_project_imgs(batch_input,batch_recontrast_data,novel_frame_ids)

        loss_project = self.compute_project_loss(batch_render_project_data,novel_frame_ids)
        loss_depth = self.compute_depth_loss(batch_render_project_data)


        batch_splating_data  = self.render_splating_imgs(batch_recontrast_data,batch_render_data)

        # print_memory(f"After render_gaussian_imgs {batch_idx}")
        loss_gaussian = self.compute_gaussian_loss(batch_splating_data,range(len(novel_frame_ids)))
        self.log(f'{stage}/gs', loss_gaussian.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}/proj', loss_project.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}/depth', loss_depth.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'{stage}/norm', loss_norm.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        loss_all = loss_gaussian + loss_project + loss_depth + loss_norm
        psnr, ssim, lpips = self.compute_reconstruction_metrics(batch_splating_data,stage,range(len(novel_frame_ids)))

        del batch_input,batch_recontrast_data, batch_render_data, batch_render_project_data, batch_splating_data, psnr, ssim, lpips
        # print_memory(f"After compute_reconstruction_metrics {batch_idx}")
        # gc.collect()
        # torch.cuda.empty_cache()
        return loss_all

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
            print('nan_prams: ',nan_params)
            print('inf_prams: ',inf_params)
            
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
            print('nan_prams: ',nan_params)
            print('inf_prams: ',inf_params)
            
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

        
        # 打印关键信息
        # print(f"\nStep {current_step}:")
        # print(f"Max Weight: {max_weight_name} = {max_weight:.6e}")
        # print(f"Max Grad:   {max_grad_name} = {max_grad:.6e}")
        
        # 检测并报告NaN
        if len(nan_params)>0 or len(inf_params)>0:
            print('nan_prams: ',nan_params)
            print('inf_prams: ',inf_params)
            
            sys.exit(-1)
    
    def on_before_optimizer_step(self, optimizer):
        """在优化器步骤前检查梯度"""
        valid_gradients = True
        
        for name, param in self.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"NaN 梯度: {name}")
                    valid_gradients = False
                if torch.isinf(param.grad).any():
                    print(f"Inf 梯度: {name}")
                    valid_gradients = False
        
        if not valid_gradients:
            print("检测到无效梯度，跳过本次更新")
            # 清零梯度并跳过步骤
            optimizer.zero_grad()
            return False
        return True
                        
    def configure_optimizers(self):
        # parameters_to_train = []
        # for name, parameters in self.model.named_parameters():
        #     train_flag = True
        #     for freeze_name in self.freeze_patterns:
        #         if freeze_name in name:
        #             parameters.requires_grad = False
        #             train_flag = False
        #     if train_flag:
        #         print(f'Training {name}')
        #         parameters_to_train.append(parameters)

        # optimizer = optim.AdamW(parameters_to_train, lr=self.learning_rate)
        # optimizer = torch.optim.SGD(parameters_to_train, lr=self.learning_rate,momentum=self.momentum,weight_decay=self.weight_decay)
        for name, parameters in self.model.named_parameters():
            if parameters.requires_grad:
                print(f'Training {name}')

        if self.auto_scale_lr:
            num_devices = self.trainer.num_devices
            scale_devices = max(1, log2(num_devices))  # 至少按1计算
            base_lr = self.learning_rate *scale_devices
        else:
            base_lr = self.learning_rate

        # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=base_lr,momentum=self.momentum,weight_decay=self.weight_decay)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=base_lr,betas=(0.9,0.98),eps=1e-7,weight_decay=self.weight_decay)

        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=self.scheduler_step_size,gamma=self.scheduler_gamma)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, 
                    T_0=self.lr_restart_epoch,  # 每5个epoch重启
                    T_mult=self.lr_restart_mult,
                    eta_min=base_lr*self.lr_min_factor
                )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }
    
    def get_recontrast_data(self, inputs, recontrast_frame_ids=0):
        """
        This function computes recontrast data for each viewpoint.
        """

        outputs = {}

        image_list = []
        c2e_extr_list = []

        frame_id = recontrast_frame_ids
        for cam_id in range(self.num_cams):
            if frame_id==0:
                c2e_extr = inputs['c2e_extr'][:, cam_id, ...]
            else:
                cam_T_cam = inputs[('cam_T_cam', 0, frame_id)][:, cam_id, ...]
                c2e_extr = torch.matmul(inputs['c2e_extr'][:, cam_id, ...], torch.linalg.inv(cam_T_cam))

            image_list.append(inputs[(f'color_aug', frame_id)][:,cam_id,...])
            c2e_extr_list.append(c2e_extr)
        image_list = torch.stack(image_list,dim=1)
        depth_maps, rot_maps, scale_maps, opacity_maps, sh_maps = self.model(image_list)
        del image_list 

        batch_size = depth_maps.shape[0]
        frame_camrea = self.num_cams

        c2e_extr_list = torch.stack(c2e_extr_list, dim=1)

        bfc_depth_maps = rearrange(depth_maps.squeeze(-1), 'b c h w -> (b c) h w ')
        bfc_K = rearrange(inputs['K'], 'b c i j -> (b c) i j ')
        bfc_c2e = rearrange(c2e_extr_list, 'b c i j -> (b c) i j ')
        bfc_sh = rearrange(sh_maps, 'b c h w p d -> (b c) h w p d') # height weight points d_sh

        # bfc_xyz = self._unproject_depth_map_to_points_map(bfc_depth_maps, bfc_K, bfc_c2e) 
        bf_e2c = torch.linalg.inv(bfc_c2e)
        bfc_xyz = depth2pc(bfc_depth_maps, bf_e2c, bfc_K)#.detach()

        c2w_rotations = rearrange(bfc_c2e[:, :3, :3], "b i j -> b () () () i j")

        bfc_sh = rotate_sh(bfc_sh, c2w_rotations)

        outputs['depth_maps'] = rearrange(bfc_depth_maps, '(b c) h w -> b (c h w)', b=batch_size, c=frame_camrea)#.contiguous()
        outputs['xyz'] = rearrange(bfc_xyz, '(b c) p k -> b (c p) k', b=batch_size, c=frame_camrea)#.contiguous()
        outputs['rot_maps'] = rearrange(rot_maps, 'b c h w d -> b (c h w) d', d=4)#.contiguous()

        # if self.stage == 'train' or self.stage == 'val':
        #     print('scale:',(scale_maps.amin().item(),scale_maps.mean().item(),scale_maps.amax().item()))
        #     print('opacity:',(opacity_maps.amin().item(),opacity_maps.mean().item(),opacity_maps.amax().item()))
        #     print('depth',bfc_depth_maps.amin().item(),bfc_depth_maps.mean().item(),bfc_depth_maps.amax().item())
            # training_scale_thr = self.init_scale_thr
            # scale_maps = torch.clamp(scale_maps, min=1e-6, max=training_scale_thr)
            # scale_maps = torch.clamp_min(scale_maps, min=1e-6)

        outputs['scale_maps'] = rearrange(scale_maps, 'b c h w d -> b (c h w) d', d=3)#.contiguous()
        outputs['opacity_maps'] = rearrange(opacity_maps, 'b c h w d -> b (c h w) d')#.contiguous()
        outputs['sh_maps'] = rearrange(bfc_sh, '(b c) h w p d -> b (c h w) d p', b=batch_size, c=frame_camrea)#.contiguous()
        del bfc_K, bfc_c2e, bfc_depth_maps, bfc_xyz, rot_maps, scale_maps, opacity_maps, bfc_sh,
        return outputs    

    # def get_render_data(self, inputs):

    #     zfar = self.max_depth
    #     znear = self.min_depth
    #     outputs = {}
    #     # frame_id = self.render_frame_ids
    #     for frame_id in self.render_frame_ids:
    #         for cam_id in range(self.num_cams):
                
    #             color_aug = inputs[(f'color_aug', frame_id)][:, cam_id, ...]
    #             bs, _, height, width = color_aug.shape

    #             if frame_id==0:
    #                 e2c_extr = torch.linalg.inv(inputs['c2e_extr'][:, cam_id, ...])
    #             else:
    #                 cam_T_cam = inputs[('cam_T_cam', 0, frame_id)][:, cam_id, ...]
    #                 e2c_extr = torch.matmul(cam_T_cam, torch.linalg.inv(inputs['c2e_extr'][:, cam_id, ...]))

    #             FovX_list = []
    #             FovY_list = []
    #             world_view_transform_list = []
    #             full_proj_transform_list = []
    #             camera_center_list = []
    #             for i in range(bs):
    #                 K_i = inputs['K'][:, cam_id, ...][i,:]
    #                 e2c_extr_i = e2c_extr[i,:]
    #                 FovX = focal2fov(K_i[0, 0], width)
    #                 FovY = focal2fov(K_i[1, 1], height)
    #                 projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, K=K_i, h=height, w=width).transpose(0, 1).cuda()
    #                 world_view_transform = e2c_extr_i.transpose(0, 1)
    #                 camera_center = torch.linalg.inv(world_view_transform)[3, :3].unsqueeze(0)
    #                 world_view_transform = world_view_transform.unsqueeze(0)
    #                 # full_proj_transform: (E^T K^T) = (K E)^T
    #                 full_proj_transform = (world_view_transform.bmm(projection_matrix.unsqueeze(0)))

    #                 FovX_list.append(FovX)
    #                 FovY_list.append(FovY)

    #                 world_view_transform_list.append(world_view_transform)
    #                 full_proj_transform_list.append(full_proj_transform)
    #                 camera_center_list.append(camera_center)
    #                 del FovX, FovY, world_view_transform, full_proj_transform, camera_center
    #             outputs[('groudtruth',frame_id, cam_id)] = color_aug
    #             # outputs[('gt_aug',frame_id, cam_id)] = inputs[(f'color_aug', frame_id)][:, cam_id, ...]

    #             outputs[('FovX', frame_id, cam_id)] = torch.tensor(FovX_list).cuda()
    #             outputs[('FovY', frame_id, cam_id)] = torch.tensor(FovY_list).cuda()
    #             outputs[('world_view_transform', frame_id, cam_id)] = torch.cat(world_view_transform_list, dim=0)
    #             outputs[('full_proj_transform', frame_id, cam_id)] = torch.cat(full_proj_transform_list, dim=0)
    #             outputs[('camera_center', frame_id, cam_id)] = torch.cat(camera_center_list, dim=0)

    #             del FovX_list, FovY_list, world_view_transform_list, full_proj_transform_list, camera_center_list

    #     return outputs

    # def render_splating_imgs(self, recontrast_data, render_data):
        
    #     bs = len(recontrast_data['xyz'])
    #     outputs = {}
    #     # frame_id = self.render_frame_ids
    #     for frame_id in self.render_frame_ids:
    #         for i in range(bs):

    #             xyz_i = recontrast_data['xyz'][i]
    #             rot_i = recontrast_data['rot_maps'][i]
    #             scale_i = recontrast_data['scale_maps'][i]
    #             opacity_i = recontrast_data['opacity_maps'][i]
    #             sh_i = recontrast_data['sh_maps'][i]

    #             for cam_id in range(self.num_cams):
    #                 _, _, height, width = render_data[('groudtruth', frame_id, cam_id)].shape
    #                 novel_FovX_i = render_data[('FovX', frame_id, cam_id)][i]
    #                 novel_FovY_i = render_data[('FovY', frame_id, cam_id)][i]
    #                 novel_world_view_transform_i = render_data[('world_view_transform', frame_id, cam_id)][i]
    #                 novel_function_proj_transform_i = render_data[('full_proj_transform', frame_id, cam_id)][i]
    #                 novel_camera_center_i = render_data[('camera_center', frame_id, cam_id)][i]

    #                 render_novel_i = render(novel_FovX=novel_FovX_i,
    #                                         novel_FovY=novel_FovY_i,
    #                                         novel_height=height,
    #                                         novel_width=width,
    #                                         novel_world_view_transform=novel_world_view_transform_i,
    #                                         novel_full_proj_transform=novel_function_proj_transform_i,
    #                                         novel_camera_center=novel_camera_center_i,
    #                                         pts_xyz=xyz_i.contiguous(), 
    #                                         pts_rgb=None, 
    #                                         rotations=rot_i.contiguous(), 
    #                                         scales=scale_i.contiguous(), 
    #                                         opacity=opacity_i.contiguous(), 
    #                                         shs=sh_i.contiguous(), 
    #                                         sh_degree=self.sh_degree,
    #                                         bg_color=[1.0, 1.0, 1.0]) 

    #                 del novel_FovX_i, novel_FovY_i, novel_world_view_transform_i, novel_function_proj_transform_i, novel_camera_center_i
    #                 if ('gaussian_color', frame_id, cam_id) not in outputs:
    #                     outputs[('gaussian_color', frame_id, cam_id)] = []
    #                 outputs[('gaussian_color', frame_id, cam_id)].append(render_novel_i.clip(0.0,1.0))
                    
    #                 del render_novel_i
    #             del xyz_i, rot_i, scale_i, opacity_i, sh_i


    #         for cam_id in range(self.num_cams):
    #             outputs[('gaussian_color', frame_id, cam_id)] = torch.stack(outputs[('gaussian_color', frame_id,cam_id)],dim=0).contiguous()
    #             outputs[('groudtruth', frame_id, cam_id)] = render_data[('groudtruth', frame_id, cam_id)]

    #     return outputs

    # def get_render_data(self, inputs):

    #     outputs = {}
    #     # frame_id = self.render_frame_ids
    #     for frame_id in self.render_frame_ids:
    #         for cam_id in range(self.num_cams):
    #             if self.render_cam_mode == 'origin':
    #                 if frame_id==0:
    #                     e2c_extr = torch.linalg.inv(inputs['c2e_extr'][:, cam_id, ...])
    #                     # cam_T_cam = torch.eye(4).unsqueeze(0).repeat(len(e2c_extr),1,1).to(e2c_extr.device)
    #                 else:
    #                     cam_T_cam = inputs[('cam_T_cam', 0, frame_id)][:, cam_id, ...]
    #                     e2c_extr = torch.matmul(cam_T_cam, torch.linalg.inv(inputs['c2e_extr'][:, cam_id, ...]))
    #                 K = inputs['K'][:, cam_id]
    #                 gt_img = inputs[(f'color_aug', frame_id)][:, cam_id, ...]
    #             elif self.render_cam_mode=='shift':
    #                 assert frame_id==0
    #                 e2c_extr = torch.linalg.inv(inputs['c2e_extr'][:, cam_id, ...])
    #                 cam_T_cam = self.render_shift_T.to(e2c_extr.device).repeat(len(e2c_extr),1,1)
    #                 e2c_extr = torch.matmul(cam_T_cam, e2c_extr)
    #                 K = inputs['K'][:, cam_id]
    #                 gt_img = inputs[(f'color_aug', frame_id)][:, cam_id, ...]
    #                 outputs[('cam_T_cam',frame_id, cam_id)] = cam_T_cam
    #                 outputs[('gt_mask',frame_id, cam_id)] = inputs['mask'][:,cam_id]
    #             elif self.render_cam_mode=='scale':
    #                 if frame_id==0:
    #                     e2c_extr = torch.linalg.inv(inputs['c2e_extr'][:, cam_id, ...])
    #                     # cam_T_cam = torch.eye(4).unsqueeze(0).repeat(len(e2c_extr),1,1).to(e2c_extr.device)
    #                 else:
    #                     cam_T_cam = inputs[('cam_T_cam', 0, frame_id)][:, cam_id, ...]
    #                     e2c_extr = torch.matmul(cam_T_cam, torch.linalg.inv(inputs['c2e_extr'][:, cam_id, ...]))
    #                 K = inputs['K'][:, cam_id].clone()
    #                 K[:,:2] = K[:,:2] * self.render_scale
    #                 gt_img = F.interpolate(inputs[('color_org', frame_id)][:,cam_id,...], size=(self.render_height,self.render_width),mode = 'bilinear', align_corners=False)

    #             outputs[('groudtruth',frame_id, cam_id)] = gt_img
    #             outputs[('e2c_extr',frame_id, cam_id)] = e2c_extr
    #             # outputs[('c2e_extr',frame_id, cam_id)] = inputs['c2e_extr'][:, cam_id, ...]

    #             outputs[('K',frame_id, cam_id)] = K

    #     return outputs

    def get_render_data(self, inputs, render_frame=0, render_scale=1.0, render_shift_x=0.0, render_shift_y=0.0):

        outputs = {'render_frame': render_frame, 'render_scale': render_scale, 
                   'render_shift_x': render_shift_x, 'render_shift_y': render_shift_y}
        for cam_id in range(self.num_cams):
            cam_T_cam = inputs[('cam_T_cam', 0, render_frame)][:, cam_id, ...]
            e2c_extr = torch.linalg.inv(inputs['c2e_extr'][:, cam_id, ...])
            e2c_extr = torch.matmul(cam_T_cam, e2c_extr)

            K_scale = inputs['K'][:, cam_id].clone()
            if render_scale != 1.0:
                K_scale[:,:2] = K_scale[:,:2] * render_scale
                render_height, render_width = int(self.height * render_scale), int(self.width * render_scale)
                gt_img = F.interpolate(inputs[('color_org', render_frame)][:,cam_id,...], 
                                       size=(render_height,render_width),mode = 'bilinear', align_corners=False)
                mask = F.interpolate(inputs['mask'][:,cam_id], 
                                       size=(render_height,render_width),mode = 'bilinear', align_corners=False)
            else:
                render_height,render_width = self.height, self.width
                gt_img = inputs[(f'color_aug', render_frame)][:, cam_id, ...]
                mask = inputs['mask'][:,cam_id]

            render_shift_T = torch.eye(4,dtype=torch.float32).unsqueeze(0)
            if render_shift_x != 0.0 or render_shift_y != 0.0:
                render_shift_T[:,0,3] = render_shift_x
                render_shift_T[:,1,3] = render_shift_y
                render_shift_T = render_shift_T.to(e2c_extr.device).repeat(len(e2c_extr),1,1)
                e2c_extr = torch.matmul(render_shift_T, e2c_extr)

            outputs[('shift_T', cam_id)] = render_shift_T
            outputs[('gt_mask', cam_id)] = mask
            outputs[('groudtruth',cam_id)] = gt_img
            outputs[('e2c_extr',cam_id)] = e2c_extr
            outputs[('K',cam_id)] = K_scale
        
        outputs['render_height']=render_height
        outputs['render_width']=render_width
            
        return outputs
    
    # def render_splating_imgs(self, recontrast_data, render_data):
        
    #     bs = len(recontrast_data['xyz'])
    #     outputs = {}
    #     # frame_id = self.render_frame_ids
    #     for frame_id in self.render_frame_ids:
    #         for i in range(bs):

    #             xyz_i = recontrast_data['xyz'][i]
    #             rot_i = recontrast_data['rot_maps'][i]
    #             scale_i = recontrast_data['scale_maps'][i]
    #             opacity_i = recontrast_data['opacity_maps'][i]
    #             sh_i = recontrast_data['sh_maps'][i]

    #             e2c_extr_i, K_i = [], []
    #             for cam_id in range(self.num_cams):
    #                 e2c_extr_i.append(render_data[('e2c_extr',frame_id, cam_id)][i])
    #                 K_i.append(render_data[('K',frame_id, cam_id)][i,:3,:3])
    #             e2c_extr_i = torch.stack(e2c_extr_i, dim=0)
    #             K_i = torch.stack(K_i, dim=0)

    #             render_colors_i, render_alphas_i, meta_i = rasterization(
    #                 xyz_i,  # [N, 3]
    #                 rot_i,  # [N, 4]
    #                 scale_i,  # [N, 3]
    #                 opacity_i.squeeze(-1),  # [N]
    #                 sh_i,  # [N, K, 3]
    #                 e2c_extr_i,  # [1, 4, 4]
    #                 K_i,  # [1, 3, 3]
    #                 self.render_width,
    #                 self.render_height,
    #                 sh_degree=self.sh_degree,
    #                 render_mode="RGB",
    #                 # sparse_grad=True,
    #                 # this is to speedup large-scale rendering by skipping far-away Gaussians.
    #                 # radius_clip=3,
    #             )
    #             # render_rgb_i, render_depth_i = render_colors_i[...,:3], render_colors_i[...,3]
    #             render_rgb_i = render_colors_i[...,:3].permute(0,3,1,2)
    #             del xyz_i, rot_i, scale_i, opacity_i, sh_i, e2c_extr_i, K_i, render_colors_i, render_alphas_i, meta_i

    #             for cam_id in range(self.num_cams):
    #                 if ('gaussian_color', frame_id, cam_id) not in outputs:
    #                     outputs[('gaussian_color', frame_id, cam_id)] = []
    #                 outputs[('gaussian_color', frame_id, cam_id)].append(render_rgb_i[cam_id])

    #         for cam_id in range(self.num_cams):
    #             gaussian_color = torch.stack(outputs[('gaussian_color', frame_id,cam_id)],dim=0).contiguous()
    #             outputs[('groudtruth', frame_id, cam_id)] = render_data[('groudtruth', frame_id, cam_id)]

    #             if self.render_cam_mode=='shift':
    #                 ref_mask = render_data[('gt_mask',frame_id,cam_id)]
    #                 ref_K = render_data[('K',frame_id, cam_id)]
    #                 ref_depths = rearrange(recontrast_data['depth_maps'],'b (c h w) -> b c h w',c=self.num_cams,h=self.height,w=self.width)[:,cam_id:cam_id+1, ...]
    #                 cam_T_cam = render_data[('cam_T_cam',frame_id, cam_id)]
    #                 ref_inv_K = torch.linalg.inv(ref_K)
    #                 gaussian_color, mask_warped = self.get_virtual_image(
    #                     gaussian_color, 
    #                     ref_mask, 
    #                     ref_depths, 
    #                     ref_inv_K, 
    #                     ref_K, 
    #                     cam_T_cam
    #                 )
    #             else:
    #                 mask_warped = torch.ones_like(gaussian_color[:,0:1,...])

    #             outputs[('gaussian_color', frame_id, cam_id)] = gaussian_color
    #             outputs[('warped_mask', frame_id, cam_id)] = mask_warped.detach()

    #     return outputs
    
    def render_splating_imgs(self, recontrast_data, render_data_list):
        
        bs = len(recontrast_data['xyz'])
        outputs = {}
        # frame_id = self.render_frame_ids
        for i in range(bs):
            xyz_i = recontrast_data['xyz'][i]
            rot_i = recontrast_data['rot_maps'][i]
            scale_i = recontrast_data['scale_maps'][i]
            opacity_i = recontrast_data['opacity_maps'][i]
            sh_i = recontrast_data['sh_maps'][i]

            for render_idx,render_data in render_data_list.items():

                e2c_extr_i, K_i = [], []
                for cam_id in range(self.num_cams):
                    e2c_extr_i.append(render_data[('e2c_extr', cam_id)][i])
                    K_i.append(render_data[('K', cam_id)][i,:3,:3])
                e2c_extr_i = torch.stack(e2c_extr_i, dim=0)
                K_i = torch.stack(K_i, dim=0)
                render_width = render_data['render_width']
                render_height = render_data['render_height']
                render_colors_i, render_alphas_i, meta_i = rasterization(
                    xyz_i,  # [N, 3]
                    rot_i,  # [N, 4]
                    scale_i,  # [N, 3]
                    opacity_i.squeeze(-1),  # [N]
                    sh_i,  # [N, K, 3]
                    e2c_extr_i,  # [1, 4, 4]
                    K_i,  # [1, 3, 3]
                    render_width,
                    render_height,
                    sh_degree=self.sh_degree,
                    render_mode="RGB",
                    # sparse_grad=True,
                    # this is to speedup large-scale rendering by skipping far-away Gaussians.
                    # radius_clip=3,
                )
                # render_rgb_i, render_depth_i = render_colors_i[...,:3], render_colors_i[...,3]
                render_rgb_i = render_colors_i[...,:3].permute(0,3,1,2)

                for cam_id in range(self.num_cams):
                    if ('gaussian_color', render_idx, cam_id) not in outputs:
                        outputs[('gaussian_color', render_idx, cam_id)] = []
                    outputs[('gaussian_color', render_idx, cam_id)].append(render_rgb_i[cam_id])



        for render_idx, render_data in render_data_list.items():
            for cam_id in range(self.num_cams):
                gaussian_color = torch.stack(outputs[('gaussian_color', render_idx,cam_id)],dim=0).contiguous()

                if render_data['render_shift_x']!=0.0 or render_data['render_shift_y']!=0.0:
                    ref_mask = render_data[('gt_mask',cam_id)]
                    ref_K = render_data[('K', cam_id)]
                    ref_depths = rearrange(recontrast_data['depth_maps'],'b (c h w) -> b c h w',c=self.num_cams,h=self.height,w=self.width)[:,cam_id:cam_id+1, ...]
                    shift_T = render_data[('shift_T', cam_id)]
                    ref_inv_K = torch.linalg.inv(ref_K)
                    gaussian_color, mask_warped = self.get_virtual_image(
                        gaussian_color, 
                        ref_mask, 
                        ref_depths, 
                        ref_inv_K, 
                        ref_K, 
                        shift_T
                    )
                else:
                    mask_warped = torch.ones_like(gaussian_color[:,0:1,...])

                outputs[('gaussian_color', render_idx, cam_id)] = gaussian_color
                outputs[('groudtruth', render_idx, cam_id)] = render_data[('groudtruth', cam_id)]
                outputs[('warped_mask', render_idx, cam_id)] = mask_warped.detach()
            render_frame = render_data['render_frame']
            render_scale = render_data['render_scale']
            render_shift_x = render_data['render_shift_x']
            render_shift_y = render_data['render_shift_y']
            outputs[('render_params', render_idx)] = f'{render_frame}_{render_scale:.2f}_{render_shift_x:.2f}_{render_shift_y:.2f}'

        return outputs

    
    def render_project_imgs(self, input_data, recontrast_data, project_frame_ids):
        outputs = {}
        recontrast_frame_ids = self.recontrast_frame_ids
        
        for cam_id in range(self.num_cams):
            ref_colors = input_data[('color_aug', recontrast_frame_ids)][:, cam_id, ...]
            bs, _, height, width = ref_colors.shape
            ref_depths = rearrange(recontrast_data['depth_maps'],'b (c h w) -> b c h w',c=self.num_cams,h=height,w=width)[:,cam_id, ...]
            ref_depths = ref_depths.unsqueeze(1)
            if 'mask' in input_data:
                ref_mask = input_data['mask'][:,cam_id,]
            else:
                ref_mask = torch.ones_like(ref_depths)
            ref_K = input_data['K'][:,cam_id,]
            ref_inv_K = torch.linalg.inv(ref_K)
            outputs[('ref_colors', recontrast_frame_ids, cam_id)] = ref_colors
            outputs[('ref_depths', recontrast_frame_ids, cam_id)] = ref_depths
            if (recontrast_frame_ids==0) and ('depth' in input_data):
                outputs[('gt_depths', recontrast_frame_ids, cam_id)] = input_data['depth'][:,cam_id,...]
            # outputs[('src_mask', recontrast_frame_ids, cam_id)] = src_mask
            for frame_id in project_frame_ids:
                src_colors = input_data[('color_aug', frame_id)][:, cam_id, ...]

                cam_T_cam = input_data[('cam_T_cam', 0, frame_id)][:, cam_id, ...]
                warped_img, warped_mask = self.get_virtual_image(
                                src_colors, 
                                ref_mask, 
                                ref_depths, 
                                ref_inv_K, 
                                ref_K, 
                                cam_T_cam
                            )

                warped_img = self.get_norm_image_single(
                    ref_colors, 
                    ref_mask,
                    warped_img, 
                    warped_mask
                )
                outputs[('warped_gt', frame_id, cam_id)] = ref_colors
                outputs[('warped_pred', frame_id, cam_id)] = warped_img
                outputs[('warped_mask', frame_id, cam_id)] = warped_mask.detach()
        
        return outputs


    def get_virtual_image(self, src_img, src_mask, tar_depth, tar_invK, src_K, T):
        """
        This function warps source image to target image using backprojection and reprojection process. 
        """
        # do reconstruction for target from source   
        pix_coords = self.project(tar_depth, T, tar_invK, src_K)
        
        img_warped = F.grid_sample(src_img, pix_coords, mode='bilinear', 
                                    padding_mode='zeros', align_corners=True)
        mask_warped = F.grid_sample(src_mask, pix_coords, mode='nearest', 
                                    padding_mode='zeros', align_corners=True)

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

        norm_warp = (warp_img - w_mean) / (w_std + 1e-8) * s_std + s_mean
        return norm_warp * warp_mask.float()   

    def get_mean_std(self, feature, mask):
        """
        This function returns mean and standard deviation of the overlapped features. 
        """
        _, c, h, w = mask.size()
        mean = (feature * mask).sum(dim=(1,2,3), keepdim=True) / (mask.sum(dim=(1,2,3), keepdim=True) + 1e-8)
        var = ((feature - mean) ** 2).sum(dim=(1,2,3), keepdim=True) / (c*h*w)
        return mean, torch.sqrt(var + 1e-16)     
    
    def _unproject_depth_map_to_points_map(self, depth_map, K, c2e_extr):
        '''
        depth_map: depth map of shape (Bs, H, W)
        K: pixel -> camera intrinsic matrix of shape (Bs, 4, 4)  # 仅使用 [:3, :3] 部分
        c2e_extr: camera -> ego extrinsics matrix of shape (Bs, 4, 4)  # 仅使用 [:3, :] 部分
        return points_map: points map of shape (Bs, H, W, 3)
        '''
        if depth_map is None:
            return None

        Bs, H, W = depth_map.shape
        
        # 1. 创建像素坐标网格 (u, v)
        u = torch.arange(W, device=depth_map.device, dtype=torch.float32)
        v = torch.arange(H, device=depth_map.device, dtype=torch.float32)
        u_grid, v_grid = torch.meshgrid(u, v, indexing='xy')  # u_grid: (H, W), v_grid: (H, W)
        
        # 扩展为批量维度
        u_grid = u_grid.unsqueeze(0).expand(Bs, -1, -1)  # (Bs, H, W)
        v_grid = v_grid.unsqueeze(0).expand(Bs, -1, -1)  # (Bs, H, W)
        
        # 2. 从内参矩阵提取参数
        # K的形状为 (Bs, 4, 4)，但我们只使用前3x3
        fx = K[:, 0, 0]  # (Bs,)
        fy = K[:, 1, 1]  # (Bs,)
        cx = K[:, 0, 2]  # (Bs,)
        cy = K[:, 1, 2]  # (Bs,)
        
        # 将参数调整为匹配网格形状 (Bs, H, W)
        fx = fx.view(Bs, 1, 1).expand(-1, H, W)
        fy = fy.view(Bs, 1, 1).expand(-1, H, W)
        cx = cx.view(Bs, 1, 1).expand(-1, H, W)
        cy = cy.view(Bs, 1, 1).expand(-1, H, W)
        
        # 3. 计算归一化相机坐标 (使用显式公式避免矩阵求逆)
        # X = (u - cx) * depth / fx
        # Y = (v - cy) * depth / fy
        # Z = depth
          # (Bs, H, W)
        x = (u_grid - cx) * depth_map / fx
        y = (v_grid - cy) * depth_map / fy
        z = depth_map
        
        # 组合成相机坐标系下的点云
        cam_points = torch.stack([x, y, z], dim=-1)  # (Bs, H, W, 3)

        # 4. 转换到ego坐标系
        R = c2e_extr[:, :3, :3]  # (Bs, 3, 3)
        t = c2e_extr[:, :3, 3]   # (Bs, 3)
        
        # 重塑点云以便批量矩阵乘法
        cam_points_flat = cam_points.reshape(Bs, -1, 3)  # (Bs, H*W, 3)
        
        # 应用变换: P_ego = R @ P_cam^T + t
        ego_points_flat = torch.matmul(cam_points_flat, R.transpose(1, 2)) + t.unsqueeze(1)
        
        # 重塑回原始图像尺寸
        ego_points = ego_points_flat.reshape(Bs, H, W, 3)  # (Bs, H, W, 3)

        # 5. 处理无效深度点 (深度为0的点设为原点)
        mask = (depth_map == 0).unsqueeze(-1).expand(-1, -1, -1, 3)  # (Bs, H, W, 3)
        ego_points[mask] = 0
        del ego_points_flat, cam_points_flat, cam_points
        return ego_points


    @rank_zero_only
    def _save_reprojection_images(self, save_name, batch_data,project_frame_ids, bs_id=0):
        if os.path.exists(os.path.join(self.save_dir, save_name)):
            shutil.rmtree(os.path.join(self.save_dir, save_name))
        os.makedirs(os.path.join(self.save_dir, save_name))
        
        recontrast_frame_ids = self.recontrast_frame_ids

        for cam_id in range(self.num_cams):
            ref_colors = batch_data[('ref_colors', recontrast_frame_ids, cam_id)].detach()
            ref_depths = batch_data[('ref_depths', recontrast_frame_ids, cam_id)].detach()
            ref_depths = torch.log(ref_depths)
            max_depth = log(self.max_depth)
            min_depth = log(self.min_depth)
            norm_depths = 1 - (ref_depths - min_depth) / (max_depth - min_depth)
            self.save_image(ref_colors[bs_id], os.path.join(self.save_dir, save_name, f"{recontrast_frame_ids}_{cam_id}_gt.png"))
            self.save_image(norm_depths[bs_id], os.path.join(self.save_dir, save_name, f"{recontrast_frame_ids}_{cam_id}_depth.png"))

            for render_frame_ids in project_frame_ids:
                warped_img = batch_data[('warped_pred', render_frame_ids, cam_id)].detach()
                self.save_image(warped_img[bs_id], os.path.join(self.save_dir, save_name, f"{render_frame_ids}_{cam_id}_preds.png"))


    
    @rank_zero_only
    def _save_splating_images(self, save_name, batch_data, bs_id=0):
        if os.path.exists(os.path.join(self.save_dir, save_name)):
            shutil.rmtree(os.path.join(self.save_dir, save_name))
        os.makedirs(os.path.join(self.save_dir, save_name))
        for frame_id in range(self.render_nums):
            for cam_id in range(self.num_cams): 
                pred = batch_data[('gaussian_color', frame_id, cam_id)].detach()
                mask = batch_data[('warped_mask', frame_id, cam_id)]
                gt = batch_data[('groudtruth', frame_id, cam_id)]
                render_params =  batch_data[('render_params',frame_id)]   
                self.save_image(pred[bs_id], os.path.join(self.save_dir, save_name, f"{frame_id}_{render_params}_{cam_id}_preds.png"))
                self.save_image(mask[bs_id], os.path.join(self.save_dir, save_name, f"{frame_id}_{render_params}_{cam_id}_mask.png"))
                self.save_image(gt[bs_id], os.path.join(self.save_dir, save_name, f"{frame_id}_{render_params}_{cam_id}_gt.png")) 
    @rank_zero_only
    def save_image(self, image, path):
        """Save an image. Assumed to be in range 0-1."""

        # Create the parent directory if it doesn't already exist.
        # os.makedirs(os.path.dirname(path),exist_ok=True)

        image = image.detach().cpu().numpy().clip(min=0, max=1)
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

    # @rank_zero_only
    # def _save_wordpoints_glb(self, glbfile, inputs, recontrast_data, render_data, bs_id=0):
    #     predictions = {'images':[],'world_points_from_depth':[],'extrinsic':[]}
    #     # frame_id = self.render_frame_ids
    #     for frame_id in self.recontrast_frame_ids:
    #         for cam_id in range(self.num_cams):
    #             predictions['images'].append(render_data[('groudtruth', frame_id, cam_id)][bs_id].cpu().numpy())

    #     predictions['images'] = np.stack(predictions['images'], axis=0)
    #     predictions['world_points_from_depth'] = recontrast_data['xyz'][bs_id].detach().cpu().numpy()
    #     predictions['extrinsic'] = inputs['c2e_extr'][bs_id].cpu().numpy()
    #     os.makedirs(os.path.dirname(glbfile),exist_ok=True)
    #     glbscene = predictions_to_glb(
    #         predictions,
    #         conf_thres=0.0,
    #         filter_by_frames='all',
    #         mask_black_bg=False,
    #         mask_white_bg=False,
    #         show_cam=True,
    #         mask_sky=False,
    #         target_dir=None,
    #         prediction_mode='',
    #     )
    #     del predictions
    #     print(f'save to {glbfile}')
    #     glbscene.export(file_obj=glbfile)
    #     torch.cuda.empty_cache()


    def _filter_visible_gaussians(self,  pts_xyz, full_proj_transform, opacity):
        """ 过滤在相机视锥内且不透明的高斯点
            pts_xyz: (N, 3) 世界坐标系中的点云
            view_matrix: (4, 4) 视图矩阵 (世界->相机)
            proj_matrix: (4, 4) 投影矩阵 (相机->裁剪空间)
            opacity: (N, 1)
        """
        # 计算高斯点在屏幕空间的投影
        points_homogeneous = torch.cat([
            pts_xyz, 
            torch.ones(pts_xyz.shape[0], 1, device=pts_xyz.device)
        ], dim=1)

        clip_points = torch.mm(full_proj_transform, points_homogeneous.t()).t()

        ndc_points = clip_points[:, :3] / clip_points[:, 3:4]

        # 检查点是否在视锥内
        # 在NDC空间中，有效点满足: -1 <= x,y,z <= 1
        in_frustum = (
            (ndc_points[:, 0] >= -1) & (ndc_points[:, 0] <= 1) &
            (ndc_points[:, 1] >= -1) & (ndc_points[:, 1] <= 1) &
            (ndc_points[:, 2] >= -1) & (ndc_points[:, 2] <= 1)
        )

        # 3. 检查不透明度 > 阈值
        opaque = opacity.squeeze(-1) > 0.01  # 可调整阈值

        valid_points = in_frustum & opaque
        del in_frustum, opaque
        return valid_points

    def compute_gaussian_loss(self, batch_data, render_frame_ids):
        """
        This function computes gaussian loss.
        """
        # self occlusion mask * overlap region mask

        gaussian_loss = 0.0 
        count = 0
        # frame_id = self.render_frame_ids
        for render_idx in render_frame_ids:
            for cam_id in range(self.num_cams): 
                pred = batch_data[('gaussian_color', render_idx, cam_id)]
                gt = batch_data[('groudtruth', render_idx, cam_id)]  
                mask = batch_data[('warped_mask', render_idx, cam_id)]

                # lpips_loss = self.lpips(pred, gt, normalize=True)
                # lpips_loss = 0.0
                # l2_loss = ((pred - gt)**2)
                l1_loss = self.l1_fn(pred, gt)
                ssim_loss = self.ssim_fn(pred, gt)
                sum_loss = 0.8 * l1_loss + 0.2 * ssim_loss
                gaussian_loss += compute_masked_loss(sum_loss, mask, eps=0.0)
                count += 1
        return self.lambda_gaussian * gaussian_loss / count

    def compute_project_loss(self, batch_data,project_frame_ids):
        """
        This function computes gaussian loss.
        """
        # self occlusion mask * overlap region mask

        project_loss = 0.0 
        count = 0
        for frame_id in project_frame_ids:
            for cam_id in range(self.num_cams): 
                pred = batch_data[('warped_pred', frame_id, cam_id)]
                gt = batch_data[('warped_gt', frame_id, cam_id)]  
                mask = batch_data[('warped_mask',frame_id,cam_id)]

                # img_loss = compute_photometric_loss(pred, gt)
                l1_loss = self.l1_fn(pred, gt)
                ssim_loss = self.ssim_fn(pred, gt)
                sum_loss = 0.8 * l1_loss + 0.2 * ssim_loss
                project_loss += compute_masked_loss(sum_loss, mask, eps=0.0)
                count += 1
        return self.lambda_project * project_loss / count

    def compute_depth_loss(self, batch_data, beta=1.0, eps=1e-6):
        """
        This function computes edge-aware smoothness loss for the disparity map.
        """
        depth_loss = 0.0 
        count = 0
        for cam_id in range(self.num_cams): 
            src_color = batch_data[('ref_colors', self.recontrast_frame_ids, cam_id)]
            src_depth = batch_data[('ref_depths', self.recontrast_frame_ids, cam_id)]
            if self.recontrast_frame_ids == 0:
                gt_depth = batch_data[('gt_depths', self.recontrast_frame_ids, cam_id)]
                mask_depth = torch.logical_and(gt_depth > self.min_depth,gt_depth < self.max_depth)
            
                # print('min_mean_max',(src_depth.amin().item(),gt_depth[mask_depth].amin().item()),(src_depth.mean().item(),gt_depth[mask_depth].mean().item()),(src_depth.amax().item(),gt_depth[mask_depth].amax().item()))
                mask_depth = mask_depth.to(torch.float32)
                abs_diff = torch.abs(gt_depth - src_depth) * mask_depth
                l1loss = torch.where(abs_diff < beta, 0.5 * abs_diff * abs_diff / beta, abs_diff - 0.5 * beta)
                l1loss = torch.sum(l1loss) / (torch.sum(mask_depth) + eps)
                depth_loss += l1loss * self.lambda_depth

            mean_disp = src_depth.mean(2, True).mean(3, True)
            norm_disp = src_depth / (mean_disp + 1e-8)
            edge_loss = compute_edg_smooth_loss(src_color, norm_disp)
            depth_loss += self.lambda_edge * edge_loss

            count += 1

        return   depth_loss / count
    
    def compute_norm_loss(self, batch_data):

        scale_loss = self.lambda_scale * torch.mean(torch.norm(batch_data['scale_maps'], dim=-1))
        
        # 不透明度稀疏性正则化: 鼓励不透明度趋于0或1
        opacity_loss = self.lambda_opacity * torch.mean(torch.abs(batch_data['opacity_maps']))
        
        # 总正则化损失
        total_reg_loss = scale_loss + opacity_loss

        return total_reg_loss

    @torch.no_grad()
    def compute_reconstruction_metrics(self, batch_data, stage, render_frame_ids):
        """
        This function computes reconstruction metrics.
        """
        psnr = 0.0
        ssim = 0.0
        lpips = 0.0

        novel_count =0
        # frame_id = self.render_frame_ids
        for render_idx in render_frame_ids:
            for cam_id in range(self.num_cams): 
                pred = batch_data[('gaussian_color', render_idx, cam_id)].detach()
                gt = batch_data[('groudtruth', render_idx, cam_id)]    
                psnr += self.compute_psnr(gt, pred).mean()
                ssim += self.compute_ssim(gt, pred).mean()
                lpips += self.compute_lpips(gt, pred).mean()
                novel_count += 1

        psnr /= novel_count
        ssim /= novel_count
        lpips /= novel_count

        self.log(f"{stage}/psnr", psnr.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/ssim", ssim.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/lpips", lpips.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
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
    from dataset.vggt3dgs_data_module import VGGT3DGS_LITDataModule

    import math

    config_file = 'configs/nuscenes/vggt3dgs.yaml'
    with open(config_file) as f:
        main_cfg = yaml.load(f, Loader=yaml.FullLoader)
    main_cfg['data_cfg']['batch_size'] = 1
    main_cfg['model_cfg']['batch_size'] = main_cfg['data_cfg']['batch_size']

    pl.seed_everything(main_cfg['seed'], workers=True)

    main_datamodule = VGGT3DGS_LITDataModule(main_cfg['data_cfg'])
    main_datamodule.setup('test')
    test_dataloader = main_datamodule.test_dataloader()

    # from pytorch_lightning.loggers import TensorBoardLogger
    # 创建TensorBoard日志记录器

    litmodel = VGGT3DGS_LITModelModule(
        cfg=main_cfg['model_cfg'],
        # logger=TensorBoardLogger(save_dir='.',name='logs')
    )
    restore_ckpt = os.environ.get('VGGT3DGS_RESTORE_CKPT', '')
    if not restore_ckpt:
        raise ValueError('Set VGGT3DGS_RESTORE_CKPT before running this debug entry point.')
    load_ckpt = torch.load(restore_ckpt,map_location=f"cuda:0")
    litmodel.load_state_dict(load_ckpt['state_dict'])


    litmodel.to('cuda:0')
    for batch_inputs in test_dataloader:
        for key in batch_inputs:
            if isinstance(batch_inputs[key],torch.Tensor):
                batch_inputs[key] = batch_inputs[key].to('cuda:0')
        outputs = litmodel.predict_step(batch_inputs,0)
        recontrast_data = outputs[0]
        render_data = outputs[1]
        splating_data = outputs[-1]

        for i in range(1):
            frame_id = 0

            cam_id = 0
            means = recontrast_data['xyz'][i]
            quats = recontrast_data['rot_maps'][i]
            scales = recontrast_data['scale_maps'][i]
            opacities = recontrast_data['opacity_maps'][i]
            sh_maps = recontrast_data['sh_maps'][i]

            height, width = litmodel.height, litmodel.width

            sh_degree = 4
            K = batch_inputs['K'][i,cam_id]
            recontrast_data['K'] = K
            recontrast_data['height'] = height
            recontrast_data['width'] = width
            recontrast_data['c2w_extr'] = batch_inputs['c2e_extr'][i, cam_id, ...]

            gs_dict = {
                '_means':means,
                '_quats':quats,
                '_scales':scales,
                '_opacities':opacities,
                '_colors':sh_maps,
                'camera_K':K,
                'height':height,
                'width':width,
                'camera_c2w':batch_inputs['c2e_extr'][i, cam_id, ...],
            }
            output_root = os.environ.get('AD_VISBENCH_OUTPUT_DIR', 'outputs')
            os.makedirs(output_root, exist_ok=True)
            test_path = os.path.join(output_root, f'reconstruction_v2.0-vggt3dgs8_bs{i+1}.pt')
            torch.save(gs_dict,test_path)

            e2c_extr =  torch.linalg.inv(batch_inputs['c2e_extr'][i, cam_id, ...]) 

            cam_T_cam = torch.tensor([
                [1, 0, 0, 1.0],
                [0, 1, 0, 0.5],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=e2c_extr.dtype, device=e2c_extr.device)
            e2c_extr = cam_T_cam @ e2c_extr 

            # scale_factor =  640.0 / 518.0
            # width, height = int(width * scale_factor), int(height * scale_factor)
            # print(K,height,width)
            # K[:2] = K[:2] * scale_factor
            # print(K)
            # FovX = torch.tensor(focal2fov(K[0, 0], width)).to('cuda:0')
            # FovY = torch.tensor(focal2fov(K[1, 1], height)).to('cuda:0')
            # projection_matrix = getProjectionMatrix(znear=1.5, zfar=100, K=K, h=height, w=width).transpose(0, 1).cuda()
            # world_view_transform = e2c_extr.transpose(0, 1) 
            # # full_proj_transform: (E^T K^T) = (K E)^T
            # full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            # camera_center = world_view_transform.inverse()[3, :3] 
            # novel_world_view_transform_i = world_view_transform
            # novel_function_proj_transform_i = full_proj_transform

            # render_color = render(novel_FovX=FovX,
            #                         novel_FovY=FovY,
            #                         novel_height=height,
            #                         novel_width=width,
            #                         novel_world_view_transform=world_view_transform,
            #                         novel_full_proj_transform=full_proj_transform,
            #                         novel_camera_center=camera_center,
            #                         pts_xyz=means,#.contiguous(), 
            #                         pts_rgb=None, 
            #                         rotations=quats,#.contiguous(), 
            #                         scales=scales,#.contiguous(), 
            #                         opacity=opacities,#.contiguous(), 
            #                         shs=sh_maps,#.contiguous(), 
            #                         sh_degree=sh_degree,
            #                         bg_color=[1.0, 1.0, 1.0]) 
            # render_rgbs = render_color

            render_color, render_alphas, meta = rasterization(
                means,  # [N, 3]
                quats,  # [N, 4]
                scales,  # [N, 3]
                opacities.squeeze(1),  # [N]
                sh_maps,  # [N, K, 3]
                e2c_extr[None],  # [1, 4, 4]
                K[None,:3,:3],  # [1, 3, 3]
                width,
                height,
                sh_degree=sh_degree,
                render_mode="RGB",
                # this is to speedup large-scale rendering by skipping far-away Gaussians.
                # radius_clip=0.1,
            )
            render_rgbs = render_color[0].permute(2,0,1)

            print(render_rgbs.shape,render_rgbs.min(),render_rgbs.mean(),render_rgbs.max())
            render_rgbs_uint8 = (render_rgbs.detach().cpu().numpy().transpose(1,2,0).clip(0,1.0) * 255).astype(np.uint8)
            pil.fromarray(render_rgbs_uint8).save(f'./test_bs{i}_novel.jpg')


            gt_img = splating_data[('groudtruth', frame_id, cam_id)][i]
            pred_img = splating_data[('gaussian_color', frame_id, cam_id)][i]

            # gt_img = batch_inputs[('color_org', frame_id)][cam_id][i]
            # print(gt_img.shape)
            # new_gt_img =  F.interpolate(gt_img.unsqueeze(0), 
            #                               size=(height,width),
            #                               mode = 'bilinear',
            #                               align_corners=False)

            # new_pred_img =  F.interpolate(pred_img.unsqueeze(0), 
            #                               size=(height,width),
            #                               mode = 'bilinear',
            #                               align_corners=False)

            src_colors = render_rgbs.unsqueeze(0)
            ref_mask = batch_inputs['mask'][i:i+1,cam_id]
            ref_K = K.unsqueeze(0)
            ref_depths = rearrange(recontrast_data['depth_maps'],'b (c h w) -> b c h w',c=litmodel.num_cams,h=height,w=width)[i:i+1,cam_id, ...]
            ref_depths = ref_depths.unsqueeze(1)
            ref_inv_K = torch.linalg.inv(ref_K)
            img_warped, mask_warped = litmodel.get_virtual_image(
                src_colors, 
                ref_mask, 
                ref_depths, 
                ref_inv_K, 
                ref_K, 
                cam_T_cam
            )

            render_rgbs_uint8 = (img_warped[0].detach().cpu().numpy().transpose(1,2,0).clip(0,1.0) * 255).astype(np.uint8)
            pil.fromarray(render_rgbs_uint8).save(f'./test_bs{i}_novel_warped.jpg')
            render_rgbs_uint8 = (mask_warped.detach().cpu().numpy()[0,0].clip(0,1.0) * 255).astype(np.uint8)
            pil.fromarray(render_rgbs_uint8).save(f'./test_bs{i}_novel_warpedmask.jpg')

            # img_warped = render_rgbs.unsqueeze(0)
            new_gt_img = gt_img.unsqueeze(0)
            new_pred_img = pred_img.unsqueeze(0)
            print(img_warped.shape,new_gt_img.shape,new_pred_img.shape)
            psnr = litmodel.compute_psnr(img_warped,new_gt_img)
            ssim = litmodel.compute_ssim(img_warped,new_gt_img)
            lpips = litmodel.compute_lpips(img_warped,new_gt_img)
            print(psnr,ssim,lpips)
            psnr = litmodel.compute_psnr(new_pred_img,new_gt_img)
            ssim = litmodel.compute_ssim(new_pred_img,new_gt_img)
            lpips = litmodel.compute_lpips(new_pred_img,new_gt_img)
            print(psnr,ssim,lpips)

            render_rgbs_uint8 = (new_pred_img[0].detach().cpu().numpy().transpose(1,2,0).clip(0,1.0) * 255).astype(np.uint8)
            pil.fromarray(render_rgbs_uint8).save(f'./test_bs{i}_origin.jpg')

            render_rgbs_uint8 = (new_gt_img[0].detach().cpu().numpy().transpose(1,2,0).clip(0,1.0) * 255).astype(np.uint8)
            pil.fromarray(render_rgbs_uint8).save(f'./test_bs{i}_gt.jpg')

        break