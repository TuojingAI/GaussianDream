import os

import torch
import torch.nn as nn
from einops import rearrange
from torch_scatter import scatter_add, scatter_max


from models.gaussian_util import depth2pc
from models.vggt.models.vggt import VGGT

from models.vggt.heads.gs_dpt_head import VGGT_DPT_GS_Head
from models.vggt.heads.depth_dpt_head import DPTHead
from models.deformable_attention_3d import DeformableAttention3D

EPSILON = 1e-6

def tensor_contiguous(x):
    if not x.is_contiguous():
        return x.contiguous()
    else:
        return x

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=8, alpha=32, dropout=0.0):
        super().__init__()
        self.linear = linear_layer
        self.lora_down = nn.Linear(linear_layer.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, linear_layer.out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)
        
        self.scaling = alpha / rank
        
        # 冻结原始权重
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x):
        orig_out = self.linear(x)
        lora_out = self.lora_up(self.dropout(self.lora_down(x)))
        return orig_out + lora_out * self.scaling
    
    def merge_weights(self):
        """将LoRA权重合并到原始层中"""
        merged_weight = self.linear.weight.data + self.scaling * (
            self.lora_up.weight @ self.lora_down.weight
        )
        return merged_weight

def apply_lora(model, layer_names=None, rank=8, alpha=32, dropout=0.0):
    """
    将模型中的线性层替换为LoRALinear
    Args:
        model: 要修改的模型
        layer_names: 要替换的层名列表（支持部分匹配），None表示所有Linear层
        rank: LoRA秩
        alpha: 缩放因子
        dropout: LoRA dropout率
    """
    # 如果未指定层名，则应用到所有Linear层
    if layer_names is None:
        layer_names = []
        for name, _ in model.named_modules():
            if isinstance(_, nn.Linear):
                layer_names.append(name)
    
    # 递归替换层
    for name, module in model.named_children():
        # 完整层名路径
        full_name = f"{name}"
        
        # 检查是否需要替换
        should_replace = any(key in full_name for key in layer_names)
        
        if should_replace and isinstance(module, nn.Linear):
            # 替换为LoRALinear
            setattr(model, name, LoRALinear(module, rank, alpha, dropout))
        elif len(list(module.children())) > 0:
            # 递归处理子模块
            apply_lora(module, layer_names, rank, alpha, dropout)
    
    return model

def verify_frozen_parameters(model,filter_name="lora_"):
    trainable_params = 0
    all_params = 0
    """验证所有原始参数是否被冻结"""
    for name, param in model.named_parameters():
        param_size = param.numel()
        all_params += param_size
        if filter_name not in name and param.requires_grad:
            # print(f"警告: 原始参数 {name} 仍然可训练!")
            param.requires_grad = False  # 强制冻结
        
        if param.requires_grad:
            trainable_params +=param_size

    print("所有原始参数已冻结")
    print(f"可训练参数总量: {trainable_params}, 总参数总量: {all_params}, 占比计算： {trainable_params*100.0/all_params:.3f}")

def extract_lora_state_dict(model):
    """提取LoRA层的状态字典"""
    lora_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            prefix = name + "."
            lora_state_dict[prefix + "lora_down.weight"] = module.lora_down.weight
            lora_state_dict[prefix + "lora_up.weight"] = module.lora_up.weight
    return lora_state_dict

def load_lora_state_dict(model, state_dict):
    """加载LoRA状态字典到模型"""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            prefix = name + "."
            module.lora_down.weight = nn.Parameter(state_dict[prefix + "lora_down.weight"])
            module.lora_up.weight = nn.Parameter(state_dict[prefix + "lora_up.weight"])
    return model


class Bottleneck_Conv(nn.Module):
    def __init__(self, in_channels, mult=0.5, kernel_size=3,dropout=0.05):
        super(Bottleneck_Conv, self).__init__()
        bottle_dim = int( in_channels*mult)
        self.norm0 = nn.GroupNorm(num_groups=in_channels//32, num_channels=in_channels,eps=EPSILON)
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv2d(in_channels, bottle_dim, kernel_size=kernel_size, padding=kernel_size//2,bias=False)
        self.norm1 = nn.GroupNorm(num_groups=bottle_dim//32, num_channels=bottle_dim,eps=EPSILON)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(bottle_dim, in_channels, kernel_size=kernel_size, padding=kernel_size//2,bias=False) 

        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)


    def forward(self, x):

        out = self.norm0(x)
        out = self.dropout(out)

        out = self.conv1(out)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)

        return x + out


class VGGT4DGSModel(torch.nn.Module):
    def __init__(self, sh_degree, height, width, min_depth, max_depth):
        super(VGGT4DGSModel, self).__init__()
        self.img_size = max(height, width)
        self.height = height
        self.width = width
        self.patch_size = 14
        self.vggt_embed_dim = 1024
        self.gs_dim = 256
        self.depth_dim = 128

        self.sh_degree = sh_degree
        self.min_depth = min_depth
        self.max_depth = max_depth

        self.num_cams = 6

        vggt_model = VGGT()   
        # VGGT_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        # vggt_model.load_state_dict(torch.hub.load_state_dict_from_url(VGGT_URL))
        vggt_path = os.environ.get('VGGT_CHECKPOINT_PATH', '')
        if not vggt_path:
            raise ValueError('Set VGGT_CHECKPOINT_PATH before initializing VGGT4DGSModel.')
        vggt_model.load_state_dict(torch.load(vggt_path))

        # 应用LoRA到模型
        # self.aggregator = vggt_model.aggregator
        self.aggregator = apply_lora(vggt_model.aggregator,layer_names=['qkv','proj','fc1','fc2'], dropout=0.05)

        # self.depth_head = vggt_model.depth_head
        self.depth_feathead = DPTHead(dim_in=2 * self.vggt_embed_dim, features=self.depth_dim*2,output_dim=self.depth_dim)
        # self.depth_norm = nn.LayerNorm(self.depth_dim)
        self.depth_feat2map = nn.Sequential(
                nn.Linear(self.depth_dim,  32),
                nn.GELU(),
                nn.Linear(32, 2),
            )

        del vggt_model

        self.d_sh = (self.sh_degree + 1) ** 2

        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
        )
        for degree in range(1, self.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

        self.raw_gs_dim =  1 + 3 + 4 + 3*self.d_sh # opacity + scale + rot + d_sh

        self.gs_feathead = VGGT_DPT_GS_Head(
            dim_in=2 * self.vggt_embed_dim,
            img_dim_in=3,
            output_dim=self.gs_dim,
            features=self.gs_dim,
        )
        # self.gs_norm = nn.LayerNorm(self.gs_dim)
        self.gs_feat2params = nn.Sequential(
                nn.Linear(self.gs_dim,  self.raw_gs_dim*2),
                nn.GELU(),
                nn.Linear(self.raw_gs_dim*2, self.raw_gs_dim),
            )
        

        self.dpt_dim = self.depth_dim + self.gs_dim
        self.dpt_morm = nn.LayerNorm(self.dpt_dim)
        self.dpt_feat_proj2d = nn.Sequential(
            nn.Linear(self.dpt_dim + 5, self.dpt_dim*2),
            nn.GELU(),
        )

        self.dfmb_attn_head = DeformableAttention3D(
            embed_dim=self.dpt_dim,
            head_dim=self.dpt_dim//8,
            num_heads=8,
            pts_dim=5,
            num_points=4,
            pos_emb_ratio=0.0,
        )

        # nn.init.zeros_(self.dpt_feat_proj2d[-1].weight)
        # if self.dpt_feat_proj2d[-1].bias is not None:
        #     nn.init.zeros_(self.dpt_feat_proj2d[-1].bias)

        # self.dpt_feat_proj2d = nn.Linear(self.depth_dim+self.gs_dim+self.pts_dim, self.dpt_dim*2)
        # self.dpt_norm = nn.LayerNorm(self.dpt_dim)
        # self.dpt_feat_refine = nn.Sequential(
        #     Bottleneck_Conv(self.dpt_dim),
        #     Bottleneck_Conv(self.dpt_dim),
        #     # Bottleneck_Conv(self.dpt_dim),
        #     # nn.GELU(),
        #     nn.Conv2d(self.dpt_dim, self.depth_dim+self.gs_dim,kernel_size=3, padding=1),
        # )
        # nn.init.zeros_(self.dpt_feat_refine[-1].weight)



        # 设置前self.depth_dim+self.gs_dim个通道的恒等映射
        for i in range(self.depth_dim + self.gs_dim):
            # 设置3x3卷积核的中心为1，其他为0
            # 权重形状: (out_channels, in_channels, kernel_h, kernel_w)
            self.dpt_feat_refine[-1].weight.data[i, i, 1, 1] = 1.0  # 中心位置设为1

        # 初始化偏置为0（如果有的话）
        if self.dpt_feat_refine[-1].bias is not None:
            nn.init.zeros_(self.dpt_feat_refine[-1].bias)


        verify_frozen_parameters(self.aggregator)
        # verify_frozen_parameters(self.aggregator,filter_name="all_weight_should_freeze") # 
        # verify_frozen_parameters(self.depth_feathead,filter_name="all_weight_should_freeze")
        # verify_frozen_parameters(self.depth_feat2map,filter_name="all_weight_should_freeze")
        # verify_frozen_parameters(self.gs_feathead,filter_name="all_weight_should_freeze")
        # verify_frozen_parameters(self.gs_feat2params,filter_name="all_weight_should_freeze")
        
    def forward(self, context_images, context_cameras):
        '''
        images: Batch_size, frame_num*view_num, 3, H, W
        context_Ks: Batch_size, frame_num*view_num, 4, 4
        context_ego2cam_extrs: Batch_size, frame_num*view_num, 4, 4
        context_frames: frame_num
        render_Ks: Batch_size, frame_num, view_num, 4, 4
        render_ego2cam_extrs: Batch_size, frame_num, view_num, 4, 4
        render_frames: Batch_size, frame_num
        '''

        context_Ks, context_e2c_extrs = context_cameras

        # depth_maps: Batch_size, frame_num*view_num, H, W, 1
        # dpt_feat: Batch_size, frame_num*view_num, H, W, D
        dpt_feat = self.feature_extraction(tensor_contiguous(context_images)) 

        xyz_maps, rot_maps, scale_maps, opacity_maps, sh_maps, xyz_conf = self.get_all_kernel_maps(dpt_feat,context_Ks,context_e2c_extrs)
        dpt_feat = rearrange(dpt_feat, 'b (f c) h w d -> b f c (h w) d',c=self.num_cams)
        return dpt_feat, xyz_maps, rot_maps, scale_maps, opacity_maps, sh_maps, xyz_conf

    def feature_extraction(self, images):
        # with torch.no_grad():

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        depth_feat = self.depth_feathead(
            aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
        )

        gs_feat = self.gs_feathead(
            aggregated_tokens_list,
            images=images,
            patch_start_idx=patch_start_idx,
        )  # batch_size, view_num, H , W, D`

        return torch.cat([depth_feat, gs_feat],dim=-1)
    
    def get_gs_params(self, context_features):
        '''
        context_features: [batch_size, ..., embed_dim]
        context_pts: [batch_size, ..., 4]
        '''

        raw_gaussian = self.gs_feat2params(tensor_contiguous(context_features))

        rot_maps, scale_maps, opacity_maps, sh_maps = raw_gaussian.split((4, 3, 1, 3 * self.d_sh), dim=-1)
        rot_maps = rot_maps.clamp(min=EPSILON)
        rot_maps = rot_maps / (rot_maps.norm(dim=-1, keepdim=True))
        scale_maps = nn.functional.softplus(scale_maps,beta=1) * 0.005
        opacity_maps = nn.functional.sigmoid(opacity_maps)

        sh_maps = rearrange(sh_maps, "... (i c) -> ... i c",i=3)
        sh_maps = sh_maps * self.sh_mask
        sh_maps = rearrange(sh_maps, "... i c -> ... c i")

        return rot_maps, scale_maps, opacity_maps, sh_maps

    def get_depth_maps(self, depth_feat):
        '''
        depth_feat: batch_size, view_num,  H , W, D
        '''

        depth_maps = self.depth_feat2map(tensor_contiguous(depth_feat)) 

        depth_maps = torch.nn.functional.sigmoid(depth_maps)
        depth_map, depth_conf = depth_maps.split([1,1],dim=-1)

        min_depth = self.min_depth
        max_depth = self.max_depth
        depth_range = max_depth-min_depth
        depth_map = min_depth + depth_range * depth_map

        return depth_map, depth_conf

    def get_all_kernel_maps(self, context_dpt_feat, context_Ks, context_e2c_extrs):
        '''
        context_dpt_feat: batch_size, frame_num*view_num, H, W, D
        context_Ks: batch_size, frame_num*view_num, 4, 4
        context_e2c_extrs: batch_size, frame_num*view_num, 4, 4
        '''
        depth_feat, gs_feat = context_dpt_feat.split((self.depth_dim,self.gs_dim),dim=-1)

        # depth_feat = self.depth_norm(depth_feat)
        depth_map, depth_conf = self.get_depth_maps(depth_feat)

        depth_conf = rearrange(depth_conf.squeeze(-1), 'b (f c) h w -> b f c h w',c=self.num_cams)
        # pts_maps: Batch_size, frame_num*view_num, (H*W), 3
        pts_maps = self.unproject_depth_map_to_point_map(depth_map, context_Ks, context_e2c_extrs)
        pts_maps = rearrange(pts_maps, 'b (f c) p d -> b f c p d',c=self.num_cams)

        gs_feat = rearrange(gs_feat, 'b (f c) h w d -> b f c (h w) d',c=self.num_cams)
        # gs_feat = self.gs_norm(gs_feat)
        rot_maps, scale_maps, opacity_maps, sh_maps = self.get_gs_params(gs_feat)
    
        return pts_maps, rot_maps, scale_maps, opacity_maps, sh_maps, depth_conf
    
    def pts4d_to_emb(self, pts_3d, delta_frame):
        '''
        pts_3d: [batch_size, frame_num, num_camera, num_ctx, 3]
        delta_frame: [batch_size, frame_num]
        '''

        pts_norm = self.xyz_norm(pts_3d)
        b,f,c,p,d = pts_norm.shape
        delta_frame_expand = delta_frame.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        delta_frame_expand = delta_frame_expand.expand(-1,-1,c,p,-1)
        cat_data = torch.cat([pts_norm, delta_frame_expand], dim=-1) # batch_size, frame_num,num_camera,num_ctx, 5

        return self.pts_proj(cat_data)
    
    def pts4d_with_delta_t(self, pts_norm, delta_frame):
        '''
        pts_norm: [batch_size, frame_num, num_camera, num_ctx, 4]
        delta_frame: [batch_size, frame_num]
        '''

        b,f,c,p,d = pts_norm.shape
        delta_frame_expand = delta_frame.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        delta_frame_expand = delta_frame_expand.expand(-1,-1,c,p,-1)
        return torch.cat([pts_norm, delta_frame_expand], dim=-1) # batch_size, frame_num,num_camera,num_ctx, 5
    
    def refine_dpt_feat_with_render_infos(self, context_dpt_feat, context_pts, render_Ks, render_e2c_extrs, render_relative_frames):
        '''
        context_dpt_feat: Batch_size, frame_ctx, num_camera, num_ctx, D
        context_pts_norm: Batch_size, frame_ctx, num_camera, num_ctx, 3
        context_frames: Batch_size, frame_ctx
        render_Ks: Batch_size, frame_render, num_camera, 4, 4
        render_e2c_extrs: Batch_size, frame_render, num_camera, 4, 4
        render_relative_frames: Batch_size, frame_render, frame_ctx
        '''

        batch_size, num_render_frames = render_relative_frames.shape[:2]
        cam_2dfeats = []
        context_pts_norm = self.xyz_norm(context_pts)

        for frame_idx in range(num_render_frames):
            
            context_4dcloud = self.pts4d_with_delta_t(context_pts_norm, render_relative_frames[:,frame_idx])
            context_query = context_dpt_feat.clone()

            context_query = rearrange(context_query, 'b f c n d -> b (f c n) d')
            context_4dcloud = rearrange(context_4dcloud, 'b f c p d -> b (f c p) d')

            # proj_2dcoord: B, num_camera, num_ctx, 2
            # proj_vaildmask: B, num_camera, num_ctx
            proj_2dcoord, proj_validmask = \
                self.project_points_map_to_pixel_coord(context_4dcloud,render_Ks[:,frame_idx],render_e2c_extrs[:,frame_idx])
            
            context_query = self.dpt_morm(context_query)
            context_query = torch.cat([context_query, context_4dcloud], dim=-1)

            context_query = self.dpt_feat_proj2d(context_query)

            for batch_idx in range(batch_size):
                for cam_idx in range(self.num_cams):
                    q_valid_mask = proj_validmask[batch_idx, cam_idx, :] # filter out invalid points
                    q_indices = torch.where(q_valid_mask)[0]

                    q_proj_2dcoord = proj_2dcoord[batch_idx, cam_idx, q_indices]  # num_points, 2
                    q_dpt_feat = context_data[batch_idx, q_indices]  # num_points, feat_dim

                    cam_2d_feat = self.project_2d_featuremap(q_dpt_feat, q_proj_2dcoord) # height, width, feat_dim
                    cam_2dfeats.append(cam_2d_feat)
                    
                    # 清理循环内变量
                    del q_valid_mask, q_indices, q_proj_2dcoord, q_dpt_feat, cam_2d_feat

            # 清理中间变量释放内存
            del context_pts_emb, context_data, proj_2dcoord, proj_validmask

        # 清理传入的大tensor
        del context_dpt_feat, context_pts
        
        fbc_2dfeats = torch.stack(cam_2dfeats, dim=0) # num_frame*batch_size*num_camera, height, width, feat_dim
        del cam_2dfeats  # 释放列表内存


        # fbc_2dfeats = self.dpt_norm(fbc_2dfeats) 
        fbc_2dfeats = tensor_contiguous(rearrange(fbc_2dfeats, 'b h w d -> b d h w'))
        fbc_2dfeats = self.dpt_feat_refine(fbc_2dfeats)
        
        render_dpt_feat = rearrange(fbc_2dfeats, '(f b c) d h w -> b (f c) h w d',f=num_render_frames,c=self.num_cams)
        del fbc_2dfeats  # 释放中间变量
        render_Ks = rearrange(render_Ks, 'b f c i j -> b (f c) i j')
        render_e2c_extrs = rearrange(render_e2c_extrs, 'b f c i j -> b (f c) i j')
        result = self.get_all_kernel_maps(render_dpt_feat, render_Ks, render_e2c_extrs)
        del render_dpt_feat, render_Ks, render_e2c_extrs  # 释放中间变量
        return result


    def spatial_temporal_attention(self, context_feat, context_4dcloud, proj_2dcoord, proj_validmask):

        context_query = self.dpt_morm(context_feat)
        context_query = torch.cat([context_query, context_4dcloud], dim=-1)

        context_query = self.dpt_feat_proj2d(context_query)

        batch_size, num_camera, num_ctx = proj_validmask.shape

        for batch_idx in range(batch_size):
            for cam_idx in range(self.num_cams):
                q_valid_mask = proj_validmask[batch_idx, cam_idx, :] # filter out invalid points
                q_indices = torch.where(q_valid_mask)[0]

                q_proj_2dcoord = proj_2dcoord[batch_idx, cam_idx, q_indices]  # num_points, 2
                q_dpt_feat = context_query[batch_idx, q_indices]  # num_points, feat_dim

                context_value_cam = self.project_2d_featuremap(q_dpt_feat, q_proj_2dcoord) # height, width, feat_dim

                q_dpt_feat = self.dfmb_attn_head(q_dpt_feat,query_pts)

    def project_2d_featuremap(self, query_feat, query_2dcoord):
        '''
        query_feat: num_query, feat_dim
        query_2dcoord: num_query, 2 [u, v] 其中 u 是宽度方向，v 是高度方向
        '''
        device = query_feat.device
        feat_dim = query_feat.size(1)

        # 将坐标转换为网格索引和偏移量
        # 注意：这里 query_2dcoord 应该是 [u, v]，其中 u 对应宽度，v 对应高度
        u, v = query_2dcoord[:, 0], query_2dcoord[:, 1]
        
        # 网格中心在 (i+0.5, j+0.5)，所以我们需要调整坐标
        # 注意：v 对应行（高度），u 对应列（宽度）
        grid_v = v - 0.5  # 行坐标（高度方向）
        grid_u = u - 0.5  # 列坐标（宽度方向）
        
        # 计算四个最近网格点的索引和权重
        v0 = torch.floor(grid_v).long()  # 行索引
        u0 = torch.floor(grid_u).long()  # 列索引
        v1 = v0 + 1
        u1 = u0 + 1
        
        # 计算双线性插值权重
        w00 = (u1 - grid_u) * (v1 - grid_v)  # 左上角 (u0, v0)
        w10 = (grid_u - u0) * (v1 - grid_v)  # 右上角 (u1, v0)
        w01 = (u1 - grid_u) * (grid_v - v0)  # 左下角 (u0, v1)
        w11 = (grid_u - u0) * (grid_v - v0)  # 右下角 (u1, v1)
        
        # 确保索引在有效范围内
        v0 = torch.clamp(v0, 0, self.height - 1)  # 行索引范围
        u0 = torch.clamp(u0, 0, self.width - 1)  # 列索引范围
        v1 = torch.clamp(v1, 0, self.height - 1)
        u1 = torch.clamp(u1, 0, self.width - 1)
        
        # 初始化特征图和权重图
        feature_map = torch.zeros(self.height, self.width, feat_dim, dtype=query_2dcoord.dtype, device=device)
        weight_map = torch.zeros(self.height, self.width,dtype=query_2dcoord.dtype, device=device)
        
        # 使用index_put_高效地累加特征和权重
        # 注意索引顺序是 (行, 列)
        # 左上角 (v0, u0)
        feature_map.index_put_((v0, u0), query_feat * w00.unsqueeze(1), accumulate=True)
        weight_map.index_put_((v0, u0), w00.to(weight_map.dtype), accumulate=True)
        
        # 右上角 (v0, u1)
        feature_map.index_put_((v0, u1), query_feat * w10.unsqueeze(1), accumulate=True)
        weight_map.index_put_((v0, u1), w10.to(weight_map.dtype), accumulate=True)
        
        # 左下角 (v1, u0)
        feature_map.index_put_((v1, u0), query_feat * w01.unsqueeze(1), accumulate=True)
        weight_map.index_put_((v1, u0), w01.to(weight_map.dtype), accumulate=True)
        
        # 右下角 (v1, u1)
        feature_map.index_put_((v1, u1), query_feat * w11.unsqueeze(1), accumulate=True)
        weight_map.index_put_((v1, u1), w11.to(weight_map.dtype), accumulate=True)
        # 归一化：除以累积权重（避免除零）

        normalized_feature_map = feature_map / (weight_map.unsqueeze(-1).clamp(min=EPSILON))
        
        # 清理中间变量
        del feature_map, weight_map, u, v, grid_v, grid_u, v0, u0, v1, u1
        del w00, w10, w01, w11
        
        return normalized_feature_map


    def xyz_norm(self, xyz):
        '''
        xyz: B, num_ctx, 3
        return B, num_ctx, 4
        '''
        r = xyz.norm(dim=-1, keepdim=True)
        xyz_normed = xyz / r.clamp(min=EPSILON)
        r_norm = torch.log(r.clamp(min=EPSILON))
        return torch.cat([xyz_normed, r_norm], dim=-1)
    
    def xyz_denorm(self, xyz_normed):

        xyz_norm = xyz_normed[..., :-1]
        r_norm = xyz_normed[..., -1:]

        r = torch.exp(r_norm)
        xyz = r * xyz_norm
        return xyz

    def unproject_depth_map_to_point_map(self, depth_map, K, e2c_extr):
        b, fc, h, w, d = depth_map.shape

        btc_depth = rearrange(depth_map.squeeze(-1), "b fc h w -> (b fc) h w") 
        if len(K.shape)==5:
            btc_K = rearrange(K, "b f c i j -> (b f c) i j")
        else:
            btc_K = rearrange(K, "b fc i j -> (b fc) i j")
        if len(e2c_extr.shape)==5:
            btc_e2c = rearrange(e2c_extr, "b f c i j -> (b f c) i j")
        else:
            btc_e2c = rearrange(e2c_extr, "b fc i j -> (b fc) i j")
        pts3d = depth2pc(btc_depth, btc_e2c, btc_K)  # pts3d: b*fc, h*w, 3
        pts3d = rearrange(pts3d, "(b fc) p d -> b fc p d",b=b, fc=fc)
        
        # 清理中间变量
        del btc_depth, btc_K, btc_e2c
        return pts3d

    # def spatial_cross_attention(self, query_feat, query_pos, key_feat, key_pos,):

    def project_points_map_to_pixel_coord(self, point_map_with_t, K, e2c_extr):
        '''
        point_map: Batch_size, frame_ctx, num_camera, num_ctx, 5
        K: B, num_camera, 4, 4 (only use [:, :, :3, :3])
        e2c_extr: B, num_camera, 4, 4

        '''

        point_map = point_map_with_t[..., :4].clone()
        point_map = self.xyz_denorm(point_map)

        # 扩展点坐标增加齐次坐标维度
        pts_homogeneous = torch.cat([point_map, torch.ones_like(point_map[..., :1])], dim=-1)  # B, num_ctx, 4

        # 转换为相机坐标系: P_cam = E * P_world
        pts_expand = pts_homogeneous.unsqueeze(1).unsqueeze(-1)  # B, 1, num_ctx,  4, 1
        e2c_extr_expand = e2c_extr.unsqueeze(2)   # B, num_camera, 1,  4, 4

        pts_cam_coords = torch.matmul(e2c_extr_expand[..., :3, :], pts_expand).squeeze(-1)  # B, num_camera, num_ctx, 3

        # 应用内参矩阵进行投影: pixel_coords = K * camera_coords
        K_expand = K.unsqueeze(2)  # B, num_camera, 1, 4, 4

        pixel_coords = torch.matmul(K_expand[..., :3, :3], pts_cam_coords.unsqueeze(-1)).squeeze(-1)  # B, num_camera, num_ctx, 3
        
        # 转换为像素坐标 (u, v)

        pixel_coords_2d = pixel_coords[..., :2] / (pixel_coords[..., 2:3].clamp(min=EPSILON))  # B, num_camera, num_ctx, 2

        # 创建有效点掩码
        # 检查条件1: z坐标必须为正(点在相机前方)
        valid_z = pts_cam_coords[..., 2] > 0
        
        # coords normalize
        # pixel_coords_2d[..., 0] = pixel_coords_2d[..., 0] / self.width
        # pixel_coords_2d[..., 1] = pixel_coords_2d[..., 1] / self.height

        # 检查条件2: 投影点必须在图像范围内
        valid_u = (pixel_coords_2d[..., 0] >= 0) & (pixel_coords_2d[..., 0] < self.width)
        valid_v = (pixel_coords_2d[..., 1] >= 0) & (pixel_coords_2d[..., 1] < self.height)
        
        # 综合有效掩码
        valid_mask = valid_z & valid_u & valid_v  # B, num_camera, num_ctx

        # 清理中间变量
        del pts_homogeneous, pts_expand, e2c_extr_expand, pts_cam_coords, K_expand, pixel_coords
        del valid_z, valid_u, valid_v
        
        return pixel_coords_2d, valid_mask


if __name__ == "__main__":
    device = 'cuda:1'
    model = VGGT4DGSModel(sh_degree=4).to(device)
    x = torch.randn(1, 6, 3, 280, 518).to(device)
    model(x)
