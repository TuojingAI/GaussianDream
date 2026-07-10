
import torch
import torch.nn as nn
import os
from einops import rearrange
try:
    from torch_scatter import scatter_add, scatter_max
except ImportError:
    scatter_add = None
    scatter_max = None

from models.vggt.models.vggt import VGGT
from models.vggt.heads.dpt_head import DPTHead
from models.vggt.heads.gs_dpt_head import VGGT_DPT_GS_Head




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

def verify_frozen_parameters(model):
    trainable_params = 0
    all_params = 0
    """验证所有原始参数是否被冻结"""
    for name, param in model.named_parameters():
        param_size = param.numel()
        all_params += param_size
        if "lora_" not in name and param.requires_grad:
            # print(f"警告: 原始参数 {name} 仍然可训练!")
            param.requires_grad = False  # 强制冻结
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



class VGGT3DGSModel(torch.nn.Module):
    def __init__(self, sh_degree, min_depth, max_depth):
        super(VGGT3DGSModel, self).__init__()
        self.img_size = 518
        self.patch_size = 14
        self.embed_dim = 1024
        self.sh_degree = sh_degree
        self.min_depth = min_depth
        self.max_depth = max_depth

        vggt_model = VGGT()   
        # VGGT_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        # vggt_model.load_state_dict(torch.hub.load_state_dict_from_url(VGGT_URL))

        ckpt_path = os.environ.get("VGGT_CHECKPOINT_PATH", "")
        if os.path.exists(ckpt_path):
            print(f"Loading VGGT checkpoint from {ckpt_path}")
            vggt_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        else:
            print(f"Warning: Checkpoint not found at {ckpt_path}, using random initialization")

        # 应用LoRA到模型
        # self.aggregator = vggt_model.aggregator
        self.aggregator = apply_lora(vggt_model.aggregator,layer_names=['qkv','proj','fc1','fc2'], dropout=0.05)
        verify_frozen_parameters(self.aggregator)
        # self.depth_head = DPTHead(dim_in=2 * self.embed_dim, output_dim=2, activation="sigmoid", conf_activation="expp1")
        self.depth_head = vggt_model.depth_head
        
        # verify_frozen_parameters(self.depth_head)
        del vggt_model

        self.d_sh = (self.sh_degree + 1) ** 2

        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

        self.raw_gs_dim =  1 + 3 + 4 + 3*self.d_sh # opacity + scale + rot + d_sh

        self.gs_head = VGGT_DPT_GS_Head(
            dim_in=2 * self.embed_dim,
            img_dim_in=3,
            output_dim=self.raw_gs_dim,
            pos_embed=False,
            # activation="linear",
            # conf_activation="expp1",
        )

    def forward(self, images):
        '''
        images: Batch_size, view_num, 3, H, W
        e2c_extr: Batch_size, view_num, 4, 4
        K: Batch_size, view_num, 4, 4
        '''

        # with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            aggregated_tokens_list, patch_start_idx = self.aggregator(images.to(torch.bfloat16))
            # depth_imgs = depth_maps.permute(0,1,4,2,3).detach()
            # depth_conf_imgs = depth_conf.unsqueeze(2).detach()

        with torch.amp.autocast("cuda", enabled=False):

            depth_maps, depth_conf = self.depth_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            )

            depth_maps = torch.nn.functional.sigmoid(torch.log(depth_maps))

            # images_depth = torch.cat([images,depth_imgs,depth_conf_imgs],axis=2)
            min_depth = self.min_depth
            max_depth = self.max_depth
            depth_range = max_depth-min_depth
            depth_maps = min_depth + depth_range * depth_maps

            raw_gaussian = self.gs_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
            )  # batch_size, view_num,  H , W, D

            rot_maps, scale_maps, opacity_maps, sh_maps = raw_gaussian.split((4, 3, 1, 3 * self.d_sh), dim=-1)
            # depth_scale = depth_scale.clamp(min=-1)
            # depth_scale = (depth_scale.mean(dim=[2,3],keepdims=True)+1) * 25

            # depth_maps = depth_scale * depth_maps
            
            rot_maps = rot_maps / (rot_maps.norm(dim=-1, keepdim=True) + 1e-8)
            scale_maps = nn.functional.softplus(scale_maps,beta=1) * 0.001
            opacity_maps = nn.functional.sigmoid(opacity_maps)
            
            sh_maps = rearrange(sh_maps, "b n h w (i c) -> b n h w i c",i=3)
            sh_maps = sh_maps * self.sh_mask
            # print(depth_maps.shape,rot_maps.shape,scale_maps.shape,opacity_maps.shape,sh_maps.shape)
        return depth_maps, rot_maps, scale_maps, opacity_maps, sh_maps, aggregated_tokens_list, patch_start_idx

    
    def voxelizaton_with_fusion(self, img_feat, pts3d, voxel_size, conf=None):
        if scatter_add is None or scatter_max is None:
            raise ImportError("torch_scatter is not installed, but is required for voxelizaton_with_fusion")
        # img_feat: V, C, H, W
        # pts3d: V, H* W, 3
        V, C, H, W = img_feat.shape
        pts3d_flatten = pts3d.flatten(0, 2)

        voxel_indices = (pts3d_flatten / voxel_size).round().int()  # [B*V*N, 3]
        unique_voxels, inverse_indices, counts = torch.unique(
            voxel_indices, dim=0, return_inverse=True, return_counts=True
        )

        # Flatten confidence scores and features
        conf_flat = conf.flatten()  # [B*V*N]
        anchor_feats_flat = img_feat.permute(0, 2, 3, 1).flatten(0, 2)  # [B*V*N, ...]

        # Compute softmax weights per voxel
        conf_voxel_max, _ = scatter_max(conf_flat, inverse_indices, dim=0)
        conf_exp = torch.exp(conf_flat - conf_voxel_max[inverse_indices])
        voxel_weights = scatter_add(
            conf_exp, inverse_indices, dim=0
        )  # [num_unique_voxels]
        weights = (conf_exp / (voxel_weights[inverse_indices] + 1e-6)).unsqueeze(
            -1
        )  # [B*V*N, 1]

        # Compute weighted average of positions and features
        weighted_pts = pts3d_flatten * weights
        weighted_feats = anchor_feats_flat.squeeze(1) * weights

        # Aggregate per voxel
        voxel_pts = scatter_add(
            weighted_pts, inverse_indices, dim=0
        )  # [num_unique_voxels, 3]
        voxel_feats = scatter_add(
            weighted_feats, inverse_indices, dim=0
        )  # [num_unique_voxels, feat_dim]

        return voxel_pts, voxel_feats


if __name__ == "__main__":
    device = 'cuda:1'
    model = VGGT3DGSModel(sh_degree=4).to(device)
    x = torch.randn(1, 6, 3, 280, 518).to(device)
    model(x)


