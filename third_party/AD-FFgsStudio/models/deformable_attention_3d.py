import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from einops import rearrange

from models.vggt.heads.utils import create_uv_grid, position_grid_to_embed

class DeformableAttention3D(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        head_dim: int = 32,
        num_heads: int = 6,
        pts_dim: int = 5,
        num_points: int = 4,
        pos_emb_ratio: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = head_dim
        self.pos_emb_ratio = pos_emb_ratio
        self.attn_dim = self.head_dim*self.num_heads
        
        # 线性变换层

        self.proj_in_value = nn.Linear(embed_dim, self.head_dim*self.num_heads)
        self.proj_in_query_feat = nn.Linear(embed_dim, embed_dim)
        self.proj_in_query_pts = nn.Sequential(
            nn.Linear(2, self.embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, embed_dim//2),
        )

        self.sampling_offsets = nn.Linear(embed_dim + embed_dim//2, num_heads * num_points * 3)  # 2D偏移量 + 1Dweights

        self.linear_out_feat = nn.Linear(self.head_dim*self.num_heads, embed_dim)  

        self.linear_out_pts = nn.Linear(self.head_dim*self.num_heads, embed_dim)  
    
    def forward(
        self,
        query_feat: torch.Tensor,           
        query_project_coords: torch.Tensor,  
        value: torch.Tensor,                  
    ) -> torch.Tensor:
        """
        Args:
            query_feat: 查询特征 [num_queries, embed_dim]
            query_project_coords: 查询投影坐标（用于2D引导，未归一化） [num_queries, 2]
            value: 图像特征 [height, width, embed_dim]
            
        Returns:
            attended_features: 注意力加权后的特征 [batch_size, num_queries, embed_dim]
        """

        # 输入数据的拉伸变换
        value_feat = self.proj_in_value(value)

        query_project_coords = query_project_coords / torch.tensor([[img_w, img_h]], device=query_project_coords.device)

        # 填充一维Batchsize （仅遵循惯例，其实无必要）
        query_pos_emb = self.proj_in_query_pts(query_project_coords)

        query_norm = torch.cat([query_feat,query_pos_emb],dim=-1).unsqueeze(0)

        img_h, img_w, img_d = value.shape
        pos_embed = create_uv_grid(img_w, img_h, aspect_ratio=img_w / img_h, dtype=v_feat.dtype, device=v_feat.device)
        pos_embed = position_grid_to_embed(pos_embed, img_d)
        v_feat = v_feat + pos_embed * self.pos_emb_ratio
        v_feat = v_feat.unsqueeze(0)
        
        # 2. 计算采样偏移量和注意力权重
        sampling_offsets_weights = self.sampling_offsets(query_norm)
        sampling_offsets_weights = rearrange(sampling_offsets_weights, 'b q (h p i) -> b q h p i',h=self.num_heads, p=self.num_points, i=3)
        # [bs, nq, num_heads, num_cams, num_points, 3]

        sampling_offsets, attention_weights = sampling_offsets_weights.split((2, 1), dim=-1)
        
        sampling_locations = query_project_coords.unsqueeze(2).unsqueeze(4) + sampling_offsets

        # 检测有效采样点 (在[0,1]范围内)
        valid_mask = (sampling_locations >= 0) & (sampling_locations <= 1)  # [bs, nq, num_heads, num_points, 2]
        valid_mask = valid_mask.all(dim=-1)  # [bs, nq, num_heads, num_points]

        # 将无效位置的权重设置为负无穷 (softmax后为0)
        attention_weights = attention_weights.masked_fill(~valid_mask, float('-inf'))

        # 重新应用softmax (自动忽略无效位置)
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # 6. 可变形采样和注意力计算
        attended_features = self.multi_views_deformable_attn(v_feat, sampling_locations, attention_weights)
        
        # 7. 合并多头输出
        attended_features = self.linear_out(attended_features) # [bs nq, embed_dim]

        return query_feat + attended_features.squeeze(0)

    def multi_views_deformable_attn(self, value, sampling_locations, attention_weights):
        """
        value: [bs, H, W, value_dim] - 多尺度特征值
        sampling_locations: [bs, num_query, num_head, num_points, 2] - 采样位置
        attention_weights: [bs, num_query, num_head, num_points] - 注意力权重
        """
        
        # 将采样位置从[0,1]归一化坐标转换为[-1,1] (grid_sample要求)
        sampling_grids = 2.0 * sampling_locations - 1.0
        
        value = rearrange(value, 'b h w d -> b d h w')
        
        # 重塑采样网格 [bs * num_heads, num_query, num_points, 2]
        sampling_grids = rearrange(sampling_grids, 'b q h p d -> (b h) q p d')
        
        # 双线性插值采样 
        sampled_value = F.grid_sample(
            value,  # [bs, value_dim, H, W]
            sampling_grids,  # [bs*num_head, num_query, num_points, 2]
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )  # 输出: [bs*num_head, value_dim, num_query, num_points]
        
        # 重塑采样结果
        sampled_value = rearrange(sampled_value, '(b h) d q p -> b q h p d', h=self.num_head)

        # 应用注意力权重
        output = sampled_value * attention_weights.unsqueeze(-1)  # 逐点相乘
        output = output.sum(dim=-2)  # 在num_points维度求和
        
        # 最终输出 [bs, num_query, num_heads * head_dim]
        output = rearrange(output, 'b q h d -> b q (h d)')
        
        return output

    def _apply_pos_embed(self, x: torch.Tensor, ratio: float = 0.1) -> torch.Tensor:
        """
        Apply positional embedding to tensor x.
        """
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=patch_w / patch_h, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed


