# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from .dpt_head import DPTHead
from .track_modules.base_track_predictor import BaseTrackerPredictor


class Mlp(nn.Module):
    def __init__(self, in_dim: int, hidden_dim=None, out_dim=None, act_layer=nn.GELU):
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class TrackHead_flow(nn.Module):
    """
    Track head that uses DPT head to process tokens and BaseTrackerPredictor for tracking.
    The tracking is performed iteratively, refining predictions over multiple iterations.
    """

    def __init__(
        self,
        dim_in,
        patch_size=14,
        features=128,
        iters=4,
        predict_conf=True,
        stride=2,
        corr_levels=7,
        corr_radius=4,
        hidden_size=384,
        num_motion_tokens=-1,
    ):
        """
        Initialize the TrackHead module.

        Args:
            dim_in (int): Input dimension of tokens from the backbone.
            patch_size (int): Size of image patches used in the vision transformer.
            features (int): Number of feature channels in the feature extractor output.
            iters (int): Number of refinement iterations for tracking predictions.
            predict_conf (bool): Whether to predict confidence scores for tracked points.
            stride (int): Stride value for the tracker predictor.
            corr_levels (int): Number of correlation pyramid levels
            corr_radius (int): Radius for correlation computation, controlling the search area.
            hidden_size (int): Size of hidden layers in the tracker network.
        """
        super().__init__()

        self.patch_size = patch_size

        # Feature extractor based on DPT architecture
        # Processes tokens into feature maps for tracking
        self.feature_extractor = DPTHead(
            dim_in=dim_in,
            patch_size=patch_size,
            features=features,
            feature_only=True,  # Only output features, no activation
            down_ratio=2,  # Reduces spatial dimensions by factor of 2
            pos_embed=False,
        )

        # output forward flow directly
        self.iters = iters


        projected_motion_dim = 32
        num_velocity_channels = 3
        self.motion_key_head = Mlp(128, 256, projected_motion_dim)   
        self.motion_basis_decoder = Mlp(projected_motion_dim, 256, num_velocity_channels)
        self.grad_checkpointing = False
        self.num_motion_tokens = num_motion_tokens

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
            LayerNorm2d(128),
            nn.GELU()
        )
        

    def unpatchify(self, x, hw=None, channel_first=True, patch_size=None) -> torch.Tensor:
        hw = hw or self.img_size
        imgs = rearrange(
            x,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            p1=self.patch_size if patch_size is None else patch_size,
            p2=self.patch_size if patch_size is None else patch_size,
            h=hw[0] // (self.patch_size if patch_size is None else patch_size),
            w=hw[1] // (self.patch_size if patch_size is None else patch_size),
        )
        if not channel_first:
            imgs = rearrange(imgs, "b c h w -> b h w c")
        return imgs

    def forward_motion_predictor(self, img_embeds, motion_tokens=None):
        b, tv, c, h, w = img_embeds.shape
        if self.grad_checkpointing:
            img_embeds = checkpoint(self.output_upscaling, img_embeds)
        else:
            img_embeds = self.output_upscaling(img_embeds.view(-1, c, h, w)).view(b, tv, c, h*2, w*2)
        
        img_embeds = rearrange(img_embeds, "b tv c h w -> (b tv) (h w) c")
        img_keys = self.motion_key_head(img_embeds)

        if self.num_motion_tokens > 0:
            hyper_in_list = []
            for i in range(self.num_motion_tokens):
                hyper_in = self.motion_query_heads[i](motion_tokens[:, i])
                hyper_in_list.append(hyper_in)
            motion_token_queries = torch.stack(hyper_in_list, dim=1)
            motion_bases = self.motion_basis_decoder(motion_tokens)
            dot_product_similarity = torch.einsum(
                "b k c, b t v h w c -> b t v h w k",
                motion_token_queries,
                img_keys,
            )
            motion_weights = torch.softmax(dot_product_similarity / self.tau, dim=-1)
            forward_flow = torch.einsum(
                "b t v h w k, b k c -> b t v h w c", motion_weights, motion_bases
            )
        else:
            # if there's no motion token, directly predict the velocity from the upsampled image features
            forward_flow = self.motion_basis_decoder(img_keys)
        
        forward_flow = rearrange(forward_flow, "(b tv) (h w) c -> b tv h w c", b=b, h=h*2, w=w*2)
        return forward_flow


    def forward(self, aggregated_tokens_list, images, patch_start_idx, 
                query_points=None, iters=None, gs_params=None, motion_tokens=None):
        """
        Forward pass of the TrackHead.

        Args:
            aggregated_tokens_list (list): List of aggregated tokens from the backbone.
            images (torch.Tensor): Input images of shape (B, S, C, H, W) where:
                                   B = batch size, S = sequence length.
            patch_start_idx (int): Starting index for patch tokens.
            query_points (torch.Tensor, optional): Initial query points to track.
                                                  If None, points are initialized by the tracker.
            iters (int, optional): Number of refinement iterations. If None, uses self.iters.

        Returns:
            tuple:
                - coord_preds (torch.Tensor): Predicted coordinates for tracked points.
                - vis_scores (torch.Tensor): Visibility scores for tracked points.
                - conf_scores (torch.Tensor): Confidence scores for tracked points (if predict_conf=True).
        """
        B, S, _, H, W = images.shape # [4, 6, 3, 280, 518]
        # Extract features from tokens
        # feature_maps has shape (B, S, C, H//2, W//2) due to down_ratio=2
        feature_maps = self.feature_extractor(aggregated_tokens_list, images, patch_start_idx) # [4, 6, 128, 140, 259]

        # Use default iterations if not specified
        if iters is None:
            iters = self.iters
        forward_flow = self.forward_motion_predictor(feature_maps, motion_tokens)
        return forward_flow
