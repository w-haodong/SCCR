# -*- coding: utf-8 -*-
# models/ClickRefineNet.py
from __future__ import annotations
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClickCenterRefiner(nn.Module):
    def __init__(
            self,
            out_size: int = 64,
            base_channels: int = 32,
            num_layers: int = 4,
            return_patches: bool = True,
            use_coord_conv: bool = True,
            in_channels_feat: int = 64
    ):
        super().__init__()
        self.out_size = out_size
        self.return_patches = return_patches
        self.use_coord_conv = use_coord_conv
        self.in_channels_feat = in_channels_feat

        # --- 构建共享骨干网络 (Backbone) ---
        # 输入通道组成:
        # 1. Image Features (Default: 64)
        # 2. Heatmap (1)
        # 3. Corner Offsets (8)  <--- 新增
        # 4. CoordConv (Optional: 2)

        in_ch = in_channels_feat + 1 + 8

        if use_coord_conv:
            in_ch += 2

        layers = []
        ch = base_channels

        for i in range(num_layers):
            layers.append(nn.Conv2d(in_ch, ch, kernel_size=3, stride=2, padding=1))
            layers.append(nn.GroupNorm(4, ch))
            layers.append(nn.ReLU(inplace=True))

            in_ch = ch
            ch = min(ch * 2, 256)

        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc_center = nn.Linear(in_ch, 2)
        self.fc_corner = nn.Linear(in_ch, 8)

        # 初始化
        nn.init.constant_(self.fc_center.weight, 0)
        nn.init.constant_(self.fc_center.bias, 0)
        nn.init.constant_(self.fc_corner.weight, 0)
        nn.init.constant_(self.fc_corner.bias, 0)

    def _extract_roi_patches(
            self,
            feature_map: torch.Tensor,
            heatmap: torch.Tensor,
            offsets: torch.Tensor,  # <--- 新增参数 (B, 8, H, W)
            centers_feat: torch.Tensor,
            roi_half_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        通用采样函数，同时采样 Feature, Heatmap 和 Offsets
        """
        B, C_f, H, W = feature_map.shape
        _, C_h, _, _ = heatmap.shape
        _, C_o, _, _ = offsets.shape  # C_o Should be 8
        K = centers_feat.shape[1]
        S = self.out_size
        device = feature_map.device

        # 1. 构建基础采样网格
        y = torch.linspace(-1, 1, S, device=device)
        x = torch.linspace(-1, 1, S, device=device)
        mesh_y, mesh_x = torch.meshgrid(y, x, indexing='ij')
        base_grid = torch.stack([mesh_x, mesh_y], dim=-1)  # (S, S, 2)

        grid = base_grid.view(1, 1, S, S, 2).expand(B, K, S, S, 2)

        # 2. 处理 ROI 半径广播
        if roi_half_feat.dim() == 2:
            if roi_half_feat.shape[1] == 1:
                roi_half = roi_half_feat.view(B, 1, 1, 1, 1)
            else:
                roi_half = roi_half_feat.view(B, K, 1, 1, 1)
        else:
            roi_half = roi_half_feat.view(B, K, 1, 1, 1)

        # 3. 计算最终采样坐标 (Feature Space)
        centers_view = centers_feat.view(B, K, 1, 1, 2)
        sampling_grid_feat = centers_view + grid * roi_half

        # 4. 归一化到 [-1, 1] 用于 grid_sample
        sx = (sampling_grid_feat[..., 0] / (W - 1)) * 2.0 - 1.0
        sy = (sampling_grid_feat[..., 1] / (H - 1)) * 2.0 - 1.0
        sampling_grid_norm = torch.stack([sx, sy], dim=-1)
        flat_grid = sampling_grid_norm.view(B * K, S, S, 2)

        # 5. 执行采样
        # A. Image Features
        feat_exp = feature_map.repeat_interleave(K, dim=0)
        patches_feat = F.grid_sample(feat_exp, flat_grid, align_corners=True, padding_mode='zeros')

        # B. Heatmap
        hm_exp = heatmap.repeat_interleave(K, dim=0)
        patches_hm = F.grid_sample(hm_exp, flat_grid, align_corners=True, padding_mode='zeros')

        # C. Offsets (新增)
        offsets_exp = offsets.repeat_interleave(K, dim=0)
        patches_offsets = F.grid_sample(offsets_exp, flat_grid, align_corners=True, padding_mode='zeros')

        # 6. 拼接
        patches_out = torch.cat([patches_feat, patches_hm, patches_offsets], dim=1)

        return patches_out.view(B, K, C_f + C_h + C_o, S, S)

    def _forward_backbone(self, patches):
        B, K, C, S, _ = patches.shape
        x = patches.view(B * K, C, S, S)

        if self.use_coord_conv:
            x_range = torch.linspace(-1, 1, S, device=x.device)
            y_range = torch.linspace(-1, 1, S, device=x.device)
            yy, xx = torch.meshgrid(y_range, x_range, indexing='ij')
            xx = xx.expand(B * K, 1, S, S)
            yy = yy.expand(B * K, 1, S, S)
            coords = torch.cat([xx, yy], dim=1)
            x = torch.cat([x, coords], dim=1)

        feat = self.net(x)
        feat = self.pool(feat).flatten(1)
        return feat

    def forward(
            self,
            feature_in: torch.Tensor,
            hm_in: torch.Tensor,
            offsets_in: torch.Tensor,  # <--- 新增输入
            clicks_feat: torch.Tensor,
            roi_half_feat: torch.Tensor
    ) -> Dict[str, Any]:

        # STAGE 1: Click -> Center
        patches_1 = self._extract_roi_patches(feature_in, hm_in, offsets_in, clicks_feat, roi_half_feat)
        feat_1 = self._forward_backbone(patches_1)

        d_center = torch.tanh(self.fc_center(feat_1))  # (BK, 2)
        d_center = d_center.view(feature_in.shape[0], -1, 2)  # (B, K, 2)

        # 计算 Scale
        if roi_half_feat.dim() == 2:
            scale = roi_half_feat.unsqueeze(-1)
        else:
            scale = roi_half_feat

        pred_center_feat = clicks_feat + d_center * scale

        # STAGE 2: Center -> Corners
        patches_2 = self._extract_roi_patches(feature_in, hm_in, offsets_in, pred_center_feat, roi_half_feat)
        feat_2 = self._forward_backbone(patches_2)

        d_corners = torch.tanh(self.fc_corner(feat_2))  # (BK, 8)
        d_corners = d_corners.view(feature_in.shape[0], -1, 4, 2)

        ret = {
            "s2_d_center": d_center,
            "s2_d_corners": d_corners,
            "s2_pred_center_feat": pred_center_feat,
        }

        if self.return_patches:
            ret["s2_roi_feat_in"] = patches_2

        return ret