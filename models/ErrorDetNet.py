# models/ErrorDetNet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Type, Dict, Any


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        return x.permute(0, 3, 1, 2)


class FeatureProcessingBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 norm_layer: Type[nn.Module] = LayerNorm2d,
                 act_layer: Type[nn.Module] = nn.SiLU):
        super().__init__()
        self.process = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            act_layer(),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,
                      bias=False, padding_mode='replicate'),
            norm_layer(out_channels),
            act_layer(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.process(x)


class TopDownFusionBlock(nn.Module):
    """
    若 top_down_feat 和 lateral_feat 空间尺寸不一致，先插值对齐再 concat
    """
    def __init__(self, top_down_channels: int, lateral_channels: int,
                 out_channels: int,
                 norm_layer: Type[nn.Module] = LayerNorm2d,
                 act_layer: Type[nn.Module] = nn.SiLU):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(top_down_channels + lateral_channels, out_channels,
                      kernel_size=1, bias=False),
            norm_layer(out_channels),
            act_layer(),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,
                      bias=False, padding_mode='replicate'),
            norm_layer(out_channels),
            act_layer(),
        )

    def forward(self, top_down_feat: torch.Tensor,
                lateral_feat: torch.Tensor) -> torch.Tensor:
        if top_down_feat.shape[2:] != lateral_feat.shape[2:]:
            top_down_feat = F.interpolate(
                top_down_feat, size=lateral_feat.shape[2:],
                mode='bilinear', align_corners=False
            )
        x = torch.cat([top_down_feat, lateral_feat], dim=1)
        return self.fusion_conv(x)


class err_det_net(nn.Module):
    """
    Stage1 heads 仍然输出：
      - pred_global_hm (logits)         (B,1,Ht,Wt)
      - pred_center_offsets            (B,2,Ht,Wt)
      - pred_corner_offsets            (B,8,Ht,Wt)
      - pred_spine_mask_logits (logits)(B,1,Ht,Wt)

    关键新增：
      - s1_content_map : Stage1 的任务特征图（target_stride） -> Stage2 取 ROI 用
      - s1_fused_map   : FPN 融合后的特征（sam_stride）      -> 可选多尺度
      - s1_pyramid_maps: 每层处理后的 P[i]（sam_stride）     -> 可选多尺度
    """
    def __init__(self,
                 args,
                 sam_feature_stride: float,
                 target_feature_stride: int,
                 num_vit_block_inputs: int,
                 in_channels_per_block: int,
                 processed_block_channels: int,
                 top_down_fused_channels: int,
                 final_feature_channels: int,
                 node_feature_dim: int,
                 rnn_input_dim: int,
                 rnn_hidden_dim: int,
                 rnn_layers: int,
                 rnn_out_dim: int,
                 act_layer=nn.SiLU,
                 norm_layer=LayerNorm2d):
        super().__init__()
        self.args = args
        self.K = int(args.K)
        self.num_vit_block_inputs = int(num_vit_block_inputs)
        if self.num_vit_block_inputs < 1:
            raise ValueError("vit_input_layer_indices must provide at least one index.")

        self.target_feature_stride = int(target_feature_stride)
        self.sam_feature_stride = float(sam_feature_stride)
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        self.final_feature_channels = int(final_feature_channels)
        self.top_down_fused_channels = int(top_down_fused_channels)
        self.processed_block_channels = int(processed_block_channels)

        # ---------------- FPN ----------------
        self.feature_processors = nn.ModuleList([
            FeatureProcessingBlock(in_channels_per_block, processed_block_channels,
                                   norm_layer=norm_layer, act_layer=act_layer)
            for _ in range(self.num_vit_block_inputs)
        ])

        self.top_down_fusion_modules = nn.ModuleList([
            TopDownFusionBlock(top_down_fused_channels,
                               processed_block_channels,
                               top_down_fused_channels,
                               norm_layer=norm_layer,
                               act_layer=act_layer)
            for _ in range(self.num_vit_block_inputs - 1)
        ])

        if processed_block_channels != top_down_fused_channels:
            self.initial_top_path_adapter = nn.Sequential(
                nn.Conv2d(processed_block_channels, top_down_fused_channels,
                          kernel_size=1, bias=False),
                norm_layer(top_down_fused_channels),
                act_layer()
            )
        else:
            self.initial_top_path_adapter = nn.Identity()

        # ---------------- Content path (sam_stride -> target_stride) ----------------
        ratio = self.sam_feature_stride / float(self.target_feature_stride)
        if abs(ratio - round(ratio)) > 1e-6:
            raise ValueError(f"sam_stride/target_stride must be integer power-of-two ratio, got {ratio}")
        ratio = int(round(ratio))
        if ratio < 2:
            raise ValueError("target_feature_stride must be smaller than sam_feature_stride")
        if (ratio & (ratio - 1)) != 0:
            raise ValueError(f"sam_stride/target_stride must be power of 2, got {ratio}")

        num_2x_upsamples = int(np.log2(ratio))

        content_layers = []
        current_channels = int(top_down_fused_channels)
        for i in range(num_2x_upsamples):
            out_channels = current_channels // 2 if i < (num_2x_upsamples - 1) else self.final_feature_channels
            content_layers += [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1,
                          bias=False, padding_mode='replicate'),
                norm_layer(out_channels),
                act_layer(),
            ]
            current_channels = out_channels

        self.content_path = nn.Sequential(*content_layers)

        # ---------------- Heads ----------------
        def _make_head(in_c, out_c, head_dim):
            head = nn.Sequential(
                nn.Conv2d(in_c, head_dim, kernel_size=3, padding=1, bias=True),
                self.norm_layer(head_dim),
                self.act_layer(),
                nn.Conv2d(head_dim, out_c, kernel_size=1, stride=1, padding=0, bias=True),
            )
            head[-1].bias.data.fill_(0.0)
            return head

        self.hm_head = _make_head(self.final_feature_channels, 1, int(getattr(self.args, 'hm_head_dim', 256)))
        self.hm_head[-1].bias.data.fill_(-2.19)

        self.center_reg_head = _make_head(self.final_feature_channels, 2, int(getattr(self.args, 'center_reg_head_dim', 128)))
        self.corner_reg_head = _make_head(self.final_feature_channels, 8, int(getattr(self.args, 'corner_reg_head_dim', 128)))
        self.seg_head = _make_head(self.final_feature_channels, 1, int(getattr(self.args, 'seg_head_dim', 128)))

        # ---------------- Style path (可选) ----------------
        style_feature_channels = int(getattr(self.args, "style_feature_channels", max(8, self.final_feature_channels // 2)))
        self.style_path = nn.Sequential(
            nn.Conv2d(int(top_down_fused_channels), int(top_down_fused_channels), kernel_size=3, padding=1,
                      bias=False, padding_mode='replicate'),
            norm_layer(int(top_down_fused_channels)),
            act_layer(),
            nn.Conv2d(int(top_down_fused_channels), style_feature_channels, kernel_size=1, bias=True),
            norm_layer(style_feature_channels),
            act_layer(),
        )

        # ---------------- Connection RNN (detector only) ----------------
        self.conn_rnn = nn.LSTM(input_size=rnn_input_dim,
                                hidden_size=rnn_hidden_dim,
                                num_layers=rnn_layers,
                                bidirectional=True,
                                batch_first=True)
        self.conn_projector = nn.Linear(rnn_hidden_dim * 2, rnn_out_dim)

        # ---------------- Intrinsic/Conn streams ----------------
        intr_embed_dim = 64
        conn_embed_dim = 64

        self.num_shape_features = 8
        self.num_diff_features = 2
        self.num_hm_features = 1
        intr_input_dim = self.num_shape_features + self.num_diff_features + self.num_hm_features

        self.intr_stream = nn.Sequential(
            nn.Linear(intr_input_dim, intr_embed_dim),
            nn.ReLU(inplace=True),
        )
        self.intr_norm = nn.LayerNorm(intr_embed_dim)

        conn_input_dim = rnn_out_dim * 2
        self.conn_stream = nn.Sequential(
            nn.Linear(conn_input_dim, conn_embed_dim),
            nn.ReLU(inplace=True),
        )
        self.conn_norm = nn.LayerNorm(conn_embed_dim)

        fusion_input_dim = intr_embed_dim + conn_embed_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

        try:
            prior_prob = 0.06
            bias_value = -torch.log(torch.tensor((1.0 - prior_prob) / prior_prob))
            self.fusion_head[-1].bias.data.fill_(bias_value)
        except Exception:
            pass

        self.hm_logit_thr_feat = float(getattr(self.args, 'hm_logit_thr_feat', -3.0))

    # ---------- helpers ----------
    @staticmethod
    def _normalize_coords_on_feat(coords_feat_xy: torch.Tensor, feat_HW: Tuple[int, int]) -> torch.Tensor:
        Hf, Wf = feat_HW
        x = coords_feat_xy[..., 0]
        y = coords_feat_xy[..., 1]
        nx = (x / max(1.0, (Wf - 1))) * 2.0 - 1.0
        ny = (y / max(1.0, (Hf - 1))) * 2.0 - 1.0
        return torch.stack([nx, ny], dim=-1)

    @staticmethod
    def _sample_at_centers(features: torch.Tensor, centers_norm: torch.Tensor) -> torch.Tensor:
        B, C, Hf, Wf = features.shape
        grid = centers_norm.unsqueeze(1)  # (B,1,K,2)
        sampled = F.grid_sample(features.float(), grid, mode='bilinear',
                                align_corners=True, padding_mode='border')  # (B,C,1,K)
        sampled = sampled.squeeze(2).permute(0, 2, 1)  # (B,K,C)
        return sampled

    def _hm_confidence_logits(self, pred_hm_logits: torch.Tensor, centers_norm: torch.Tensor) -> torch.Tensor:
        r = max(0, int(getattr(self.args, 'hm_pool_radius_feat', 1)))
        pooled = F.max_pool2d(pred_hm_logits, kernel_size=2 * r + 1, stride=1, padding=r) if r > 0 else pred_hm_logits
        return self._sample_at_centers(pooled, centers_norm)  # (B,K,1)

    @staticmethod
    def _get_axes_for_diff(p_k42: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        p_k42: (B,K,4,2) order MUST be (tl,tr,bl,br)
        """
        p0 = p_k42[:, :, 0, :]
        p1 = p_k42[:, :, 1, :]
        p2 = p_k42[:, :, 2, :]
        p3 = p_k42[:, :, 3, :]

        mid_L = (p0 + p2) / 2.0
        mid_R = (p1 + p3) / 2.0
        mid_T = (p0 + p1) / 2.0
        mid_B = (p2 + p3) / 2.0

        long_axis_vec = mid_R - mid_L
        lat_axis_vec = mid_B - mid_T
        return long_axis_vec, lat_axis_vec

    # ---------- forward (standard) ----------
    def forward(self,
                vit_block_outputs: List[torch.Tensor],
                p_err: torch.Tensor = None,
                connection_features: torch.Tensor = None,
                intrinsic_shape_features: torch.Tensor = None,
                img_shape_HW: torch.Size = None) -> Dict[str, Any]:
        # 默认不进行任何消融，调用 ab_forward
        return self.ab_forward(vit_block_outputs, p_err, connection_features,
                               intrinsic_shape_features, img_shape_HW,
                               ablate_geo=False, ablate_topo=False)

    # ---------- ab_forward (for ablation study) ----------
    def ab_forward(self,
                   vit_block_outputs: List[torch.Tensor],
                   p_err: torch.Tensor = None,
                   connection_features: torch.Tensor = None,
                   intrinsic_shape_features: torch.Tensor = None,
                   img_shape_HW: torch.Size = None,
                   ablate_geo: bool = False,
                   ablate_topo: bool = False) -> Dict[str, Any]:
        """
        专用 forward 函数，用于消融实验。
        Args:
            ablate_geo (bool): 如果为 True，将几何(Intrinsic)特征强制置零。
            ablate_topo (bool): 如果为 True，将拓扑(Connection)特征强制置零。
        """

        if (not isinstance(vit_block_outputs, list)) or (len(vit_block_outputs) != self.num_vit_block_inputs):
            raise ValueError(f"Expected {self.num_vit_block_inputs} features, got {len(vit_block_outputs)}")

        # ---------- FPN ----------
        P = [self.feature_processors[i](vit_block_outputs[i]) for i in range(self.num_vit_block_inputs)]
        fused = self.initial_top_path_adapter(P[-1])
        for i in range(self.num_vit_block_inputs - 2, -1, -1):
            idx = (self.num_vit_block_inputs - 2) - i
            fused = self.top_down_fusion_modules[idx](fused, P[i])

        # 分叉 content/style
        content_map = self.content_path(fused)   # (B, Cc, Ht, Wt)  target_stride
        style_map = self.style_path(fused)       # (B, Cs, Hs, Ws)  sam_stride

        B, Cc, Ht, Wt = content_map.shape

        # ---------- Heads ----------
        hm_logits = self.hm_head(content_map)
        pred_center_offsets = self.center_reg_head(content_map)
        pred_corner_offsets = self.corner_reg_head(content_map)
        pred_spine_mask_logits = self.seg_head(content_map)

        content_features = F.adaptive_avg_pool2d(content_map, output_size=1).view(B, -1)
        style_features = F.adaptive_avg_pool2d(style_map, output_size=1).view(B, -1)

        out: Dict[str, Any] = {
            'pred_global_hm': hm_logits,
            'pred_center_offsets': pred_center_offsets,
            'pred_corner_offsets': pred_corner_offsets,
            'pred_spine_mask_logits': pred_spine_mask_logits,
            'content_features': content_features,
            'style_features': style_features,

            # ===== Stage2 用的图像特征（关键）=====
            's1_content_map': content_map,   # target_stride
            's1_fused_map': fused,           # sam_stride
            's1_pyramid_maps': P,            # sam_stride (list)
            's1_target_stride': float(self.target_feature_stride),
            's1_sam_stride': float(self.sam_feature_stride),
        }

        # detector 分支需要监督特征才算
        if (p_err is None) or (connection_features is None) or (intrinsic_shape_features is None):
            return out

        B2, Npts, _ = p_err.shape
        K = self.K
        assert B2 == B, f"Batch mismatch: p_err B={B2} vs features B={B}"
        assert Npts == 4 * K, f"p_err.shape[1]={Npts}, expected 4*K={4*K}"

        eps = 1e-8
        dtype = p_err.dtype

        p_err_reshaped = p_err.view(B, K, 4, 2)  # order (tl,tr,bl,br)
        centers_img = torch.mean(p_err_reshaped, dim=2)
        centers_feat = centers_img / float(self.target_feature_stride)
        centers_norm = self._normalize_coords_on_feat(centers_feat, (Ht, Wt)).to(device=content_map.device, dtype=dtype)

        hm_conf_logits = self._hm_confidence_logits(hm_logits, centers_norm).to(dtype)

        # Connection RNN
        h_conn_seq, _ = self.conn_rnn(connection_features)
        h_conn = self.conn_projector(h_conn_seq)

        Dc = h_conn.shape[-1]
        zero = torch.zeros(B, 1, Dc, device=h_conn.device, dtype=h_conn.dtype)
        h_up = torch.cat([zero, h_conn], dim=1)
        h_down = torch.cat([h_conn, zero], dim=1)
        h_rnn = torch.cat([h_up, h_down], dim=2)

        # shape features
        h_shape_intrinsic = intrinsic_shape_features.to(dtype)

        long_axis_err, lat_axis_err = self._get_axes_for_diff(p_err_reshaped)

        sampled_center_offsets = self._sample_at_centers(pred_center_offsets, centers_norm)   # (B,K,2)
        sampled_corner_offsets = self._sample_at_centers(pred_corner_offsets, centers_norm)   # (B,K,8)

        centers_feat_refined = centers_feat + sampled_center_offsets.detach()
        pred_corners_feat = centers_feat_refined.unsqueeze(2) + sampled_corner_offsets.detach().view(B, K, 4, 2)
        pred_corners_abs = pred_corners_feat * float(self.target_feature_stride)

        long_axis_pred, lat_axis_pred = self._get_axes_for_diff(pred_corners_abs.detach())
        cos_sim_long = F.cosine_similarity(long_axis_err, long_axis_pred, dim=-1, eps=eps).clamp(-1.0, 1.0)
        cos_sim_lat = F.cosine_similarity(lat_axis_err, lat_axis_pred, dim=-1, eps=eps).clamp(-1.0, 1.0)

        dir_diff_long = torch.acos(cos_sim_long).unsqueeze(-1)
        dir_diff_lat = torch.acos(cos_sim_lat).unsqueeze(-1)
        h_shape_difference = torch.cat([dir_diff_long, dir_diff_lat], dim=-1).to(dtype)

        h_shape_intrinsic = torch.nan_to_num(h_shape_intrinsic, nan=0.0, posinf=0.0, neginf=0.0)
        h_shape_difference = torch.nan_to_num(h_shape_difference, nan=0.0, posinf=0.0, neginf=0.0)

        hm_flag_input = (hm_conf_logits < self.hm_logit_thr_feat).float()

        h_intr_input = torch.cat([hm_flag_input, h_shape_intrinsic, h_shape_difference], dim=2)
        h_intr_embed = self.intr_stream(h_intr_input)
        h_intr_norm = self.intr_norm(h_intr_embed)

        h_conn_embed = self.conn_stream(h_rnn.to(dtype))
        h_conn_norm = self.conn_norm(h_conn_embed)

        # =========================================================
        # Ablation Masking Logic
        # =========================================================
        if ablate_geo:
            h_intr_norm = torch.zeros_like(h_intr_norm)

        if ablate_topo:
            h_conn_norm = torch.zeros_like(h_conn_norm)
        # =========================================================

        h_fused = torch.cat([h_intr_norm, h_conn_norm], dim=2)
        error_logits = self.fusion_head(h_fused)

        out.update({
            'hm_confidence_feat': hm_conf_logits.detach(),
            'hm_flag': hm_flag_input.detach(),
            'error_logits': error_logits,
            'pred_corners_abs': pred_corners_abs.detach(),
        })
        return out


__all__ = ["err_det_net", "LayerNorm2d"]