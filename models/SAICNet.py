# -*- coding: utf-8 -*-
# models/SAICNet.py

from __future__ import annotations
import math
import numpy as np
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from .ErrorDetNet import err_det_net
from .DANN import GradientReversalLayer, DomainClassifier
from .ClickRefineNet import ClickCenterRefiner
from operation.decode import DecDecoder

# 【关键】直接复用您项目中的几何计算工具，保证和训练一致
try:
    from utils.geometry import calc_connection_features_from_err, calc_intrinsic_shape_features_np
except ImportError:
    print("[SAICNet] Warning: utils.geometry not found. Re-prediction might fail.")


    def calc_connection_features_from_err(pts, K):
        return np.zeros((K - 1, 2), dtype=np.float32)


    def calc_intrinsic_shape_features_np(pts):
        return np.zeros((len(pts) // 4, 4), dtype=np.float32)


class saic_net(nn.Module):
    def __init__(self, peft_encoder: nn.Module, args):
        super().__init__()
        self.args = args
        self.encoder = peft_encoder
        self.vit_input_layer_indices = args.vit_input_layer_indices

        # -------------------- Stage1 Config --------------------
        self.target_feature_stride = float(args.target_feature_stride)
        feature_stride = 16.0
        num_vit_block_inputs = len(self.vit_input_layer_indices)

        self.err_det_net = err_det_net(
            args=args,
            sam_feature_stride=feature_stride,
            target_feature_stride=args.target_feature_stride,
            num_vit_block_inputs=num_vit_block_inputs,
            in_channels_per_block=768,
            processed_block_channels=256,
            top_down_fused_channels=256,
            final_feature_channels=64,
            node_feature_dim=args.node_feature_dim,
            rnn_input_dim=args.rnn_input_dim,
            rnn_hidden_dim=args.rnn_hidden_dim,
            rnn_layers=args.rnn_layers,
            rnn_out_dim=args.rnn_out_dim
        )

        self.grl = GradientReversalLayer()
        self.domain_classifier = DomainClassifier(input_features=64)

        self._decoder = DecDecoder(
            K=int(args.K),
            candidate_topk=int(getattr(args, "decode_candidate_topk", 32)),
            conf_thresh=0.05,
            seg_thr=0.0,
        )

        self.K = int(args.K)

        # -------------------- Stage2 Config --------------------
        self.s2_topk_global = int(getattr(args, "s2_topk_global", 15))
        self.s2_R_default = float(getattr(args, "s2_R_default", 24.0))
        self.s2_R_min = float(getattr(args, "s2_R_min", 12.0))
        self.s2_R_max = float(getattr(args, "s2_R_max", 80.0))

        self.s2_roi_scale = float(getattr(args, "s2_roi_scale", 1.8))
        self.s2_roi_out = int(getattr(args, "s2_roi_out", 64))

        self.s2_click_frac = float(getattr(args, "s2_click_frac", 0.85))

        self.s2_detach_stage1_hm = bool(getattr(args, "s2_detach_stage1_hm", True))
        self.s2_return_patches = bool(getattr(args, "s2_return_patches", True))

        self.click_refiner = ClickCenterRefiner(
            out_size=self.s2_roi_out,
            base_channels=int(getattr(args, "s2_base_channels", 32)),
            num_layers=int(getattr(args, "s2_num_layers", 4)),
            return_patches=self.s2_return_patches,
            in_channels_feat=64
        )

    # -------------------- Helpers --------------------
    def _get_base_template(self, device):
        return torch.tensor([[-0.7, -0.7], [0.7, -0.7], [0.7, 0.7], [-0.7, 0.7]], device=device).view(1, 1, 4, 2)

    @staticmethod
    def _grid_sample_at_points(feat_bchw, xy_norm_bk2):
        grid = xy_norm_bk2.unsqueeze(1)
        samp = F.grid_sample(feat_bchw.float(), grid.float(), mode="bilinear", align_corners=True,
                             padding_mode="border")
        return samp.squeeze(2).permute(0, 2, 1)

    @staticmethod
    def _normalize_grid_xy(xy_feat, Hf, Wf):
        x = xy_feat[..., 0];
        y = xy_feat[..., 1]
        nx = x / max(1.0, float(Wf - 1)) * 2.0 - 1.0
        ny = y / max(1.0, float(Hf - 1)) * 2.0 - 1.0
        return torch.stack([nx, ny], dim=-1)

    @torch.no_grad()
    def _estimate_R_abs_from_stage1_topk(self, outputs):
        hm_prob = outputs["pred_global_hm"].detach().sigmoid()
        wh = outputs["pred_corner_offsets"].detach()
        B, _, Hf, Wf = hm_prob.shape
        xs, ys, sc = self._decoder.decode_peaks(hm_prob, topk=self.s2_topk_global, conf_thresh=0.01)
        centers_feat = torch.stack([xs, ys], dim=-1)
        centers_norm = self._normalize_grid_xy(centers_feat, Hf, Wf)
        wh_s = self._grid_sample_at_points(wh, centers_norm)
        d_mean = torch.norm(wh_s.view(B, -1, 4, 2), dim=-1).mean(dim=2)
        w = sc.clamp_min(0.0)
        denom = w.sum(dim=1, keepdim=True).clamp_min(1e-6)
        r_feat_avg = (d_mean * w).sum(dim=1, keepdim=True) / denom
        r_abs = r_feat_avg * self.target_feature_stride
        return torch.nan_to_num(r_abs, nan=self.s2_R_default).clamp(self.s2_R_min, self.s2_R_max).view(B, 1)

    # -------------------- Forward --------------------
    # 【修改 1】forward 中加入消融参数
    def forward(self, batch, phase="infer", alpha=None, stage2_grad=True,
                ablate_geo=False, ablate_topo=False, **kwargs):
        x = batch["input_image"]

        # 【修改 2】调用 err_det_net.ab_forward 并传入消融参数
        outputs = self.err_det_net.ab_forward(
            vit_block_outputs=self.encoder(x, self.vit_input_layer_indices),
            p_err=batch.get("p_err", None),
            connection_features=batch.get("connection_features", None),
            intrinsic_shape_features=batch.get("intrinsic_shape_features", None),
            img_shape_HW=x.shape[2:],
            ablate_geo=ablate_geo,  # 传入
            ablate_topo=ablate_topo  # 传入
        )

        if alpha is not None and "content_features" in outputs:
            rev = self.grl(outputs["content_features"], alpha)
            outputs["domain_logits"] = self.domain_classifier(rev)

        R_abs_b1 = self._estimate_R_abs_from_stage1_topk(outputs)

        if "p_gt" in batch:
            B = x.shape[0]
            pgt = batch["p_gt"].float().to(x.device).view(B, self.K, 4, 2)
            clicks_abs = pgt.mean(dim=2).unsqueeze(2)
        else:
            hm = outputs["pred_global_hm"].sigmoid()
            wh = outputs["pred_corner_offsets"]
            reg = outputs["pred_center_offsets"]
            pts = self._decoder.ctdet_decode(hm, wh, reg)
            if pts is not None:
                if isinstance(pts, np.ndarray): pts = torch.from_numpy(pts).to(x.device)
                if pts.dim() == 2: pts = pts.unsqueeze(0)
                clicks_abs = pts[..., 2:4] * self.target_feature_stride
            else:
                clicks_abs = torch.zeros((x.shape[0], self.K, 2), device=x.device)
            outputs["is_prediction_prompt"] = True

        roi_half_abs = R_abs_b1 * self.s2_roi_scale
        stride = self.target_feature_stride

        if "p_gt" in batch:
            B = x.shape[0]
            pgt = batch["p_gt"].float().to(x.device).view(B, self.K, 4, 2)
            centers = pgt.mean(dim=2, keepdim=True)
            quads_limit = centers + self.s2_click_frac * (pgt - centers)
            weights = torch.distributions.Dirichlet(
                torch.full((B, self.K, 4), 1.0, device=pgt.device)).sample().unsqueeze(-1)
            clicks_abs = (weights * quads_limit).sum(dim=2)
            outputs["s2_gt_centers_abs"] = centers.squeeze(2)

        feature_in = outputs["s1_content_map"]
        hm_in = outputs["pred_global_hm"].sigmoid()
        offsets_in = outputs["pred_corner_offsets"]

        if self.s2_detach_stage1_hm:
            feature_in = feature_in.detach()
            hm_in = hm_in.detach()
            offsets_in = offsets_in.detach()

        clicks_feat = clicks_abs / stride
        roi_half_feat = roi_half_abs / stride

        s2_out = self.click_refiner(feature_in, hm_in, offsets_in, clicks_feat, roi_half_feat)
        outputs.update(s2_out)

        d_center = outputs["s2_d_center"]

        if roi_half_abs.dim() == 2 and roi_half_abs.shape[1] == 1:
            roi_half_abs_k = roi_half_abs.view(-1, 1, 1)
        else:
            roi_half_abs_k = roi_half_abs.view(-1, self.K, 1)

        pred_center_abs = clicks_abs + d_center * roi_half_abs_k
        outputs["s2_pred_centers_abs"] = pred_center_abs

        d_corners = outputs["s2_d_corners"]
        pred_center_abs_exp = pred_center_abs.unsqueeze(2)
        refined_corners = pred_center_abs_exp + d_corners * roi_half_abs_k.unsqueeze(-1)

        outputs["s2_pred_corners_abs"] = refined_corners
        outputs["s2_clicks_abs"] = clicks_abs
        outputs["s2_roi_half_abs"] = roi_half_abs

        if "s2_valid_mask" not in outputs:
            outputs["s2_valid_mask"] = torch.ones((x.shape[0], self.K), device=x.device)

        return outputs

    # ==========================================================
    # 交互接口 1: Refine (无需消融，这是Stage2)
    # ==========================================================
    @torch.no_grad()
    def inference_interactive(self, cached_features, target_idx, new_x, new_y):
        self.eval()

        feature_in = cached_features["s1_content_map"]
        hm_in = cached_features["pred_global_hm"].sigmoid()
        offsets_in = cached_features["pred_corner_offsets"]

        roi_all = cached_features["s2_roi_half_abs"]

        if roi_all.shape[1] > 1:
            target_roi = roi_all[:, target_idx: target_idx + 1].clone()
        else:
            target_roi = roi_all.clone()
        target_roi = torch.clamp(target_roi, min=24.0)
        roi_half_abs = target_roi.expand(-1, self.K)

        stride = self.target_feature_stride
        B = feature_in.shape[0]

        single_click = torch.tensor([[[new_x, new_y]]], device=feature_in.device, dtype=torch.float32)
        click_abs = single_click.expand(B, self.K, 2)

        click_feat = click_abs / stride
        roi_half_feat = roi_half_abs / stride

        s2_out = self.click_refiner(feature_in, hm_in, offsets_in, click_feat, roi_half_feat)

        d_center = s2_out['s2_d_center']
        d_corners = s2_out['s2_d_corners']

        if roi_half_abs.dim() == 2:
            roi_half_k = roi_half_abs.unsqueeze(-1)
        else:
            roi_half_k = roi_half_abs

        pred_center_abs = click_abs + d_center * roi_half_k
        pred_corners_abs = pred_center_abs.unsqueeze(2) + d_corners * roi_half_k.unsqueeze(-1)

        if "s2_pred_centers_abs" in cached_features:
            cached_features["s2_pred_centers_abs"][:, target_idx] = pred_center_abs[:, 0]
        else:
            cached_features["s2_pred_centers_abs"] = pred_center_abs.clone()

        target_center = pred_center_abs[0, 0].cpu().numpy()
        target_corner = pred_corners_abs[0, 0].cpu().numpy()

        return {
            "refined_center": target_center,
            "refined_corners": target_corner
        }

    # ==========================================================
    # 交互接口 2: 重预测错误 (调用原生几何函数)
    # ==========================================================
    # 【修改 3】re_predict_errors 中加入消融参数
    @torch.no_grad()
    def re_predict_errors(self, cached_vit_outputs, new_p_err_abs, img_shape_HW,
                          ablate_geo=False, ablate_topo=False):
        # new_p_err_abs: (B, K, 4, 2) Tensor
        B, K, _, _ = new_p_err_abs.shape
        device = new_p_err_abs.device

        # 1. 转换为 Numpy，以便调用 utils.geometry
        p_err_np = new_p_err_abs[0].cpu().numpy().reshape(-1, 2)  # (68, 2)

        # 2. 调用原生工具函数计算特征
        conn_feat_np = calc_connection_features_from_err(p_err_np, self.K).astype(np.float32)
        intr_feat_np = calc_intrinsic_shape_features_np(p_err_np).astype(np.float32)

        # 3. 转回 Tensor 并增加 Batch 维度
        conn_feat = torch.from_numpy(conn_feat_np).unsqueeze(0).to(device)  # (1, K-1, D)
        intr_feat = torch.from_numpy(intr_feat_np).unsqueeze(0).to(device)  # (1, K, D)

        # 4. 将 p_err 展平为 (B, 68, 2) 以匹配 ErrorDetNet 的输入
        p_err_flat = new_p_err_abs.view(B, -1, 2)

        # 【修改 4】调用 ab_forward 传入消融参数
        outputs = self.err_det_net.ab_forward(
            vit_block_outputs=cached_vit_outputs,
            p_err=p_err_flat,
            connection_features=conn_feat,
            intrinsic_shape_features=intr_feat,
            img_shape_HW=img_shape_HW,
            ablate_geo=ablate_geo,  # 传入
            ablate_topo=ablate_topo  # 传入
        )

        if "error_logits" in outputs:
            return outputs["error_logits"]
        else:
            return torch.zeros((1, self.K, 2), device=device)