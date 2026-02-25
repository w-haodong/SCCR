# -*- coding: utf-8 -*-
# operation/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bce_logits_with_mask(logits, targets, mask=None, pos_weight=None):
    loss = F.binary_cross_entropy_with_logits(
        logits.float(), targets.float(),
        reduction='none',
        pos_weight=pos_weight
    )
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum().clamp_min(1.0)
    return loss.mean()


class _HeatmapFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, neg_power=4.0, epsilon=1e-6):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.neg_power = neg_power
        self.eps = epsilon

    def forward(self, pred_logits, gt):
        p = torch.clamp(torch.sigmoid(pred_logits.float()), min=self.eps, max=1 - self.eps)
        gt = gt.float()
        pos = gt.eq(1).float()
        neg = gt.lt(1).float()
        neg_w = torch.pow(1 - gt, self.neg_power)

        pos_loss = -self.alpha * torch.log(p) * torch.pow(1 - p, self.gamma) * pos
        neg_loss = -(1 - self.alpha) * torch.log(1 - p) * torch.pow(p, self.gamma) * neg_w * neg
        return (pos_loss.sum() + neg_loss.sum()) / (pos.sum() + self.eps)


class _DenseRegL1Loss(nn.Module):
    def __init__(self, beta=1.0, eps=1e-6):
        super().__init__()
        self.beta = beta
        self.eps = eps

    def forward(self, pred, gt_map, weight_map):
        pred = pred.float()
        gt_map = gt_map.float()
        w = weight_map.float()

        if w.ndim == 3:
            w = w.unsqueeze(1)
        if w.size(1) != 1:
            raise ValueError(f"[DenseReg] weight_map must be (B,1,H,W), got {tuple(w.shape)}")

        diff = F.smooth_l1_loss(pred, gt_map, reduction='none', beta=self.beta)
        diff = diff.mean(dim=1, keepdim=True)

        denom = w.sum().clamp_min(self.eps)
        return (diff * w).sum() / denom


def _soft_dice_loss_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits.float())
    t = targets.float()
    probs = probs.view(probs.size(0), -1)
    t = t.view(t.size(0), -1)
    inter = (probs * t).sum(dim=1)
    union = probs.sum(dim=1) + t.sum(dim=1)
    dice = (2 * inter + eps) / (union + eps)
    return (1.0 - dice).mean()


class SAIC_Loss(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args

        # Stage 1 weights
        self.lambda_hm = getattr(args, 'lambda_hm', 1.0)
        self.lambda_det = getattr(args, 'lambda_det', 1.0)
        self.lambda_center_reg = getattr(args, 'lambda_center_reg', 1.0)
        self.lambda_corner_reg = getattr(args, 'lambda_corner_reg', 0.1)
        self.lambda_cons = getattr(args, 'lambda_cons', 0.2)

        # Seg
        self.lambda_seg = getattr(args, 'lambda_seg', 1.0)
        self.lambda_seg_dice = getattr(args, 'lambda_seg_dice', 0.0)

        # Stage 2 (Cascade) Weights
        self.lambda_s2_click_l1 = float(getattr(args, "lambda_s2_click_l1", 100))

        self.hm_warmup_epochs = getattr(args, 'hm_warmup_epochs', 0)
        self.eps = 1e-6

        self.hm_focal = _HeatmapFocalLoss()
        self.dense_reg = _DenseRegL1Loss(beta=1.0)

    # ---------- detector ----------
    def _compute_pos_weight_2ch(self, labels, conn_valid_mask):
        B, K, C = labels.shape
        device = labels.device

        y_intr = labels[..., 0].reshape(-1)
        pos_intr = y_intr.sum()
        tot_intr = y_intr.numel()

        y_conn = labels[..., 1] * conn_valid_mask
        pos_conn = y_conn.sum()
        tot_conn = conn_valid_mask.sum().clamp_min(1.0)

        pw_intr = ((tot_intr - pos_intr).clamp_min(1.0) / (pos_intr + self.eps)).to(device)
        pw_conn = ((tot_conn - pos_conn).clamp_min(1.0) / (pos_conn + self.eps)).to(device)
        return torch.stack([pw_intr, pw_conn], dim=0)

    def _detector_loss(self, outputs, batch):
        if 'error_logits' not in outputs:
            return torch.tensor(0.0, device=batch['input_image'].device)

        logits = outputs['error_logits']
        labels = batch['error_labels'].to(logits.dtype)

        B, K, C = labels.shape
        mask = torch.ones_like(labels)
        mask[:, -1, 1] = 0.0

        conn_valid_mask = torch.ones((B, K), device=logits.device, dtype=logits.dtype)
        conn_valid_mask[:, -1] = 0.0

        posw = self._compute_pos_weight_2ch(labels, conn_valid_mask)
        return _bce_logits_with_mask(logits, labels, mask, posw)

    # ---------- stage1 losses ----------
    def _heatmap_loss(self, outputs, batch):
        if 'pred_global_hm' not in outputs:
            return torch.tensor(0.0, device=batch['gt_global_hm'].device)
        return self.hm_focal(outputs['pred_global_hm'], batch['gt_global_hm'])

    def _center_reg_loss(self, outputs, batch):
        if 'pred_center_offsets' not in outputs:
            return torch.tensor(0.0, device=batch['input_image'].device)
        if ('gt_center_reg_map' not in batch) or ('gt_reg_weight_map' not in batch):
            return torch.tensor(0.0, device=batch['input_image'].device)
        return self.dense_reg(
            outputs['pred_center_offsets'],
            batch['gt_center_reg_map'].to(outputs['pred_center_offsets'].dtype),
            batch['gt_reg_weight_map'].to(outputs['pred_center_offsets'].dtype),
        )

    def _corner_reg_loss(self, outputs, batch):
        if 'pred_corner_offsets' not in outputs:
            return torch.tensor(0.0, device=batch['input_image'].device)
        if ('gt_corner_reg_map' not in batch) or ('gt_reg_weight_map' not in batch):
            return torch.tensor(0.0, device=batch['input_image'].device)
        return self.dense_reg(
            outputs['pred_corner_offsets'],
            batch['gt_corner_reg_map'].to(outputs['pred_corner_offsets'].dtype),
            batch['gt_reg_weight_map'].to(outputs['pred_corner_offsets'].dtype),
        )

    def _seg_loss(self, outputs, batch):
        if ('pred_spine_mask_logits' not in outputs) or ('gt_spine_mask' not in batch):
            z = torch.tensor(0.0, device=batch['input_image'].device)
            return z, z

        logits = outputs['pred_spine_mask_logits']
        gt = batch['gt_spine_mask'].to(logits.dtype)

        with torch.no_grad():
            pos = gt.sum()
            tot = gt.numel()
            neg = (tot - pos).clamp_min(1.0)
            pw = (neg / (pos + 1e-6)).clamp(min=1.0, max=100.0).to(logits.device)

        bce = F.binary_cross_entropy_with_logits(logits.float(), gt.float(), reduction='mean', pos_weight=pw)
        dice = _soft_dice_loss_from_logits(logits, gt) if self.lambda_seg_dice > 0 else torch.tensor(0.0, device=logits.device)
        return bce, dice

    def _ori_consistency_loss(self, batch):
        if ('p_err' not in batch) or ('error_labels' not in batch):
            return torch.tensor(0.0, device=batch['input_image'].device)

        pe = batch['p_err']
        B = pe.shape[0]
        K = pe.shape[1] // 4
        p = pe.view(B, K, 4, 2)
        centers = p.mean(dim=2)

        mid_left = (p[:, :, 0, :] + p[:, :, 2, :]) * 0.5
        mid_right = (p[:, :, 1, :] + p[:, :, 3, :]) * 0.5
        a = F.normalize(mid_right - mid_left, dim=-1, eps=1e-6)

        d = torch.zeros_like(centers)
        if K >= 3:
            d[:, 1:-1] = centers[:, 2:] - centers[:, :-2]
        if K >= 2:
            d[:, 0] = centers[:, 1] - centers[:, 0]
            d[:, -1] = centers[:, -1] - centers[:, -2]
        t = F.normalize(d, dim=-1, eps=1e-6)
        n = torch.stack([-t[..., 1], t[..., 0]], dim=-1)
        cos_delta = (a * n).sum(dim=-1)

        labels = batch['error_labels'].float()
        mask = (1.0 - labels[..., 0]) * torch.cat(
            [1.0 - labels[:, :-1, 1], torch.ones(B, 1, device=labels.device)], dim=1
        )
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pe.device)
        return ((1.0 - cos_delta) * mask).sum() / mask.sum().clamp_min(1.0)

    # ============================================================
    # Stage2: Cascade Loss
    # [FIXED] Correct tensor broadcasting for roi_half
    # ============================================================
    def _s2_cascade_loss(self, outputs, batch):
        loss_zero = torch.tensor(0.0, device=batch["input_image"].device)

        if self.lambda_s2_click_l1 <= 0:
            return loss_zero, loss_zero

        required = ["s2_d_center", "s2_d_corners", "s2_clicks_abs"]
        if any(k not in outputs for k in required):
            return loss_zero, loss_zero

        # Inputs
        clicks = outputs["s2_clicks_abs"]           # (B, K, 2)
        roi_half = outputs["s2_roi_half_abs"]       # (B, 1) usually
        
        d_center_pred = outputs["s2_d_center"]      # (B, K, 2)
        d_corners_pred = outputs["s2_d_corners"]    # (B, K, 4, 2)

        # GT
        if "p_gt" not in batch:
             return loss_zero, loss_zero
        
        pgt = batch["p_gt"].float().to(clicks.device)
        B, K, _ = clicks.shape
        pgt = pgt.view(B, K, 4, 2)
        gt_centers = pgt.mean(dim=2) # (B, K, 2)

        valid_mask = outputs.get("s2_valid_mask", torch.ones((B, K), device=clicks.device))

        # [FIXED] Handle Radius Broadcast correctly
        # roi_half could be (B, 1) or (B, K) or (B, 1, 1)
        
        # 1. Ensure at least 3 dims (B, N, 1)
        if roi_half.dim() == 2:
            roi_half = roi_half.unsqueeze(2) # (B, N, 1)

        # 2. Expand to (B, K, 1) if necessary
        # If shape is (B, 1, 1), expand to (B, K, 1)
        if roi_half.size(1) == 1:
            roi_half_k = roi_half.expand(B, K, 1)
        else:
            roi_half_k = roi_half

        # 3. Create expand version for corners (B, K, 1, 1)
        roi_half_exp = roi_half_k.unsqueeze(-1)
        
        # Ensure values are safe
        roi_half_k = roi_half_k.clamp(min=1.0)
        roi_half_exp = roi_half_exp.clamp(min=1.0)
        
        # --- Task 1: Center Regression Loss ---
        gt_d_center = (gt_centers - clicks) / roi_half_k
        gt_d_center = gt_d_center.clamp(-1.0, 1.0) 
        
        loss_center_map = F.l1_loss(d_center_pred, gt_d_center, reduction='none').mean(dim=-1) # (B, K)
        loss_center = (loss_center_map * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)

        # --- Task 2: Corner Regression Loss ---
        gt_d_corners = (pgt - gt_centers.unsqueeze(2)) / roi_half_exp
        gt_d_corners = gt_d_corners.clamp(-1.0, 1.0)

        loss_corner_map = F.l1_loss(d_corners_pred, gt_d_corners, reduction='none').mean(dim=(2, 3)) # (B, K)
        loss_corner = (loss_corner_map * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)

        # Total S2 Loss
        total_l1 = loss_center + loss_corner
        
        return loss_zero, total_l1

    # ---------- forward ----------
    def forward(self, outputs, batch, epoch: int):
        l_hm = self._heatmap_loss(outputs, batch)
        l_ctr = self._center_reg_loss(outputs, batch)
        l_crn = self._corner_reg_loss(outputs, batch)

        l_seg_bce, l_seg_dice = self._seg_loss(outputs, batch)
        l_seg = self.lambda_seg * l_seg_bce + self.lambda_seg_dice * l_seg_dice

        # Warmup for Stage 1 Heatmap
        if epoch <= self.hm_warmup_epochs:
            total = (self.lambda_hm * l_hm +
                     self.lambda_center_reg * l_ctr +
                     self.lambda_corner_reg * l_crn +
                     l_seg)
            stats = {
                'hm_loss': l_hm.detach(),
                'center_reg_loss': l_ctr.detach(),
                'corner_reg_loss': l_crn.detach(),
                'seg_bce': l_seg_bce.detach(),
                'seg_dice': l_seg_dice.detach(),
                'total_loss': total.detach(),
            }
            return total, stats

        l_det = self._detector_loss(outputs, batch)
        l_cons = self._ori_consistency_loss(batch) * self.lambda_cons

        stage1_total = (self.lambda_hm * l_hm +
                        self.lambda_center_reg * l_ctr +
                        self.lambda_corner_reg * l_crn +
                        self.lambda_det * l_det +
                        l_cons +
                        l_seg)

        # Stage 2 Loss (Cascade)
        l_s2_dummy, l_s2_l1 = self._s2_cascade_loss(outputs, batch)
        stage2_total = self.lambda_s2_click_l1 * l_s2_l1

        total = stage1_total + stage2_total

        stats = {
            'hm_loss': l_hm.detach(),
            'center_reg_loss': l_ctr.detach(),
            'corner_reg_loss': l_crn.detach(),
            'det_loss': l_det.detach(),
            'cons_loss': l_cons.detach(),
            'seg_loss': l_seg.detach(),
            # Stage 2 stats
            's2_l1_loss': l_s2_l1.detach(),
            'total_loss': total.detach(),
        }
        return total, stats