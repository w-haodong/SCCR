# -*- coding: utf-8 -*-
# ============================================================
# File: operation/decode.py
# 说明：
#   DecDecoder：全局椎体中心热图 + 四角点 offset 的解码器
#   新增：
#     - candidate_topk：先取较多的候选 peak
#     - conf_thresh：最小响应阈值
#     - seg_thr：结合分割概率图，对候选 peak 做过滤
#   注意：
#     - 仍然保持 pts 的列布局不变：
#         0: cx, 1: cy,
#         2-3: tl_x, tl_y,
#         4-5: tr_x, tr_y,
#         6-7: br_x, br_y,
#         8-9: bl_x, bl_y,
#         10: score
#
#   ✅ 新增（不影响旧逻辑）：
#     - decode_peaks(): 只做 NMS+topk+seg过滤，返回 peak 坐标（torch）
#     - decode_peak1(): 返回 top1 peak（torch）
# ============================================================

import torch
import torch.nn.functional as F


class DecDecoder(object):
    """
    解码规则：
      1) heat: 全局椎体中心热图（已 sigmoid）
      2) NMS: 3x3 max-pool 抑制非极大值
      3) 先在全图上取 candidate_topk 个候选 peak，
         再结合：
           - conf_thresh：最低响应阈值
           - seg_thr：在 seg_prob 中的最小概率约束（中心必须落在高置信分割带内）
         得到最终 K 个候选
      4) 用 reg/head 回归得到亚像素中心和 4 个角点
    """

    def __init__(self, K=17, candidate_topk=32, conf_thresh=0.0, seg_thr=0.0):
        self.K = int(K)
        self.candidate_topk = int(max(candidate_topk, self.K))
        self.conf_thresh = float(conf_thresh)
        self.seg_thr = float(seg_thr)

    # -------------------- 基础工具 --------------------
    def _nms(self, heat, kernel=3):
        hmax = F.max_pool2d(
            heat, (kernel, kernel),
            stride=1,
            padding=(kernel - 1) // 2
        )
        keep = (hmax == heat).float()
        return heat * keep

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def _topk(self, scores):
        """
        从每张图中取 candidate_topk 个 peak（暂不考虑 seg/阈值约束）
        """
        batch, cat, height, width = scores.size()
        Kc = min(self.candidate_topk, height * width)

        topk_scores, topk_inds = torch.topk(
            scores.view(batch, cat, -1), Kc
        )  # [B,cat,Kc]

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds // width).int().float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(
            topk_scores.view(batch, -1), Kc
        )  # [B,Kc]
        topk_inds = self._gather_feat(
            topk_inds.view(batch, -1, 1), topk_ind
        ).view(batch, Kc)
        topk_ys = self._gather_feat(
            topk_ys.view(batch, -1, 1), topk_ind
        ).view(batch, Kc)
        topk_xs = self._gather_feat(
            topk_xs.view(batch, -1, 1), topk_ind
        ).view(batch, Kc)

        return topk_score, topk_inds, topk_ys, topk_xs

    # ============================================================
    # ✅ 新增：仅 peak 解码（不回归角点）
    # ============================================================
    @torch.no_grad()
    def decode_peaks(
        self,
        heat,                 # [B,1,H,W] sigmoid 概率图
        seg_prob=None,        # 可选 [B,1,Hs,Ws] / [B,Hs,Ws]
        topk=10,
        conf_thresh=None,
        seg_thr=None,
        nms_kernel=3,
        return_inds=False,
    ):
        """
        只做：NMS + candidate_topk + conf/seg过滤 + 取 topk
        返回：
          xs:    [B,topk]  float (heat坐标系)
          ys:    [B,topk]
          score: [B,topk]
          (可选) inds: [B,topk] flatten index
        """
        assert heat.dim() == 4 and heat.size(1) == 1, f"heat must be [B,1,H,W], got {tuple(heat.shape)}"
        B, _, H, W = heat.shape
        device = heat.device

        topk = int(max(1, topk))
        Kc = int(max(self.candidate_topk, topk))

        _conf = float(self.conf_thresh if conf_thresh is None else conf_thresh)
        _segthr = float(self.seg_thr if seg_thr is None else seg_thr)

        # 1) align seg to heat
        seg_aligned = None
        if seg_prob is not None:
            if seg_prob.dim() == 4:
                seg = seg_prob
            elif seg_prob.dim() == 3:
                seg = seg_prob.unsqueeze(1)
            else:
                raise ValueError(f"Unexpected seg_prob shape: {seg_prob.shape}")
            if seg.shape[-2:] != (H, W):
                seg = F.interpolate(seg, size=(H, W), mode="bilinear", align_corners=False)
            seg_aligned = seg  # [B,1,H,W]

        # 2) NMS
        heat_nms = self._nms(heat, kernel=int(nms_kernel))

        # 3) candidates：直接 topk flatten
        Kc_eff = min(Kc, H * W)
        cand_scores, cand_inds = torch.topk(heat_nms.view(B, -1), Kc_eff)  # [B,Kc]
        cand_ys = (cand_inds // W).int().float()
        cand_xs = (cand_inds % W).int().float()

        xs_out = torch.zeros((B, topk), device=device, dtype=torch.float32)
        ys_out = torch.zeros((B, topk), device=device, dtype=torch.float32)
        sc_out = torch.zeros((B, topk), device=device, dtype=torch.float32)
        ind_out = torch.zeros((B, topk), device=device, dtype=torch.long)

        for b in range(B):
            s = cand_scores[b]
            ys = cand_ys[b]
            xs = cand_xs[b]
            ind = cand_inds[b]

            valid = torch.ones_like(s, dtype=torch.bool, device=device)
            if _conf > 0:
                valid &= (s >= _conf)
            if (seg_aligned is not None) and (_segthr > 0):
                segb = seg_aligned[b, 0]
                yi = ys.long().clamp(0, H - 1)
                xi = xs.long().clamp(0, W - 1)
                valid &= (segb[yi, xi] >= _segthr)

            if not valid.any():
                valid = torch.ones_like(valid)

            eff = s.clone()
            eff[~valid] = -1e6

            take = min(topk, Kc_eff)
            sc, idx = torch.topk(eff, take)

            if take < topk:
                pad = topk - take
                idx = torch.cat([idx, idx[-1:].repeat(pad)], dim=0)
                sc = torch.cat([sc, sc[-1:].repeat(pad)], dim=0)

            xs_out[b] = xs[idx]
            ys_out[b] = ys[idx]
            sc_out[b] = sc
            ind_out[b] = ind[idx]

        if return_inds:
            return xs_out, ys_out, sc_out, ind_out
        return xs_out, ys_out, sc_out

    @torch.no_grad()
    def decode_peak1(self, heat, seg_prob=None, conf_thresh=None, seg_thr=None, nms_kernel=3):
        """
        返回每张图 top1 peak 的 (x,y,score) —— torch
        """
        xs, ys, sc = self.decode_peaks(
            heat=heat,
            seg_prob=seg_prob,
            topk=1,
            conf_thresh=conf_thresh,
            seg_thr=seg_thr,
            nms_kernel=nms_kernel,
            return_inds=False,
        )
        return xs[:, 0], ys[:, 0], sc[:, 0]

    # -------------------- 主解码函数（原样保留，不改） --------------------
    def ctdet_decode(self, heat, wh, reg, seg_prob=None):
        """
        heat:     [B, 1, H, W]    — 全局中心热图（概率）
        wh:       [B, 8, H, W]    — 四角点偏移
        reg:      [B, 2, H, W]    — 中心 offsets
        seg_prob: [B,1,Hs,Ws] 或 [B,Hs,Ws] — 脊柱分割概率（可选）

        return:
          pts: [K, 11]  (当 B=1 时)
            0: cx, 1: cy,
            2-3: tl_x, tl_y,
            4-5: tr_x, tr_y,
            6-7: br_x, br_y,
            8-9: bl_x, bl_y,
            10: score
        """
        batch, c, height, width = heat.size()
        device = heat.device

        # -------- 1) 对齐分割到 heat 的分辨率 --------
        seg_aligned = None
        if seg_prob is not None:
            if seg_prob.dim() == 4:
                seg = seg_prob
            elif seg_prob.dim() == 3:
                seg = seg_prob.unsqueeze(1)
            else:
                raise ValueError(f"Unexpected seg_prob shape: {seg_prob.shape}")

            if seg.shape[-2:] != (height, width):
                seg = F.interpolate(seg, size=(height, width),
                                    mode="bilinear", align_corners=False)
            seg_aligned = seg  # [B,1,H,W]

        # -------- 2) NMS 抑制非极大值 --------
        heat_nms = self._nms(heat)

        # -------- 3) 先取 candidate_topk 个候选 peak --------
        cand_scores, cand_inds, cand_ys, cand_xs = self._topk(heat_nms)  # [B,Kc]

        Kc = cand_scores.size(1)

        final_inds = torch.zeros((batch, self.K), dtype=torch.long, device=device)
        final_scores = torch.zeros((batch, self.K), dtype=torch.float32, device=device)
        final_ys = torch.zeros((batch, self.K), dtype=torch.float32, device=device)
        final_xs = torch.zeros((batch, self.K), dtype=torch.float32, device=device)

        for b in range(batch):
            scores_b = cand_scores[b]  # [Kc]
            ys_b = cand_ys[b]
            xs_b = cand_xs[b]

            valid_mask = torch.ones_like(scores_b, dtype=torch.bool, device=device)

            if self.conf_thresh > 0.0:
                valid_mask &= (scores_b >= self.conf_thresh)

            if seg_aligned is not None and self.seg_thr > 0.0:
                seg_b = seg_aligned[b, 0]  # [H,W]
                ys_int = ys_b.long().clamp(0, height - 1)
                xs_int = xs_b.long().clamp(0, width - 1)
                seg_vals = seg_b[ys_int, xs_int]
                valid_mask &= (seg_vals >= self.seg_thr)

            if not valid_mask.any():
                valid_mask = torch.ones_like(valid_mask, dtype=torch.bool, device=device)

            eff_scores = scores_b.clone()
            eff_scores[~valid_mask] = -1e6

            topk = min(self.K, Kc)
            top_scores_b, idx_local = torch.topk(eff_scores, topk)

            if topk < self.K:
                pad = self.K - topk
                idx_local = torch.cat(
                    [idx_local, idx_local[-1:].repeat(pad)],
                    dim=0
                )
                top_scores_b = torch.cat(
                    [top_scores_b, top_scores_b[-1:].repeat(pad)],
                    dim=0
                )

            chosen_inds = cand_inds[b][idx_local]
            chosen_ys = ys_b[idx_local]
            chosen_xs = xs_b[idx_local]

            final_inds[b] = chosen_inds
            final_scores[b] = top_scores_b
            final_ys[b] = chosen_ys
            final_xs[b] = chosen_xs

        # -------- 4) 按最终索引回归中心和四角点 --------
        scores = final_scores.view(batch, self.K, 1)  # [B,K,1]

        reg_feat = self._tranpose_and_gather_feat(reg, final_inds)
        reg_feat = reg_feat.view(batch, self.K, 2)

        xs = final_xs.view(batch, self.K, 1) + reg_feat[:, :, 0:1]
        ys = final_ys.view(batch, self.K, 1) + reg_feat[:, :, 1:2]

        wh_feat = self._tranpose_and_gather_feat(wh, final_inds)
        wh_feat = wh_feat.view(batch, self.K, 2 * 4)

        # 这里假定 wh 的 layout 为：
        #   [tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y]
        # 假定 wh_feat layout: [tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y]
        tl_x = xs + wh_feat[:, :, 0:1]
        tl_y = ys + wh_feat[:, :, 1:2]
        tr_x = xs + wh_feat[:, :, 2:3]
        tr_y = ys + wh_feat[:, :, 3:4]
        bl_x = xs + wh_feat[:, :, 4:5]
        bl_y = ys + wh_feat[:, :, 5:6]
        br_x = xs + wh_feat[:, :, 6:7]
        br_y = ys + wh_feat[:, :, 7:8]

        pts = torch.cat(
            [
                xs, ys,
                tl_x, tl_y,
                tr_x, tr_y,
                bl_x, bl_y,
                br_x, br_y,
                scores
            ],
            dim=2
        )

        if pts.shape[0] == 1:
            pts = pts.squeeze(0)

        return pts.data.cpu().numpy()
