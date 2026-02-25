# -*- coding: utf-8 -*-
import os
import cv2
import math
import numpy as np
import torch
import torch.utils.data as data
from scipy.io import loadmat

from operation import transform
from utils.draw_gaussian import draw_umich_gaussian, gaussian_radius
from utils.geometry import (
    angle_sort_all,
    calc_connection_features_from_err,
    calc_intrinsic_shape_features_np,
)

def collater(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

class CorrectionDataset(data.Dataset):
    def __init__(self, args, phase="train", domain=None):
        super().__init__()
        self.args = args
        self.phase = phase
        self.domain = domain

        self.num_vertebrae = int(args.K)
        self.down_ratio = int(args.target_feature_stride)
        self.input_h = int(args.input_h)
        self.input_w = int(args.input_w)

        assert (self.input_h % self.down_ratio == 0 and self.input_w % self.down_ratio == 0), \
            f"input_h={self.input_h}, input_w={self.input_w} must be divisible by down_ratio={self.down_ratio}"

        self.output_h = self.input_h // self.down_ratio
        self.output_w = self.input_w // self.down_ratio

        # --------------------- paths ---------------------
        self.data_dir = args.data_dir
        phase_root = os.path.join(self.data_dir, phase)

        if domain is None:
            base = phase_root
            base_lp = os.path.join(base, "labels_processed")
            if not os.path.isdir(base_lp):
                for cand_domain in ["source", "target"]:
                    cand_base = os.path.join(phase_root, cand_domain)
                    cand_lp = os.path.join(cand_base, "labels_processed")
                    if os.path.isdir(cand_lp):
                        print(f"[{phase.upper()}] Warning: '{base_lp}' not found, switch to '{cand_lp}'")
                        base = cand_base
                        break
        else:
            base_candidate = os.path.join(phase_root, domain)
            lp_candidate = os.path.join(base_candidate, "labels_processed")
            lp_phase_root = os.path.join(phase_root, "labels_processed")

            if os.path.isdir(lp_candidate):
                base = base_candidate
            elif os.path.isdir(lp_phase_root):
                print(f"[{phase.upper()}-{domain}] Warning: '{lp_candidate}' not found, fallback '{lp_phase_root}'")
                base = phase_root
            else:
                raise FileNotFoundError(f"Cannot find labels_processed under {lp_candidate} or {lp_phase_root}")

        self.img_dir = os.path.join(base, "images")
        self.label_dir = os.path.join(base, "labels_processed")

        if not os.path.isdir(self.label_dir):
            raise FileNotFoundError(f"Label dir not found: {self.label_dir}")

        self.label_files = sorted([f for f in os.listdir(self.label_dir) if f.endswith(".mat")])

        tag = phase.upper() if domain is None else f"{phase.upper()}-{domain}"
        print(f"[{tag}] samples={len(self.label_files)}")
        print(f"   -> img_dir:   {self.img_dir}")
        print(f"   -> label_dir: {self.label_dir}")
        print(f"   -> Input: ({self.input_h},{self.input_w}), HM: ({self.output_h},{self.output_w})")

        # Aug
        if domain == 'source':
            min_scale, max_scale = 0.65, 1.0
        else:
            min_scale, max_scale = 0.98, 1.0

        self.train_aug = transform.Compose(
            [
                transform.ConvertImgFloat(),
                transform.PhotometricDistort(),
                transform.RandomScale(scale_range=(min_scale, max_scale)),
                transform.RandomRotate(angle_range=(-15, 15), prob=0.5),
                transform.RandomMirror_w(),
            ]
        )
        self.eval_aug = transform.Compose([transform.ConvertImgFloat()])

        self.mask_close_ky = 0
        self.mask_dilate_k = 0
        self.tl_idx, self.tr_idx, self.bl_idx, self.br_idx = 0, 1, 3, 2

    def __len__(self):
        return len(self.label_files)

    def _letterbox(self, image, pts_list):
        H, W = image.shape[:2]
        T_h, T_w = self.input_h, self.input_w
        scale = min(float(T_h) / float(H), float(T_w) / float(W))
        new_w = int(round(W * scale))
        new_h = int(round(H * scale))

        img_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((T_h, T_w, 3), dtype=image.dtype)
        top = (T_h - new_h) // 2
        left = (T_w - new_w) // 2
        canvas[top: top + new_h, left: left + new_w] = img_resized

        out_pts = []
        for pts in pts_list:
            if pts is None or pts.size == 0:
                out_pts.append(np.empty((0, 2), np.float32))
            else:
                t = pts.astype(np.float32).copy().reshape(-1, 2)
                t[:, 0] = t[:, 0] * scale + left
                t[:, 1] = t[:, 1] * scale + top
                out_pts.append(t)

        meta = {
            "orig_h": float(H), "orig_w": float(W),
            "input_h": float(T_h), "input_w": float(T_w),
            "scale": float(scale), "top": float(top), "left": float(left),
        }
        return np.ascontiguousarray(canvas), out_pts, meta

    def _build_global_heatmap(self, p_gt_aug_4K2):
        K = self.num_vertebrae
        hm = np.zeros((1, self.output_h, self.output_w), dtype=np.float32)
        pts_gt = p_gt_aug_4K2.reshape(K, 4, 2)

        for v in range(K):
            quad_gt = pts_gt[v]
            if np.any(np.isnan(quad_gt)) or np.all(quad_gt == 0):
                continue

            center_orig = np.mean(quad_gt, axis=0)
            center_feat = center_orig / self.down_ratio

            w_orig = quad_gt[:, 0].max() - quad_gt[:, 0].min()
            h_orig = quad_gt[:, 1].max() - quad_gt[:, 1].min()
            w_feat = w_orig / self.down_ratio
            h_feat = h_orig / self.down_ratio

            bbox_h_est = max(1, np.ceil(h_feat))
            bbox_w_est = max(1, np.ceil(w_feat))

            radius = gaussian_radius((math.ceil(bbox_h_est), math.ceil(bbox_w_est)), min_overlap=0.7)
            radius = max(0, int(radius))

            ct_int = np.clip(center_feat, 0, [self.output_w - 1, self.output_h - 1]).astype(np.int32)
            draw_umich_gaussian(hm[0], ct_int, radius=radius)

        return np.ascontiguousarray(hm)

    # =========================================================
    # ✅ Dense regression GT (NEW): maps + gaussian weights
    # =========================================================
    @staticmethod
    def _gaussian2d(shape, sigma):
        h, w = int(shape[0]), int(shape[1])
        m = (h - 1) / 2.0
        n = (w - 1) / 2.0
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        g = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma + 1e-12))
        eps = np.finfo(g.dtype).eps
        g[g < eps * g.max()] = 0
        return g.astype(np.float32)

    def _apply_gaussian_update_dense_maps(
        self,
        center_map_2hw: np.ndarray,     # (2,Hf,Wf)
        corner_map_8hw: np.ndarray,     # (8,Hf,Wf)
        weight_map_1hw: np.ndarray,     # (1,Hf,Wf)
        center_feat_xy: np.ndarray,     # (2,) float
        quad_feat_4x2: np.ndarray,      # (4,2) float
        ct_int_xy: np.ndarray,          # (2,) int
        radius: int
    ):

        Hf, Wf = weight_map_1hw.shape[1], weight_map_1hw.shape[2]
        x0, y0 = int(ct_int_xy[0]), int(ct_int_xy[1])
        r = int(max(0, radius))

        x1 = max(0, x0 - r)
        x2 = min(Wf - 1, x0 + r)
        y1 = max(0, y0 - r)
        y2 = min(Hf - 1, y0 + r)
        if x2 < x1 or y2 < y1:
            return

        diam = 2 * r + 1
        sigma = max(1e-6, diam / 6.0)
        g_full = self._gaussian2d((diam, diam), sigma=sigma)

        gx1 = x1 - (x0 - r)
        gx2 = gx1 + (x2 - x1) + 1
        gy1 = y1 - (y0 - r)
        gy2 = gy1 + (y2 - y1) + 1
        g = g_full[gy1:gy2, gx1:gx2]  # (h,w)

        w_patch = weight_map_1hw[0, y1:y2 + 1, x1:x2 + 1]  # (h,w)
        upd = g > w_patch
        if not np.any(upd):
            return

        w_patch[upd] = g[upd]
        weight_map_1hw[0, y1:y2 + 1, x1:x2 + 1] = w_patch

        # meshgrid -> (h,w)
        xs = np.arange(x1, x2 + 1, dtype=np.float32)
        ys = np.arange(y1, y2 + 1, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)  # (h,w)

        dx = float(center_feat_xy[0]) - xx
        dy = float(center_feat_xy[1]) - yy

        cmx = center_map_2hw[0, y1:y2 + 1, x1:x2 + 1]
        cmy = center_map_2hw[1, y1:y2 + 1, x1:x2 + 1]
        cmx[upd] = dx[upd]
        cmy[upd] = dy[upd]
        center_map_2hw[0, y1:y2 + 1, x1:x2 + 1] = cmx
        center_map_2hw[1, y1:y2 + 1, x1:x2 + 1] = cmy

        offsets_8 = (quad_feat_4x2 - center_feat_xy.reshape(1, 2)).reshape(8).astype(np.float32)
        for c in range(8):
            patch = corner_map_8hw[c, y1:y2 + 1, x1:x2 + 1]
            patch[upd] = offsets_8[c]
            corner_map_8hw[c, y1:y2 + 1, x1:x2 + 1] = patch

    def _build_dense_regression_gt(self, p_gt_aug_4K2):
        """
        return:
          gt_corner_reg_map: (8,Hf,Wf)
          gt_center_reg_map: (2,Hf,Wf)
          gt_reg_weight_map: (1,Hf,Wf)
        """
        K = self.num_vertebrae
        Hf, Wf = self.output_h, self.output_w
        dr = float(self.down_ratio)

        corner_map = np.zeros((8, Hf, Wf), dtype=np.float32)
        center_map = np.zeros((2, Hf, Wf), dtype=np.float32)
        weight_map = np.zeros((1, Hf, Wf), dtype=np.float32)

        pts_gt = np.asarray(p_gt_aug_4K2, np.float32).reshape(K, 4, 2)

        for v in range(K):
            quad = pts_gt[v]
            if np.any(np.isnan(quad)) or np.all(quad == 0):
                continue

            center_orig = quad.mean(axis=0)
            center_feat = center_orig / dr
            quad_feat = quad / dr

            w_orig = quad[:, 0].max() - quad[:, 0].min()
            h_orig = quad[:, 1].max() - quad[:, 1].min()
            w_feat = w_orig / dr
            h_feat = h_orig / dr
            bbox_h_est = max(1, int(np.ceil(h_feat)))
            bbox_w_est = max(1, int(np.ceil(w_feat)))

            radius = gaussian_radius((math.ceil(bbox_h_est), math.ceil(bbox_w_est)), min_overlap=0.7)
            radius = int(max(0, radius))

            ct_int = np.clip(center_feat, 0, [Wf - 1, Hf - 1]).astype(np.int32)

            self._apply_gaussian_update_dense_maps(
                center_map_2hw=center_map,
                corner_map_8hw=corner_map,
                weight_map_1hw=weight_map,
                center_feat_xy=center_feat.astype(np.float32),
                quad_feat_4x2=quad_feat.astype(np.float32),
                ct_int_xy=ct_int,
                radius=radius
            )

        return np.ascontiguousarray(corner_map), np.ascontiguousarray(center_map), np.ascontiguousarray(weight_map)

    def build_spine_mask_68_ring_feat(self, p_gt_aug_4K2: np.ndarray) -> np.ndarray:
        """
        68点围起来的整体脊柱mask（feature scale）：
        - 右边界：每椎体 (tr, br) 串起来
        - 左边界：每椎体 (bl, tl) 反向串回来
        """
        K = self.num_vertebrae
        out_h, out_w = self.output_h, self.output_w
        dr = float(self.down_ratio)

        pts = np.asarray(p_gt_aug_4K2, np.float32).reshape(K, 4, 2)

        valid = []
        for v in range(K):
            quad = pts[v]
            if np.any(np.isnan(quad)) or np.all(quad == 0):
                continue
            valid.append(v)

        mask = np.zeros((out_h, out_w), np.uint8)
        if len(valid) < 2:
            return mask[None].astype(np.float32)

        v0 = valid[0]
        poly = []

        poly.append(pts[v0, self.tl_idx] / dr)

        for v in valid:
            poly.append(pts[v, self.tr_idx] / dr)
            poly.append(pts[v, self.br_idx] / dr)

        for v in reversed(valid):
            poly.append(pts[v, self.bl_idx] / dr)
            poly.append(pts[v, self.tl_idx] / dr)

        poly = np.asarray(poly, np.float32)
        poly[:, 0] = np.clip(poly[:, 0], 0, out_w - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, out_h - 1)

        poly_i = poly.astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [poly_i], 1)

        return mask[None].astype(np.float32)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.label_files[index])
        try:
            mat = loadmat(label_path)
        except Exception as e:
            print(f"Load mat failed: {label_path} -> {e}")
            return None

        img_name = mat["img_name"].item()
        img_path = os.path.join(self.img_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            print(f"Read image failed: {img_path}")
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.ascontiguousarray(image)

        p_gt_raw = mat.get("p2_gt", None)
        p_err_raw = mat.get("p2_erroneous", None)
        if p_gt_raw is None or p_err_raw is None:
            return None

        p_gt_raw = p_gt_raw.astype(np.float32).reshape(-1, 2)
        p_err_raw = p_err_raw.astype(np.float32).reshape(-1, 2)

        p_gt_sorted = angle_sort_all(p_gt_raw)
        p_err_sorted = angle_sort_all(p_err_raw)

        intrinsic_labels = mat["intrinsic_labels"].astype(np.float32)
        connection_labels = mat["connection_labels"].astype(np.float32)
        if connection_labels.ndim == 1:
            connection_labels = connection_labels[:, None]
        connection_labels = np.pad(connection_labels, ((0, 1), (0, 0)), "constant", constant_values=0)
        combined_error_labels = np.concatenate([intrinsic_labels, connection_labels], axis=1)

        # letterbox
        image_sq, [p_gt_sq, p_err_sq], meta = self._letterbox(image, [p_gt_sorted, p_err_sorted])

        # augment on canvas
        combined = np.concatenate([p_gt_sq, p_err_sq], axis=0).astype(np.float32)
        if self.phase == "train":
            image_aug, combined_aug = self.train_aug(image_sq, combined)
        else:
            image_aug, combined_aug = self.eval_aug(image_sq, combined)

        image_aug = np.ascontiguousarray(image_aug)
        combined_aug = np.ascontiguousarray(combined_aug)

        num_gt = p_gt_sq.shape[0]
        p_gt_aug = combined_aug[:num_gt].astype(np.float32, copy=False)
        p_err_aug = combined_aug[num_gt:].astype(np.float32, copy=False)

        # HM
        gt_global_hm = self._build_global_heatmap(p_gt_aug)

        gt_corner_reg_map, gt_center_reg_map, gt_reg_weight_map = self._build_dense_regression_gt(p_gt_aug)

        # 68点围起来的 mask（feature scale）
        gt_spine_mask = self.build_spine_mask_68_ring_feat(p_gt_aug)

        connection_features = calc_connection_features_from_err(p_err_aug, self.num_vertebrae)
        intrinsic_shape_features = calc_intrinsic_shape_features_np(p_err_aug)

        img_tensor = torch.from_numpy(np.transpose(image_aug, (2, 0, 1))).float()
        img_tensor.div_(255.0).sub_(0.5)

        out = {
            "input_image": img_tensor,
            "img_path": img_path,
            "p_err": torch.from_numpy(p_err_aug).float(),
            "error_labels": torch.from_numpy(np.ascontiguousarray(combined_error_labels)).float(),
            "connection_features": torch.from_numpy(np.ascontiguousarray(connection_features)).float(),
            "intrinsic_shape_features": torch.from_numpy(np.ascontiguousarray(intrinsic_shape_features)).float(),
            "p_gt": torch.from_numpy(p_gt_aug).float(),

            "gt_global_hm": torch.from_numpy(gt_global_hm).float(),

            "gt_corner_reg_map": torch.from_numpy(gt_corner_reg_map).float(),   # (8,Hf,Wf)
            "gt_center_reg_map": torch.from_numpy(gt_center_reg_map).float(),   # (2,Hf,Wf)
            "gt_reg_weight_map": torch.from_numpy(gt_reg_weight_map).float(),   # (1,Hf,Wf)

            "gt_spine_mask": torch.from_numpy(gt_spine_mask).float(),
            "meta": meta,
        }
        return out
