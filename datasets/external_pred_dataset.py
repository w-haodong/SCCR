# -*- coding: utf-8 -*-
# datasets/external_pred_dataset.py

import os
import re
import cv2
import numpy as np
import torch
import torch.utils.data as data
from scipy.io import loadmat

# [关键] 导入您的 transform 模块，确保预处理逻辑一致
from operation import transform

from utils.geometry import (
    calc_connection_features_from_err,
    calc_intrinsic_shape_features_np,
)


def _norm_stem(s: str) -> str:
    s = os.path.splitext(s)[0]
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def _strip_known_prefixes(stem: str, prefixes=("pl_",)) -> str:
    for p in prefixes:
        if stem.lower().startswith(p.lower()):
            return stem[len(p):]
    return stem


def _read_image_unicode(path: str):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


class ExternalPredCorrectionDataset(data.Dataset):
    def __init__(
            self,
            args,
            images_dir: str,
            pred_dir: str,
            gt_dir: str,
            pred_key: str = "pr_landmarks",
            gt_key: str = "p2",
            strip_pred_prefixes=("pl_",),
            image_exts=(".jpg", ".jpeg", ".png", ".bmp"),
            debug_print_examples: bool = True,
    ):
        super().__init__()
        self.args = args
        self.images_dir = images_dir
        self.pred_dir = pred_dir
        self.gt_dir = gt_dir
        self.pred_key = pred_key
        self.gt_key = gt_key

        self.num_vertebrae = int(args.K)
        self.input_h = int(args.input_h)
        self.input_w = int(args.input_w)
        self.down_ratio = int(args.target_feature_stride)

        # [关键] 复用 transform，确保行为一致
        self.eval_aug = transform.Compose([transform.ConvertImgFloat()])

        # --- 1) 收集 images ---
        img_map = {}
        if os.path.isdir(images_dir):
            for fn in os.listdir(images_dir):
                if fn.lower().endswith(image_exts):
                    stem = _norm_stem(fn)
                    img_map[stem] = os.path.join(images_dir, fn)

        # --- 2) 收集 gt mats ---
        gt_map = {}
        if os.path.isdir(gt_dir):
            for fn in os.listdir(gt_dir):
                if fn.lower().endswith(".mat"):
                    stem0 = fn
                    if stem0.lower().endswith(".jpg.mat"):
                        stem0 = stem0[:-len(".jpg.mat")]
                    else:
                        stem0 = os.path.splitext(stem0)[0]
                    stem = _norm_stem(stem0)
                    gt_map[stem] = os.path.join(gt_dir, fn)

        # --- 3) 收集 pred mats ---
        pred_files = []
        if os.path.isdir(pred_dir):
            pred_files = sorted([f for f in os.listdir(pred_dir) if f.lower().endswith(".mat")])

        matched = []
        miss_img = 0
        miss_gt = 0

        pred_stems_dbg = []
        img_stems_dbg = list(img_map.keys())[:5]

        for fn in pred_files:
            pred_path = os.path.join(pred_dir, fn)
            stem_raw = os.path.splitext(fn)[0]
            stem_raw2 = _strip_known_prefixes(stem_raw, prefixes=strip_pred_prefixes)
            stem = _norm_stem(stem_raw2)

            pred_stems_dbg.append(stem)

            img_path = img_map.get(stem, None)
            gt_path = gt_map.get(stem, None)

            if img_path is None:
                miss_img += 1
                continue
            if gt_path is None:
                miss_gt += 1
                continue

            matched.append({
                "stem": stem,
                "img_path": img_path,
                "pred_path": pred_path,
                "gt_path": gt_path,
                "pred_filename": fn,
            })

        print(f"[ExternalPredDataset] pred mats={len(pred_files)}")
        print(f"[ExternalPredDataset] matched  ={len(matched)}")
        print(f"[ExternalPredDataset] miss_img ={miss_img}")
        print(f"[ExternalPredDataset] miss_gt  ={miss_gt}")

        if debug_print_examples:
            print("[ExternalPredDataset][DEBUG] Example pred stems (first 5):")
            for s in pred_stems_dbg[:5]:
                print(f"  - '{s}'")
            print("[ExternalPredDataset][DEBUG] Example image stems (first 5):")
            for s in img_stems_dbg:
                print(f"  - '{s}'")

        if len(matched) == 0:
            raise RuntimeError("[ExternalPredDataset] No matched samples.")

        self.items = matched

    def __len__(self):
        return len(self.items)

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
            if pts is None:
                out_pts.append(None)
                continue
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

    def _load_mat_points(self, mat_path: str, key: str):
        mat = loadmat(mat_path)
        if key in mat:
            arr = mat[key]
            return np.asarray(arr, np.float32)
        return None

    def __getitem__(self, index):
        it = self.items[index]
        img_path = it["img_path"]
        pred_path = it["pred_path"]
        gt_path = it["gt_path"]

        # 1. Load Image
        img_bgr = _read_image_unicode(img_path)
        if img_bgr is None:
            raise RuntimeError(f"Read image failed: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = np.ascontiguousarray(img_rgb)

        # 2. Load Points
        pred_pts = self._load_mat_points(pred_path, self.pred_key)
        if pred_pts is None:
            raise RuntimeError(f"[Pred] key='{self.pred_key}' not found: {pred_path}")
        pred_pts = pred_pts.astype(np.float32).reshape(-1, 2)

        gt_pts = self._load_mat_points(gt_path, self.gt_key)
        if gt_pts is None:
            gt_pts = self._load_mat_points(gt_path, "p2_gt")
        if gt_pts is None:
            raise RuntimeError(f"[GT] key='{self.gt_key}' not found: {gt_path}")
        gt_pts = gt_pts.astype(np.float32).reshape(-1, 2)

        # -----------------------------------------------------------
        # [HARD REQUIREMENT: Z-ORDER -> LOOP-ORDER CONVERSION HERE]
        # 原始 .mat 是 Z形 (TL, TR, BL, BR) -> 必须转为 Loop (TL, TR, BR, BL)
        # -----------------------------------------------------------

        # 临时 reshape 成 (17, 4, 2) 方便操作
        gt_loop = gt_pts.reshape(self.num_vertebrae, 4, 2).copy()
        pred_loop = pred_pts.reshape(self.num_vertebrae, 4, 2).copy()

        # 交换 index 2 和 3 (BL <-> BR)
        gt_loop[:, [2, 3]] = gt_loop[:, [3, 2]]
        pred_loop[:, [2, 3]] = pred_loop[:, [3, 2]]

        # 变回 (-1, 2)
        gt_pts = gt_loop.reshape(-1, 2)
        pred_pts = pred_loop.reshape(-1, 2)

        # -----------------------------------------------------------

        # 3. Sort (已经转为 Loop，无需 angle_sort_all)
        gt_sorted = gt_pts
        pred_sorted = pred_pts

        # 4. Letterbox
        img_sq, [gt_sq, pred_sq], meta = self._letterbox(img_rgb, [gt_sorted, pred_sorted])

        # 5. Transform
        combined_sq = np.concatenate([gt_sq, pred_sq], axis=0).astype(np.float32)
        image_aug, combined_aug = self.eval_aug(img_sq, combined_sq)

        num_gt = gt_sq.shape[0]
        gt_aug = combined_aug[:num_gt]
        pred_aug = combined_aug[num_gt:]

        # 6. Normalize & ToTensor
        image_aug = np.ascontiguousarray(image_aug)
        img_tensor = torch.from_numpy(np.transpose(image_aug, (2, 0, 1))).float()
        img_tensor.div_(255.0).sub_(0.5)

        # 7. Features (现在数据已是 Loop，直接计算特征，特征就是闭环逻辑的)
        connection_features = calc_connection_features_from_err(pred_aug, self.num_vertebrae)
        intrinsic_shape_features = calc_intrinsic_shape_features_np(pred_aug)

        out = {
            "input_image": img_tensor,
            "img_path": img_path,
            "stem": it["stem"],
            "pred_path": pred_path,
            "pred_filename": it["pred_filename"],

            # 输出给 Evaluator 的数据已经是 Loop 顺序了
            "p_err": torch.from_numpy(np.ascontiguousarray(pred_aug)).float(),
            "p_gt": torch.from_numpy(np.ascontiguousarray(gt_aug)).float(),

            "connection_features": torch.from_numpy(np.ascontiguousarray(connection_features)).float(),
            "intrinsic_shape_features": torch.from_numpy(np.ascontiguousarray(intrinsic_shape_features)).float(),
            "meta": meta,
        }
        return out