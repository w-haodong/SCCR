# -*- coding: utf-8 -*-
# operation/test_qt.py
# FINAL VERSION: Pure Geometric Extension (No Mask Noise) + 4-Color Vis

import sys
import os
import copy
import traceback
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSizePolicy, QSpinBox
)
from PyQt5.QtCore import Qt, QPointF, QRectF, QTimer
from PyQt5.QtGui import QPainter, QPen, QColor, QFont, QBrush, QImage, QPolygonF

from models.SAICNet import saic_net
from datasets.dataset import CorrectionDataset


def collater(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


class ImageCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.img_rgb = None
        self.p_err_centers = None
        self.p_err_corners = None

        self.node_colors = None
        self.conn_colors = None

        # 调试点击位置 (粉色叉号)
        self.debug_click_pos = None

        self.scale = 1.0
        self.offset = QPointF(0, 0)

        # center hover/selected
        self.hover_idx = -1
        self.selected_idx = -1

        # corner hover/selected: (k, i)
        self.hover_corner_k = -1
        self.hover_corner_i = -1
        self.selected_corner_k = -1
        self.selected_corner_i = -1

        # dragging state
        self.is_dragging_center = False
        self.is_dragging_corner = False

        self.refine_callback = None

        # 命中阈值（图像坐标系像素）
        self.center_hit_thresh = 40.0
        self.corner_hit_thresh = 12.0

    def set_data(self, img_rgb, p_err_raw, error_logits):
        if img_rgb is not None:
            self.img_rgb = img_rgb.copy()
        else:
            self.img_rgb = None

        try:
            if p_err_raw is not None:
                pts = p_err_raw.reshape(-1, 4, 2).astype(np.float32)
                self.p_err_corners = pts
                self.p_err_centers = pts.mean(axis=1)
            else:
                self.p_err_corners = None
                self.p_err_centers = None
        except Exception:
            self.p_err_corners = None
            self.p_err_centers = None

        K = 17
        if self.p_err_centers is not None:
            K = len(self.p_err_centers)

        if error_logits is not None:
            error_logits = np.nan_to_num(error_logits, nan=0.0)
            if error_logits.ndim == 3:
                error_logits = error_logits[0]

            if error_logits.shape[0] != K:
                self.node_colors = np.zeros(K, dtype=bool)
                self.conn_colors = np.zeros(K, dtype=bool)
            else:
                probs = 1.0 / (1.0 + np.exp(-error_logits))
                self.node_colors = probs[:, 0] > 0.5
                self.conn_colors = probs[:, 1] > 0.5
        else:
            self.node_colors = np.zeros(K, dtype=bool)
            self.conn_colors = np.zeros(K, dtype=bool)

        # reset hover/selected when new data comes
        self.hover_idx = -1
        self.selected_idx = -1
        self.hover_corner_k = self.hover_corner_i = -1
        self.selected_corner_k = self.selected_corner_i = -1
        self.is_dragging_center = False
        self.is_dragging_corner = False

        self.update()

    def update_data_interactive(self, new_corners, new_logits):
        self.set_data(self.img_rgb, new_corners, new_logits)

    def set_debug_click(self, x, y):
        self.debug_click_pos = (x, y)
        self.update()

    def register_callback(self, func):
        self.refine_callback = func

    def win2img(self, pos):
        if self.scale == 0:
            return None
        x = float(pos.x() - self.offset.x()) / self.scale
        y = float(pos.y() - self.offset.y()) / self.scale
        return (x, y)

    def img2win(self, x, y):
        if x is None or y is None or np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y):
            return QPointF(-100, -100)
        wx = float(x) * self.scale + self.offset.x()
        wy = float(y) * self.scale + self.offset.y()
        return QPointF(wx, wy)

    # ----------------------------
    # hit test
    # ----------------------------
    def find_center_hit(self, x, y):
        if self.p_err_centers is None:
            return -1
        dists = np.linalg.norm(self.p_err_centers - np.array([x, y], dtype=np.float32), axis=1)
        min_idx = int(np.argmin(dists))
        if dists[min_idx] < self.center_hit_thresh:
            return min_idx
        return -1

    def find_corner_hit(self, x, y):
        """
        返回 (k, i)；没命中返回 (-1, -1)
        """
        if self.p_err_corners is None:
            return -1, -1
        pts = self.p_err_corners.reshape(-1, 2)  # (K*4,2)
        if pts.size == 0:
            return -1, -1

        # 过滤nan
        valid = ~np.any(np.isnan(pts), axis=1)
        if not np.any(valid):
            return -1, -1

        vpts = pts[valid]
        d = vpts - np.array([x, y], dtype=np.float32)[None, :]
        dists = np.sqrt((d * d).sum(axis=1))
        j_local = int(np.argmin(dists))
        if dists[j_local] >= self.corner_hit_thresh:
            return -1, -1

        valid_indices = np.where(valid)[0]
        j = int(valid_indices[j_local])  # index in K*4
        k = j // 4
        i = j % 4
        return k, i

    def _clear_hover(self):
        self.hover_idx = -1
        self.hover_corner_k = -1
        self.hover_corner_i = -1

    # ----------------------------
    # mouse events
    # ----------------------------
    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        img_xy = self.win2img(event.pos())
        if not img_xy:
            return
        x, y = img_xy

        # 角点优先（避免点到角点却拖了中心）
        ck, ci = self.find_corner_hit(x, y)
        if ck != -1:
            self.selected_corner_k = ck
            self.selected_corner_i = ci
            self.is_dragging_corner = True

            # 取消中心选择
            self.selected_idx = -1
            self.is_dragging_center = False
            self.update()
            return

        # 否则尝试中心点
        idx = self.find_center_hit(x, y)
        if idx != -1:
            self.selected_idx = idx
            self.is_dragging_center = True

            # 取消角点选择
            self.selected_corner_k = -1
            self.selected_corner_i = -1
            self.is_dragging_corner = False
            self.update()
            return

    def mouseMoveEvent(self, event):
        img_xy = self.win2img(event.pos())
        if not img_xy:
            return
        x, y = img_xy

        # 拖角点：仅更新 p_err_corners，不改 centers，不触发任何回调
        if self.is_dragging_corner and self.selected_corner_k != -1:
            k = self.selected_corner_k
            i = self.selected_corner_i
            if self.p_err_corners is not None and 0 <= k < len(self.p_err_corners) and 0 <= i < 4:
                self.p_err_corners[k, i, 0] = float(x)
                self.p_err_corners[k, i, 1] = float(y)
            self.update()
            return

        # 拖中心点：仅移动中心
        if self.is_dragging_center and self.selected_idx != -1:
            self.p_err_centers[self.selected_idx] = [float(x), float(y)]
            self.update()
            return

        # hover（非拖动状态）
        old_center = self.hover_idx
        old_ck, old_ci = self.hover_corner_k, self.hover_corner_i

        ck, ci = self.find_corner_hit(x, y)
        if ck != -1:
            self.hover_corner_k, self.hover_corner_i = ck, ci
            self.hover_idx = -1
        else:
            self.hover_corner_k, self.hover_corner_i = -1, -1
            self.hover_idx = self.find_center_hit(x, y)

        if old_center != self.hover_idx or old_ck != self.hover_corner_k or old_ci != self.hover_corner_i:
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton:
            return

        # 释放角点：不触发纠正
        if self.is_dragging_corner:
            self.is_dragging_corner = False
            self.selected_corner_k = -1
            self.selected_corner_i = -1
            self.update()
            return

        # 释放中心：触发纠正
        if self.is_dragging_center and self.selected_idx != -1:
            if self.refine_callback:
                curr_pt = self.p_err_centers[self.selected_idx]
                self.set_debug_click(curr_pt[0], curr_pt[1])
                print(f"\n[GUI] Release Node {self.selected_idx} at ({curr_pt[0]:.1f}, {curr_pt[1]:.1f})")
                try:
                    new_corners, new_logits = self.refine_callback(
                        self.selected_idx, float(curr_pt[0]), float(curr_pt[1])
                    )
                    if new_corners is not None:
                        self.update_data_interactive(new_corners, new_logits)
                    else:
                        self.update()
                except Exception:
                    traceback.print_exc()

        self.is_dragging_center = False
        self.selected_idx = -1
        self.update()

    # ----------------------------
    # paint
    # ----------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(40, 40, 40))

        if self.img_rgb is None:
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "No Image")
            return

        H, W, C = self.img_rgb.shape
        win_w, win_h = self.width(), self.height()
        if H == 0 or W == 0:
            return

        try:
            bytes_per_line = self.img_rgb.strides[0]
            qimg = QImage(self.img_rgb.data, W, H, bytes_per_line, QImage.Format_RGB888)

            self.scale = min(win_w / W, win_h / H) * 0.95
            disp_w = W * self.scale
            disp_h = H * self.scale
            ox = (win_w - disp_w) / 2
            oy = (win_h - disp_h) / 2
            self.offset = QPointF(ox, oy)

            painter.drawImage(QRectF(ox, oy, disp_w, disp_h), qimg)
        except Exception as e:
            print(f"Paint Error: {e}")
            return

        if self.p_err_centers is None:
            return

        # 4 色角点
        corner_colors = [
            QColor(255, 255, 0),    # 0: Red
            QColor(255, 255, 0),    # 1: Green
            QColor(255, 255, 0),  # 2: Blue
            QColor(255, 255, 0),  # 3: Yellow
        ]

        pen_err = QPen(QColor(255, 0, 0), 2, Qt.DashLine)
        pen_ok = QPen(QColor(0, 255, 0), 2, Qt.SolidLine)
        pen_conn_err = QPen(QColor(255, 0, 0), 2, Qt.DashLine)
        pen_conn_ok = QPen(QColor(0, 255, 0), 2, Qt.SolidLine)

        brush_hover = QBrush(QColor(255, 255, 255))
        brush_err = QBrush(QColor(255, 0, 0))
        brush_ok = QBrush(QColor(0, 255, 0))

        K = len(self.p_err_centers)

        # 1) 连线（按中心点）
        for k in range(K - 1):
            p1 = self.img2win(*self.p_err_centers[k])
            p2 = self.img2win(*self.p_err_centers[k + 1])
            is_err = bool(self.conn_colors[k]) if self.conn_colors is not None and len(self.conn_colors) > k else False
            painter.setPen(pen_conn_err if is_err else pen_conn_ok)
            painter.drawLine(p1, p2)

        # 2) 椎体框 + 角点 + 中心点
        for k in range(K):
            is_err = bool(self.node_colors[k]) if self.node_colors is not None and len(self.node_colors) > k else False

            # A) 画框 + 角点
            if self.p_err_corners is not None:
                pts = self.p_err_corners[k]
                if not np.any(np.isnan(pts)):
                    poly_pts = [self.img2win(p[0], p[1]) for p in pts]
                    poly = QPolygonF(poly_pts)

                    painter.setPen(pen_err if is_err else pen_ok)
                    painter.setBrush(Qt.NoBrush)
                    painter.drawPolygon(poly)

                    # 角点
                    for i in range(4):
                        if i >= len(poly_pts):
                            continue
                        pt_win = poly_pts[i]
                        is_corner_active = (
                            (k == self.selected_corner_k and i == self.selected_corner_i) or
                            (k == self.hover_corner_k and i == self.hover_corner_i)
                        )
                        r = 5 if is_corner_active else 3
                        painter.setPen(Qt.NoPen)
                        painter.setBrush(QBrush(corner_colors[i]))
                        painter.drawEllipse(pt_win, r, r)

            # B) 画中心点（中心点拖动会触发纠正）
            center = self.img2win(*self.p_err_centers[k])
            is_center_active = (k == self.selected_idx) or (k == self.hover_idx)

            painter.setPen(Qt.NoPen)
            painter.setBrush(brush_hover if is_center_active else (brush_err if is_err else brush_ok))
            painter.drawEllipse(center, 6 if is_center_active else 4, 6 if is_center_active else 4)

            painter.setPen(QColor(255, 255, 0))
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            painter.drawText(center + QPointF(10, -5), str(k))



class SAIC_GUI(QWidget):
    def __init__(self, args, peft_encoder):
        super().__init__()
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.dataset = None
        self.total = 0
        self.idx = 0
        self.model = None
        self.cache = None
        self.domain_options = ['source', 'target']
        self.current_domain_idx = 0

        self._auto_running = False
        self._auto_state = None

        self.init_ui()
        self.init_model(peft_encoder)
        self.init_data(domain=self.domain_options[self.current_domain_idx])
        self.load(0)

    def init_ui(self):
        self.setWindowTitle("SAIC-Net: Interactive Correction")
        self.resize(1200, 900)
        l = QVBoxLayout(self)

        top = QHBoxLayout()
        self.lbl = QLabel("Loading...")
        self.lbl.setFont(QFont("Arial", 14, QFont.Bold))
        top.addWidget(self.lbl)

        self.btn_domain = QPushButton(f"Data: {self.domain_options[self.current_domain_idx].upper()}")
        self.btn_domain.setMinimumHeight(40)
        self.btn_domain.setStyleSheet("font-weight: bold; font-size: 14px; background-color: #4CAF50; color: white;")
        self.btn_domain.clicked.connect(self.switch_domain)
        top.addWidget(self.btn_domain)

        l.addLayout(top)

        self.canvas = ImageCanvas()
        self.canvas.register_callback(self.infer_and_update)
        l.addWidget(self.canvas)

        bot = QHBoxLayout()
        b_p = QPushButton("<< Prev")
        b_p.clicked.connect(self.prev)
        b_p.setMinimumHeight(40)
        bot.addWidget(b_p)

        b_n = QPushButton("Next >>")
        b_n.clicked.connect(self.next)
        b_n.setMinimumHeight(40)
        bot.addWidget(b_n)

        b_r = QPushButton("Reset")
        b_r.clicked.connect(self.reset)
        b_r.setMinimumHeight(40)
        bot.addWidget(b_r)

        self.btn_auto_start = QPushButton("Auto Correct")
        self.btn_auto_start.setMinimumHeight(40)
        self.btn_auto_start.setStyleSheet(
            "font-weight: bold; font-size: 14px; background-color: #2196F3; color: white;")
        self.btn_auto_start.clicked.connect(self.auto_correct_start)
        bot.addWidget(self.btn_auto_start)

        l.addLayout(bot)

    def switch_domain(self):
        self.current_domain_idx = (self.current_domain_idx + 1) % len(self.domain_options)
        new_domain = self.domain_options[self.current_domain_idx]
        self.btn_domain.setText(f"Data: {new_domain.upper()}")
        self.init_data(domain=new_domain)
        self.load(0)

    def init_model(self, enc):
        print("Loading Model...")
        self.model = saic_net(enc, self.args).to(self.device)
        self.model.eval()
        candidates = [
            "best_model.pth", "latest_model.pth",
            os.path.join(self.args.work_dir, "best_model.pth"),
            "checkpoints/best_model.pth"
        ]
        ckpt_path = None
        for p in candidates:
            if os.path.exists(p):
                ckpt_path = p
                break

        if ckpt_path:
            print(f"✅ Found checkpoint: {ckpt_path}")
            st = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(st.get("saic", st), strict=False)
        else:
            print("❌ No checkpoint found!")

    def init_data(self, domain='source'):
        da = copy.copy(self.args)
        self.dataset = CorrectionDataset(da, phase="test", domain=domain)
        self.total = len(self.dataset)
        print(f"[GUI] Dataset '{domain}' loaded with {self.total} samples.")

    def load(self, idx):
        if idx < 0 or idx >= self.total: return

        if self._auto_running:
            self.auto_correct_stop()

        self.idx = idx
        self.lbl.setText(f"Sample: {idx + 1} / {self.total}")
        self.canvas.debug_click_pos = None

        try:
            sample = self.dataset[idx]
            batch = collater([sample])
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)

            im = batch['input_image'][0].permute(1, 2, 0).detach().cpu().numpy()
            im = ((im * 0.5 + 0.5) * 255).astype(np.uint8)
            rgb = np.ascontiguousarray(im)

            with torch.no_grad():
                x = batch["input_image"]
                vit_out = self.model.encoder(x, self.model.vit_input_layer_indices)
                out = self.model(batch)

                self.cache = {
                    's1_content_map': out['s1_content_map'],
                    'pred_global_hm': out['pred_global_hm'],
                    'pred_corner_offsets': out['pred_corner_offsets'],
                    's2_roi_half_abs': out['s2_roi_half_abs'],
                    'vit_outputs': vit_out,
                    'img_shape': x.shape[2:],
                    'pred_spine_mask_logits': out.get('pred_spine_mask_logits', None),
                }

                err_logits = out['error_logits'].cpu().numpy() if 'error_logits' in out else None
                p_err = batch['p_err'][0].cpu().numpy() if 'p_err' in batch else None
                self.canvas.set_data(rgb, p_err, err_logits)
        except Exception:
            traceback.print_exc()

    def infer_and_update(self, idx, x, y):
        try:
            with torch.no_grad():
                res = self.model.inference_interactive(self.cache, idx, x, y)
            refined_corn = res['refined_corners']

            current_corners = self.canvas.p_err_corners.copy()
            current_corners[idx] = refined_corn
            centers = current_corners.mean(axis=1)
            sorted_corners = current_corners[np.argsort(centers[:, 1])]

            new_p_err_tensor = torch.from_numpy(sorted_corners).unsqueeze(0).to(self.device)
            with torch.no_grad():
                new_logits = self.model.re_predict_errors(
                    cached_vit_outputs=self.cache['vit_outputs'],
                    new_p_err_abs=new_p_err_tensor,
                    img_shape_HW=self.cache['img_shape']
                )
            return sorted_corners, new_logits.cpu().numpy()
        except Exception:
            traceback.print_exc()
            return None, None

    def prev(self):
        if self.idx > 0: self.load(self.idx - 1)

    def next(self):
        if self.idx < self.total - 1: self.load(self.idx + 1)

    def reset(self):
        self.load(self.idx)

    def _get_segmentation_data(self):
        # 仍然保留，用于计算中间间隙的中线查询，但不用于头尾补全
        logits = self.cache.get('pred_spine_mask_logits', None)
        if logits is None or self.canvas.img_rgb is None: return None, None
        H_real, W_real, _ = self.canvas.img_rgb.shape
        with torch.no_grad():
            prob_tensor = torch.sigmoid(logits)
            prob_high_res = torch.nn.functional.interpolate(prob_tensor, size=(H_real, W_real), mode='bilinear',
                                                            align_corners=False)
            mask = (prob_high_res[0, 0].cpu().numpy() > 0.3).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 30))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels == largest_label).astype(np.uint8) * 255
        else:
            return mask, []
        ys, xs = np.where(mask > 0)
        if len(ys) == 0: return mask, []
        y_min_raw, y_max_raw = int(ys.min()), int(ys.max())
        shrink_ratio = 0.01
        margin = (y_max_raw - y_min_raw) * shrink_ratio
        target_y_top = y_min_raw + margin
        target_y_bot = y_max_raw - margin
        centerline_pts = []
        win_h = int(H_real * 0.035)
        if win_h < 5: win_h = 5
        for cy in range(y_min_raw, y_max_raw + 1, 5):
            y_start = max(y_min_raw, cy - win_h)
            y_end = min(y_max_raw, cy + win_h)
            window_mask = mask[y_start:y_end, :]
            win_ys_local, win_xs = np.where(window_mask > 0)
            if len(win_ys_local) == 0: continue
            mean_x = np.mean(win_xs)
            mean_y = np.mean(win_ys_local + y_start)
            centerline_pts.append((mean_x, mean_y))
        if len(centerline_pts) >= 2:
            centerline_pts.sort(key=lambda p: p[1])
            centerline_pts = [p for p in centerline_pts if target_y_top <= p[1] <= target_y_bot]
            if len(centerline_pts) < 2: return mask, centerline_pts
            pts_arr = np.array(centerline_pts)
            # Top Anchor
            n_fit = min(len(pts_arr), 10)
            k, b = np.polyfit(pts_arr[:n_fit, 1], pts_arr[:n_fit, 0], 1)
            target_x = k * target_y_top + b
            if abs(pts_arr[0, 1] - target_y_top) < 1.0:
                centerline_pts[0] = (target_x, float(target_y_top))
            else:
                centerline_pts.insert(0, (target_x, float(target_y_top)))
            # Bottom Anchor
            k, b = np.polyfit(pts_arr[-n_fit:, 1], pts_arr[-n_fit:, 0], 1)
            target_x = k * target_y_bot + b
            if abs(pts_arr[-1, 1] - target_y_bot) < 1.0:
                centerline_pts[-1] = (target_x, float(target_y_bot))
            else:
                centerline_pts.append((target_x, float(target_y_bot)))
        return mask, centerline_pts

    def _get_centerline_x_at_y(self, target_y, centerline_pts):
        # 仅用于中间间隙
        if not centerline_pts: return None
        pts_arr = np.array(centerline_pts)
        ys = pts_arr[:, 1]
        xs = pts_arr[:, 0]
        if target_y < ys.min():
            v = pts_arr[:8]
            k, b = np.polyfit(v[:, 1], v[:, 0], 1)
            return float(k * target_y + b)
        elif target_y > ys.max():
            v = pts_arr[-8:]
            k, b = np.polyfit(v[:, 1], v[:, 0], 1)
            return float(k * target_y + b)
        else:
            return float(np.interp(target_y, ys, xs))

    def _get_node_geometry(self, k):
        # 不打散点序，只按索引分组
        if self.canvas.p_err_corners is None:
            return None, None, None, 0.0

        corners = self.canvas.p_err_corners[k] # (4, 2)
        center = np.mean(corners, axis=0)      # 几何中心

        # 1. 直接按索引取前两个和后两个
        # 假设：0,1 是一个终板（如上终板），2,3 是另一个终板（如下终板）
        pair_a = corners[:2]  # 前两点
        pair_b = corners[2:]  # 后两点

        # 2. 计算这两组的中点
        mid_a = np.mean(pair_a, axis=0)
        mid_b = np.mean(pair_b, axis=0)

        # 3. 确定谁是 Top 谁是 Bot
        # 在图像坐标系中，Y值越小越靠上。
        # 即使椎体倾斜，我们只需要知道哪一组在物理上方，哪一组在下方
        if mid_a[1] < mid_b[1]:
            top_mid = mid_a
            bot_mid = mid_b
        else:
            top_mid = mid_b
            bot_mid = mid_a

        # 4. 计算高度
        height = np.linalg.norm(top_mid - bot_mid)

        return top_mid, bot_mid, center, height

    def _find_nearest_error_node(self, target_x, target_y):
        if self.canvas.node_colors is None: return -1
        err_indices = np.where(self.canvas.node_colors)[0]
        if len(err_indices) == 0: return -1
        target = np.array([target_x, target_y])
        centers = self.canvas.p_err_centers[err_indices]
        dists = np.linalg.norm(centers - target, axis=1)
        min_local_idx = np.argmin(dists)
        return err_indices[min_local_idx]


    # ============================================================
    # 自动纠正：纯几何延伸
    # ============================================================

    def auto_correct_start(self):
        if self._auto_running: return
        if self.cache is None: return
        mask, centerline = self._get_segmentation_data()
        self._auto_state = {
            'steps': 0, 'max_steps': int(200),
            'centerline': centerline, 'last_action': None
        }
        self._auto_running = True
        print("\n[AUTO] 启动：纯几何延伸补全 + 4色可视化")
        QTimer.singleShot(10, self._auto_step)

    def auto_correct_stop(self):
        if self._auto_running: print("[AUTO] 停止。")
        self._auto_running = False
        self._auto_state = None

    def _auto_step(self):
        if not self._auto_running: return
        st = self._auto_state
        if st is None or st['steps'] >= st['max_steps']: self.auto_correct_stop(); return
        if self.canvas.node_colors is None: self.auto_correct_stop(); return

        green_indices = np.where(~self.canvas.node_colors)[0]
        centers = self.canvas.p_err_centers
        if len(green_indices) == 0: self.auto_correct_stop(); return

        sorted_args = np.argsort(centers[green_indices, 1])
        green_sorted = green_indices[sorted_args]
        action_taken = False

        # 中线信息（仅用于 trigger，不用于坐标计算）
        if st['centerline'] and len(st['centerline']) > 0:
            cl_start_y = st['centerline'][0][1]
            cl_end_y = st['centerline'][-1][1]
            cl_len = max(cl_end_y - cl_start_y, 1.0)
        else:
            cl_start_y, cl_end_y, cl_len = 0, 10000, 10000

        # A. Gap Check (保持)
        for i in range(len(green_sorted) - 1):
            upper_k = green_sorted[i]
            lower_k = green_sorted[i + 1]
            u_top, u_bot, u_cen, u_h = self._get_node_geometry(upper_k)
            l_top, l_bot, l_cen, l_h = self._get_node_geometry(lower_k)
            gap_dist = np.linalg.norm(u_bot - l_top)
            avg_height = (u_h + l_h) / 2.0
            if gap_dist > avg_height:
                curr_y = u_cen[1]
                progress = (curr_y - cl_start_y) / cl_len
                progress = np.clip(progress, 0.0, 1.0)
                dynamic_ratio = 0.3 + (0.5 - 0.3) * progress
                step_distance = u_h * (1.0 + dynamic_ratio)
                target_y = u_cen[1] + step_distance
                if target_y < l_cen[1]:
                    target_x = self._get_centerline_x_at_y(target_y, st['centerline'])
                    if target_x is not None:
                        err_node = self._find_nearest_error_node(target_x, target_y)
                        if err_node != -1:
                            self._execute_click(err_node, target_x, target_y, st)
                            action_taken = True
                            break
        if action_taken: return

        # =========================================================
        # B. 检查两端 (Ends Check) - 纯几何向量延伸
        # =========================================================

        f_k = green_sorted[0]  # 最上面
        l_k = green_sorted[-1]  # 最下面

        # 1. 获取几何数据: top_mid, bot_mid, center, height
        f_top, f_bot, f_cen, f_h = self._get_node_geometry(f_k)
        l_top, l_bot, l_cen, l_h = self._get_node_geometry(l_k)

        dist_top = f_cen[1] - cl_start_y
        dist_bot = cl_end_y - l_cen[1]

        target_k = -1
        target_x, target_y = 0, 0

        # 1. 头部补全
        if dist_top > f_h * 0.8 and dist_top > dist_bot:
            print(f"[AUTO-Top] 头部缺失 (Geom).")
            # 向量 = 上 - 下 (指向正上方)
            vec = f_top - f_bot
            # 目标 = 上终板 + 向量 * 1.3
            target_pt = f_top + vec * 1.03
            target_x, target_y = target_pt[0], target_pt[1]
            target_k = self._find_nearest_error_node(target_x, target_y)

        # 2. 尾部补全
        elif dist_bot > l_h * 0.8:
            print(f"[AUTO-Bot] 尾部缺失 (Geom).")
            # 向量 = 下 - 上 (指向正下方)
            vec = l_bot - l_top
            # 目标 = 下终板 + 向量 * 1.3
            target_pt = l_bot + vec * 1.03
            target_x, target_y = target_pt[0], target_pt[1]
            target_k = self._find_nearest_error_node(target_x, target_y)

        if target_k != -1:
            self._execute_click(target_k, target_x, target_y, st)
            action_taken = True

        if not action_taken:
            print("[AUTO] 扫描完毕，未发现可纠正的间隙。")
            self.auto_correct_stop()

    def _execute_click(self, k, x, y, st):
        curr_action = (int(k), int(x), int(y))
        if st['last_action'] == curr_action:
            print("[AUTO] 陷入重复操作，强制停止。")
            self.auto_correct_stop()
            return
        self.canvas.set_debug_click(x, y)
        st['last_action'] = curr_action
        st['steps'] += 1
        print(f"[AUTO] 执行纠正 -> 将节点 {k} 移至 ({x:.1f}, {y:.1f})")
        new_corners, new_logits = self.infer_and_update(int(k), float(x), float(y))
        if new_corners is not None:
            self.canvas.update_data_interactive(new_corners, new_logits)

        QTimer.singleShot(100, self._auto_step)
