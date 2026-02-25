# utils/vis_s2_click_center.py
import numpy as np
import cv2


def _to_np(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def vis_s2_click_center(img_bgr, clicks_abs, pred_centers_abs, gt_centers_abs=None, radius=4, draw_line=True):
    """
    img_bgr: (H,W,3) uint8
    clicks_abs/pred_centers_abs/gt_centers_abs: (K,2) abs pixel coords (numpy or torch)
    Color:
      click: blue (255,0,0)
      pred center: white (255,255,255)
      gt center: green (0,255,0) (optional)
    """
    img = img_bgr.copy()
    c = _to_np(clicks_abs)
    p = _to_np(pred_centers_abs)
    g = _to_np(gt_centers_abs)

    if c is None or p is None:
        return img

    K = c.shape[0]
    for k in range(K):
        if np.abs(c[k]).sum() < 1e-4:
            continue

        cx, cy = int(round(float(c[k, 0]))), int(round(float(c[k, 1])))
        px, py = int(round(float(p[k, 0]))), int(round(float(p[k, 1])))

        cv2.circle(img, (cx, cy), radius, (255, 0, 0), -1)              # click
        cv2.circle(img, (px, py), radius, (255, 255, 255), -1)          # pred center

        if draw_line:
            cv2.line(img, (cx, cy), (px, py), (255, 255, 255), 1)

        if g is not None and np.abs(g[k]).sum() > 1e-4:
            gx, gy = int(round(float(g[k, 0]))), int(round(float(g[k, 1])))
            cv2.circle(img, (gx, gy), radius, (0, 255, 0), 1)           # gt center (hollow)

    return img
