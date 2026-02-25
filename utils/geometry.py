import numpy as np
import math
import torch
import cv2 # 用于可视化

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    
    r = min(r1, r2, r3)
    return r if not np.isnan(r) else 0

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def rearrange_by_angle_np(pts4):
    """
    1. arctan2 获得循环顺序 [A,B,C,D]
    2. 找到全局“最高点” (min Y)
    3. 在循环中找到它，检查其邻居 (长/短)
    4. 用 X 轴规则确定 TL, TR
    5. 确定 BL, BR
    6. 按 [TL, TR, BR, BL] 顺序返回
    """
    
    # 1. 计算质心 (Centroid) - 这是4个角点的平均中心
    c = np.mean(pts4, axis=0)
    
    # 2. 计算相对于质心的角度
    ang = np.arctan2(pts4[:,1]-c[1], pts4[:,0]-c[0])
    
    # 3. 对角度排序，得到循环顺序索引 (e.g., [3, 0, 1, 2])
    circular_indices = np.argsort(ang)
    
    # 4. 找到“最高点” (Y轴最小) 的 *全局* 索引
    top_point_idx = np.argmin(pts4[:, 1]) # e.g., index 0
    
    # 5. 在 *循环顺序* 中找到这个“最高点”的位置
    # e.g., top_point_idx=0, circular_indices=[3, 0, 1, 2] -> circular_pos=1
    circular_pos = np.where(circular_indices == top_point_idx)[0][0]
    
    # 6. 获取这个“最高点” (p_top)
    p_top = pts4[top_point_idx]
    
    # 7. 获取它在循环列表中的邻居 (使用 % 4 来处理环绕)
    neighbor_idx_1 = circular_indices[(circular_pos - 1) % 4]
    neighbor_idx_2 = circular_indices[(circular_pos + 1) % 4]
    p_neighbor_1 = pts4[neighbor_idx_1]
    p_neighbor_2 = pts4[neighbor_idx_2]
    
    # (获取对角点)
    diag_idx = circular_indices[(circular_pos + 2) % 4]
    p_diag = pts4[diag_idx]
    
    # 8. 计算到两个邻居的距离，判断长短
    dist_1 = np.linalg.norm(p_top - p_neighbor_1)
    dist_2 = np.linalg.norm(p_top - p_neighbor_2)
    
    p_long_neighbor = p_neighbor_1 if dist_1 > dist_2 else p_neighbor_2
    p_short_neighbor = p_neighbor_2 if dist_1 > dist_2 else p_neighbor_1
    
    # 9. 应用你的X轴规则 (在 p_top 和 p_long_neighbor 之间)
    # "x轴上，谁在左，就是左上点"
    
    if p_top[0] < p_long_neighbor[0]:
        # p_top 在左边 -> p_top 是 TL
        p_TL = p_top
        p_TR = p_long_neighbor
    else:
        # p_long_neighbor 在左边 -> p_long_neighbor 是 TL
        p_TL = p_long_neighbor
        p_TR = p_top
        
    # 10. 识别 BL 和 BR
    if np.array_equal(p_TL, p_top):
        p_BL = p_short_neighbor
        p_BR = p_diag
    else:
        p_BR = p_short_neighbor
        p_BL = p_diag
        
    # 11. 按 [TL, TR, BR, BL] 顺序返回
    new_order_pts = np.stack([
        p_TL, # p0 (Top-Left)
        p_TR, # p1 (Top-Right)
        p_BR, # p2 (Bottom-Right)
        p_BL  # p3 (Bottom-Left)
    ])
    
    return new_order_pts


def angle_sort_all(flat_pts):
    """ (Numpy 版本，用于 Dataset) 
    使用修复后的、保留形状的按邻接关系排序。
    """
    if flat_pts is None or flat_pts.size == 0: 
        return flat_pts
    K = flat_pts.shape[0] // 4
    out = np.zeros_like(flat_pts)
    for v in range(K):
        s = v*4
        pts_4 = flat_pts[s:s+4]
        if np.any(np.isnan(pts_4)) or np.all(pts_4 == 0):
            out[s:s+4] = pts_4
            continue
        out[s:s+4] = rearrange_by_angle_np(pts_4)
    
    return out # (K*4, 2)

def calc_connection_features_from_err(p_err_points_np, K):
    """ 
    为 RNN 计算连接特征。
    输出: (K-1, 3) -> [dx_norm, dy_norm, L_norm]
    """
    num_v = K; num_conn = num_v - 1
    
    if p_err_points_np is None or p_err_points_np.shape[0] != num_v*4 or np.all(p_err_points_np == 0):
        return np.zeros((max(0, num_conn), 3), dtype=np.float32)
        
    try:
        p = p_err_points_np.reshape((num_v, 4, 2))
        centers = np.mean(p, axis=1) # (K, 2)
        
        # 修复后的排序 p0,p1,p2,p3 对应 [TL, TR, BR, BL]
        p0,p1,p2,p3 = p[:,0],p[:,1],p[:,2],p[:,3]
        
        # p0-p1 (长, Top), p1-p2 (短, Right), p2-p3 (长, Bottom), p3-p0 (短, Left)
        edge01 = np.linalg.norm(p0-p1, axis=-1) # 长
        edge12 = np.linalg.norm(p1-p2, axis=-1) # 短
        edge23 = np.linalg.norm(p2-p3, axis=-1) # 长
        edge30 = np.linalg.norm(p3-p0, axis=-1) # 短
        
        avg_size = (edge01+edge12+edge23+edge30)/4.0
        Global = np.mean(avg_size[avg_size>0]) + 1e-6 # 避免除以0

        conn = np.zeros((num_conn, 3), dtype=np.float32)
        
        for i in range(num_conn):
            v = centers[i+1] - centers[i]
            L = np.linalg.norm(v)
            if L < 1e-6: 
                conn[i]=0
            else:
                conn[i,0]=v[0]/L; conn[i,1]=v[1]/L; conn[i,2]=L/Global
                
        return conn
        
    except Exception as e:
        # 增加异常打印，以防万一
        print(f"[Error in calc_connection_features_from_err]: {e}")
        return np.zeros((max(0, num_conn), 3), dtype=np.float32)


def calc_intrinsic_shape_features_np(p_k42, eps: float = 1e-8) -> np.ndarray:
    """
    (Numpy) 提取8个归一化的、基于中点的形状特征。
    输入: (K*4, 2) 或 (K, 4, 2)
    输出: (K, 8)
    """
    if p_k42.ndim == 2:
        K = p_k42.shape[0] // 4
        p_k42 = p_k42.reshape((K, 4, 2))
    else:
        K = p_k42.shape[0]
        
    out_features = np.zeros((K, 8), dtype=np.float32)
    
    # 1. 基本组件
    p0 = p_k42[:, 0, :]; p1 = p_k42[:, 1, :]
    p2 = p_k42[:, 2, :]; p3 = p_k42[:, 3, :]
    
    # 2. 边向量
    e_top = p1 - p0
    e_right = p2 - p1
    e_bottom = p3 - p2 # 注意方向 p3-p2
    e_left = p0 - p3  # 注意方向 p0-p3
    
    # 3. 边长
    e_top_len = np.linalg.norm(e_top, axis=-1)
    e_right_len = np.linalg.norm(e_right, axis=-1)
    e_bottom_len = np.linalg.norm(e_bottom, axis=-1)
    e_left_len = np.linalg.norm(e_left, axis=-1)
    
    # 4. 中点轴
    mid_L = (p0 + p3) / 2.0; mid_R = (p1 + p2) / 2.0
    mid_T = (p0 + p1) / 2.0; mid_B = (p2 + p3) / 2.0
    
    long_axis_vec = mid_R - mid_L
    lat_axis_vec = mid_B - mid_T
    
    long_axis_len = np.linalg.norm(long_axis_vec, axis=-1)
    lat_axis_len = np.linalg.norm(lat_axis_vec, axis=-1)

    # 5. 辅助函数：安全的 log(a/b)
    def safe_log_ratio(a, b, eps):
        return np.log((a + eps) / (b + eps))
        
    # 6. 辅助函数：安全的 cosine similarity
    def safe_cosine(v1, v2, eps):
        dot = np.sum(v1 * v2, axis=-1)
        norm = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
        return np.clip(dot / (norm + eps), -1.0, 1.0)

    # --- (开始计算8个特征) ---
    
    # 特征 1-2: 对边比率
    out_features[:, 0] = safe_log_ratio(e_top_len, e_bottom_len, eps)
    out_features[:, 1] = safe_log_ratio(e_left_len, e_right_len, eps)
    
    # 特征 3: 中点轴比率
    out_features[:, 2] = safe_log_ratio(long_axis_len, lat_axis_len, eps)
    
    # 特征 4: 中点轴正交性
    out_features[:, 3] = safe_cosine(long_axis_vec, lat_axis_vec, eps)
    
    # 特征 5-8: 内角
    out_features[:, 4] = safe_cosine(-e_left, e_top, eps)
    out_features[:, 5] = safe_cosine(-e_top, e_right, eps)
    out_features[:, 6] = safe_cosine(-e_right, -e_bottom, eps)
    out_features[:, 7] = safe_cosine(e_left, e_bottom, eps)
    
    return np.nan_to_num(out_features, nan=0.0, posinf=0.0, neginf=0.0)
