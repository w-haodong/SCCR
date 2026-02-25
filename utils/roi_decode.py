# utils/roi_decode.py
import torch
import torch.nn.functional as F

@torch.no_grad()
def _as_4d(hm: torch.Tensor):
    """
    把 hm 统一成 (BK, C, R, R)；返回 (hm4d, B, K, R)
    支持输入 (B,K,C,R,R) 或 (BK,C,R,R)
    """
    if hm.dim() == 5:
        B, K, C, R, _ = hm.shape
        return hm.reshape(B * K, C, R, R).contiguous(), B, K, R
    if hm.dim() == 4:
        BK, C, R, _ = hm.shape
        return hm.contiguous(), 1, BK, R
    raise ValueError(f"hm_prob must be 4D or 5D, got {hm.shape}")

@torch.no_grad()
def _as_off_5d(off: torch.Tensor, B: int, K: int, R: int):
    """
    把 off 统一成 (BK, 4, 2, R, R)；支持输入 (B,K,4,2,R,R) 或 (BK,4,2,R,R)
    """
    if off.dim() == 6:
        return off.reshape(B * K, 4, 2, R, R).contiguous()
    if off.dim() == 5:
        return off.contiguous()
    raise ValueError(f"off_map must be 5D or 6D, got {off.shape}")

@torch.no_grad()
def _nms_maxpool(hm4d: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """CornerNet 风格 NMS，输入必须是 (N,C,H,W)。"""
    if hm4d.dim() != 4:
        raise ValueError(f"NMS expects 4D tensor, got {hm4d.shape}")
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(hm4d, kernel_size=kernel, stride=1, padding=pad)
    keep = (hmax == hm4d).float()
    return hm4d * keep

@torch.no_grad()
def decode_roi_corners_with_offset(hm_prob: torch.Tensor,
                                   off_map: torch.Tensor):
    """
    用“热图峰值 + 偏移图(像素单位)”解码四个角点坐标（ROI 像素坐标系）。
    - hm_prob: (B,K,4,R,R) 或 (BK,4,R,R)  —— 概率(已 sigmoid)
    - off_map: (B,K,4,2,R,R) 或 (BK,4,2,R,R) —— (dx,dy)，建议像素尺度
    返回: List[Tensor]，每个元素形状 (K,4,2)
    """
    # 统一形状
    hm4d, B, K, R = _as_4d(hm_prob)              # (BK,4,R,R)
    off5d = _as_off_5d(off_map, B, K, R)         # (BK,4,2,R,R)

    BK = B * K
    device = off5d.device

    # —— 关键：在可视化/后处理阶段统一用 float32，避免 AMP 的半精度 dtype 冲突
    hm4d = hm4d.float()
    off5d = off5d.float()

    # NMS
    hm_nms = _nms_maxpool(hm4d, kernel=3)        # (BK,4,R,R)

    # 峰值坐标 (整数网格)
    flat = hm_nms.view(BK, 4, -1)
    _, inds = flat.max(dim=-1)                   # (BK,4)
    ys = (inds // R).float()                     # (BK,4)
    xs = (inds %  R).float()                     # (BK,4)

    # 构造 grid（注意 dtype/device 必须与 input 一致）
    xs_n = (xs / (R - 1)) * 2 - 1               # [-1,1]
    ys_n = (ys / (R - 1)) * 2 - 1
    grid = torch.stack([xs_n, ys_n], dim=-1)    # (BK,4,2)
    grid = grid.view(BK * 4, 1, 1, 2).to(device=device, dtype=off5d.dtype)

    # 采样偏移 (dx,dy)
    off_feat = off5d.view(BK * 4, 2, R, R)       # (BK*4,2,R,R)
    sampled = F.grid_sample(off_feat, grid,
                            mode='bilinear',
                            padding_mode='border',
                            align_corners=True)  # (BK*4,2,1,1)
    sampled = sampled.squeeze(-1).squeeze(-1)    # (BK*4,2)
    sampled = sampled.view(BK, 4, 2)             # (BK,4,2)

    # 最终角点 = 峰值 + 偏移（ROI 像素坐标）
    pred_x = xs.unsqueeze(-1) + sampled[..., 0:1]
    pred_y = ys.unsqueeze(-1) + sampled[..., 1:2]
    pred   = torch.cat([pred_x, pred_y], dim=-1) # (BK,4,2)

    if hm_prob.dim() == 5:
        pred = pred.view(B, K, 4, 2)
        return [pred[b] for b in range(B)]
    else:
        return [pred]
