import torch
import argparse
import os
import sys

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ---- 让 dinov3 作为顶层包可见（models/dino/dinov3/...）----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DINO_ROOT = os.path.join(_THIS_DIR, 'models', 'dino')  # 该目录下应有 dinov3/
if _DINO_ROOT not in sys.path:
    sys.path.insert(0, _DINO_ROOT)

from operation.train import Trainer
from PyQt5.QtWidgets import QApplication # 需要这个来启动应用
from models.sam.segment_anything import build_sam_encoder_only
from peft import get_peft_model, LoraConfig
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser(
        description='SAIC-Net V-Global: HM + Detector with Canonical Patch + Corner Regression Head'
    )

    # ---------------- 路径与环境 ----------------
    parser.add_argument('--work_dir', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='')


    # 训练模式：单域 or 双域联合 + 对抗
    parser.add_argument('--train_mode', type=str, default='joint',
                        choices=['single', 'joint'],
                        help="Training mode: 'single' (only data_dir) or 'joint' (source+target DA).")

    parser.add_argument('--source_data_dir', type=str, default=None,
                        help='Root dir for source domain (used when train_mode=joint).')
    parser.add_argument('--target_data_dir', type=str, default=None,
                        help='Root dir for target domain (used when train_mode=joint).')

    # 选择编码器
    parser.add_argument('--encoder', type=str, default='dino', choices=['sam', 'dino'],
                        help='Backbone encoder type: sam or dino')

    # SAM 权重
    parser.add_argument(
        '--sam_checkpoint',
        type=str,
        default='./models/sam/work_dir/sam_vit_b_01ec64.pth'
    )

    # DINO 权重
    parser.add_argument(
        '--dino_checkpoint',
        type=str,
        default='./models/pth/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
        help="Path to DINOv3 ViT-B/16 weights."
    )

    parser.add_argument('--vit_input_layer_indices', type=int, nargs='+', default=[0, 2, 5, 11])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=4)

    # ---------------- 训练超参 ----------------
    parser.add_argument('--phase', type=str, default='test_auto', choices=['train', 'eval', 'test'])
    parser.add_argument('--resume', type=str, default='latest_model.pth',
                        help='Checkpoint filename to load for evaluation (relative to work_dir) or absolute path.')
    parser.add_argument('--num_epoch', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--init_lr', type=float, default=2e-4)
    parser.add_argument('--accumulation_steps', type=int, default=4)

    # 热图预热（仅 HM + Reg）
    parser.add_argument('--hm_warmup_epochs', type=int, default=0,
                        help='Warmup epochs for Task A (heatmap + corner regression).')

    # ---------------- 图像 & 模型 ----------------
    parser.add_argument('--input_h', type=int, default=1792)
    parser.add_argument('--input_w', type=int, default=512)
    parser.add_argument('--K', type=int, default=17)
    parser.add_argument('--target_feature_stride', type=int, default=2,
                        help='Down-ratio for the global heatmap (e.g., 1024->256)')
    parser.add_argument('--node_feature_dim', type=int, default=128)  # 占位

    # RNN（连接特征）
    parser.add_argument('--rnn_input_dim', type=int, default=3,
                        help='Input dim for RNN (dx_norm, dy_norm, L_norm)')
    parser.add_argument('--rnn_hidden_dim', type=int, default=64)
    parser.add_argument('--rnn_layers', type=int, default=2)
    parser.add_argument('--rnn_out_dim', type=int, default=128)  # 稍后在代码中覆盖为双向 *2

    # 全局热图
    parser.add_argument('--hm_pool_radius_feat', type=int, default=3,
                        help='Max-pool radius (feature px) before sampling HM confidence.')
    parser.add_argument('--hm_logit_thr_feat', type=float, default=-1.0,
                        help='Logit threshold for HM response -> binary flag (1 = below threshold).')

    # Canonical Patch（CPC）
    parser.add_argument('--patch_size', type=int, default=24)
    parser.add_argument('--patch_scale_w', type=float, default=0.45)
    parser.add_argument('--patch_scale_h', type=float, default=0.65)

    # 角点/中心回归头
    parser.add_argument('--corner_reg_head_dim', type=int, default=128,
                        help='Channels in the corner regression head.')
    parser.add_argument('--center_reg_head_dim', type=int, default=128,
                        help='Channels in the center offset regression head.')

    # 损失权重
    parser.add_argument('--lambda_hm', type=float, default=1.0)
    parser.add_argument('--lambda_det', type=float, default=1.0)
    parser.add_argument('--lambda_cons', type=float, default=0.2,
                        help='Weight for p_err non-threshold consistency regularizer.')
    parser.add_argument('--lambda_corner_reg', type=float, default=0.1,
                        help='Weight for the corner offset regression loss (wh_loss).')
    parser.add_argument('--lambda_center_reg', type=float, default=1.0,
                        help='Weight for the center offset regression loss (off_loss).')

    # 对抗域自适应（DANN）
    parser.add_argument('--w_domain', type=float, default=0.1,
                        help='Weight for domain adversarial loss.')
    parser.add_argument('--w_sup_target', type=float, default=1.0,
                        help='Weight for supervised loss on target domain.')
    parser.add_argument('--da_use_schedule', action='store_true',
                        help='If set, use DANN schedule for alpha; else alpha=1.0 constant.')

    # 可视化
    parser.add_argument('--vis_every', type=int, default=200)
    parser.add_argument('--vis_max_k', type=int, default=17)

    args = parser.parse_args()

    # 双向 RNN 输出维
    args.rnn_out_dim = args.rnn_hidden_dim * 2

    # 根据 encoder 类型设置主干步长（patch size）与名字
    if args.encoder == 'dino':
        args.backbone_stride = 16  # ViT-B/16
        args.backbone_variant = 'dinov3_vitb16'
    else:
        args.backbone_stride = 16  # SAM ViT-B 也是 16
        args.backbone_variant = 'sam_vit_b'

    # train_mode = single 时，source/target 默认都用 data_dir
    if args.train_mode == 'single':
        if args.source_data_dir is None:
            args.source_data_dir = args.data_dir
        if args.target_data_dir is None:
            args.target_data_dir = args.data_dir

    return args


def _build_encoder(args, device):
    """
    根据 args.encoder 构建编码器并加载权重。
    - SAM: 使用 build_sam_encoder_only(checkpoint_path=args.sam_checkpoint)
    - DINO: 通过顶层包 dinov3.hub.backbones.dinov3_vitb16 加载
    """
    if args.encoder == 'sam':
        print(f"[Encoder] Using SAM encoder ({args.backbone_variant}).")
        print(f"[Encoder] Loading checkpoint: {args.sam_checkpoint}")
        enc = build_sam_encoder_only(checkpoint_path=args.sam_checkpoint).to(device)
        return enc

    # -------- DINOv3 路线 --------
    print(f"[Encoder] Using DINOv3 encoder ({args.backbone_variant}).")
    print(f"[Encoder] Loading checkpoint: {args.dino_checkpoint}")
    try:
        from dinov3.hub.backbones import dinov3_vitb16
    except Exception as e:
        raise ImportError(
            "Cannot import dinov3_vitb16 from dinov3.hub.backbones. "
            "请确认 models/dino/dinov3 目录结构正确，且 main.py 顶部已将 models/dino 加入 sys.path。"
        ) from e

    dino_model = dinov3_vitb16(
        pretrained=True,
        weights=args.dino_checkpoint,
        check_hash=False,
    )
    dino_model = dino_model.to(device)
    return dino_model


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 训练模式检查 & 路径说明
    print(f"Training mode: {args.train_mode}")
    if args.train_mode == 'joint':
        if args.source_data_dir is None and args.target_data_dir is None:
            print("[INFO] 未显式指定 --source_data_dir / --target_data_dir，"
                  "默认认为 data_dir 下存在 train/val/test 以及其下的 {source,target} 子目录。")
            args.source_data_dir = args.data_dir
            args.target_data_dir = args.data_dir
        elif args.source_data_dir is None and args.target_data_dir is not None:
            print("[WARN] 只指定了 --target_data_dir，自动将 --source_data_dir 设为同一路径。")
            args.source_data_dir = args.target_data_dir
        elif args.source_data_dir is not None and args.target_data_dir is None:
            print("[WARN] 只指定了 --source_data_dir，自动将 --target_data_dir 设为同一路径。")
            args.target_data_dir = args.source_data_dir

        print(f"  Source data root: {args.source_data_dir}")
        print(f"  Target data root: {args.target_data_dir}")
    else:
        print(f"  Single data dir: {args.data_dir}")

    print(f"Running on device: {device}")
    print(f"V-Global: HM + Detector + Canonical Patch + CornerReg Head")
    print(f"Heatmap warmup epochs: {args.hm_warmup_epochs}")
    print(f"Target feature stride: {args.target_feature_stride} -> feature {(args.input_h // args.target_feature_stride)}")
    print(f"Backbone encoder: {args.encoder}  | variant: {args.backbone_variant}  | stride (patch): {args.backbone_stride}")
    print(f"RNN Input Dim: {args.rnn_input_dim}")
    print(f"HM pool radius (feat px): {args.hm_pool_radius_feat}")
    print(f"HM logit threshold (binary flag): {args.hm_logit_thr_feat}")
    print(f"Patch: size={args.patch_size}, scale_w={args.patch_scale_w}, scale_h={args.patch_scale_h}")
    print(f"CenterReg Head (off_loss): dim={args.center_reg_head_dim}, lambda={args.lambda_center_reg}")
    print(f"CornerReg Head (wh_loss): dim={args.corner_reg_head_dim}, lambda={args.lambda_corner_reg}")
    print(f"p_err Cons. Loss: lambda={args.lambda_cons}")
    if args.train_mode == 'joint':
        print(f"Domain Adv: w_domain={args.w_domain}, w_sup_target={args.w_sup_target}, da_use_schedule={args.da_use_schedule}")

    # 构建编码器并加载权重（SAM 或 DINO）
    encoder = _build_encoder(args, device = torch.device(args.device if torch.cuda.is_available() else "cpu"))

    # LoRA（SAM/DINO 都包一层）
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["qkv"], lora_dropout=0.1, bias="none")
    print("Wrapping the image encoder with PEFT (LoRA)...")
    peft_encoder = get_peft_model(encoder, lora_config)
    peft_encoder.print_trainable_parameters()

    if args.phase in ('train'):
        trainer = Trainer(args)
        trainer.setup(peft_encoder, args)
        trainer.train()
    elif args.phase in ('test_auto'):
        from operation.test_auto import SAIC_GUI
        # --- Qt 启动逻辑 ---
        print("Starting Qt GUI...")
        app = QApplication(sys.argv)

        # 实例化我们的 GUI 窗口
        gui = SAIC_GUI(args, peft_encoder)
        gui.show()

        # 进入事件循环，阻塞直到窗口关闭
        sys.exit(app.exec_())

    print("Process finished.")


if __name__ == '__main__':
    main()

