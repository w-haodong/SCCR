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
    parser.add_argument('--work_dir', type=str, default='E:/work/scoliosis_jz/whd_jz_10/')
    parser.add_argument('--data_dir', type=str, default='E:/datasets/scoliosis_jz/create_data/2026_0109_data/data_processed_v51/')


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
        default='E:/work/scoliosis_jz/whd_jz_8/models/sam/work_dir/sam_vit_b_01ec64.pth'
    )

    # DINO 权重
    parser.add_argument(
        '--dino_checkpoint',
        type=str,
        default='E:/work/scoliosis_jz/whd_jz_8//models/pth/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
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
    elif args.phase in ('test_click'):
        from operation.test_click import SAIC_GUI
        # --- Qt 启动逻辑 ---
        print("Starting Qt GUI...")
        app = QApplication(sys.argv)

        # 实例化我们的 GUI 窗口
        gui = SAIC_GUI(args, peft_encoder)
        gui.show()

        # 进入事件循环，阻塞直到窗口关闭
        sys.exit(app.exec_())
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
    elif args.phase in ('test_auto_corr_with_fullPred'):
        from operation.test_auto_corr_with_fullPred import SAIC_GUI
        # --- Qt 启动逻辑 ---
        print("Starting Qt GUI...")
        app = QApplication(sys.argv)

        # 实例化我们的 GUI 窗口
        gui = SAIC_GUI(args, peft_encoder)
        gui.show()

        # 进入事件循环，阻塞直到窗口关闭
        sys.exit(app.exec_())
        # +++ 修改 2: 添加 eval 分支 +++

    elif args.phase in ('test_ann'):
        from operation.test_auto_annotation import SAIC_GUI
        # --- Qt 启动逻辑 ---
        print("Starting Qt GUI...")
        app = QApplication(sys.argv)

        # 实例化我们的 GUI 窗口
        gui = SAIC_GUI(args, peft_encoder)
        gui.show()

        # 进入事件循环，阻塞直到窗口关闭
        sys.exit(app.exec_())
        # +++ 修改 2: 添加 eval 分支 +++
    elif args.phase == 'eval':
        print("Starting Headless Evaluation (Auto Correction)...")
        from operation.eval_auto import Evaluator
        # 实例化 Evaluator
        evaluator = Evaluator(args, peft_encoder)

        # 确定权重路径
        ckpt_path = args.resume
        # 如果不是绝对路径，也不是相对路径存在的文件，则去 work_dir 找
        if not os.path.isfile(ckpt_path):
            potential_path = os.path.join(args.work_dir, args.resume)
            if os.path.isfile(potential_path):
                ckpt_path = potential_path
            else:
                print(f"[Warning] Checkpoint not found at {ckpt_path} or {potential_path}")

        # 加载权重并运行
        evaluator.load_weights(ckpt_path)
        evaluator.evaluate()
    elif args.phase == 'eval_acc':
        print("Starting Headless Evaluation (Error Detection Metrics + Plots)...")
        from operation.eval_acc import ErrorEvaluator
        NODE_THR = 0.5  # 节点错误判定阈值 (原 0.3)
        CONN_THR = 0.5  # 连接错误判定阈值 (原 0.7)
        # 实例化 ErrorEvaluator
        evaluator = ErrorEvaluator(
            args,
            peft_encoder,
            node_thr=NODE_THR,  # 传入节点阈值
            conn_thr=CONN_THR,  # 传入连接阈值
             )

        # 确定权重路径（优先用 args.resume；若不是文件则拼到 work_dir）
        ckpt_path = args.resume
        if not os.path.isfile(ckpt_path):
            potential_path = os.path.join(args.work_dir, args.resume)
            if os.path.isfile(potential_path):
                ckpt_path = potential_path
            else:
                print(f"[Warning] Checkpoint not found at {ckpt_path} or {potential_path}")

        # 加载权重并运行
        if os.path.isfile(ckpt_path):
            evaluator.load_weights(ckpt_path)
        else:
            print("[Warning] Running evaluation without loading weights (random/init weights).")

        evaluator.evaluate()
    elif args.phase == 'eval_corr':
        print("Starting Headless Evaluation (Auto Correction Metrics)...")
        # 你的新文件名：operation/eval_corr.py
        from operation.eval_corr import Evaluator

        evaluator = Evaluator(args, peft_encoder)

        # 确定权重路径（优先 args.resume；否则拼到 work_dir）
        ckpt_path = args.resume
        if not os.path.isfile(ckpt_path):
            potential_path = os.path.join(args.work_dir, args.resume)
            if os.path.isfile(potential_path):
                ckpt_path = potential_path
            else:
                print(f"[Warning] Checkpoint not found at {ckpt_path} or {potential_path}")

        # 加载权重并运行
        if os.path.isfile(ckpt_path):
            evaluator.load_weights(ckpt_path)
        else:
            print("[Warning] Running evaluation without loading weights (random/init weights).")

        evaluator.evaluate()
    elif args.phase == 'eval_auto_pred_with_corr':
        print("Starting Headless Evaluation (Auto Correction Metrics)...")
        # 你的新文件名：operation/eval_corr.py
        from operation.eval_auto_pred_with_corr import Evaluator

        evaluator = Evaluator(args, peft_encoder)

        # 确定权重路径（优先 args.resume；否则拼到 work_dir）
        ckpt_path = args.resume
        if not os.path.isfile(ckpt_path):
            potential_path = os.path.join(args.work_dir, args.resume)
            if os.path.isfile(potential_path):
                ckpt_path = potential_path
            else:
                print(f"[Warning] Checkpoint not found at {ckpt_path} or {potential_path}")

        # 加载权重并运行
        if os.path.isfile(ckpt_path):
            evaluator.load_weights(ckpt_path)
        else:
            print("[Warning] Running evaluation without loading weights (random/init weights).")

        evaluator.evaluate()


    elif args.phase == "eval_auto_corr_other_pred":
        print("Starting Headless Evaluation (Auto Correction on External Predictions)...")
        # =========================================================
        # ✅ 外部数据路径
        # =========================================================
        aid_dataset = "target"
        external_root = r"E:\投稿\脊柱纠正\res_pr_other\\" + aid_dataset

        images_dir = os.path.join(external_root, "images")
        # 1. 定义模型根目录：用于存放 heatmap 等子文件夹
        pred_dir_name = "contour"
        pred_root_dir = os.path.join(external_root, pred_dir_name)  # E:\...\HTN

        # 2. 定义 mat 文件目录：你的 mat 实际上在 point 子文件夹
        pred_mat_dir = os.path.join(pred_root_dir, "point")  # E:\...\HTN\point
        gt_dir = os.path.join(external_root, "labels")

        pred_key = "pr_landmarks"
        gt_key = "p2"

        corrected_dir = os.path.join(external_root, f"{pred_dir_name}_corrected")
        weight_path = os.path.join(args.work_dir, "latest_model.pth")
        # =========================================================

        # 【新增】定义阈值变量，方便修改对比
        NODE_THR = 0.05  # 节点错误判定阈值 (原 0.3)
        CONN_THR = 0.95  # 连接错误判定阈值 (原 0.7)

        from datasets.external_pred_dataset import ExternalPredCorrectionDataset
        # ✅ Dataset 读取 mat，所以传 pred_mat_dir

        dset = ExternalPredCorrectionDataset(
            args=args,
            images_dir=images_dir,
            pred_dir=pred_mat_dir,  # 指向 .../point
            gt_dir=gt_dir,
            pred_key=pred_key,
            gt_key=gt_key,
            strip_pred_prefixes=("pl_",),
            debug_print_examples=True,
        )

        from operation.eval_auto_other_pred_with_p import Evaluator
        # ✅ Evaluator 查找 heatmap，所以传根目录 pred_root_dir (内部会自动拼 /heatmap)
        # ✅ 传入自定义阈值
        evaluator = Evaluator(
            args=args,
            peft_encoder=peft_encoder,
            dataset=dset,
            pred_key=pred_key,
            corrected_dir=corrected_dir,
            results_subdir=f"eval_{pred_dir_name}_results",
            pred_dir=pred_root_dir,  # 传入根目录
            node_thr=NODE_THR,  # 传入节点阈值
            conn_thr=CONN_THR,  # 传入连接阈值
            aid_dataset=aid_dataset
        )
        ok = evaluator.load_weights(weight_path)
        if not ok:
            raise RuntimeError(f"Load weights failed: {weight_path}")

        evaluator.evaluate()
    elif args.phase == "eval_auto_corr_other_pred_xh":
        print("Starting Headless Evaluation Loop (Auto Correction on External Predictions)...")

        # 引入必要的包（移到循环外）
        from datasets.external_pred_dataset import ExternalPredCorrectionDataset
        from operation.eval_auto_other_pred_with_p import Evaluator
        import gc

        # =========================================================
        # ✅ 循环配置
        # =========================================================
        target_datasets = ["source","target"]
        target_algorithms = ["vf_ld", "htn", "biss", "contour", "sam_seg"]

        # 定义基础路径前缀
        base_root = r"E:\投稿\脊柱纠正\res_pr_other"
        weight_path = os.path.join(args.work_dir, "latest_model.pth")

        # 定义常量参数
        pred_key = "pr_landmarks"  # ⚠️注意：如果不同算法生成的mat文件内部key不同，这里需要做字典映射
        gt_key = "p2"
        NODE_THR = 0.05
        CONN_THR = 0.95

        # =========================================================
        # 🔄 双层循环开始
        # =========================================================
        for aid_dataset in target_datasets:
            print(f"\n{'=' * 60}")
            print(f"🚀 Processing Dataset: {aid_dataset}")
            print(f"{'=' * 60}")

            external_root = os.path.join(base_root, aid_dataset)
            images_dir = os.path.join(external_root, "images")
            gt_dir = os.path.join(external_root, "labels")

            for pred_dir_name in target_algorithms:
                print(f"\n   >> Algorithm: {pred_dir_name}")

                # 1. 动态构建路径
                pred_root_dir = os.path.join(external_root, pred_dir_name)  # E:\...\dataset\algo
                pred_mat_dir = os.path.join(pred_root_dir, "point")  # E:\...\dataset\algo\point

                # 2. 结果保存路径区分算法
                corrected_dir = os.path.join(external_root, f"{pred_dir_name}_corrected")
                results_subdir = f"eval_{pred_dir_name}_results"

                # 🛠️ 安全检查：如果文件夹不存在，跳过
                if not os.path.exists(pred_mat_dir):
                    print(f"      [Warning] Path not found, skipping: {pred_mat_dir}")
                    continue

                try:
                    # 3. 初始化 Dataset
                    dset = ExternalPredCorrectionDataset(
                        args=args,
                        images_dir=images_dir,
                        pred_dir=pred_mat_dir,
                        gt_dir=gt_dir,
                        pred_key=pred_key,
                        gt_key=gt_key,
                        strip_pred_prefixes=("pl_",),
                        debug_print_examples=False,  # 批量跑建议关闭debug打印
                    )

                    print(f"      Loaded {len(dset)} samples.")

                    # 4. 初始化 Evaluator
                    evaluator = Evaluator(
                        args=args,
                        peft_encoder=peft_encoder,
                        dataset=dset,
                        pred_key=pred_key,
                        corrected_dir=corrected_dir,
                        results_subdir=results_subdir,
                        pred_dir=pred_root_dir,
                        node_thr=NODE_THR,
                        conn_thr=CONN_THR,
                        aid_dataset=aid_dataset
                    )

                    # 5. 加载权重
                    ok = evaluator.load_weights(weight_path)
                    if not ok:
                        print(f"      [Error] Load weights failed for {weight_path}")
                        continue

                    # 6. 开始评估
                    evaluator.evaluate()

                    print(f"      ✅ Finished {aid_dataset} - {pred_dir_name}")

                    # 🧹 清理显存和对象，防止循环过程中内存泄漏
                    del evaluator
                    del dset
                    torch.cuda.empty_cache()
                    gc.collect()

                except Exception as e:
                    import traceback
                    print(f"      ❌ Error processing {aid_dataset} - {pred_dir_name}: {e}")
                    traceback.print_exc()

    print("Process finished.")


if __name__ == '__main__':
    main()
