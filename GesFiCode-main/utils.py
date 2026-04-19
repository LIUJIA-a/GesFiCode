import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import random
import numbers
from collections import Counter
from tqdm import tqdm
from torchvision.transforms import functional as F1
from sklearn.metrics import confusion_matrix
from algorithm import *
from dataloader import *
from mytransforms import Physical_Mask_Augment, Frequency_Axis_Flip


def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def accuracy(network, loader, weights, usedpredict='p'):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for inputs, labels, pdlables, item in loader:
            x = inputs.cuda().float()
            y = labels.cuda().long()
            if usedpredict == 'p':
                p = network.predict(x)
            else:
                p = network.predict1(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset:
                                        weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.cuda()
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() *
                            batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() *
                            batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return correct / total


# ═══════════════════════════════════════════════════════════════════════════
# Widar 数据集构建器
# ═══════════════════════════════════════════════════════════════════════════

# Widar 统一的 6 类手势映射（所有实验共享，保证训练/测试标签一致）
WIDAR_GESTURE_MAP = {'G01': 0, 'G02': 1, 'G03': 2, 'G04': 3, 'G05': 4, 'G06': 5}
WIDAR_GESTURES = ['G01', 'G02', 'G03', 'G04', 'G05', 'G06']


def build_widar_loaders(args, img_transform, img_transformte):
    """
    根据实验类型构建 Widar 的训练/测试 DataLoader。

    支持的实验类型 (args.experiment):
      - in_domain   : 全量数据 (E1+E2+E3, G01-G06) 随机 80/20 划分
      - cross_user  : E1 内，按用户划分
      - cross_env   : E1+E2 训练，E3 测试
      - cross_loc   : E1 内，按位置划分
      - cross_ori   : E1 内，按方向划分
    """
    data_dir = args.data_path
    gmap = WIDAR_GESTURE_MAP

    if args.experiment == 'in_domain':
        # 全量数据，transform=None（稍后通过 TransformSubset 分别包装）
        full_dataset = WidarDataset(
            data_dir, transform=None,
            allowed_gestures=WIDAR_GESTURES,
            gesture_map=gmap
        )
        # 80/20 随机划分（固定种子保证可复现）
        total = len(full_dataset)
        train_size = int(0.8 * total)
        test_size = total - train_size
        generator = torch.Generator().manual_seed(42)
        train_subset, test_subset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size], generator=generator
        )
        dataset_source = TransformSubset(train_subset, img_transform)
        dataset_target = TransformSubset(test_subset, img_transformte)

    elif args.experiment == 'cross_user':
        # E1 内，按用户划分
        train_users = ['U05', 'U10', 'U11', 'U12', 'U13', 'U14', 'U15']
        test_users = ['U16', 'U17']
        dataset_source = WidarDataset(
            data_dir, transform=img_transform,
            allowed_envs=['E1'], allowed_users=train_users,
            allowed_gestures=WIDAR_GESTURES, gesture_map=gmap
        )
        dataset_target = WidarDataset(
            data_dir, transform=img_transformte,
            allowed_envs=['E1'], allowed_users=test_users,
            allowed_gestures=WIDAR_GESTURES, gesture_map=gmap
        )

    elif args.experiment == 'cross_env':
        # E1+E2 训练，E3 测试
        dataset_source = WidarDataset(
            data_dir, transform=img_transform,
            allowed_envs=['E1', 'E2'],
            allowed_gestures=WIDAR_GESTURES, gesture_map=gmap
        )
        dataset_target = WidarDataset(
            data_dir, transform=img_transformte,
            allowed_envs=['E3'],
            allowed_gestures=WIDAR_GESTURES, gesture_map=gmap
        )

    elif args.experiment == 'cross_loc':
        # E1 内，按位置划分
        dataset_source = WidarDataset(
            data_dir, transform=img_transform,
            allowed_envs=['E1'], allowed_locs=['L1', 'L2', 'L3', 'L4'],
            allowed_gestures=WIDAR_GESTURES, gesture_map=gmap
        )
        dataset_target = WidarDataset(
            data_dir, transform=img_transformte,
            allowed_envs=['E1'], allowed_locs=['L5'],
            allowed_gestures=WIDAR_GESTURES, gesture_map=gmap
        )

    elif args.experiment == 'cross_ori':
        # E1 内，按方向划分
        dataset_source = WidarDataset(
            data_dir, transform=img_transform,
            allowed_envs=['E1'], allowed_oris=['O1', 'O2', 'O3', 'O4'],
            allowed_gestures=WIDAR_GESTURES, gesture_map=gmap
        )
        dataset_target = WidarDataset(
            data_dir, transform=img_transformte,
            allowed_envs=['E1'], allowed_oris=['O5'],
            allowed_gestures=WIDAR_GESTURES, gesture_map=gmap
        )
    else:
        raise ValueError(f"Unknown experiment type: {args.experiment}")

    return dataset_source, dataset_target


def build_widar_source_eval(args, img_transformte):
    """
    构建源域评估数据集（使用 test transform，无增强，确定性评估）。
    与训练集同分布但不做随机增强。
    """
    data_dir = args.data_path
    gmap = WIDAR_GESTURE_MAP

    if args.experiment == 'in_domain':
        full_dataset = WidarDataset(
            data_dir, transform=None,
            allowed_gestures=WIDAR_GESTURES,
            gesture_map=gmap
        )
        total = len(full_dataset)
        train_size = int(0.8 * total)
        test_size = total - train_size
        generator = torch.Generator().manual_seed(42)
        train_subset, _ = torch.utils.data.random_split(
            full_dataset, [train_size, test_size], generator=generator
        )
        return TransformSubset(train_subset, img_transformte)

    elif args.experiment == 'cross_user':
        train_users = ['U05', 'U10', 'U11', 'U12', 'U13', 'U14', 'U15']
        return WidarDataset(
            data_dir, transform=img_transformte,
            allowed_envs=['E1'], allowed_users=train_users,
            allowed_gestures=WIDAR_GESTURES, gesture_map=gmap
        )

    elif args.experiment == 'cross_env':
        return WidarDataset(
            data_dir, transform=img_transformte,
            allowed_envs=['E1', 'E2'],
            allowed_gestures=WIDAR_GESTURES, gesture_map=gmap
        )

    elif args.experiment == 'cross_loc':
        return WidarDataset(
            data_dir, transform=img_transformte,
            allowed_envs=['E1'], allowed_locs=['L1', 'L2', 'L3', 'L4'],
            allowed_gestures=WIDAR_GESTURES, gesture_map=gmap
        )

    elif args.experiment == 'cross_ori':
        return WidarDataset(
            data_dir, transform=img_transformte,
            allowed_envs=['E1'], allowed_oris=['O1', 'O2', 'O3', 'O4'],
            allowed_gestures=WIDAR_GESTURES, gesture_map=gmap
        )
    else:
        raise ValueError(f"Unknown experiment type: {args.experiment}")


def _assign_random_domain_labels(dataset, K):
    """
    M2 消融用：随机分配伪域标签（均匀分布到 0..K-1），替代 K-Means 聚类。
    返回 Counter 统计各域样本数。
    """
    N = len(dataset)
    random_labels = np.random.randint(0, K, size=N)
    indices = np.arange(N)
    dataset.set_labels_by_index(random_labels, indices, 'pdlabel')
    return Counter(random_labels.tolist())


def trainer(trainmodel, img_transform, img_transformte, device, opta, scheduler,
            total_epoch=200, log_dir="log", args=None):
    num_classes = args.num_classes if args else 8
    ones = torch.sparse.torch.eye(num_classes)
    ones = ones.to(device)
    bestac = 0.0

    ablation = args.ablation if (args and hasattr(args, 'ablation')) else 'full'
    print(f'\n[Ablation Mode] {ablation}')

    # ── 构建数据集 ──────────────────────────────────────────────────────────
    if args is not None and args.dataset == 'widar':
        dataset_source, dataset_target = build_widar_loaders(
            args, img_transform, img_transformte
        )
        # 源域评估集（无增强，确定性评估）
        dataset_source_eval = build_widar_source_eval(args, img_transformte)
    else:
        # 原有 XRF55 逻辑
        train_list = args.data_path if (args and args.data_path) else \
            r'C:\Users\G\Downloads\GesFiCode-main-v2\GesFiCode-main\Processed_Data'
        dataset_source = datatrcsie(
            data_list=train_list,
            transform=img_transform
        )
        test_list = train_list
        dataset_target = datatecsie(
            data_list=test_list,
            transform=img_transformte
        )
        # XRF55 源域评估集
        dataset_source_eval = datatrcsie(
            data_list=train_list,
            transform=img_transformte
        )

    batch_size = args.batch_size if args else 32
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset_source, batch_size=batch_size,
        shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=dataset_target, batch_size=batch_size,
        shuffle=False, num_workers=2
    )
    source_eval_loader = torch.utils.data.DataLoader(
        dataset=dataset_source_eval, batch_size=batch_size,
        shuffle=False, num_workers=2
    )
    lengthtr = len(train_loader)
    lengthte = len(test_loader)
    print(f'Train set size: {len(dataset_source)} samples, {lengthtr} batches')
    print(f'Test set size:  {len(dataset_target)} samples, {lengthte} batches')
    print(f'Source eval size: {len(dataset_source_eval)} samples')

    # ── 物理增强实例化 ────────────────────────────────────────────────────────
    var_pct = args.variance_percentile if args else 30
    mask_r = args.mask_ratio if args else 0.15
    p_size = args.patch_size if args else 16
    phys_mask_aug = Physical_Mask_Augment(
        variance_percentile=var_pct,
        mask_ratio=mask_r,
        patch_size=p_size
    )
    freq_flip = Frequency_Axis_Flip()

    acc_file = os.path.join(log_dir, "acc.txt")
    bestacc_file = os.path.join(log_dir, "bestacc.txt")

    with open(acc_file, "w") as f:
        for round in range(total_epoch):
            # 更新模型内的 round 计数器（驱动 GRL alpha 自适应衰减）
            trainmodel._round = round

            trainmodel.train()
            print(f'\n======== ROUND {round} ========  [Ablation: {ablation}]')

            # ============================================================
            # M0 (Baseline): 纯 CE 分类，所有 epoch 走 4 个 step
            # ============================================================
            if ablation == 'M0':
                print('==== M0 Baseline: CE Only ====')
                loss_list = ['total', 'cls']
                print_row(['epoch'] + [item + '_loss' for item in loss_list], colwidth=15)
                for step in range(4):
                    for inputs, labels, pdlables, item in train_loader:
                        loss_result_dict = trainmodel.update_baseline(
                            inputs, labels, opta
                        )
                    print_row([step] + [loss_result_dict.get(item, 0) for item in loss_list], colwidth=15)

            else:
                # ══════════════════════════════════════════════════════════
                # 阶段①: Feature Update & Contrastive Pre-training
                # ══════════════════════════════════════════════════════════
                print('==== Stage 1: SupCon Pre-training ====')

                # M1 消融：永远不传 SupCon 视图（包括前 3 个 epoch）
                use_supcon = (ablation not in ('M1',))

                if round <= 2 and use_supcon:
                    loss_list = ['total', 'cls', 'supcon']
                    print_row(['epoch'] + [item + '_loss' for item in loss_list], colwidth=15)
                    for step in range(2):
                        for inputs, labels, pdlables, item in train_loader:
                            # 方差先验掩码增强，构造双视图
                            x_raw = inputs.cuda().float()
                            x_view1, x_view2 = phys_mask_aug(x_raw)
                            loss_result_dict = trainmodel.update_a(
                                inputs, labels, pdlables, opta,
                                x_view1=x_view1, x_view2=x_view2
                            )
                        print_row([step] + [loss_result_dict.get(item, 0) for item in loss_list], colwidth=15)
                else:
                    loss_list = ['total', 'cls']
                    print_row(['epoch'] + [item + '_loss' for item in loss_list], colwidth=15)
                    for step in range(2):
                        for inputs, labels, pdlables, item in train_loader:
                            loss_result_dict = trainmodel.update_a(
                                inputs, labels, pdlables, opta,
                                x_view1=None, x_view2=None
                            )
                        print_row([step] + [loss_result_dict.get(item, 0) for item in loss_list], colwidth=15)

                # ══════════════════════════════════════════════════════════
                # 阶段②: Latent Domain Characterization & PCL
                # ══════════════════════════════════════════════════════════
                # M2 消融：跳过整个 Stage②，随机分配域标签
                if ablation == 'M2':
                    print('==== Stage 2: SKIPPED (M2 ablation) ====')
                    # 每个 epoch 重新随机分配域标签
                    K = args.latent_domain_num
                    Cpd = _assign_random_domain_labels(dataset_source, K)
                    print(f'  Random domain assignment: {Cpd}')
                else:
                    skip_grl_s2 = (ablation == 'M3')
                    print('==== Stage 2: Latent Domain PCL ====')
                    if skip_grl_s2:
                        print('  (GRL disabled for M3 ablation)')
                    loss_list = ['total', 'dis', 'proto', 'ent_pen']
                    print_row(['epoch'] + [item + '_loss' for item in loss_list], colwidth=15)
                    for step in range(1):
                        for inputs, labels, pdlables, item in train_loader:
                            loss_result_dict = trainmodel.update_d(
                                inputs, labels, pdlables, opta,
                                skip_grl=skip_grl_s2
                            )
                        print_row([step] + [loss_result_dict.get(item, 0) for item in loss_list], colwidth=15)

                    Cpd = trainmodel.set_dlabel(train_loader)

                # ══════════════════════════════════════════════════════════
                # 阶段③: Domain-invariant & Hard Negative Contrastive
                # ══════════════════════════════════════════════════════════
                skip_grl_s3 = (ablation == 'M3')
                use_hardnce = (ablation not in ('M4',))

                print('==== Stage 3: Domain-invariant + HardNCE ====')
                if skip_grl_s3:
                    print('  (GRL disabled for M3 ablation)')
                if not use_hardnce:
                    print('  (HardNCE disabled for M4 ablation)')

                loss_list = ['total', 'cls', 'dis', 'hard']
                print_row(['epoch'] + [item + '_loss' for item in loss_list], colwidth=15)
                for step in range(1):
                    for inputs, labels, pdlables, item in train_loader:
                        # 频轴翻转构造困难负样本（M4 消融时不构造）
                        if use_hardnce:
                            x_raw = inputs.cuda().float()
                            x_mirrored = freq_flip(x_raw)
                        else:
                            x_mirrored = None

                        step_vals = trainmodel.update(
                            inputs, labels, pdlables, opta, Cpd,
                            x_mirrored=x_mirrored,
                            skip_grl=skip_grl_s3
                        )
                    print_row([step] + [step_vals.get(item, 0) for item in loss_list], colwidth=15)

            # ══════════════════════════════════════════════════════════════
            # 评估：源域精度 + 目标域精度
            # ══════════════════════════════════════════════════════════════
            src_acc = accuracy(trainmodel, source_eval_loader, None)
            tgt_acc = accuracy(trainmodel, test_loader, None)
            print(f'  Source acc: {src_acc:.6f}  |  Target acc: {tgt_acc:.6f}')

            if tgt_acc > bestac:
                bestac = tgt_acc
            print(f'  Best target acc: {bestac:.6f}')

            f.write(f"Round {round}: src_acc={src_acc:.6f}, tgt_acc={tgt_acc:.6f}, best_tgt={bestac:.6f}\n")
            f.flush()

            # 学习率衰减
            scheduler.step()

    with open(bestacc_file, "w") as f:
        f.write(f"{bestac:.6f}\n")

    print(f"\nResults saved to {log_dir}")
    print(f"acc.txt: {acc_file}")
    print(f"bestacc.txt: {bestacc_file}")
