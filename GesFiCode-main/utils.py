import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import random
import numbers
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
        # E1 域内数据，transform=None（稍后通过 TransformSubset 分别包装）
        full_dataset = WidarDataset(
            data_dir, transform=None,
            allowed_envs=['E1'],
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


def trainer(trainmodel, img_transform, img_transformte, device, opta, scheduler,
            total_epoch=200, log_dir="log", args=None):
    num_classes = args.num_classes if args else 8
    ones = torch.sparse.torch.eye(num_classes)
    ones = ones.to(device)
    bestac = 0.0

    # ── 构建数据集 ──────────────────────────────────────────────────────────
    if args is not None and args.dataset == 'widar':
        dataset_source, dataset_target = build_widar_loaders(
            args, img_transform, img_transformte
        )
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

    batch_size = args.batch_size if args else 32
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset_source, batch_size=batch_size,
        shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=dataset_target, batch_size=batch_size,
        shuffle=False, num_workers=2
    )
    lengthtr = len(train_loader)
    lengthte = len(test_loader)
    print(f'Train set size: {len(dataset_source)} samples, {lengthtr} batches')
    print(f'Test set size:  {len(dataset_target)} samples, {lengthte} batches')

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
            print(f'\n======== ROUND {round} ========')

            # ══════════════════════════════════════════════════════════════
            # 阶段①: Feature Update & Contrastive Pre-training (epoch <= 2)
            # ══════════════════════════════════════════════════════════════
            print('==== Stage 1: SupCon Pre-training ====')
            if round <= 2:
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

            # ══════════════════════════════════════════════════════════════
            # 阶段②: Latent Domain Characterization & PCL
            # ══════════════════════════════════════════════════════════════
            print('==== Stage 2: Latent Domain PCL ====')
            loss_list = ['total', 'dis', 'proto', 'ent_pen']
            print_row(['epoch'] + [item + '_loss' for item in loss_list], colwidth=15)
            for step in range(1):
                for inputs, labels, pdlables, item in train_loader:
                    loss_result_dict = trainmodel.update_d(
                        inputs, labels, pdlables, opta
                    )
                print_row([step] + [loss_result_dict.get(item, 0) for item in loss_list], colwidth=15)

            Cpd = trainmodel.set_dlabel(train_loader)

            # ══════════════════════════════════════════════════════════════
            # 阶段③: Domain-invariant & Hard Negative Contrastive
            # ══════════════════════════════════════════════════════════════
            print('==== Stage 3: Domain-invariant + HardNCE ====')
            loss_list = ['total', 'cls', 'dis', 'hard']
            print_row(['epoch'] + [item + '_loss' for item in loss_list], colwidth=15)
            for step in range(1):
                for inputs, labels, pdlables, item in train_loader:
                    # 频轴翻转构造困难负样本
                    x_raw = inputs.cuda().float()
                    x_mirrored = freq_flip(x_raw)
                    step_vals = trainmodel.update(
                        inputs, labels, pdlables, opta, Cpd,
                        x_mirrored=x_mirrored
                    )
                print_row([step] + [step_vals.get(item, 0) for item in loss_list], colwidth=15)

            acc = accuracy(trainmodel, test_loader, None)
            print(acc)
            if acc > bestac:
                bestac = acc
            print(bestac)

            f.write(f"Round {round}: acc={acc:.6f}, bestacc={bestac:.6f}\n")
            f.flush()

            # 学习率衰减
            scheduler.step()

    with open(bestacc_file, "w") as f:
        f.write(f"{bestac:.6f}\n")

    print(f"\nResults saved to {log_dir}")
    print(f"acc.txt: {acc_file}")
    print(f"bestacc.txt: {bestacc_file}")
