"""
验证脚本：检查 Widar 数据集在各实验模式下的加载是否正确。
运行方式：python verify_widar.py
"""
import sys
import os

# 将代码目录加入 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'GesFiCode-main'))

from dataloader import WidarDataset, TransformSubset
import torch.utils.data

DATA_DIR = '/mnt/nj-1/usr/liujia7/workspace/datasets/widar_processed_data/Processed_Widar_6AP'
GESTURES = ['G01', 'G02', 'G03', 'G04', 'G05', 'G06']
GMAP = {'G01': 0, 'G02': 1, 'G03': 2, 'G04': 3, 'G05': 4, 'G06': 5}

def verify_experiment(name, train_kwargs, test_kwargs, expected_train=None, expected_test=None):
    print(f'\n{"="*60}')
    print(f'实验: {name}')
    print(f'{"="*60}')

    train_ds = WidarDataset(DATA_DIR, transform=None, gesture_map=GMAP, **train_kwargs)
    test_ds = WidarDataset(DATA_DIR, transform=None, gesture_map=GMAP, **test_kwargs)

    print(f'  训练集: {len(train_ds)} 样本')
    print(f'  测试集: {len(test_ds)} 样本')
    print(f'  训练集标签分布: {sorted(set(train_ds.img_labels))}')
    print(f'  测试集标签分布: {sorted(set(test_ds.img_labels))}')

    # 检查标签范围
    max_label = max(max(train_ds.img_labels), max(test_ds.img_labels))
    min_label = min(min(train_ds.img_labels), min(test_ds.img_labels))
    assert min_label == 0, f'标签最小值应为 0，实际为 {min_label}'
    assert max_label == 5, f'标签最大值应为 5 (6类)，实际为 {max_label}'

    if expected_train is not None:
        assert len(train_ds) == expected_train, \
            f'训练集大小预期 {expected_train}，实际 {len(train_ds)}'
    if expected_test is not None:
        assert len(test_ds) == expected_test, \
            f'测试集大小预期 {expected_test}，实际 {len(test_ds)}'

    # 检查无重叠（通过文件路径）
    train_paths = set(train_ds.img_paths)
    test_paths = set(test_ds.img_paths)
    overlap = train_paths & test_paths
    assert len(overlap) == 0, f'训练/测试集存在 {len(overlap)} 个重叠样本！'
    print(f'  [OK] 训练/测试无重叠')
    print(f'  [OK] 标签范围 [0, 5] 正确')
    if expected_train:
        print(f'  [OK] 训练集大小 {expected_train} 正确')
    if expected_test:
        print(f'  [OK] 测试集大小 {expected_test} 正确')

    return train_ds, test_ds


def verify_in_domain():
    """验证 in_domain: 全量数据 80/20 随机划分"""
    print(f'\n{"="*60}')
    print(f'实验: in_domain (随机 80/20 划分)')
    print(f'{"="*60}')

    full_ds = WidarDataset(DATA_DIR, transform=None,
                           allowed_gestures=GESTURES, gesture_map=GMAP)
    total = len(full_ds)
    train_size = int(0.8 * total)
    test_size = total - train_size

    generator = torch.Generator().manual_seed(42)
    train_sub, test_sub = torch.utils.data.random_split(
        full_ds, [train_size, test_size], generator=generator
    )
    train_ds = TransformSubset(train_sub, transform=None)
    test_ds = TransformSubset(test_sub, transform=None)

    print(f'  全量: {total} 样本')
    print(f'  训练集: {len(train_ds)} 样本 ({len(train_ds)/total*100:.1f}%)')
    print(f'  测试集: {len(test_ds)} 样本 ({len(test_ds)/total*100:.1f}%)')
    assert len(train_ds) + len(test_ds) == total
    print(f'  [OK] 总数一致 ({total})')

    # 验证 TransformSubset 可以取数据
    img, label, pdlabel, idx = train_ds[0]
    print(f'  [OK] TransformSubset 取样正常, label={label}, idx={idx}')


if __name__ == '__main__':
    # Cross-User: E1, 训练 U05,U10-U15 (7人), 测试 U16,U17 (2人)
    # 每人 G01-G06: 6手势 × 5位置 × 5方向 × 5重复 = 750 样本
    verify_experiment(
        'cross_user',
        train_kwargs=dict(allowed_envs=['E1'],
                          allowed_users=['U05','U10','U11','U12','U13','U14','U15'],
                          allowed_gestures=GESTURES),
        test_kwargs=dict(allowed_envs=['E1'],
                         allowed_users=['U16','U17'],
                         allowed_gestures=GESTURES),
        expected_train=7 * 750,   # 5250
        expected_test=2 * 750     # 1500
    )

    # Cross-Env: 训练 E1+E2, 测试 E3
    verify_experiment(
        'cross_env',
        train_kwargs=dict(allowed_envs=['E1', 'E2'], allowed_gestures=GESTURES),
        test_kwargs=dict(allowed_envs=['E3'], allowed_gestures=GESTURES),
    )

    # Cross-Loc: E1 内, 训练 L1-L4, 测试 L5
    verify_experiment(
        'cross_loc',
        train_kwargs=dict(allowed_envs=['E1'],
                          allowed_locs=['L1','L2','L3','L4'],
                          allowed_gestures=GESTURES),
        test_kwargs=dict(allowed_envs=['E1'],
                         allowed_locs=['L5'],
                         allowed_gestures=GESTURES),
    )

    # Cross-Ori: E1 内, 训练 O1-O4, 测试 O5
    verify_experiment(
        'cross_ori',
        train_kwargs=dict(allowed_envs=['E1'],
                          allowed_oris=['O1','O2','O3','O4'],
                          allowed_gestures=GESTURES),
        test_kwargs=dict(allowed_envs=['E1'],
                         allowed_oris=['O5'],
                         allowed_gestures=GESTURES),
    )

    # In-domain
    verify_in_domain()

    print(f'\n{"="*60}')
    print('所有验证通过！')
    print(f'{"="*60}')
