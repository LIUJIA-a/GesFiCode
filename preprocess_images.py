"""
CSI图像预处理脚本
将相位图和DFS图合并成3通道图像，供网络训练使用

使用方法:
    python preprocess_images.py

生成:
    WIDAR_COMBINED/  - 合并后的图像文件夹
"""

import os
from PIL import Image
import numpy as np
from tqdm import tqdm


def combine_images(phase_dir, dfs_dir, output_dir):
    """
    合并相位图和DFS图为3通道RGB图像

    通道设计:
        通道0: 相位图 (包含绝对位置/角度信息)
        通道1: DFS图 (包含动态变化信息)
        通道2: 局部对比度增强 (基于相位图，增强纹理特征)

    参数:
        phase_dir: 相位图文件夹路径
        dfs_dir: DFS图文件夹路径
        output_dir: 输出文件夹路径
    """
    os.makedirs(output_dir, exist_ok=True)

    phase_files = [f for f in os.listdir(phase_dir) if f.endswith('.jpg')]
    phase_files = sorted(phase_files)

    print(f'发现 {len(phase_files)} 张相位图')
    print(f'相位图目录: {phase_dir}')
    print(f'DFS图目录: {dfs_dir}')
    print(f'输出目录: {output_dir}')
    print('-' * 50)

    processed = 0
    missing = 0

    for filename in tqdm(phase_files, desc='处理进度'):
        phase_path = os.path.join(phase_dir, filename)
        dfs_path = os.path.join(dfs_dir, filename)
        output_path = os.path.join(output_dir, filename)

        if not os.path.exists(dfs_path):
            missing += 1
            continue

        phase_img = Image.open(phase_path)
        dfs_img = Image.open(dfs_path)

        if phase_img.size != dfs_img.size:
            dfs_img = dfs_img.resize(phase_img.size, Image.LANCZOS)

        phase_arr = np.array(phase_img)
        dfs_arr = np.array(dfs_img)

        if phase_arr.ndim == 3:
            phase_arr = phase_arr[:, :, 0]
        if dfs_arr.ndim == 3:
            dfs_arr = dfs_arr[:, :, 0]

        phase_arr = phase_arr.astype(np.float32)
        dfs_arr = dfs_arr.astype(np.float32)

        phase_min, phase_max = phase_arr.min(), phase_arr.max()
        dfs_min, dfs_max = dfs_arr.min(), dfs_arr.max()

        phase_norm = (phase_arr - phase_min) / (phase_max - phase_min + 1e-8)
        dfs_norm = (dfs_arr - dfs_min) / (dfs_max - dfs_min + 1e-8)

        local_var = compute_local_contrast(phase_norm, window_size=5)

        rgb_image = np.stack([phase_norm, dfs_norm, local_var], axis=-1)
        rgb_image = (rgb_image * 255).astype(np.uint8)

        Image.fromarray(rgb_image).save(output_path)
        processed += 1

    print('-' * 50)
    print(f'处理完成!')
    print(f'成功: {processed} 张')
    print(f'缺失: {missing} 张 (DFS文件不存在)')


def compute_local_contrast(image, window_size=5):
    """
    计算局部对比度作为第三通道

    使用局部标准差来量化纹理强度，帮助网络捕获空间变化

    参数:
        image: 归一化后的灰度图 [H, W], 值域 [0, 1]
        window_size: 窗口大小

    返回:
        local_contrast: 局部对比度图 [H, W], 值域 [0, 1]
    """
    from scipy.ndimage import uniform_filter

    image = image.astype(np.float64)

    local_mean = uniform_filter(image, size=window_size)
    local_sqr_mean = uniform_filter(image ** 2, size=window_size)
    local_std = np.sqrt(np.maximum(local_sqr_mean - local_mean ** 2, 0))

    contrast = (local_std - local_std.min()) / (local_std.max() - local_std.min() + 1e-8)

    return contrast


def combine_images_simple(phase_dir, dfs_dir, output_dir):
    """
    简化版本：直接用三个归一化通道

    通道设计:
        通道0: 相位图
        通道1: DFS图
        通道2: DFS图的边缘检测 (Sobel算子)
    """
    from scipy.ndimage import sobel

    os.makedirs(output_dir, exist_ok=True)

    phase_files = [f for f in os.listdir(phase_dir) if f.endswith('.jpg')]
    phase_files = sorted(phase_files)

    print(f'发现 {len(phase_files)} 张相位图 (简化版本)')
    print(f'相位图目录: {phase_dir}')
    print(f'DFS图目录: {dfs_dir}')
    print(f'输出目录: {output_dir}')
    print('-' * 50)

    for filename in tqdm(phase_files, desc='处理进度'):
        phase_path = os.path.join(phase_dir, filename)
        dfs_path = os.path.join(dfs_dir, filename)
        output_path = os.path.join(output_dir, filename)

        if not os.path.exists(dfs_path):
            continue

        phase_arr = np.array(Image.open(phase_path)).astype(np.float32)
        dfs_arr = np.array(Image.open(dfs_path)).astype(np.float32)

        phase_norm = (phase_arr - phase_arr.min()) / (phase_arr.max() - phase_arr.min() + 1e-8)
        dfs_norm = (dfs_arr - dfs_arr.min()) / (dfs_arr.max() - dfs_arr.min() + 1e-8)

        edge = sobel(dfs_norm)
        edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8)

        rgb_image = np.stack([phase_norm, dfs_norm, edge], axis=-1)
        rgb_image = (rgb_image * 255).astype(np.uint8)

        Image.fromarray(rgb_image).save(output_path)

    print('-' * 50)
    print(f'处理完成! 共处理 {len(phase_files)} 张')


if __name__ == '__main__':
    BASE_DIR = r'C:\Users\G\Downloads\GesFiCode-main\GesFiCode-main'

    PHASE_DIR = os.path.join(BASE_DIR, 'WIDAR_STIFMM-20260410T101126Z-3-001', 'WIDAR_STIFMM')
    DFS_DIR = os.path.join(BASE_DIR, 'WIDAR_STIFMM_DFS-20260410T100850Z-3-001', 'WIDAR_STIFMM_DFS')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'WIDAR_COMBINED')

    print('=' * 50)
    print('CSI图像预处理 - 相位图与DFS图合并')
    print('=' * 50)
    print()

    print('请选择合成方式:')
    print('1: [相位, DFS, 局部对比度] - 推荐，增强纹理特征')
    print('2: [相位, DFS, 边缘检测] - 简化版本')
    print()
    choice = input('请输入选项 (1/2，默认1): ').strip() or '1'

    if choice == '2':
        combine_images_simple(PHASE_DIR, DFS_DIR, OUTPUT_DIR)
    else:
        combine_images(PHASE_DIR, DFS_DIR, OUTPUT_DIR)

    print()
    print('=' * 50)
    print('预处理完成!')
    print(f'合并后的图像位于: {OUTPUT_DIR}')
    print()
    print('使用方法:')
    print(f"  DATA_PATH = '{OUTPUT_DIR}'")
    print()
    print('注意: 如果原代码中 DATA_PATH 不是 "WIDAR_STIFMM"，')
    print('      请修改 csimain.py 或相应文件中的 DATA_PATH 变量。')
    print('=' * 50)
