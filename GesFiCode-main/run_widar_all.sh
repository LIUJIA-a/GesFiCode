#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Widar 数据集全部实验一键运行脚本
# 5 个实验分配 GPU 0-4 并行后台运行，每个 50 epoch
# 用法: bash run_widar_all.sh
# ═══════════════════════════════════════════════════════════════════════════

DATA_PATH="/mnt/nj-1/usr/liujia7/workspace/datasets/widar_processed_data/Processed_Widar_6AP"
MAX_EPOCH=50

echo "========================================"
echo " Widar 实验启动"
echo " 数据路径: ${DATA_PATH}"
echo " Epoch: ${MAX_EPOCH}"
echo "========================================"

# 1. In-domain: E1 内 80/20 随机划分
CUDA_VISIBLE_DEVICES=0 nohup python csimain.py \
  --dataset widar \
  --data_path ${DATA_PATH} \
  --experiment in_domain \
  --exp_id widar_in_domain \
  --max_epoch ${MAX_EPOCH} \
  > nohup_in_domain.log 2>&1 &
echo "[GPU 0] in_domain   PID: $!"

# 2. Cross-User: E1 内，训练 U05,U10-U15，测试 U16,U17
CUDA_VISIBLE_DEVICES=1 nohup python csimain.py \
  --dataset widar \
  --data_path ${DATA_PATH} \
  --experiment cross_user \
  --exp_id widar_cross_user \
  --max_epoch ${MAX_EPOCH} \
  > nohup_cross_user.log 2>&1 &
echo "[GPU 1] cross_user  PID: $!"

# 3. Cross-Env: 训练 E1+E2，测试 E3
CUDA_VISIBLE_DEVICES=2 nohup python csimain.py \
  --dataset widar \
  --data_path ${DATA_PATH} \
  --experiment cross_env \
  --exp_id widar_cross_env \
  --max_epoch ${MAX_EPOCH} \
  > nohup_cross_env.log 2>&1 &
echo "[GPU 2] cross_env   PID: $!"

# 4. Cross-Loc: E1 内，训练 L1-L4，测试 L5
CUDA_VISIBLE_DEVICES=3 nohup python csimain.py \
  --dataset widar \
  --data_path ${DATA_PATH} \
  --experiment cross_loc \
  --exp_id widar_cross_loc \
  --max_epoch ${MAX_EPOCH} \
  > nohup_cross_loc.log 2>&1 &
echo "[GPU 3] cross_loc   PID: $!"

# 5. Cross-Ori: E1 内，训练 O1-O4，测试 O5
CUDA_VISIBLE_DEVICES=4 nohup python csimain.py \
  --dataset widar \
  --data_path ${DATA_PATH} \
  --experiment cross_ori \
  --exp_id widar_cross_ori \
  --max_epoch ${MAX_EPOCH} \
  > nohup_cross_ori.log 2>&1 &
echo "[GPU 4] cross_ori   PID: $!"

echo ""
echo "========================================"
echo " 全部 5 个实验已提交后台运行"
echo " 日志文件: nohup_*.log"
echo " 结果目录: log/widar_*/"
echo " 查看进度: tail -f nohup_cross_user.log"
echo "========================================"
