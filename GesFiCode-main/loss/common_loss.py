# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.mean(torch.sum(entropy, dim=1))
    return entropy


def Entropylogits(input, redu='mean'):
    input_ = F.softmax(input, dim=1)
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    if redu == 'mean':
        entropy = torch.mean(torch.sum(entropy, dim=1))
    elif redu == 'None':
        entropy = torch.sum(entropy, dim=1)
    return entropy


# ── SupContrast Loss (SupCon) ─────────────────────────────────────────────────
class SupConLoss(torch.nn.Module):
    """
    监督对比损失：同类拉近，异类推远。
    feat1/feat2 为同一 batch 的两个增强视图，labels 为手势标签。
    标准 NT-Xent：
    L = -log exp(sim(z_i,z_j^+)/τ) / ∑_{k} exp(sim(z_i,z_k)/τ)
    其中 z_j^+ 是同标签的其他样本（含 feat2 中对应的那一个）。
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, feat1, feat2, labels):
        # (B, D)  →  concat → (2B, D)，每张图片两个增强视图
        z = F.normalize(torch.cat([feat1, feat2], dim=0), dim=1)
        labels = labels.repeat(2)                     # (2B,)
        sim = torch.matmul(z, z.T) / self.tau         # (2B, 2B)

        loss = torch.zeros(z.size(0), device=z.device)
        for i in range(z.size(0)):
            pos_mask = (labels == labels[i]).float()
            pos_mask[i] = 0                            # 去掉自身
            numerator   = (torch.exp(sim[i]) * pos_mask).sum()
            denominator = torch.exp(sim[i]).sum()
            loss[i] = -torch.log(numerator / (denominator + 1e-8) + 1e-8)
        return loss.mean()


# ── ProtoNCE Loss (Latent Domain PCL) ────────────────────────────────────────
class ProtoNCELoss(torch.nn.Module):
    """
    原型对比损失：强制特征靠近其伪域原型，远离其他域原型。
    对每个样本 i，惩罚其在所有原型上的概率分布与 one-hot 的差距。
    L_i = -log exp(sim(z_i, p_{y_i})/τ) / ∑_{k} exp(sim(z_i, p_k)/τ)
    其中 y_i 是样本 i 被分配的伪域标签（由 set_dlabel 聚类得到）。
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.tau = temperature

    def forward(self, features, prototypes, domain_labels=None):
        """
        features:      (B, D)    归一化后的特征
        prototypes:    (K, D)    各伪域原型，归一化
        domain_labels: (B,) [可选]  样本对应的伪域标签；若不提供则取 argmax
        返回标量损失。
        """
        features   = F.normalize(features, dim=1)
        prototypes = F.normalize(prototypes, dim=1)
        sim = torch.matmul(features, prototypes.T) / self.tau  # (B, K)

        if domain_labels is None:
            domain_labels = sim.argmax(dim=1)   # (B,)  取最近原型作为伪标签

        numerator = torch.zeros(features.size(0), device=features.device)
        for i in range(features.size(0)):
            k = domain_labels[i].item()
            numerator[i] = torch.exp(sim[i, k])
        denominator = torch.exp(sim).sum(dim=1)  # 每个样本对所有 K 个原型的归一化分母
        return -torch.log((numerator + 1e-8) / (denominator + 1e-8) + 1e-8).mean()


# ── Mirror Hard Negative InfoNCE ─────────────────────────────────────────────
class InfoNCE_HardNegative(torch.nn.Module):
    """
    方向解耦损失（Directional Decoupling Loss）：
    通过最小化余弦相似度，强制原始特征与频域翻转（反向运动）特征在潜空间中
    相互推远至正交，以锐化模型对运动方向（多普勒正负）的敏感度。

    物理意义：频轴翻转等价于动作方向反转（Push↔Pull），该损失迫使模型
    区分正向与反向运动，而非将它们视为同一手势。

    注意：对于标签不区分方向的数据集（如 Widar 的 Push&Pull 合并类），
    应通过 --beta 0.0 禁用此损失。
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.tau = temperature  # 保留接口兼容，当前实现未使用 temperature

    def forward(self, feat_orig, feat_mirrored):
        """
        feat_orig:     (B, D)      原始频谱特征
        feat_mirrored: (B, D)      频轴翻转后特征
        返回非负标量损失。相似度为正时惩罚，推至正交（sim→0）后停止。
        """
        z1 = F.normalize(feat_orig, dim=1)      # (B, D)
        z2 = F.normalize(feat_mirrored, dim=1)  # (B, D)

        # 逐样本余弦相似度
        sim_hard_neg = torch.sum(z1 * z2, dim=1)  # (B,)

        # ReLU margin: 只惩罚正相关性，推至正交即停止
        loss = torch.mean(F.relu(sim_hard_neg))
        return loss
