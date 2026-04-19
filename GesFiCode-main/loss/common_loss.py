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
    镜像困难负样本对比损失（Widar3.0 专用）：
    - 正样本对：同一 batch 内 sample_i 的原始频谱特征与其频轴翻转版本（feat_mirrored[i]）。
      它们来自同一手势，理应在特征空间中高度相似，损失应把它们拉近。
    - 困难负样本：同一 batch 内的其他样本（feat_orig[j], j≠i）。

    标准 InfoNCE（L_i ≥ 0 恒成立）：
    L_i = -log exp(sim(z_i, z_i^mir)/τ) / ∑_{k} exp(sim(z_i, z_k^mir)/τ)

    注意：这里的"推远"是指通过拉近正样本对而间接实现的——拉近了正确对，
    相对上就等于推开了错误对。
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, feat_orig, feat_mirrored):
        """
        feat_orig:     (B, D)      原始频谱特征
        feat_mirrored: (B, D)      频轴翻转后特征
        返回非负标量损失。
        """
        z1 = F.normalize(feat_orig, dim=1)      # (B, D)
        z2 = F.normalize(feat_mirrored, dim=1)  # (B, D)

        # cross-similarity：每行 i 是 z1[i] 与所有 z2[k] 的相似度
        sim_cross = torch.matmul(z1, z2.T) / self.tau  # (B, B)
        # 对角线 z1[i] · z2[i] 是正样本对；其余 z1[i] · z2[k≠i] 是 batch 内困难负样本
        pos_sim = torch.diag(sim_cross)                        # (B,)
        denominator = torch.exp(sim_cross).sum(dim=1)          # (B,)
        loss = -pos_sim + torch.log(denominator + 1e-8)       # 等价于 -log(pos / den)
        return loss.mean()
