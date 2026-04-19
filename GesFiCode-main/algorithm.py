from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist

from model import *
from network import Adver_network, common_network
from base import Algorithm
from loss.common_loss import Entropylogits, SupConLoss, ProtoNCELoss, InfoNCE_HardNegative
import argparse


class GeneFi(Algorithm):

    def __init__(self, args):
        super(GeneFi, self).__init__(args)
        self.args = args

        # ── 共享 Backbone ────────────────────────────────────────────────────
        self.featurizer = FeatureNet()
        in_features = 512

        # ── 阶段①: Feature Update & SupCon 预训练 ────────────────────────────
        self.abottleneck = common_network.feat_bottleneck(
            in_features, args.bottleneck, args.layer)
        self.aclassifier = common_network.feat_classifier(
            args.num_classes, args.bottleneck, args.classifier)

        # ── 阶段②: Latent Domain Characterization & PCL ─────────────────────
        self.dbottleneck = common_network.feat_bottleneck(
            in_features, args.bottleneck, args.layer)
        self.ddiscriminator = Adver_network.Discriminator(
            args.bottleneck, args.dis_hidden, args.num_classes)
        self.dclassifier = common_network.feat_classifier(
            args.latent_domain_num, args.bottleneck, args.classifier)

        # ── 阶段③: Domain-invariant & Hard Negative Contrastive ──────────────
        self.bottleneck = common_network.feat_bottleneck(
            in_features, args.bottleneck, args.layer)
        self.classifier = common_network.feat_classifier(
            args.num_classes, args.bottleneck, args.classifier)
        self.discriminator = Adver_network.Discriminator(
            args.bottleneck, args.dis_hidden, args.latent_domain_num)

        # ── 新增损失函数 ─────────────────────────────────────────────────────
        self.supcon_loss = SupConLoss(temperature=args.supcon_tau)
        self.proto_loss = ProtoNCELoss(temperature=args.proto_tau)
        self.hardnce_loss = InfoNCE_HardNegative(temperature=args.hardnce_tau)

        # ── 原型存储（阶段② PCL 更新）────────────────────────────────────────
        self.current_domain_prototypes = None

        # ── GRL alpha 记录（用于动态调度）──────────────────────────────────
        self._round = 0

    # ══════════════════════════════════════════════════════════════════════
    # M0 Baseline: 纯 ResNet18 + CE Loss（使用 Stage③ 的 bottleneck+classifier）
    # ══════════════════════════════════════════════════════════════════════
    def update_baseline(self, inputs, labels, opt):
        """M0 消融: 纯 CE 分类，使用 Stage③ 头以便 predict() 兼容。"""
        all_x = inputs.cuda().float()
        all_y = labels.cuda().long()
        all_z = self.bottleneck(self.featurizer(all_x))
        cls_loss = F.cross_entropy(self.classifier(all_z), all_y)
        opt.zero_grad()
        cls_loss.backward()
        opt.step()
        return {'total': cls_loss.item(), 'cls': cls_loss.item()}

    # ══════════════════════════════════════════════════════════════════════
    # 阶段①: Feature Update & Contrastive Pre-training (仅前 3 个 epoch)
    # ══════════════════════════════════════════════════════════════════════
    def update_a(self, inputs, labels, pdlables, opt, x_view1=None, x_view2=None):
        """
        阶段①训练逻辑。

        若提供 x_view1/x_view2（Physical_Mask_Augment 输出），则走 SupCon 分支；
        否则退化为纯分类损失（原 GeneFi update_a 行为）。
        """
        all_x = inputs.cuda().float()
        all_y = labels.cuda().long()
        all_z = self.abottleneck(self.featurizer(all_x))
        cls_loss = F.cross_entropy(self.aclassifier(all_z), all_y)

        if x_view1 is not None and x_view2 is not None:
            f1 = self.abottleneck(self.featurizer(x_view1.cuda().float()))
            f2 = self.abottleneck(self.featurizer(x_view2.cuda().float()))
            supcon = self.supcon_loss(f1, f2, all_y)
            loss = cls_loss + self.args.gamma * supcon
            loss_dict = {'total': loss.item(), 'cls': cls_loss.item(), 'supcon': supcon.item()}
        else:
            loss = cls_loss
            loss_dict = {'total': loss.item(), 'cls': cls_loss.item()}

        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss_dict

    # ══════════════════════════════════════════════════════════════════════
    # 阶段②: Latent Domain Characterization & PCL
    # ══════════════════════════════════════════════════════════════════════
    def update_d(self, inputs, labels, pdlabels, opt, skip_grl=False):
        """
        阶段②训练逻辑：
        - GRL 域判别损失（ddiscriminator + ReverseLayerF）[skip_grl=True 时跳过]
        - 原型对比损失 ProtoNCE（强制特征靠近伪域原型 + 伪标签引导）
        - 域分配熵惩罚（防止域坍塌：鼓励各域样本量均衡）
        """
        all_x = inputs.cuda().float()
        all_c = pdlabels.cuda().long()
        z1 = self.dbottleneck(self.featurizer(all_x))

        # GRL 域判别损失（M3 消融时跳过）
        if not skip_grl:
            alpha_dyn = self.args.alpha1 * (1.0 / (1.0 + 0.1 * self._round))
            disc_in = Adver_network.ReverseLayerF.apply(z1, alpha_dyn)
            disc_out = self.ddiscriminator(disc_in)
            disc_loss = F.cross_entropy(disc_out, labels.cuda().long())
        else:
            disc_loss = torch.tensor(0.0, device=all_x.device)

        # 原型对比损失：传入伪域标签，使每个样本只靠近自己的原型
        if self.current_domain_prototypes is not None:
            prototypes = self.current_domain_prototypes.cuda()
            proto = self.proto_loss(z1, prototypes, domain_labels=all_c)
        else:
            proto = torch.tensor(0.0, device=all_x.device)

        # ── 域分配熵惩罚 ───────────────────────────────────────────────────
        # ��罚域分布的极端不均衡：H(p) = -∑ p_k log(p_k)
        # N_K 是当前 batch 各域样本数，归一化为分布 q_k
        # loss_ent 越大说明分布越集中（坍塌），越小说明分布越均匀
        N = all_c.size(0)
        K = self.args.latent_domain_num
        counts = torch.bincount(all_c, minlength=K).float()
        q = counts / (N + 1e-8)
        H_q = -(q * torch.log(q + 1e-8)).sum()  # 熵
        # 均均匀分布的熵是 log(K)，归一化到 [0, 1]
        ent_penalty = (torch.log(torch.tensor(K, dtype=torch.float32, device=all_x.device)) - H_q) / torch.log(
            torch.tensor(K, dtype=torch.float32, device=all_x.device))
        ent_penalty = ent_penalty * self.args.lam_ent  # 乘系数

        # 保证 loss 始终有连接到模型参数的梯度路径
        # （skip_grl=True 且 prototypes=None 时, disc_loss 和 proto 都是常量）
        loss = disc_loss + self.args.lam_pcl * proto + ent_penalty + 0.0 * z1.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        return {
            'total': loss.item(),
            'dis': disc_loss.item(),
            'proto': proto.item(),
            'ent_pen': ent_penalty.item()
        }

    # ══════════════════════════════════════════════════════════════════════
    # 更新伪域标签与原型（PCL 聚类）
    # ══════════════════════════════════════════════════════════════════════
    def set_dlabel(self, loader):
        self.dbottleneck.eval()
        self.dclassifier.eval()
        self.featurizer.eval()

        start_test = True
        with torch.no_grad():
            iter_test = iter(loader)
            for _ in range(len(loader)):
                inputs, labels, pdlables, item = next(iter_test)
                inputs = inputs.cuda().float()
                index = item
                feas = self.dbottleneck(self.featurizer(inputs))
                outputs = self.dclassifier(feas)
                if start_test:
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    all_index = index
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                    all_output = torch.cat(
                        (all_output, outputs.float().cpu()), 0)
                    all_index = np.hstack((all_index, index))
        all_output = nn.Softmax(dim=1)(all_output)

        # ── 257 维特征（拼接常数列）用于余弦距离聚类 ──────────────────────────
        all_fea_257 = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea_257 = (all_fea_257.t() / torch.norm(all_fea_257, p=2, dim=1)).t()
        all_fea_257 = all_fea_257.float().cpu().numpy()

        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()
        initc = aff.transpose().dot(all_fea_257)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea_257, initc, 'cosine')
        pred_label = dd.argmin(axis=1)

        for _ in range(1):
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_fea_257)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dd = cdist(all_fea_257, initc, 'cosine')
            pred_label = dd.argmin(axis=1)

        # ── 更新原型（使用 256 维原始特征，与 z1 维度一致）─────────────────────
        domain_num = self.args.latent_domain_num
        all_fea_np = all_fea.float().cpu().numpy()
        prototypes = []
        for d in range(domain_num):
            mask = (pred_label == d)
            if mask.sum() > 0:
                proto = all_fea_np[mask].mean(axis=0)
                proto = proto / (np.linalg.norm(proto) + 1e-8)
            else:
                proto = np.zeros(all_fea_np.shape[1])
            prototypes.append(proto)
        self.current_domain_prototypes = torch.tensor(np.array(prototypes), dtype=torch.float32)

        loader.dataset.set_labels_by_index(pred_label, all_index, 'pdlabel')
        print(Counter(pred_label))
        self.dbottleneck.train()
        self.dclassifier.train()
        self.featurizer.train()
        return Counter(pred_label)

    # ══════════════════════════════════════════════════════════════════════
    # 阶段③: Domain-invariant & Hard Negative Contrastive Learning
    # ══════════════════════════════════════════════════════════════════════
    def update(self, inputs, labels, pdlables, opt, Cpd, x_mirrored=None, skip_grl=False):
        """
        阶段③训练逻辑：
        - 基础手势分类损失
        - 域不变对抗损失（GRL + weighted CE）[skip_grl=True 时跳过]
        - 镜像困难负样本对比损失（可选）

        注意：各损失分别 backward()，避免某个损失数量级过大掩盖其他梯度。
        """
        all_x = inputs.cuda().float()
        all_y = labels.cuda().long()
        all_z = self.bottleneck(self.featurizer(all_x))

        # ① 分类损失（主导信号）
        cls_loss = F.cross_entropy(self.classifier(all_z), all_y)

        # ② GRL 域对齐损失（M3 消融时跳过）
        if not skip_grl:
            cpsum = sum(Cpd.values())
            alpha_dyn = self.args.alpha * (1.0 / (1.0 + 0.1 * self._round))
            disc_input = Adver_network.ReverseLayerF.apply(all_z, alpha_dyn)
            disc_out = self.discriminator(disc_input)
            disc_labels = pdlables.cuda().long()
            disc_loss = torch.tensor(0.0, device=all_x.device)
            for i in range(len(disc_labels)):
                k = pdlables[i]
                disc_loss = disc_loss + (cpsum / (len(Cpd) * Cpd[k.item()])) * \
                           F.cross_entropy(disc_out[i].unsqueeze(0), disc_labels[i].unsqueeze(0))
            disc_loss = disc_loss / len(disc_labels)
        else:
            disc_loss = torch.tensor(0.0, device=all_x.device)

        # ③ 镜像困难负样本对比损失（若提供翻转样本）
        if x_mirrored is not None:
            x_mir = x_mirrored.cuda().float()
            feat_mir = self.bottleneck(self.featurizer(x_mir))
            hard_loss = self.hardnce_loss(all_z, feat_mir)
            total_loss = cls_loss + disc_loss + self.args.beta * hard_loss
            loss_dict = {'total': total_loss.item(), 'cls': cls_loss.item(),
                         'dis': disc_loss.item(), 'hard': hard_loss.item()}
        else:
            total_loss = cls_loss + disc_loss
            loss_dict = {'total': total_loss.item(), 'cls': cls_loss.item(), 'dis': disc_loss.item()}

        opt.zero_grad()
        total_loss.backward()
        opt.step()
        return loss_dict

    def predict(self, x):
        return self.classifier(self.bottleneck(self.featurizer(x)))

    def predict1(self, x):
        return self.ddiscriminator(self.dbottleneck(self.featurizer(x)))
