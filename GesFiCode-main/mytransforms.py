import numpy as np
from PIL import Image,ImageFilter
import random
import torchvision.transforms as transforms
import torch

class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0, p=0.5):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w, c = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
            N = np.repeat(N, c, axis=2)
            img = N + img
            img[img > 255] = 255                       # 避免有值超过255而反转
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img
    
class Addblur(object):

    def __init__(self, p=0.5,blur="normal"):
        #         self.density = density
        self.p = p
        self.blur= blur

    def __call__(self, img):
        if random.uniform(0, 1) < self.p: 
            if self.blur== "normal":
                img = img.filter(ImageFilter.BLUR)
                return img
            if self.blur== "Gaussian":
                img = img.filter(ImageFilter.GaussianBlur)
                return img
            if self.blur== "mean":
                img = img.filter(ImageFilter.BoxBlur)
                return img

        else:
            return img

class ReBlur(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img_transform = transforms.Compose([
                            transforms.Resize([224, 56]),
                            transforms.Resize([224, 224]),])
            img = img_transform(img)
        return img
    
class RandomShift(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            if random.uniform(0, 1) < 0.5:
                img_transform = transforms.Compose([transforms.RandomCrop(224, padding=(32,0),padding_mode='reflect')])
            else:
                img_transform = transforms.Compose([transforms.RandomCrop(224, padding=(16,0),padding_mode='reflect')])
            img = img_transform(img)
        return img

class RandomSpi(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            finalImg = Image.new('RGB', (448, 224))
            finalImg.paste(img, (0, 0))
            finalImg.paste(img, (224, 0))
            img_transform = transforms.Compose([transforms.RandomCrop(224)])
            img = img_transform(finalImg)
        return img
    
class RandomCpr(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            if random.uniform(0, 1) < 0.5:
                img_transform = transforms.Compose([transforms.Pad(padding=(24,0),padding_mode='edge')])
            else:
                img_transform = transforms.Compose([transforms.Pad(padding=(48,0),padding_mode='edge')])
            img = img_transform(img)
        return img
    
class RandomComPre(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            rpadnum = random.randint(24, 112)
            img_transform = transforms.Compose([transforms.Pad(padding=(0,rpadnum),padding_mode='reflect')])
            img = img_transform(img)
        return img


# ── Physical Augmentation for CSI Time-Frequency Images ─────────────────────

class Physical_Mask_Augment(torch.nn.Module):
    """
    基于方差先验的掩码增强（阶段① SupCon 预训练专用）。

    工作原理：
    1. 将 batch 内每个样本视为同一时刻的多视角视图，先拼接求方差图。
    2. 在方差低的区域（信息量少）随机生成掩码块，强制模型关注高动态区域。
    3. 对两个视图分别应用独立掩码，构造对比增强对。

    输入 x: Tensor (B, C, H, W)，值域 [0, 1]（ToTensor 后的结果）
    返回 x_view1, x_view2: 各自经过随机掩码增强的 Tensor
    """

    def __init__(self, variance_percentile=30, mask_ratio=0.15, patch_size=16):
        super().__init__()
        self.vp = variance_percentile
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        mask_h = H // self.patch_size
        mask_w = W // self.patch_size

        # 全局方差图（跨 batch 和 channel）
        var_map = x.var(dim=(0, 1))  # (H, W)

        # 统计各 patch 的平均方差
        var_patches = var_map.unfold(0, self.patch_size, self.patch_size) \
                               .unfold(1, self.patch_size, self.patch_size)  # (mask_h, mask_w, ps, ps)
        patch_mean = var_patches.mean(dim=(2, 3))  # (mask_h, mask_w)

        # 低方差阈值：方差低于给定百分位的 patch 视为可掩区域
        threshold = torch.quantile(patch_mean, self.vp / 100.0)
        valid_mask = (patch_mean <= threshold).float()

        # 随机生成掩码块（从有效区域中随机选取一定比例）
        rand = torch.rand_like(patch_mean)
        mask_idx = (rand < self.mask_ratio) & (valid_mask > 0)

        # 生成二值掩码图
        mask = torch.zeros_like(patch_mean)
        mask[mask_idx] = 1.0
        mask = mask.repeat_interleave(self.patch_size, dim=0).repeat_interleave(self.patch_size, dim=1)
        mask = mask[:H, :W].unsqueeze(0).unsqueeze(0).expand(B, C, H, W).to(x.device)

        # 对 x_view1 和 x_view2 分别施加独立掩码
        x_view1 = x * (1 - mask)
        x_view2 = x * (1 - mask.flip(2))   # flip 频轴使第二个视图掩码位置略有不同
        return x_view1, x_view2


class Frequency_Axis_Flip(torch.nn.Module):
    """
    频轴翻转（阶段③ 镜像困难负样本专用）。

    沿高度轴（频率轴）对频谱图做镜像翻转，模拟手势在不同频率响应下的
    对称干扰（如 Widar3.0 中某些手势的频率镜像特征），构成困难负样本对。

    输入 x: Tensor (B, C, H, W)
    返回翻转后的 Tensor (B, C, H, W)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.flip(2)  # 沿 H（频率）轴翻转