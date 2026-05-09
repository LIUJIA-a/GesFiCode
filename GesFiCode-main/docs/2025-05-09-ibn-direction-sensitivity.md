# v20260509: IBN + Direction Sensitivity Auto-Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve cross-user generalization by (1) replacing shallow BatchNorm with InstanceNorm in ResNet18 (IBN hybrid), and (2) automatically detecting direction-sensitive gestures via SSIM-based flip similarity to control HardNCE loss behavior.

**Architecture:** IBN replaces BN2d with IN2d in ResNet18's layer1/layer2 (shallow layers capture style/domain info, IN removes it). Direction sensitivity is computed pre-training by measuring SSIM between each sample and its frequency-axis flip, aggregated per gesture class. Classes with SSIM significantly below the mean (controlled by `--direction_k`) are flagged as direction-sensitive; for these, flipped samples become hard negatives (pushed away). For direction-agnostic classes, flipped samples remain positive pairs (pulled together, current behavior).

**Tech Stack:** PyTorch, scipy.ndimage (for SSIM computation), PIL/numpy

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `model.py` | Modify | Add IBN: replace BN2d→IN2d in ResNet18 layer1/layer2 |
| `utils.py` | Modify | Add `compute_flip_sensitivity()` function; modify Stage③ in `trainer()` to pass `direction_mask` |
| `loss/common_loss.py` | Modify | Modify `InfoNCE_HardNegative.forward()` to accept `direction_mask` and split positive/negative treatment |
| `algorithm.py` | Modify | Modify `update()` to accept and use `direction_mask` |
| `csimain.py` | Modify | Add `--direction_k` argument |

---

### Task 1: Add `--direction_k` argument to `csimain.py`

**Files:**
- Modify: `csimain.py:96-97` (after `--data_fraction`)

- [ ] **Step 1: Add the argument**

After line 97 (`help="fraction of training data to use (0-1)"`), add:

```python
    parser.add_argument('--direction_k', type=float, default=1.0,
                        help="Direction sensitivity threshold: k in tau = mean - k*std. "
                             "Higher k = fewer classes marked as direction-sensitive. "
                             "Set very high (e.g. 99) to disable direction detection entirely.")
```

- [ ] **Step 2: Verify syntax**

Run: `cd /mnt/nj-1/usr/liujia7/workspace/v20260509/ZLt_Repo/GesFiCode-main && python csimain.py --help 2>&1 | grep direction_k`

Expected: Shows the help text for `--direction_k`.

---

### Task 2: IBN Hybrid in `model.py`

**Files:**
- Modify: `model.py`

- [ ] **Step 1: Replace BN2d with IN2d in ResNet18 layer1/layer2**

Replace the entire `model.py` with:

```python
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models


def _replace_bn_with_in(module):
    """Recursively replace all BatchNorm2d with InstanceNorm2d (no affine, no tracking)."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.InstanceNorm2d(child.num_features, affine=False))
        else:
            _replace_bn_with_in(child)


class FeatureNet(nn.Module):
    def __init__(self, num_classes=10):
        super(FeatureNet, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        children = list(resnet.children())
        # children[0]=conv1, [1]=bn1, [2]=relu, [3]=maxpool, [4]=layer1, [5]=layer2, [6]=layer3, [7]=layer4, [8]=avgpool, [9]=fc
        
        # IBN: replace BN2d with IN2d in shallow layers (conv1's bn, layer1, layer2)
        # children[1] is the top-level bn1
        children[1] = nn.InstanceNorm2d(64, affine=False)
        # layer1 = children[4], layer2 = children[5]
        _replace_bn_with_in(children[4])
        _replace_bn_with_in(children[5])
        # layer3 = children[6], layer4 = children[7] keep BatchNorm2d (semantic layers)

        self.features = nn.Sequential(*children[:-1])  # everything except fc

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out
```

- [ ] **Step 2: Verify the model loads and runs**

Run: `cd /mnt/nj-1/usr/liujia7/workspace/v20260509/ZLt_Repo/GesFiCode-main && python -c "from model import FeatureNet; import torch; m = FeatureNet(); x = torch.randn(2,3,224,224); print(m(x).shape)"`

Expected: `torch.Size([2, 512])`

---

### Task 3: Add `compute_flip_sensitivity()` to `utils.py`

**Files:**
- Modify: `utils.py` (add function after imports, before `print_row`)

- [ ] **Step 1: Add the function**

After the import block (line 12: `from mytransforms import ...`), add:

```python
def compute_flip_sensitivity(dataset_source, args):
    """
    Compute per-class SSIM-based flip sensitivity.
    
    For each gesture class, compute mean SSIM between original spectrogram
    and its frequency-axis (vertical) flip across training samples.
    Classes with SSIM < mean - k*std are flagged as direction-sensitive.
    
    Returns:
        direction_sensitive: dict mapping class_label (int) -> bool
            True = direction-sensitive (flip should be hard negative)
            False = direction-agnostic (flip should be positive pair)
    """
    from scipy.ndimage import uniform_filter
    from PIL import Image
    
    print('\n[Direction Sensitivity] Computing SSIM-based flip sensitivity...')
    
    # Collect per-class SSIM values
    # We need raw images (before transform), so we access the dataset's file list
    class_ssims = {}  # class_label -> list of SSIMs
    
    # Get the root dataset to access file paths
    ds = dataset_source
    while hasattr(ds, 'subset'):
        ds = ds.subset
    while hasattr(ds, 'dataset'):
        ds = ds.dataset
    
    # Sample up to max_samples per class for efficiency
    max_samples = getattr(args, 'direction_max_samples', 50)
    
    # Iterate through dataset to get (image_path, label) pairs
    # Support both WidarDataset and XRF55 (datatrcsie) which have .imgs attribute
    if hasattr(ds, 'imgs'):
        file_label_pairs = [(path, label) for path, label in ds.imgs]
    else:
        print('  WARNING: Dataset has no .imgs attribute, skipping direction detection')
        return {}
    
    # Group by class
    from collections import defaultdict
    class_files = defaultdict(list)
    for path, label in file_label_pairs:
        class_files[label].append(path)
    
    def _ssim_gray(img1, img2, win_size=7):
        """Compute SSIM on grayscale images using uniform_filter."""
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2
        g1 = img1.mean(axis=2).astype(np.float64)
        g2 = img2.mean(axis=2).astype(np.float64)
        mu1 = uniform_filter(g1, win_size)
        mu2 = uniform_filter(g2, win_size)
        sig1_sq = uniform_filter(g1 * g1, win_size) - mu1 * mu1
        sig2_sq = uniform_filter(g2 * g2, win_size) - mu2 * mu2
        sig12 = uniform_filter(g1 * g2, win_size) - mu1 * mu2
        ssim_map = ((2 * mu12 + C1) * (2 * sig12 + C2)) / \
                   ((mu1 * mu1 + mu2 * mu2 + C1) * (sig1_sq + sig2_sq + C2))
        return ssim_map.mean()
    
    for label, files in sorted(class_files.items()):
        sampled = files[:max_samples]
        ssim_vals = []
        for fpath in sampled:
            img = np.array(Image.open(fpath).resize((224, 224)))
            flipped = img[::-1, :, :].copy()
            ssim_vals.append(_ssim_gray(img, flipped))
        mu = np.mean(ssim_vals)
        class_ssims[label] = mu
        print(f'  Class {label}: SSIM={mu:.4f} (n={len(sampled)})')
    
    # Compute threshold: tau = mean - k * std
    all_mus = list(class_ssims.values())
    global_mean = np.mean(all_mus)
    global_std = np.std(all_mus)
    k = args.direction_k
    tau = global_mean - k * global_std
    
    direction_sensitive = {}
    for label, mu in class_ssims.items():
        direction_sensitive[label] = (mu < tau)
    
    sensitive_classes = [l for l, s in direction_sensitive.items() if s]
    agnostic_classes = [l for l, s in direction_sensitive.items() if not s]
    print(f'  Threshold tau = {global_mean:.4f} - {k}*{global_std:.4f} = {tau:.4f}')
    print(f'  Direction-sensitive classes: {sensitive_classes}')
    print(f'  Direction-agnostic classes:  {agnostic_classes}')
    
    return direction_sensitive
```

**IMPORTANT BUG FIX:** The `_ssim_gray` function above has a bug — `mu12` is used but not defined. The correct variable name is `mu1 * mu2`. The actual code should be:

```python
    def _ssim_gray(img1, img2, win_size=7):
        """Compute SSIM on grayscale images using uniform_filter."""
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2
        g1 = img1.mean(axis=2).astype(np.float64)
        g2 = img2.mean(axis=2).astype(np.float64)
        mu1 = uniform_filter(g1, win_size)
        mu2 = uniform_filter(g2, win_size)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sig1_sq = uniform_filter(g1 * g1, win_size) - mu1_sq
        sig2_sq = uniform_filter(g2 * g2, win_size) - mu2_sq
        sig12 = uniform_filter(g1 * g2, win_size) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sig12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sig1_sq + sig2_sq + C2))
        return ssim_map.mean()
```

Use the corrected version when implementing.

- [ ] **Step 2: Verify the function runs**

Run: `cd /mnt/nj-1/usr/liujia7/workspace/v20260509/ZLt_Repo/GesFiCode-main && python -c "
from utils import compute_flip_sensitivity
import argparse
args = argparse.Namespace(direction_k=1.0, direction_max_samples=10, dataset='xrf55',
                          data_path='/mnt/nj-1/usr/liujia7/workspace/datasets/xrf55_processed_data/Processed_Data_Scene_1')
from dataloader import datatrcsie
ds = datatrcsie(data_list=args.data_path, transform=None)
result = compute_flip_sensitivity(ds, args)
print(result)
"`

Expected: Prints per-class SSIM values and the direction-sensitive/agnostic classification.

---

### Task 4: Modify `InfoNCE_HardNegative` in `loss/common_loss.py`

**Files:**
- Modify: `loss/common_loss.py` — replace the `InfoNCE_HardNegative` class

- [ ] **Step 1: Replace the InfoNCE_HardNegative class**

Replace the existing `InfoNCE_HardNegative` class (starting from `class InfoNCE_HardNegative`) with:

```python
class InfoNCE_HardNegative(torch.nn.Module):
    """
    Direction-aware mirror contrastive loss.

    Behavior depends on direction_mask (per-sample):
    - direction_mask[i] = True  (direction-sensitive):
        Flipped sample is a HARD NEGATIVE -> push away.
        Loss = max(0, margin - distance(z_i, z_i^mir)) + standard negative term
    - direction_mask[i] = False (direction-agnostic):
        Flipped sample is a POSITIVE PAIR -> pull together (original InfoNCE behavior).
        Loss = -log exp(sim(z_i, z_i^mir)/tau) / sum_k exp(sim(z_i, z_k^mir)/tau)

    If direction_mask is None, all samples are treated as direction-agnostic
    (backward compatible with original behavior).
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, feat_orig, feat_mirrored, direction_mask=None):
        """
        feat_orig:       (B, D)  original spectrogram features
        feat_mirrored:   (B, D)  frequency-axis flipped features
        direction_mask:  (B,) bool tensor or None
                         True = direction-sensitive (push flip away)
                         False = direction-agnostic (pull flip close)
        """
        z1 = F.normalize(feat_orig, dim=1)
        z2 = F.normalize(feat_mirrored, dim=1)
        B = z1.size(0)

        sim_cross = torch.matmul(z1, z2.T) / self.tau  # (B, B)

        if direction_mask is None:
            # Original behavior: all positive pairs
            pos_sim = torch.diag(sim_cross)
            denominator = torch.exp(sim_cross).sum(dim=1)
            loss = -pos_sim + torch.log(denominator + 1e-8)
            return loss.mean()

        # Split into agnostic (positive pair) and sensitive (hard negative)
        agnostic_mask = ~direction_mask  # (B,) True = agnostic

        loss = torch.zeros(B, device=z1.device)

        # Agnostic samples: standard InfoNCE (pull flip close)
        if agnostic_mask.any():
            pos_sim_ag = torch.diag(sim_cross)  # (B,)
            denom_ag = torch.exp(sim_cross).sum(dim=1)  # (B,)
            loss_ag = -pos_sim_ag + torch.log(denom_ag + 1e-8)
            loss[agnostic_mask] = loss_ag[agnostic_mask]

        # Sensitive samples: push flip away
        # Use negative InfoNCE: maximize distance to own flip, minimize to others
        # L_i = -log( sum_{k!=i} exp(sim(z_i, z_k^mir)/tau) / sum_k exp(sim(z_i, z_k^mir)/tau) )
        # Equivalently: L_i = log(1 + exp(sim(z_i, z_i^mir)/tau) / sum_{k!=i} exp(sim(z_i, z_k^mir)/tau))
        # Simpler: just use -InfoNCE = treat diagonal as negative
        if direction_mask.any():
            pos_sim_sens = torch.diag(sim_cross)  # (B,) — this is the "bad" pair to push away
            denom_sens = torch.exp(sim_cross).sum(dim=1)
            # Push away: maximize pos_sim denominator contribution, i.e. make pos_sim small
            # loss = pos_sim (we want to minimize sim with flip = make pos_sim negative/small)
            # Use: loss = max(0, pos_sim - margin) where margin could be 0
            # Simpler approach: loss = pos_sim (gradient pushes sim down)
            loss_sens = pos_sim_sens  # directly penalize high similarity with flip
            loss[direction_mask] = loss_sens[direction_mask]

        return loss.mean()
```

---

### Task 5: Modify `algorithm.py` — `update()` accepts `direction_mask`

**Files:**
- Modify: `algorithm.py` — change `update()` method signature and pass mask to loss

- [ ] **Step 1: Modify the `update` method**

In `algorithm.py`, change the `update` method signature from:

```python
    def update(self, inputs, labels, pdlables, opt, Cpd, x_mirrored=None, skip_grl=False):
```

to:

```python
    def update(self, inputs, labels, pdlables, opt, Cpd, x_mirrored=None, skip_grl=False, direction_mask=None):
```

Then in the same method, change the hard loss computation from:

```python
            hard_loss = self.hardnce_loss(all_z, feat_mir)
```

to:

```python
            hard_loss = self.hardnce_loss(all_z, feat_mir, direction_mask=direction_mask)
```

---

### Task 6: Modify `trainer()` in `utils.py` — compute and use direction sensitivity

**Files:**
- Modify: `utils.py` — two changes in `trainer()` function

- [ ] **Step 1: Compute direction sensitivity after dataset construction**

In `trainer()`, after the line `print(f'Source eval size: {len(dataset_source_eval)} samples')` (around the data loader setup area), add:

```python
    # ── Direction sensitivity auto-detection ──────────────────────────────
    direction_sensitive = compute_flip_sensitivity(dataset_source, args)
```

- [ ] **Step 2: Build direction_mask in Stage③ training loop**

In the Stage③ training loop, where `x_mirrored` is constructed, change the block from:

```python
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
```

to:

```python
                    for inputs, labels, pdlables, item in train_loader:
                        # 频轴翻转构造困难负样本（M4 消融时不构造）
                        if use_hardnce:
                            x_raw = inputs.cuda().float()
                            x_mirrored = freq_flip(x_raw)
                            # Build per-sample direction mask from pre-computed sensitivity
                            if direction_sensitive:
                                dir_mask = torch.tensor(
                                    [direction_sensitive.get(l.item(), False) for l in labels],
                                    dtype=torch.bool, device=x_raw.device
                                )
                            else:
                                dir_mask = None
                        else:
                            x_mirrored = None
                            dir_mask = None

                        step_vals = trainmodel.update(
                            inputs, labels, pdlables, opta, Cpd,
                            x_mirrored=x_mirrored,
                            skip_grl=skip_grl_s3,
                            direction_mask=dir_mask
                        )
```

---

### Task 7: Verification — dry run

- [ ] **Step 1: Syntax check all modified files**

Run:
```bash
cd /mnt/nj-1/usr/liujia7/workspace/v20260509/ZLt_Repo/GesFiCode-main
python -c "import csimain" 2>&1 | head -5
```

Expected: No import errors (may print usage/help if argparse triggers).

- [ ] **Step 2: Quick training smoke test (2 epochs)**

Run a 2-epoch XRF55 cross_user test to verify nothing crashes:

```bash
cd /mnt/nj-1/usr/liujia7/workspace/v20260509/ZLt_Repo/GesFiCode-main
CUDA_VISIBLE_DEVICES=6 python csimain.py \
    --dataset xrf55 \
    --data_path /mnt/nj-1/usr/liujia7/workspace/datasets/xrf55_processed_data/Processed_Data_Scene_1 \
    --max_epoch 2 \
    --direction_k 1.0 \
    --exp_id smoke_test_v20260509 2>&1 | tail -30
```

Expected: Should print direction sensitivity analysis, then run 2 epochs without errors. Accuracy values don't matter for smoke test.

---

## Summary of Changes

| Change | Purpose | Risk |
|--------|---------|------|
| IBN in ResNet18 layer1/layer2 | Remove user-specific style info in shallow features | May slightly reduce in-domain accuracy |
| SSIM-based direction detection | Auto-identify direction-sensitive gestures | Depends on SSIM discrimination; k=1.0 is conservative (only strong outliers flagged) |
| Direction-aware HardNCE loss | Push away flips for sensitive gestures, pull close for agnostic | New gradient signal; beta weight controls impact |
| `--direction_k` hyperparameter | Control sensitivity threshold | k→large disables detection (safe fallback) |
