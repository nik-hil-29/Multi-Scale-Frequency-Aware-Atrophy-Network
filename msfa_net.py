"""
MSFA-Net: Multi-Scale Frequency-Aware Atrophy Network
OASIS MRI Alzheimer's Disease — 4-Class Image Classification

Dataset: kaggle.com/datasets/ninadaithal/imagesoasis
  80,000 preprocessed 2D axial T1-weighted MRI brain slices
  Slices 100–160 along z-axis, skull-stripped, grayscale ~128×128

Classes:
  0 — Non Demented      : 67,200  (83.9%)
  1 — Very Mild Demented: 13,700  (17.1%)
  2 — Mild Demented     :  5,200   (6.5%)
  3 — Moderate Demented :    488   (0.6%)
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from pathlib import Path
from sklearn.metrics import (classification_report, confusion_matrix,
                             balanced_accuracy_score)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: DATASET
# ─────────────────────────────────────────────────────────────────────────────

KAGGLE_DATA_ROOT = '/kaggle/input/imagesoasis/Data'

CLASS_NAMES  = ['Non Demented', 'Very mild Dementia',
                'Mild Dementia', 'Moderate Dementia']
CLASS_COUNTS = [67200, 13700, 5002, 488]
CDR_VALUES   = [0.0, 0.5, 1.0, 2.0]


def build_transforms(img_size=128, mode='train'):
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05),
                                    scale=(0.95, 1.05)),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
            ], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])


class OASISMRIDataset(Dataset):
    FOLDER_MAP = {
        0: ['Non Demented',        'NonDemented',      'non_demented'],
        1: ['Very mild Dementia',  'Very Mild Dementia','VeryMildDemented'],
        2: ['Mild Dementia',       'Mild Demented',    'MildDemented'],
        3: ['Moderate Dementia',   'Moderate Demented','ModerateDemented'],
    }

    def __init__(self, root: str, transform=None):
        self.root      = Path(root)
        self.transform = transform
        self.samples   = []
        self._load()

    def _load(self):
        for label, folder_names in self.FOLDER_MAP.items():
            found = False
            for fname in folder_names:
                folder = self.root / fname
                if folder.exists():
                    for ext in ('*.jpg', '*.jpeg', '*.png'):
                        for img_path in folder.glob(ext):
                            self.samples.append((str(img_path), label))
                    found = True
                    break
            if not found:
                print(f"[WARN] Class {label} folder not found in {self.root}")
        print(f"Loaded {len(self.samples)} images from {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_sample_weights(self):
        counts  = np.bincount([s[1] for s in self.samples],
                              minlength=len(CLASS_NAMES))
        class_w = 1.0 / (counts + 1e-6)
        return [class_w[s[1]] for s in self.samples]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: ARCHITECTURE COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

class LearnableWaveletFrequencyBlock(nn.Module):
    """
    Learnable wavelet frequency decomposition.
    Separates low-frequency (global structure) and high-frequency (fine detail)
    sub-bands via Haar-initialized learnable filters, processed in parallel branches.
    Shape: (B, 1, H, W) → (B, d_out, H/2, W/2)
    """
    def __init__(self, d_out: int = 64):
        super().__init__()
        haar_lo = torch.tensor([[[[1, 1], [1, 1]]]]) / 2.0
        haar_hi = torch.tensor([[[[1, -1], [-1, 1]]]]) / 2.0

        self.ll_filter = nn.Parameter(haar_lo.clone())
        self.hh_filter = nn.Parameter(haar_hi.clone())

        self.ll_branch = nn.Sequential(
            nn.Conv2d(1, d_out // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(d_out // 2),
            nn.GELU(),
        )
        self.hh_branch = nn.Sequential(
            nn.Conv2d(1, d_out // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(d_out // 2),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(d_out, d_out, 1, bias=False),
            nn.BatchNorm2d(d_out),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor):
        ll = F.conv2d(x, self.ll_filter, stride=2, padding=0)
        hh = F.conv2d(x, self.hh_filter, stride=2, padding=0)

        min_h = min(ll.shape[2], hh.shape[2])
        min_w = min(ll.shape[3], hh.shape[3])
        ll = ll[:, :, :min_h, :min_w]
        hh = hh[:, :, :min_h, :min_w]

        ll_feat = self.ll_branch(ll)
        hh_feat = self.hh_branch(hh)
        return self.fuse(torch.cat([ll_feat, hh_feat], dim=1))


class AsymmetricDilatedAtrophyAttention(nn.Module):
    """
    Asymmetric dilated attention with separate horizontal and vertical
    dilation paths to capture elongated anatomical structures.
    Also applies channel-wise SE attention.
    Shape: (B, C, H, W) → (B, C, H, W)
    """
    def __init__(self, channels: int, reduction: int = 8,
                 dil_h: int = 2, dil_v: int = 3):
        super().__init__()
        mid = max(channels // reduction, 8)

        self.h_path = nn.Sequential(
            nn.Conv2d(channels, mid, (1, 3), padding=(0, dil_h),
                      dilation=(1, dil_h), bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, (1, 3), padding=(0, dil_h),
                      dilation=(1, dil_h), bias=False),
        )
        self.v_path = nn.Sequential(
            nn.Conv2d(channels, mid, (3, 1), padding=(dil_v, 0),
                      dilation=(dil_v, 1), bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, (3, 1), padding=(dil_v, 0),
                      dilation=(dil_v, 1), bias=False),
        )
        self.mix = nn.Parameter(torch.tensor(0.5))
        self.se  = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        H, W   = x.shape[2], x.shape[3]
        attn_h = torch.sigmoid(self.h_path(x))
        attn_v = torch.sigmoid(self.v_path(x))

        attn_h = F.interpolate(attn_h, size=(H, W), mode='bilinear',
                               align_corners=False)
        attn_v = F.interpolate(attn_v, size=(H, W), mode='bilinear',
                               align_corners=False)

        mix         = torch.sigmoid(self.mix)
        spatial_attn = mix * attn_h + (1 - mix) * attn_v
        ch_attn      = self.se(x).unsqueeze(-1).unsqueeze(-1)

        return x * spatial_attn * ch_attn


class FrequencySpatialDualAttention(nn.Module):
    """
    Dual attention operating simultaneously in spatial and frequency domains.
    Spatial branch: WHERE in the image matters.
    Frequency branch: WHICH frequency bins matter (DCT-proxy via grouped conv).
    Shape: (B, C, H, W) → (B, C, H, W)
    """
    def __init__(self, channels: int, n_freq_bins: int = 8):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid(),
        )

        assert channels % n_freq_bins == 0
        self.n_freq_bins = n_freq_bins
        self.freq_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, n_freq_bins),
            nn.Softmax(dim=-1),
        )
        self.freq_proj = nn.Conv2d(channels, channels, 1,
                                   groups=n_freq_bins, bias=False)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape

        avg_p   = x.mean(dim=1, keepdim=True)
        max_p   = x.max(dim=1, keepdim=True).values
        sp_attn = self.spatial(torch.cat([avg_p, max_p], dim=1))

        freq_w = self.freq_attn(x)
        freq_w = freq_w.repeat_interleave(C // self.n_freq_bins, dim=1)
        freq_w = freq_w.unsqueeze(-1).unsqueeze(-1)

        x_freq = self.freq_proj(x) * freq_w

        return x * sp_attn + x_freq


class InvertedResidualBlock(nn.Module):
    """
    MobileNetV3-style inverted residual block with optional ADAA injection.
    Expand → Depthwise → ADAA → Squeeze
    """
    def __init__(self, in_ch: int, out_ch: int, expand: int = 4,
                 stride: int = 1, use_adaa: bool = True):
        super().__init__()
        mid_ch = in_ch * expand
        self.use_skip = (stride == 1 and in_ch == out_ch)

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.Hardswish(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1,
                      groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.Hardswish(inplace=True),
            AsymmetricDilatedAtrophyAttention(mid_ch) if use_adaa else nn.Identity(),
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_skip:
            return x + out
        return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: MSFA-NET MODEL
# ─────────────────────────────────────────────────────────────────────────────

class MSFANet(nn.Module):
    """
    Input : (B, 1, 128, 128) — grayscale axial MRI slice
    Output: logits (B, 4) [+ L2-normalised embedding (B, 128) if requested]

    Stage 1  LWFDB              → (B, 64,  64, 64)
    Stage 2  7× InvResidual+ADAA → (B, 256, 16, 16)
    Stage 3  FSDA               → (B, 256, 16, 16)
    Stage 4  AvgPool + MaxPool  → (B, 512)
    Stage 5a Embedding head     → (B, 128)  unit sphere
    Stage 5b Classification head → (B, 4)
    """

    def __init__(self, n_classes: int = 4, img_size: int = 128,
                 emb_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.n_classes = n_classes
        self.emb_dim   = emb_dim

        self.lwfdb = LearnableWaveletFrequencyBlock(d_out=64)

        self.backbone = nn.Sequential(
            InvertedResidualBlock(64,  96,  expand=4, stride=1, use_adaa=True),
            InvertedResidualBlock(96,  96,  expand=4, stride=1, use_adaa=True),
            InvertedResidualBlock(96,  128, expand=4, stride=2, use_adaa=True),
            InvertedResidualBlock(128, 128, expand=4, stride=1, use_adaa=True),
            InvertedResidualBlock(128, 160, expand=4, stride=1, use_adaa=True),
            InvertedResidualBlock(160, 256, expand=4, stride=2, use_adaa=True),
            InvertedResidualBlock(256, 256, expand=4, stride=1, use_adaa=False),
        )

        self.fsda = FrequencySpatialDualAttention(256, n_freq_bins=8)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        self.embed_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, emb_dim),
            nn.LayerNorm(emb_dim),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        x     = self.lwfdb(x)
        x     = self.backbone(x)
        x     = self.fsda(x)

        avg_p = self.gap(x).flatten(1)
        max_p = self.gmp(x).flatten(1)
        feat  = torch.cat([avg_p, max_p], dim=1)

        emb    = F.normalize(self.embed_head(feat), dim=-1)
        logits = self.cls_head(feat)

        if return_embedding:
            return logits, emb
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss.
    CB weighting (Cui et al., CVPR 2019) × Focal modulation (Lin et al., ICCV 2017).
    Addresses extreme class imbalance and hard-sample dominance simultaneously.
    """
    def __init__(self, samples_per_class, beta: float = 0.9999,
                 gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        effective_num = [1.0 - beta**s for s in samples_per_class]
        cb_weights    = [(1.0 - beta) / (e + 1e-6) for e in effective_num]
        cb_sum        = sum(cb_weights)
        cb_weights    = [w / cb_sum * len(samples_per_class) for w in cb_weights]
        self.register_buffer('cb_weights',
                             torch.tensor(cb_weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        log_p   = F.log_softmax(logits, dim=-1)
        ce      = F.nll_loss(log_p, targets, weight=self.cb_weights, reduction='none')
        p_t     = torch.exp(-ce)
        focal_w = (1 - p_t) ** self.gamma
        return (focal_w * ce).mean()


class OrdinalCosineMarginalLoss(nn.Module):
    """
    Ordinal Cosine Margin Loss.
    Enforces CDR-distance ordering in the embedding space:
      - Same-class pairs pulled together (cos_sim → 1).
      - Different-class pairs pushed apart with penalty
        scaled by ordinal distance |CDR_i - CDR_j|.
    """
    def __init__(self, cdr_values=None, pos_margin: float = 0.1,
                 neg_margin: float = 0.5):
        super().__init__()
        cdr   = cdr_values or CDR_VALUES
        cdr_t = torch.tensor(cdr, dtype=torch.float32)
        dist  = (cdr_t.unsqueeze(0) - cdr_t.unsqueeze(1)).abs()
        dist  = dist / dist.max()
        self.register_buffer('ord_dist', dist)
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor):
        B = embeddings.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=embeddings.device)

        cos_sim = embeddings @ embeddings.t()
        ord_w   = self.ord_dist[targets.unsqueeze(0), targets.unsqueeze(1)]

        eye      = torch.eye(B, device=embeddings.device)
        pos_mask = (targets.unsqueeze(0) == targets.unsqueeze(1)).float() - eye
        neg_mask = (targets.unsqueeze(0) != targets.unsqueeze(1)).float()

        pos_loss   = F.relu(self.pos_margin - cos_sim) * pos_mask
        neg_thresh = self.neg_margin * (1 - ord_w)
        neg_loss   = F.relu(cos_sim - neg_thresh) * neg_mask * ord_w

        denom = (pos_mask.sum() + neg_mask.sum()).clamp(min=1)
        return (pos_loss.sum() + neg_loss.sum()) / denom


class MSFALoss(nn.Module):
    """Combined loss: L = CBFL(logits, y) + λ * OCML(embeddings, y)"""
    def __init__(self, samples_per_class=None, beta=0.9999, gamma=2.0,
                 lambda_ocml=0.25):
        super().__init__()
        spc = samples_per_class or CLASS_COUNTS
        self.cbfl        = ClassBalancedFocalLoss(spc, beta, gamma)
        self.ocml        = OrdinalCosineMarginalLoss()
        self.lambda_ocml = lambda_ocml

    def forward(self, logits, embeddings, targets):
        L_cbfl = self.cbfl(logits, targets)
        L_ocml = self.ocml(embeddings, targets)
        loss   = L_cbfl + self.lambda_ocml * L_ocml
        return loss, {'L_cbfl': L_cbfl.item(), 'L_ocml': L_ocml.item()}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(data_root: str = KAGGLE_DATA_ROOT,
                      img_size: int = 128,
                      batch_size: int = 64, val_split: float = 0.15,
                      test_split: float = 0.10, num_workers: int = 2):
    full_ds = OASISMRIDataset(data_root, transform=build_transforms(img_size, 'train'))

    n       = len(full_ds)
    n_test  = int(n * test_split)
    n_val   = int(n * val_split)
    n_train = n - n_val - n_test

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42))

    val_ds.dataset  = OASISMRIDataset(data_root,
                                       transform=build_transforms(img_size, 'val'))
    test_ds.dataset = OASISMRIDataset(data_root,
                                       transform=build_transforms(img_size, 'val'))

    all_weights   = full_ds.get_sample_weights()
    train_weights = [all_weights[i] for i in train_ds.indices]
    sampler       = WeightedRandomSampler(train_weights, len(train_weights),
                                          replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.autocast(device_type='cuda'):
                logits, embeds = model(imgs, return_embedding=True)
                loss, _ = criterion(logits, embeds, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, embeds = model(imgs, return_embedding=True)
            loss, _ = criterion(logits, embeds, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits, embeds = model(imgs, return_embedding=True)
        loss, _        = criterion(logits, embeds, labels)

        total_loss += loss.item() * imgs.size(0)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    ba     = balanced_accuracy_score(labels, preds)
    acc    = (preds == labels).mean()

    return total_loss / len(loader.dataset), acc, ba, preds, labels


def build_and_train(data_root: str = KAGGLE_DATA_ROOT,
                    epochs: int = 60,
                    batch_size: int = 64, img_size: int = 128,
                    lr: float = 1e-3, seed: int = 42,
                    save_dir: str = '/kaggle/working'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f"Device: {device}  |  AMP: {use_amp}")

    train_loader, val_loader, test_loader = build_dataloaders(
        data_root, img_size, batch_size)

    model     = MSFANet(n_classes=4, img_size=img_size).to(device)
    criterion = MSFALoss(CLASS_COUNTS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs,
        steps_per_epoch=len(train_loader), pct_start=0.1)
    scaler = torch.amp.GradScaler() if use_amp else None

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nMSFA-Net | Trainable parameters: {n_params:,}\n")
    print(f"{'Epoch':>6} {'TrLoss':>8} {'TrAcc':>7} "
          f"{'VlLoss':>8} {'VlAcc':>7} {'VlBA':>7}")
    print("─" * 50)

    best_ba, best_epoch = 0.0, 0
    for epoch in range(1, epochs + 1):
        tr_l, tr_a = train_epoch(model, train_loader, optimizer,
                                  criterion, device, scaler)
        vl_l, vl_a, vl_ba, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if vl_ba > best_ba:
            best_ba, best_epoch = vl_ba, epoch
            torch.save(model.state_dict(), f'{save_dir}/msfa_best.pt')

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6} {tr_l:>8.4f} {tr_a:>7.4f} "
                  f"{vl_l:>8.4f} {vl_a:>7.4f} {vl_ba:>7.4f}")

    print(f"\nBest Balanced Accuracy: {best_ba:.4f} at epoch {best_epoch}")

    model.load_state_dict(torch.load(f'{save_dir}/msfa_best.pt',
                                     map_location=device, weights_only=True))
    _, ts_acc, ts_ba, preds, labels = evaluate(
        model, test_loader, criterion, device)

    print(f"\nTest Accuracy: {ts_acc:.4f} | Test Balanced Acc: {ts_ba:.4f}")
    print("\nPer-class Report:")
    print(classification_report(labels, preds, target_names=CLASS_NAMES))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))
    return model


if __name__ == '__main__':
    if os.path.exists(KAGGLE_DATA_ROOT):
        print("Kaggle environment detected.")
        model = build_and_train()
    else:
        model = MSFANet().eval()
        x = torch.randn(2, 1, 128, 128)
        logits, emb = model(x, return_embedding=True)
        print(f"logits: {logits.shape} | embedding: {emb.shape}")
        print("Sanity check passed.")
