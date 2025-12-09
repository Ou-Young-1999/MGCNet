import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

def _contrastive_loss_single(anchor_features, contrast_features, labels, exclude_self, temperature=0.1):
    """
    Multi-positive InfoNCE loss (SupCon-style) with numerical stability.
    Assumes anchor_features[i] and contrast_features[i] are views of the same instance.
    Labels indicate class identity; same-class samples are treated as positives.
    """
    device = anchor_features.device
    B = anchor_features.size(0)

    # Normalize features
    anchor_features = F.normalize(anchor_features, p=2, dim=1)
    contrast_features = F.normalize(contrast_features, p=2, dim=1)

    # Cosine similarity matrix [B, B]
    sim_matrix = torch.matmul(anchor_features, contrast_features.T) / temperature

    # Positive mask: same label
    labels = labels.contiguous().view(-1, 1)
    pos_mask = (labels == labels.T).float().to(device)

    if exclude_self:
        # Only applicable when anchor == contrast (same tensor)
        pos_mask.fill_diagonal_(0)

    # Log-probabilities with numerical stability
    logits = sim_matrix
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    # Count positives per anchor (at least 1 to avoid div by zero)
    pos_counts = pos_mask.sum(dim=1).clamp(min=1)

    # Warn if any anchor has no positive (shouldn't happen if self is included)
    # if (pos_counts < 1).any():
    #     warnings.warn("Some anchors have no positive samples!")

    # Compute loss: average over positives for each anchor
    loss_per_anchor = -(pos_mask * log_probs).sum(dim=1) / pos_counts

    return loss_per_anchor.mean()


def intra_time_contrastive_loss(t_features, labels, temperature=0.1):
    """Modality-intrinsic contrastive loss for time-domain features."""
    return _contrastive_loss_single(t_features, t_features, labels, True, temperature)


def intra_freq_contrastive_loss(f_features, labels, temperature=0.1):
    """Modality-intrinsic contrastive loss for frequency-domain features."""
    return _contrastive_loss_single(f_features, f_features, labels, True, temperature)


def inter_modal_contrastive_loss(t_features, f_features, labels, temperature=0.1):
    """Cross-modal contrastive loss: time<->freq within same class."""
    loss_t2f = _contrastive_loss_single(t_features, f_features, labels, False, temperature)
    loss_f2t = _contrastive_loss_single(f_features, t_features, labels, False, temperature)
    return (loss_t2f + loss_f2t) * 0.5


# ----------------------------
# 辅助模块
# ----------------------------

class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, padding=3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# ----------------------------
# 双向门控模块 (BGM)
# ----------------------------
class BGM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # 用于生成门控权重的 MLP（简单用 1x1 conv 或 linear）
        self.gate_time_from_freq = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, C, 1, 1]
            nn.Flatten(),
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )
        self.gate_freq_from_time = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # [B, C, 1]
            nn.Flatten(),
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )

        # 下采样层（时域：stride=2；频域：stride=2）
        self.downsample_time = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )
        self.downsample_freq = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_time, x_freq):
        """
        x_time: [B, C, L]
        x_freq: [B, C, H, W]
        Returns:
            out_time: [B, C, L//2]
            out_freq: [B, C, H//2, W//2]
        """

        # 生成门控
        gate_for_time = self.gate_time_from_freq(x_freq)  # [B, C]
        gate_for_freq = self.gate_freq_from_time(x_time)  # [B, C]

        # 应用门控（广播）
        x_time_gated = x_time * gate_for_time.unsqueeze(-1)  # [B, C, L]
        x_freq_gated = x_freq * gate_for_freq.unsqueeze(-1).unsqueeze(-1)  # [B, C, H, W]

        # 下采样
        out_time = self.downsample_time(x_time_gated)  # [B, C, L//2]
        out_freq = self.downsample_freq(x_freq_gated)  # [B, C, H//2, W//2]

        return out_time, out_freq


# ----------------------------
# 多模态门控融合 (GF)
# ----------------------------
class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, t_feat, f_feat):
        gate = self.gate(torch.cat([t_feat, f_feat], dim=1))  # [B, dim]
        fused = gate * t_feat + (1 - gate) * f_feat
        return self.proj(fused)


# ----------------------------
# 主模型：GMCNet
# ----------------------------
class GMCNet(nn.Module):
    def __init__(self, num_classes=2, temperature=0.1):
        super().__init__()
        self.temperature = temperature

        # === 时域分支初始处理 ===
        self.time_init = nn.Sequential(
            Conv1DBlock(1, 32, kernel_size=25, padding=12)
        )  # 输出: [B, 64, 1250]

        # === 频域分支初始处理 ===
        self.freq_init = nn.Sequential(
            Conv2DBlock(3, 32, kernel_size=3, padding=1),
        )  # 输出: [B, 64, 128, 128]

        # === 多级 BGM + 下采样 ===
        self.bgm1 = BGM(channels=32)
        self.bgm2 = BGM(channels=64)
        self.bgm3 = BGM(channels=128)

        # 下采样模块（用于与 BGM 输出拼接）
        self.time_down1 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )  # 2500 → 1250
        self.freq_down1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.time_down2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )  # 1250 → 625
        self.freq_down2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.time_down3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )   # 625 → 313
        self.freq_down3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # 通道扩展卷积（拼接后调整通道）
        self.expand1_time = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.expand1_freq = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.expand2_time = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.expand2_freq = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.expand3_time = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        self.expand3_freq = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.gru = nn.GRU(
            input_size=256,          # t3 的通道数
            hidden_size=128,
            num_layers=1,
            batch_first=True,        # 输入形状: [B, L, C]
            bidirectional=True      # 可设为 True，但需调整输出维度
        )

        self.gated_fusion = GatedFusion(256)

        # 最终分类头
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_time, x_freq, labels=None, 
                use_intra_time=False, 
                use_intra_freq=False, 
                use_inter_modal=False):
        """
        x_time: [B, 1, 2500]
        x_freq: [B, 3, 256, 256]
        """
        x_time = x_time.transpose(1, 2)
        # 初始特征提取
        t = self.time_init(x_time)  # [B, 64, 1250]
        f = self.freq_init(x_freq)  # [B, 64, 128, 128]

        # ===== 第一级交互 =====
        t_b1, f_b1 = self.bgm1(t, f)  # t_b1: [B,64,625], f_b1: [B,64,64,64]
        t_d1 = self.time_down1(t)  # [B,64,625]
        f_d1 = self.freq_down1(f)  # [B,64,64,64]

        t1 = torch.cat([t_b1, t_d1], dim=1)  # [B,128,625]
        f1 = torch.cat([f_b1, f_d1], dim=1)  # [B,128,64,64]
        t1 = self.expand1_time(t1)  # [B,128,625]
        f1 = self.expand1_freq(f1)  # [B,128,64,64]

        # ===== 第二级交互 =====
        t_b2, f_b2 = self.bgm2(t1, f1)  # t_b2: [B,128,313], f_b2: [B,128,32,32]
        t_d2 = self.time_down2(t1)  # [B,128,313]
        f_d2 = self.freq_down2(f1)  # [B,128,32,32]
        t2 = torch.cat([t_b2, t_d2], dim=1)  # [B,256,313]
        f2 = torch.cat([f_b2, f_d2], dim=1)  # [B,256,32,32]
        t2 = self.expand2_time(t2)  # [B,256,313]
        f2 = self.expand2_freq(f2)  # [B,256,32,32]

        # ===== 第三级交互 =====
        t_b3, f_b3 = self.bgm3(t2, f2)
        t_d3 = self.time_down3(t2)
        f_d3 = self.freq_down3(f2)

        t3 = torch.cat([t_b3, t_d3], dim=1)
        f3 = torch.cat([f_b3, f_d3], dim=1)
        t3 = self.expand3_time(t3)
        f3 = self.expand3_freq(f3)

        t3_seq = t3.transpose(1, 2)  # [B, L, 512]
        gru_out, _ = self.gru(t3_seq)  # gru_out: [B, L, 512]
        t_global = gru_out[:, -1, :]   # 取最后一个时间步: [B, 512]

        # Global average pooling
        f_global = F.adaptive_avg_pool2d(f3, 1).squeeze(-1).squeeze(-1)  # [B,512]

        # 融合
        fused = self.gated_fusion(t_global, f_global)
        logits = self.classifier(fused)

        # 有监督对比损失
        if labels is not None:
            scl_losses = {}
            total_scl = 0.0

            if use_intra_time:
                loss_t = intra_time_contrastive_loss(t_global, labels, self.temperature)
                scl_losses['intra_time'] = loss_t
                total_scl += loss_t

            if use_intra_freq:
                loss_f = intra_freq_contrastive_loss(f_global, labels, self.temperature)
                scl_losses['intra_freq'] = loss_f
                total_scl += loss_f

            if use_inter_modal:
                loss_cross = inter_modal_contrastive_loss(t_global, f_global, labels, self.temperature)
                scl_losses['inter_modal'] = loss_cross
                total_scl += loss_cross

            return logits, total_scl, scl_losses

        return logits, t_global, f_global

if __name__ == "__main__":
    model = GMCNet()
    ecg = torch.randn(16, 2500, 1)  # batch=16
    img = torch.randn(16, 3, 256,256)  # batch=16
    y = model(ecg,img)
    print(y.shape)  # 应该是 [16, 2]

    # 计算 FLOPs 和参数量
    flops, params = profile(model, inputs=(ecg,img))
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")  # 将 FLOPs 转换为 GFLOPs
    print(f"Parameters: {params / 1e6:.2f} M")  # 将参数量转换为百万