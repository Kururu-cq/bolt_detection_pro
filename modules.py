import torch
import torch.nn as nn
import torch.nn.functional as F


# === LDAN: 轻量级双分支注意力 ===
class LDAN(nn.Module):
    def __init__(self, in_channels, num_classes, num_keypoints):
        super().__init__()
        # 共享层
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU())

        # 全局分支 (Channel Attention)
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 128, 1),
            nn.Sigmoid()
        )

        # 局部分支 (Spatial Attention)
        self.local_conv = nn.Conv2d(128, 1, 7, 1, 3)

        # 预测头
        self.cls_head = nn.Linear(128 * 7 * 7, num_classes)
        self.box_head = nn.Linear(128 * 7 * 7, num_classes * 4)
        self.kp_head = nn.Conv2d(128, num_keypoints, 1)

    def forward(self, x):
        # x: [N_proposals, 256, 7, 7]
        feat = self.stem(x)  # -> 128

        # Dual Attention
        glob_attn = self.global_branch(feat)

        spat_map = self.local_conv(feat)
        spat_attn = torch.sigmoid(spat_map)

        feat_refined = feat * glob_attn * spat_attn

        # Flatten for Box/Cls
        feat_flat = feat_refined.flatten(1)

        cls_logits = self.cls_head(feat_flat)
        box_deltas = self.box_head(feat_flat)
        kp_heatmaps = self.kp_head(feat_refined)

        return cls_logits, box_deltas, kp_heatmaps, feat_flat


# === AGCAK: 自适应几何约束 ===
class AGCAK(nn.Module):
    def __init__(self, num_keypoints=6):
        super().__init__()
        # 形状分类器
        self.shape_cls = nn.Linear(128 * 7 * 7, 2)  # 0:Hex, 1:Rect

    def compute_loss(self, keypoints, shape_types):
        """
        keypoints: [N, K, 2] (归一化或绝对坐标均可，这里假设绝对)
        shape_types: [N] GT Shape
        """
        losses = {}

        # 1. 辅助分类损失 (Shape Classification Loss) 在Detector中计算

        # 2. 几何约束损失
        # 六边形 (Hexagon)
        hex_mask = (shape_types == 0)
        if hex_mask.sum() > 0:
            kps = keypoints[hex_mask]
            # 边长方差 (理想正六边形边长相等)
            dists = torch.norm(kps[:, 1:] - kps[:, :-1], dim=2)
            # 添加闭合边 (点6到点1)
            dist_last = torch.norm(kps[:, 0] - kps[:, -1], dim=1).unsqueeze(1)
            dists = torch.cat([dists, dist_last], dim=1)
            losses['loss_geo_hex'] = torch.var(dists)

        # 矩形 (Rectangle)
        rect_mask = (shape_types == 1)
        if rect_mask.sum() > 0:
            kps = keypoints[rect_mask][:, :4]  # 只取前4点
            # 对角线相等约束
            diag1 = torch.norm(kps[:, 0] - kps[:, 2], dim=1)
            diag2 = torch.norm(kps[:, 1] - kps[:, 3], dim=1)
            losses['loss_geo_rect'] = F.mse_loss(diag1, diag2)

        return losses

    def forward(self, feature_flat):
        return self.shape_cls(feature_flat)