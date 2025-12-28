import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign

# 依然使用你的 HRNet
from .backbone import HRNet

# === 配置常量 ===
ROI_SIZE = 14
FEAT_CHANNELS = 256
HIDDEN_DIM = 256


# === 1. LDAN (保持高精度反卷积设计) ===
class LDAN(nn.Module):
    def __init__(self, in_channels, num_classes, num_keypoints):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, HIDDEN_DIM, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(HIDDEN_DIM)
        self.relu = nn.ReLU(inplace=True)

        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(HIDDEN_DIM, HIDDEN_DIM, 1), nn.Sigmoid()
        )
        self.local_branch = nn.Sequential(
            nn.Conv2d(HIDDEN_DIM, 1, 7, 1, 3), nn.Sigmoid()
        )

        self.cls_head = nn.Linear(HIDDEN_DIM * ROI_SIZE * ROI_SIZE, num_classes)
        self.box_head = nn.Linear(HIDDEN_DIM * ROI_SIZE * ROI_SIZE, num_classes * 4)

        # 反卷积提升分辨率
        self.kps_deconv = nn.Sequential(
            nn.ConvTranspose2d(HIDDEN_DIM, HIDDEN_DIM, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Conv2d(HIDDEN_DIM, num_keypoints, kernel_size=1)
        )

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.bn1(feat)
        feat = self.relu(feat)

        glob_attn = self.global_branch(feat)
        loc_attn = self.local_branch(feat)
        feat_refined = feat * glob_attn * loc_attn

        feat_flat = feat_refined.flatten(1)
        cls_logits = self.cls_head(feat_flat)
        box_deltas = self.box_head(feat_flat)
        kp_heatmaps = self.kps_deconv(feat_refined)

        return cls_logits, box_deltas, kp_heatmaps, feat_flat


# === 2. AGCAK (几何约束) ===
class AGCAK(nn.Module):
    def __init__(self, num_keypoints=8):
        super().__init__()
        input_dim = HIDDEN_DIM * ROI_SIZE * ROI_SIZE
        self.shape_cls = nn.Linear(input_dim, 2)

    def compute_loss(self, keypoints, shape_types):
        losses = {}
        scale_factor = 5.0  # 适当的权重

        hex_mask = (shape_types == 0)
        if hex_mask.sum() > 0:
            kps = keypoints[hex_mask]
            dists = torch.norm(kps[:, 1:] - kps[:, :-1], dim=2)
            dist_last = torch.norm(kps[:, 0] - kps[:, -1], dim=1).unsqueeze(1)
            dists = torch.cat([dists, dist_last], dim=1)
            losses['loss_geo_hex'] = torch.var(dists) * scale_factor

        rect_mask = (shape_types == 1)
        if rect_mask.sum() > 0:
            kps = keypoints[rect_mask][:, :4]
            diag1 = torch.norm(kps[:, 0] - kps[:, 2], dim=1)
            diag2 = torch.norm(kps[:, 1] - kps[:, 3], dim=1)
            losses['loss_geo_rect'] = F.mse_loss(diag1, diag2) * scale_factor
        return losses

    def forward(self, feature_flat):
        return self.shape_cls(feature_flat)


# === 3. ROI Heads (含坐标回归 Loss) ===
class BoltRoIHeads(RoIHeads):
    def __init__(self, box_roi_pool, num_classes, num_keypoints, *args, **kwargs):
        super().__init__(box_roi_pool, box_head=None, box_predictor=None,
                         fg_iou_thresh=0.5, bg_iou_thresh=0.5,
                         batch_size_per_image=512, positive_fraction=0.25,
                         bbox_reg_weights=None,
                         score_thresh=0.05, nms_thresh=0.5, detections_per_img=100)

        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.ldan = LDAN(FEAT_CHANNELS, num_classes, num_keypoints)
        self.agcak = AGCAK(num_keypoints)
        self.grid_x = None
        self.grid_y = None

    def heatmap_to_coord_differentiable(self, heatmaps, boxes):
        # ... (保持原有的可微分坐标解码逻辑不变) ...
        device = heatmaps.device
        N, K, H, W = heatmaps.shape
        if self.grid_x is None or self.grid_x.device != device or self.grid_x.shape[0] != H:
            rng_x = torch.arange(W, device=device, dtype=torch.float32)
            rng_y = torch.arange(H, device=device, dtype=torch.float32)
            self.grid_y, self.grid_x = torch.meshgrid(rng_y, rng_x, indexing='ij')

        heatmaps_flat = heatmaps.view(N, K, -1)
        heatmap_probs = F.softmax(heatmaps_flat, dim=2).view(N, K, H, W)

        expected_x = torch.sum(self.grid_x * heatmap_probs, dim=[2, 3])
        expected_y = torch.sum(self.grid_y * heatmap_probs, dim=[2, 3])
        max_vals, _ = torch.max(heatmaps_flat, dim=2)

        box_w = boxes[:, 2] - boxes[:, 0]
        box_h = boxes[:, 3] - boxes[:, 1]

        x_global = boxes[:, 0].unsqueeze(1) + (expected_x + 0.5) * (box_w.unsqueeze(1) / W)
        y_global = boxes[:, 1].unsqueeze(1) + (expected_y + 0.5) * (box_h.unsqueeze(1) / H)
        return torch.stack([x_global, y_global, max_vals], dim=2)

    def forward(self, features, proposals, image_shapes, targets=None):
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        class_logits, box_regression, kp_heatmaps, feat_flat = self.ldan(box_features)
        shape_logits = self.agcak(feat_flat)

        result = []
        losses = {}

        if self.training:
            labels_tensor = torch.cat(labels, dim=0)
            regression_targets_tensor = torch.cat(regression_targets, dim=0)
            proposals_tensor = torch.cat(proposals, dim=0)

            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels_tensor,
                                                          regression_targets_tensor)
            losses.update({'loss_classifier': loss_classifier, 'loss_box_reg': loss_box_reg})

            gt_keypoints = []
            gt_shapes = []
            for i, (proposal, matched_idx) in enumerate(zip(proposals, matched_idxs)):
                t = targets[i]
                valid_idxs = matched_idx[matched_idx >= 0]
                if len(valid_idxs) > 0:
                    gt_keypoints.append(t['keypoints'][valid_idxs])
                    gt_shapes.append(t['shape_type'][valid_idxs])
                else:
                    dev = class_logits.device
                    gt_keypoints.append(torch.zeros((0, self.num_keypoints, 3), device=dev))
                    gt_shapes.append(torch.zeros((0,), device=dev))

            gt_keypoints = torch.cat(gt_keypoints, dim=0)
            gt_shapes = torch.cat(gt_shapes, dim=0)

            pos_inds = torch.where(labels_tensor > 0)[0]
            if len(pos_inds) > 0:
                pos_heatmaps = kp_heatmaps[pos_inds]
                pos_gt_kps = gt_keypoints[pos_inds]
                pos_gt_shapes = gt_shapes[pos_inds]
                losses['loss_shape'] = F.cross_entropy(shape_logits[pos_inds], pos_gt_shapes)

                # 计算坐标回归 Loss
                pred_kps_coord = self.heatmap_to_coord_differentiable(pos_heatmaps, proposals_tensor[pos_inds])
                vis_mask = pos_gt_kps[:, :, 2] > 0
                if vis_mask.sum() > 0:
                    losses['loss_kp_reg'] = F.l1_loss(
                        pred_kps_coord[vis_mask][:, :2],
                        pos_gt_kps[vis_mask][:, :2]
                    ) * 2.0

                losses.update(self.agcak.compute_loss(pos_gt_kps[:, :, :2], pos_gt_shapes))
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            if len(boxes) > 0:
                final_boxes_tensor = torch.cat(boxes, dim=0)
                final_features = self.box_roi_pool(features, boxes, image_shapes)
                _, _, final_heatmaps, _ = self.ldan(final_features)
                all_keypoints = self.heatmap_to_coord_differentiable(final_heatmaps, final_boxes_tensor)

                kp_cursor = 0
                for i in range(len(boxes)):
                    n_box = len(boxes[i])
                    img_kps = all_keypoints[kp_cursor: kp_cursor + n_box]
                    kp_cursor += n_box
                    result.append({
                        "boxes": boxes[i], "labels": labels[i], "scores": scores[i], "keypoints": img_kps
                    })
            else:
                for i in range(len(boxes)):
                    result.append({
                        "boxes": torch.empty((0, 4)), "labels": torch.empty((0,), dtype=torch.int64),
                        "scores": torch.empty((0,)), "keypoints": torch.empty((0, self.num_keypoints, 3))
                    })
        return result, losses


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    classification_loss = F.cross_entropy(class_logits, labels)
    pos_inds = torch.where(labels > 0)[0]
    labels_pos = labels[pos_inds]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)
    box_loss = F.smooth_l1_loss(box_regression[pos_inds, labels_pos], regression_targets[pos_inds], beta=1.0 / 9,
                                reduction='sum')
    if labels.numel() > 0:
        box_loss = box_loss / labels.numel()
    else:
        box_loss = box_loss * 0.0
    return classification_loss, box_loss


# === 4. 核心 BoltDetector (参数针对性优化) ===
class BoltDetector(FasterRCNN):
    def __init__(self, num_classes, num_keypoints):
        backbone = HRNet()
        backbone.out_channels = 256

        # 1. 增加了一个更小的尺度 16，专门抓小螺栓
        anchor_sizes = ((16, 32, 64, 128, 256),)
        aspect_ratios = ((0.5, 1.0, 2.0),)
        anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
        head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])

        # 2. 核心修改：降低 fg_iou_thresh 到 0.5
        # 这能大幅减少“漏检”，特别是对于正视六边形这种容易被忽略的目标
        rpn = RegionProposalNetwork(
            anchor_generator, head,
            fg_iou_thresh=0.5,  # 之前是 0.7 (太严了)
            bg_iou_thresh=0.3,
            batch_size_per_image=256, positive_fraction=0.5,
            pre_nms_top_n={"training": 2000, "testing": 1000},
            post_nms_top_n={"training": 2000, "testing": 1000},
            nms_thresh=0.7)

        roi_heads = BoltRoIHeads(
            MultiScaleRoIAlign(featmap_names=['0'], output_size=ROI_SIZE, sampling_ratio=2),
            num_classes, num_keypoints
        )

        super().__init__(backbone, num_classes=num_classes, rpn_anchor_generator=None,
                         box_roi_pool=None, box_head=None, box_predictor=None)

        self.rpn = rpn
        self.roi_heads = roi_heads

        self.transform = GeneralizedRCNNTransform(
            min_size=800, max_size=1333,
            image_mean=(0.485, 0.456, 0.406), image_std=(0.229, 0.224, 0.225)
        )