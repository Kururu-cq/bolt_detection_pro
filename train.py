import torch
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign

from .backbone import HRNet
from .modules import LDAN, AGCAK


class BoltRoIHeads(RoIHeads):
    def __init__(self, box_roi_pool, num_classes, num_keypoints, *args, **kwargs):
        # 初始化父类，关闭默认的 Head，因为我们要替换它们
        super().__init__(box_roi_pool, box_head=None, box_predictor=None,
                         fg_iou_thresh=0.5, bg_iou_thresh=0.5,
                         batch_size_per_image=512, positive_fraction=0.25,
                         bbox_reg_weights=None,
                         score_thresh=0.05, nms_thresh=0.5, detections_per_img=100)

        self.num_classes = num_classes
        self.num_keypoints = num_keypoints

        # 核心模块集成
        self.ldan = LDAN(256, num_classes, num_keypoints)
        self.agcak = AGCAK(num_keypoints)

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        重写前向传播，嵌入 LDAN 和 AGCAK
        """
        # 1. 训练阶段：采样 (Matching & Sampling)
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)

        # 2. ROI Align
        box_features = self.box_roi_pool(features, proposals, image_shapes)

        # 3. LDAN 前向
        class_logits, box_regression, kp_heatmaps, feat_flat = self.ldan(box_features)

        # 4. AGCAK 前向 (形状预测)
        shape_logits = self.agcak(feat_flat)

        result = []
        losses = {}

        if self.training:
            # === 计算损失 ===
            # A. 分类与回归损失
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)

            # B. 关键点损失 & C. 形状损失
            gt_keypoints = []
            gt_shapes = []

            for i, (proposal, matched_idx) in enumerate(zip(proposals, matched_idxs)):
                t = targets[i]
                valid_idxs = matched_idx[matched_idx >= 0]

                if len(valid_idxs) > 0:
                    gt_keypoints.append(t['keypoints'][valid_idxs])
                    gt_shapes.append(t['shape_type'][valid_idxs])
                else:
                    device = class_logits.device
                    gt_keypoints.append(torch.zeros((0, self.num_keypoints, 3), device=device))
                    gt_shapes.append(torch.zeros((0,), device=device))

            gt_keypoints = torch.cat(gt_keypoints, dim=0)
            gt_shapes = torch.cat(gt_shapes, dim=0)

            # 只计算前景的额外损失
            pos_inds = torch.where(labels > 0)[0]

            if len(pos_inds) > 0:
                # 形状分类损失
                losses['loss_shape'] = F.cross_entropy(shape_logits[pos_inds], gt_shapes[pos_inds])
                # 几何约束损失 (AGCAK)
                losses.update(self.agcak.compute_loss(gt_keypoints[pos_inds][:, :, :2], gt_shapes[pos_inds]))

            losses.update({'loss_classifier': loss_classifier, 'loss_box_reg': loss_box_reg})
        else:
            # 推理阶段
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            result = [{'boxes': boxes, 'labels': labels, 'scores': scores}]

        return result, losses


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    classification_loss = F.cross_entropy(class_logits, labels)

    # 只计算正样本的回归损失
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction='sum',
    )
    if labels.numel() > 0:
        box_loss = box_loss / labels.numel()
    else:
        box_loss = box_loss * 0

    return classification_loss, box_loss


class BoltDetector(FasterRCNN):
    def __init__(self, num_classes, num_keypoints):
        # 1. 骨干
        backbone = HRNet()
        backbone.out_channels = 256

        # 2. RPN
        # 注意 sizes 的写法：必须是 ((...),) 元组套元组
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256),), aspect_ratios=((0.5, 1.0, 2.0),))

        head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])

        # === 修复：这里使用整数代替字典，避免 list vs int 错误 ===
        rpn = RegionProposalNetwork(
            anchor_generator, head,
            fg_iou_thresh=0.7, bg_iou_thresh=0.3,
            batch_size_per_image=256, positive_fraction=0.5,
            pre_nms_top_n=2000,  # 改为 int
            post_nms_top_n=2000,  # 改为 int
            nms_thresh=0.7)

        # 3. ROI Heads (自定义)
        roi_heads = BoltRoIHeads(
            MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2),
            num_classes, num_keypoints
        )

        # 初始化父类
        # === 修复：num_classes 不能为 None ===
        super().__init__(backbone, num_classes=num_classes, rpn_anchor_generator=None,
                         box_roi_pool=None, box_head=None, box_predictor=None)

        # 覆盖组件
        self.rpn = rpn
        self.roi_heads = roi_heads

        # 变换
        self.transform = GeneralizedRCNNTransform(min_size=800, max_size=1333,
                                                  image_mean=[0.485, 0.456, 0.406],
                                                  image_std=[0.229, 0.224, 0.225])