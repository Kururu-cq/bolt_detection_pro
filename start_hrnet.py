import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import time
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import KeypointRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import timm

try:
    from bolt_dataset import BoltDataset
except ImportError:
    from dataset import BoltDataset
import transforms as T


class GlobalConfig:
    IMG_DIR = r"F:\bolt_detection_pro\data\images"
    TRAIN_JSON = r"F:\bolt_detection_pro\data\annotations\train.json"
    VAL_JSON = r"F:\bolt_detection_pro\data\annotations\val.json"
    CHECKPOINT_DIR = "./checkpoints"

    BATCH_SIZE = 4
    NUM_WORKERS = 0
    NUM_CLASSES = 4
    NUM_KEYPOINTS = 8
    NUM_EPOCHS = 100
    LR = 0.001  # 基础值，下面会覆盖
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class HRNetBackbone(torch.nn.Module):
    def __init__(self, model_name='hrnet_w32'):
        super().__init__()
        self.body = timm.create_model(model_name, features_only=True, pretrained=True)
        all_channels = self.body.feature_info.channels()
        self.out_channels = all_channels[0]
        print(f"ℹ️ HRNet Backbone: 锁定第0层高清特征, 通道数 {self.out_channels}")

    def forward(self, x):
        features = self.body(x)
        return {'0': features[0]}


def get_hrnet_model(num_classes, num_keypoints):
    print("🏗️ 构建 HRNet (稳定版) 模型...")
    backbone = HRNetBackbone('hrnet_w32')

    # 保持小目标 Anchor
    anchor_generator = AnchorGenerator(sizes=((16, 32, 64, 128, 256),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    # 👇👇👇 改回 14，求稳 👇👇👇
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    keypoint_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=14, sampling_ratio=2)
    # 👆👆👆 ================= 👆👆👆

    model = KeypointRCNN(backbone,
                         num_classes=num_classes,
                         num_keypoints=num_keypoints,
                         rpn_anchor_generator=anchor_generator,
                         box_roi_pool=roi_pooler,
                         keypoint_roi_pool=keypoint_roi_pooler)
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    # 冻结 BN (必保留)
    for module in model.backbone.body.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()

    total_loss = 0
    num_batches = len(data_loader)

    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print(f"❌ Loss 无限大 ({loss_value})")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss_value

        if i % 5 == 0:
            loss_kp = loss_dict['loss_keypoint'].item()
            loss_box = loss_dict['loss_box_reg'].item()
            loss_rpn = loss_dict.get('loss_objectness', torch.tensor(0)).item() + loss_dict.get('loss_rpn_box_reg',
                                                                                                torch.tensor(0)).item()
            print(
                f"  [Epoch {epoch + 1}] {i}/{num_batches} | Total: {loss_value:.3f} (KP: {loss_kp:.3f} | Box: {loss_box:.3f} | RPN: {loss_rpn:.3f})")

    return total_loss / num_batches


@torch.no_grad()
def evaluate_loss(model, data_loader, device):
    model.train()
    total_loss = 0
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
    return total_loss / len(data_loader)


def main():
    print(f"🔥 启动 HRNet 修复版训练 | 过滤(0,0) + 降速")

    if not os.path.exists(GlobalConfig.CHECKPOINT_DIR):
        os.makedirs(GlobalConfig.CHECKPOINT_DIR)

    dataset_train = BoltDataset(GlobalConfig.IMG_DIR, GlobalConfig.TRAIN_JSON, T.get_transforms(True))
    dataset_val = BoltDataset(GlobalConfig.IMG_DIR, GlobalConfig.VAL_JSON, T.get_transforms(False))

    loader_train = DataLoader(dataset_train, batch_size=GlobalConfig.BATCH_SIZE,
                              shuffle=True, num_workers=GlobalConfig.NUM_WORKERS,
                              collate_fn=T.collate_fn)

    loader_val = DataLoader(dataset_val, batch_size=GlobalConfig.BATCH_SIZE,
                            shuffle=False, num_workers=GlobalConfig.NUM_WORKERS,
                            collate_fn=T.collate_fn)

    model = get_hrnet_model(GlobalConfig.NUM_CLASSES, GlobalConfig.NUM_KEYPOINTS)
    model.to(GlobalConfig.DEVICE)

    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]

    # 👇👇👇 调整后的学习率 👇👇👇
    param_groups = [
        {"params": backbone_params, "lr": 0.0002},  # 主干慢
        {"params": head_params, "lr": 0.0005},  # 头部也慢下来，防止 KP 震荡
    ]

    optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    train_losses = []

    print("🚀 开始训练 (请务必删除旧的 checkpoints)...")

    for epoch in range(GlobalConfig.NUM_EPOCHS):
        # 简化的 Warmup
        if epoch == 0:
            optimizer.param_groups[0]['lr'] = 0.0002 * 0.1
            optimizer.param_groups[1]['lr'] = 0.0005 * 0.1
        elif epoch == 1:
            optimizer.param_groups[0]['lr'] = 0.0002
            optimizer.param_groups[1]['lr'] = 0.0005

        t_loss = train_one_epoch(model, optimizer, loader_train, GlobalConfig.DEVICE, epoch)
        v_loss = evaluate_loss(model, loader_val, GlobalConfig.DEVICE)

        lr_scheduler.step()
        train_losses.append(t_loss)

        print(f"✅ Epoch [{epoch + 1}] | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(GlobalConfig.CHECKPOINT_DIR, f"model_hrnet_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"💾 模型已保存: {save_path}")

    plt.figure()
    plt.plot(train_losses)
    plt.title('Training Curve')
    plt.savefig(os.path.join(GlobalConfig.CHECKPOINT_DIR, 'loss_curve.png'))


if __name__ == "__main__":
    main()