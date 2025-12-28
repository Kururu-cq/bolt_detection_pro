import torch
import cv2
import numpy as np
import os
import glob
import random
import torchvision
import timm
from torchvision.models.detection import KeypointRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision import transforms as T
from PIL import Image, ImageOps  # 👈 [修改1] 必须引入 ImageOps

# ==========================================
# 1. 基础配置
# ==========================================
CHECKPOINT_DIR = './checkpoints'
# 请确认你的图片文件夹路径
IMAGE_FOLDER = r"F:\bolt_detection_pro\data\images"

NUM_CLASSES = 4
NUM_KEYPOINTS = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 可视化颜色配置
VIS_CONFIG = {
    "font_scale": 0.8,
    "color_map": {1: (0, 255, 0), 2: (0, 165, 255), 3: (0, 0, 255)},  # 绿, 橙, 红
    "class_names": {1: 'FRONT', 2: 'SIDE', 3: 'MISS'}
}


# ==========================================
# 2. 模型定义 (必须与 start_hrnet.py 完全一致)
# ==========================================
class HRNetBackbone(torch.nn.Module):
    def __init__(self, model_name='hrnet_w32'):
        super().__init__()
        # 预测模式：pretrained=False
        self.body = timm.create_model(model_name, features_only=True, pretrained=False)
        all_channels = self.body.feature_info.channels()
        self.out_channels = all_channels[0]

    def forward(self, x):
        features = self.body(x)
        return {'0': features[0]}


def get_hrnet_model(num_classes, num_keypoints):
    backbone = HRNetBackbone('hrnet_w32')

    # [修改2] 必须和训练代码一致 (小目标 Anchor)
    anchor_generator = AnchorGenerator(sizes=((16, 32, 64, 128, 256),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    # [修改3] 必须和训练代码一致 (ROI Size 14)
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    keypoint_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=14, sampling_ratio=2)

    model = KeypointRCNN(backbone,
                         num_classes=num_classes,
                         num_keypoints=num_keypoints,
                         rpn_anchor_generator=anchor_generator,
                         box_roi_pool=roi_pooler,
                         keypoint_roi_pool=keypoint_roi_pooler)
    return model


# ==========================================
# 3. 加载权重
# ==========================================
def load_best_model():
    model = get_hrnet_model(NUM_CLASSES, NUM_KEYPOINTS)

    pth_files = glob.glob(os.path.join(CHECKPOINT_DIR, "*hrnet*.pth"))
    if not pth_files: pth_files = glob.glob("./*.pth")

    if not pth_files:
        print(f"❌ [错误] 没找到权重文件！请先训练模型。")
        return None

    latest_pth = max(pth_files, key=os.path.getmtime)
    print(f"📂 加载权重: {os.path.basename(latest_pth)}")

    checkpoint = torch.load(latest_pth, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model


# ==========================================
# 4. 推理主函数
# ==========================================
def run_inference():
    # 1. 加载模型
    model = load_best_model()
    if model is None: return

    # 2. 随机抽取图片
    search_path = os.path.join(IMAGE_FOLDER, "*.jpg")
    img_list = glob.glob(search_path)
    if not img_list:
        print(f"❌ [错误] 在 {IMAGE_FOLDER} 里没找到图片！")
        return

    img_path = random.choice(img_list)
    print(f"🎲 随机测试: {os.path.basename(img_path)}")

    # 3. 读取图片 + [关键修复] EXIF 旋转 + Normalize
    try:
        img_temp = Image.open(img_path)
        # 👇👇👇 这一行是解决置信度 0.08 的关键！ 👇👇👇
        original_img = ImageOps.exif_transpose(img_temp)
        # 👆👆👆 ===================================== 👆👆👆
        original_img = original_img.convert("RGB")
    except Exception as e:
        print(f"❌ 读取图片失败: {e}")
        return

    # [修改4] Normalize 必须存在且参数正确
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(original_img).to(DEVICE)

    # 4. 推理
    print(f"🔍 检测中...")
    with torch.no_grad():
        predictions = model([img_tensor])

    pred = predictions[0]
    # 转成 OpenCV 格式 (RGB -> BGR)
    cv_img = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)

    # 5. NMS 过滤
    boxes = pred['boxes']
    scores = pred['scores']
    keep = torchvision.ops.nms(boxes, scores, 0.3)

    boxes = boxes[keep].cpu().numpy()
    labels = pred['labels'][keep].cpu().numpy()
    scores = scores[keep].cpu().numpy()
    keypoints = pred['keypoints'][keep].cpu().numpy()

    print(f"🎯 发现 {len(scores)} 个目标，最高置信度: {scores.max() if len(scores) > 0 else 0:.4f}")

    # 6. 绘制结果
    threshold = 0.5
    found_any = False

    for i in range(len(boxes)):
        if scores[i] < threshold: continue
        found_any = True

        x1, y1, x2, y2 = boxes[i].astype(int)
        cls_id = int(labels[i])

        color = VIS_CONFIG["color_map"].get(cls_id, (200, 200, 200))
        label_str = f"{VIS_CONFIG['class_names'].get(cls_id)} {scores[i]:.2f}"

        # 画框
        cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, 2)
        # 写字
        cv2.putText(cv_img, label_str, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 画点
        kps = keypoints[i]
        points_to_draw = 8 if cls_id != 1 else 6
        for k in range(points_to_draw):
            kx, ky, v = kps[k]
            # 只要坐标大于0就画
            if kx > 0:
                cv2.circle(cv_img, (int(kx), int(ky)), 4, color, -1)

    # 7. 保存并显示
    save_name = f"result_{os.path.basename(img_path)}"
    cv2.imwrite(save_name, cv_img)
    print(f"💾 结果已保存: {save_name}")

    if found_any:
        try:
            cv2.imshow("Result", cv_img)
            print("⌨️  按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            pass
    else:
        print("😱 这张图没检测到任何置信度 > 0.5 的目标")
        # 调试信息：看看是不是因为阈值太高
        if len(scores) > 0:
            print(f"   (调试) 最高分其实是: {scores.max():.4f}，属于类别: {labels[scores.argmax()]}")


if __name__ == '__main__':
    run_inference()