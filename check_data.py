import torch
import numpy as np
import cv2
import os
import random

# === 修正导入部分 ===
try:
    # 优先尝试从 bolt_dataset 导入
    from bolt_dataset import BoltDataset

    print("✅ 成功导入: from bolt_dataset import BoltDataset")
except ImportError:
    try:
        # 如果失败，尝试从 dataset 导入
        from dataset import BoltDataset

        print("✅ 成功导入: from dataset import BoltDataset")
    except ImportError:
        print("❌ [错误] 找不到 dataset.py 或 bolt_dataset.py，请确认文件名！")
        exit()

# 引入你的 transforms (确保 transforms.py 在同一目录下)
try:
    import transforms as T
except ImportError:
    import my_transforms as T  # 兼容旧文件名

# 1. 配置路径 (请确认路径是否正确)
IMG_DIR = r"F:\bolt_detection_pro\data\images"
JSON_FILE = r"F:\bolt_detection_pro\data\annotations\train.json"


def check_data():
    if not os.path.exists(IMG_DIR) or not os.path.exists(JSON_FILE):
        print(f"❌ [错误] 路径不存在，请检查:\n{IMG_DIR}\n{JSON_FILE}")
        return

    # 使用训练时的 transforms (带 Normalize)
    # 这一步是为了检查 Normalize 是否导致坐标和图片对不上
    dataset = BoltDataset(IMG_DIR, JSON_FILE, T.get_transforms(True))

    print(f"🔍 正在检查数据集... 共 {len(dataset)} 张图片")

    # 随机抽一张看
    idx = random.randint(0, len(dataset) - 1)
    print(f"🎲 随机抽取第 {idx} 张图片进行检查...")
    img_tensor, target = dataset[idx]

    # === 关键步骤：反标准化 (把 Normalize 过的图还原) ===
    # 必须用和你 transforms.py 里一模一样的均值方差
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Tensor (C,H,W) -> Numpy (H,W,C)
    img_np = img_tensor.permute(1, 2, 0).numpy()

    # 反归一化: pixel = (pixel * std) + mean
    img_np = (img_np * std + mean)

    # 限制在 0-1 之间并转为 0-255
    img_np = np.clip(img_np, 0, 1)
    img_cv = (img_np * 255).astype(np.uint8).copy()

    # 转成 BGR 以便 OpenCV 显示
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # === 画出标注数据 ===
    boxes = target['boxes'].numpy()
    keypoints = target['keypoints'].numpy()

    print(f"📦 发现 {len(boxes)} 个框")

    if len(boxes) == 0:
        print("⚠️ 警告: 这张图没有标注框！")

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        # 画绿框 (厚度2)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 画红点
        # keypoints shape: [N, 8, 3] -> (x, y, visibility)
        kps = keypoints[i]
        for kp_idx, kp in enumerate(kps):
            x, y, v = map(int, kp)

            # v=0: 不在图内, v=1: 遮挡, v=2: 可见
            # 通常只要 v > 0 我们就认为是有标记的
            if v > 0:
                cv2.circle(img_cv, (x, y), 4, (0, 0, 255), -1)
            else:
                pass
                # print(f"  - 关键点 {kp_idx} 不可见 (v=0)")

    # 保存一下，防止窗口弹不出来
    cv2.imwrite("debug_check_data.jpg", img_cv)
    print("💾 检查结果已保存为 debug_check_data.jpg")

    try:
        cv2.imshow("Data Check (Press Q to exit)", img_cv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print("⚠️ 无法弹出窗口 (可能是在服务器环境)，请查看生成的 jpg 图片。")


if __name__ == "__main__":
    check_data()