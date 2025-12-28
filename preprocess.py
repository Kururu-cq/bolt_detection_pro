import os
import json
import shutil
import glob
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# === 配置 ===
RAW_DIR = r'F:\bolt_detection_pro\data\raw'
OUT_IMG_DIR = r'F:\bolt_detection_pro\data\images'
OUT_ANN_DIR = r'F:\bolt_detection_pro\data\annotations'
TRAIN_RATIO = 0.8


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# === 新增：智能类别判断逻辑 ===
def get_category_id(label, num_pts, bbox):
    """
    根据标签名、点数和几何形状综合判断类别
    Returns:
        1: 正视 (Front)
        2: 侧视 (Side)
        3: 缺失 (Miss)
    """
    label = label.upper()

    # 1. 绝对规则：标签包含关键字
    if 'MISS' in label: return 3
    if 'SIDE' in label or 'CE' in label: return 2  # 如果标签叫 "bolt_side"
    if 'FRONT' in label or 'ZHENG' in label: return 1  # 如果标签叫 "bolt_front"

    # 2. 绝对规则：点数特征
    if num_pts >= 7: return 2  # 7-8个点肯定是侧视

    # 3. 几何规则：处理 6 个点的情况 (区分正视还是侧视)
    # 正视通常接近正方形 (Aspect Ratio ~ 1.0)
    # 侧视通常是长条形
    if num_pts == 6:
        w, h = bbox[2], bbox[3]
        if h == 0: return 1
        ratio = w / h

        # 设定阈值：如果长宽比在 0.7 到 1.4 之间，认为是正视(正方形/六边形)
        # 否则认为是侧视(扁长条或竖长条)
        if 0.7 < ratio < 1.45:
            return 1  # 正视
        else:
            return 2  # 侧视

    # 4. 默认兜底 (点数少于6等情况)
    # 倾向于归为侧视，因为正视特征很标准，不符合标准的归为侧视更安全
    return 2


def parse_labelme_json(json_path, img_dir_out, image_id, ann_id_start):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # --- 图片处理 ---
    img_name = os.path.basename(data.get('imagePath', ''))
    if not img_name:
        img_name = os.path.basename(json_path).replace('.json', '.jpg')

    raw_img_path = os.path.join(os.path.dirname(json_path), img_name)
    if not os.path.exists(raw_img_path):
        raw_img_path = json_path.replace('.json', '.jpg')
        if not os.path.exists(raw_img_path):
            return None, [], ann_id_start

    dst_img_path = os.path.join(img_dir_out, img_name)
    shutil.copy2(raw_img_path, dst_img_path)

    try:
        with Image.open(dst_img_path) as img:
            width, height = img.size
    except:
        return None, [], ann_id_start

    image_info = {"id": image_id, "file_name": img_name, "width": width, "height": height}

    # --- 标注处理 ---
    annotations = []
    ann_id = ann_id_start

    # 提取所有矩形框和点
    shapes = data.get('shapes', [])
    rects = [s for s in shapes if s['shape_type'] == 'rectangle']
    points = [s for s in shapes if s['shape_type'] == 'point']

    for rect in rects:
        # 1. 基础信息
        pts = rect['points']
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        bbox = [min(x1, x2), min(y1, y2), abs(x1 - x2), abs(y1 - y2)]  # [x,y,w,h]
        area = bbox[2] * bbox[3]

        # 2. 匹配关键点
        margin = max(bbox[2], bbox[3]) * 0.2
        x_min, y_min = bbox[0] - margin, bbox[1] - margin
        x_max, y_max = bbox[0] + bbox[2] + margin, bbox[1] + bbox[3] + margin

        matched_points = []
        for p in points:
            px, py = p['points'][0]
            if x_min <= px <= x_max and y_min <= py <= y_max:
                matched_points.append(p)

        # 排序关键点 (尝试按 label 后缀 p1, p2... 排序，否则按 y 坐标)
        try:
            matched_points.sort(key=lambda x: int(x['label'].split('_')[-1].replace('p', '')))
        except:
            matched_points.sort(key=lambda x: x['points'][0][1])

        num_pts = len(matched_points)

        # 3. === 智能判定类别 ===
        category_id = get_category_id(rect['label'], num_pts, bbox)

        # 4. 填充关键点 (补齐到 8 个)
        final_kps = []
        for i in range(8):
            if i < len(matched_points):
                px, py = matched_points[i]['points'][0]
                final_kps.extend([px, py, 2])  # 2=可见
            else:
                final_kps.extend([0, 0, 0])  # 0=不存在

        ann = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0,
            "keypoints": final_kps,
            "num_keypoints": num_pts,
            "shape_type": 0  # 废弃字段，留空即可
        }
        annotations.append(ann)
        ann_id += 1

    return image_info, annotations, ann_id


def main():
    print("🚀 [Preprocess] 开始处理数据 (智能区分 6点正视 vs 6点侧视)...")

    if os.path.exists(OUT_ANN_DIR):
        shutil.rmtree(OUT_ANN_DIR)
    ensure_dir(OUT_IMG_DIR)
    ensure_dir(OUT_ANN_DIR)

    json_files = glob.glob(os.path.join(RAW_DIR, "*.json"))
    if not json_files:
        print(f"❌ 错误: {RAW_DIR} 为空！")
        return

    train_files, val_files = train_test_split(json_files, train_size=TRAIN_RATIO, random_state=42)

    def process_files(files, mode):
        coco_output = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "BOLT_FRONT", "keypoints": [f"p{i}" for i in range(1, 9)]},
                {"id": 2, "name": "BOLT_SIDE", "keypoints": [f"p{i}" for i in range(1, 9)]},
                {"id": 3, "name": "MISS", "keypoints": []}
            ]
        }

        img_id = 1
        ann_id = 1

        print(f"🔄 处理 {mode} 集...")
        for jf in tqdm(files):
            img_info, anns, ann_id = parse_labelme_json(jf, OUT_IMG_DIR, img_id, ann_id)
            if img_info and anns:
                coco_output["images"].append(img_info)
                coco_output["annotations"].extend(anns)
                img_id += 1

        with open(os.path.join(OUT_ANN_DIR, f"{mode}.json"), 'w') as f:
            json.dump(coco_output, f)

    process_files(train_files, "train")
    process_files(val_files, "val")
    print("🎉 完成！请务必检查 data/annotations 下的新文件。")


if __name__ == "__main__":
    main()