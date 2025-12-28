import torch
import os
from PIL import Image, ImageOps
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import numpy as np


class BoltDataset(Dataset):
    def __init__(self, root, json_file, transforms=None):
        self.root = root
        self.coco = COCO(json_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        # 你的全局配置是 8 个点，这里必须硬性规定
        self.num_keypoints = 8

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.root, path)

        # 1. 修复 EXIF 旋转
        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)
        img = img.convert('RGB')

        boxes = []
        labels = []
        keypoints = []
        shape_types = []

        for ann in coco_annotation:
            boxes.append(ann['bbox'])
            labels.append(ann['category_id'])

            # === 核心修改：处理 6点/8点 混合情况 ===
            if 'keypoints' in ann:
                # 原始数据
                kp_raw = np.array(ann['keypoints']).reshape(-1, 3)
                num_points = kp_raw.shape[0]

                # 创建一个标准的 (8, 3) 容器，默认全 0
                kp_final = np.zeros((self.num_keypoints, 3), dtype=np.float32)

                # 把有的点填进去
                # 如果 num_points > 8，只取前8个；如果 < 8，剩下的保持为0
                valid_count = min(num_points, self.num_keypoints)
                kp_final[:valid_count, :] = kp_raw[:valid_count, :]

                # ⚠️ 强制清洗规则 ⚠️
                # 1. 对于原本就没有的点 (比如 6点数据的 index 6,7)，因为初始化是0，所以 x=0,y=0,v=0。
                #    v=0 意味着 Loss 不会计算它。安全！

                # 2. 对于原本就有，但可能是幽灵点 (0,0) 的数据
                #    强制把它们的 v 设为 0
                for i in range(valid_count):
                    px, py, pv = kp_final[i]
                    if px <= 1 and py <= 1:  # 如果点在左上角
                        kp_final[i, 2] = 0  # 设为不可见

                keypoints.append(kp_final)
            else:
                # 如果完全没关键点，填一个全0的占位
                keypoints.append(np.zeros((self.num_keypoints, 3), dtype=np.float32))

            if 'shape_type' in ann:
                shape_types.append(ann['shape_type'])
            else:
                shape_types.append(0)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        if boxes.shape[0] > 0:
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        if len(keypoints) > 0:
            keypoints = torch.as_tensor(np.array(keypoints), dtype=torch.float32)
        else:
            # 极端情况：图里没标关键点，也要返回正确的 shape
            keypoints = torch.zeros((0, self.num_keypoints, 3), dtype=torch.float32)

        shape_types = torch.as_tensor(shape_types, dtype=torch.int64)

        image_id = torch.tensor([img_id])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["keypoints"] = keypoints
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["shape_type"] = shape_types

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)