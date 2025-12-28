import torch
import os


class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    IMG_DIR = os.path.join(DATA_DIR, 'images')
    TRAIN_ANN = os.path.join(DATA_DIR, 'annotations', 'train.json')
    VAL_ANN = os.path.join(DATA_DIR, 'annotations', 'val.json')
    SAVE_DIR = os.path.join(BASE_DIR, 'checkpoints')

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 0

    NUM_CLASSES = 3

    # === 修改这里 ===
    NUM_KEYPOINTS = 8  # 改为8以适配矩形的最新的点数定义

    BATCH_SIZE = 4
    EPOCHS = 50
    LR = 0.002
    WEIGHT_DECAY = 0.0001
    LR_STEPS = [30, 45]

    LOSS_WEIGHTS = {
        'loss_classifier': 1.0,
        'loss_box_reg': 1.0,
        'loss_keypoint': 1.0,
        'loss_shape': 0.5,
        'loss_geo_hex': 0.5,
        'loss_geo_rect': 0.5
    }

    @staticmethod
    def ensure_dirs():
        os.makedirs(Config.SAVE_DIR, exist_ok=True)