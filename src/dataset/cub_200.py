from copy import deepcopy
from functools import lru_cache
from PIL import Image

import numpy as np
from random import random
import torch
from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision.transforms import ToTensor, Compose, Resize, functional as Fvision, InterpolationMode
from scipy.io import loadmat

from utils import path_exists, use_seed
from utils.image import square_bbox
from utils.path import DATASETS_PATH, TMP_PATH


PADDING_BBOX = 0.05
JITTER_BBOX = 0.05
RANDOM_FLIP = True
RANDOM_JITTER = True
EVAL_IMG_SIZE = (256, 256)

KP_LEFT_RIGHT_PERMUTATION = np.array([1, 2, 3, 4, 5, 6, 11, 12, 13, 10, 7, 8, 9, 14, 15]) - 1
KP_NAMES = ['Back', 'Beak', 'Belly', 'Breast', 'Crown', 'FHead', 'LEye', 'LLeg', 'LWing',
            'Nape', 'REye', 'RLeg', 'RWing', 'Tail', 'Throat']
KP_COLOR_MAPPING = [
    (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 0), (255, 0, 255),
    (255, 255, 0), (0, 0, 255), (0, 128, 255), (128, 0, 255), (0, 128, 0),
    (128, 0, 0), (0, 0, 128), (128, 128, 0), (0, 128, 128), (128, 0, 128),
]


class CUB200Dataset(TorchDataset):
    root = DATASETS_PATH
    name = 'cub_200'
    n_channels = 3
    n_views = 1

    def __init__(self, split, img_size, **kwargs):
        kwargs = deepcopy(kwargs)
        self.split = split
        try:
            self.data_path = path_exists(DATASETS_PATH / self.name / 'images')
        except FileNotFoundError:
            self.data_path = path_exists(TMP_PATH / 'datasets' / self.name / 'images')
        path = self.data_path.parent / 'cachedir' / 'cub' / 'data' / f'{split}_cub_cleaned.mat'
        self.data = loadmat(path, struct_as_record=False, squeeze_me=True)['images']
        if self.split in ['val', 'test']:  # XXX images are sorted by model so we shuffle
            with use_seed(123):
                np.random.shuffle(self.data)

        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.padding_mode = kwargs.pop('padding_mode', 'edge')
        self.random_flip = kwargs.pop('random_flip', RANDOM_FLIP) and self.split == 'train'
        self.random_jitter = kwargs.pop('random_jitter', RANDOM_JITTER) and self.split == 'train'
        self.eval_mode = kwargs.pop('eval_mode', False)
        assert len(kwargs) == 0, kwargs

    def __len__(self):
        return len(self.data) if self.split != 'val' else 5

    def __getitem__(self, idx):
        data = self.data[idx]

        img = Image.open(self.data_path / data.rel_path).convert('RGB')
        mask = Image.fromarray(data.mask * 255)
        # Retrieve the GT bbox, XXX bounding box has values in [0, max(H, W)] included
        bbox = np.asarray([data.bbox.x1, data.bbox.y1, data.bbox.x2, data.bbox.y2], np.float32)
        # Increase bbox size with borders and jitter the bbox
        bw, bh = bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1
        bbox += np.asarray([PADDING_BBOX * s for s in [-bw, -bh, bw, bh]], dtype=np.float32)
        if self.random_jitter:
            bbox += np.asarray([JITTER_BBOX * s * (1 - 2 * random()) for s in [bw, bh, bw, bh]], dtype=np.float32)
        # Adjust bbox to a square bbox
        bbox = square_bbox(bbox)
        # Pad image if bbox is outside the image scope, and adjust bbox to new image size
        p_left, p_top = max(0, -bbox[0]), max(0, -bbox[1])
        p_right, p_bottom = max(0, bbox[2] - img.size[0]), max(0, bbox[3] - img.size[1])
        if sum([p_left, p_top, p_right, p_bottom]) > 0:
            img = Fvision.pad(img, (p_left, p_top, p_right, p_bottom), padding_mode=self.padding_mode)
            mask = Fvision.pad(mask, (p_left, p_top, p_right, p_bottom), padding_mode='constant')
        adj_bbox = bbox + np.asarray([p_left, p_top, p_left, p_top], dtype=np.uint16)
        # Crop image, it follows the classical python convention where final values are excluded
        # E.g., img.crop((0, 0, 1, 1)) corresponds to the pixel at [0, 0]
        img, mask = img.crop(adj_bbox), mask.crop(adj_bbox)

        # Adjust GT keypoints
        keypoints = data.parts.T.copy().astype(np.float32)
        visible = keypoints[:, 2].astype(bool)
        keypoints[visible, :2] -= np.asarray([bbox[0], bbox[1]], dtype=np.float32)
        scale = max(EVAL_IMG_SIZE if self.eval_mode else self.img_size) / max(img.size)
        keypoints[visible, :2] *= scale
        keypoints = torch.from_numpy(keypoints)

        # Horizontal flip
        hflip = self.random_flip and np.random.binomial(1, p=0.5)
        if hflip:
            img, mask = map(Fvision.hflip, [img, mask])
            size = EVAL_IMG_SIZE[0] if self.eval_mode else self.img_size[0]
            keypoints[:, 0] = size - keypoints[:, 0] - 1
            keypoints = keypoints[KP_LEFT_RIGHT_PERMUTATION]

        img = self.transform(img)
        mask = self.transform_mask(mask)
        poses = torch.cat([torch.eye(3), torch.Tensor([[0], [0], [2.732]])], dim=1)
        return {'imgs': img, 'masks': mask, 'poses': poses}, {'kps': keypoints}

    @property
    @lru_cache()
    def transform(self):
        return Compose([Resize(self.img_size[0]), ToTensor()])

    @property
    @lru_cache()
    def transform_mask(self):
        size = EVAL_IMG_SIZE[0] if self.eval_mode else self.img_size[0]
        return Compose([Resize(size, interpolation=InterpolationMode.NEAREST), ToTensor()])
