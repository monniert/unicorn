from copy import deepcopy
from functools import lru_cache
from PIL import Image

import numpy as np
from random import random
from scipy import io as scio
import torch
from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision.transforms import (ToTensor, Compose, Resize, RandomCrop, CenterCrop, functional as Fvision,
                                    RandomHorizontalFlip)

from utils import path_exists
from utils.image import square_bbox
from utils.path import DATASETS_PATH
from .torch_transforms import SquarePad, Resize as ResizeCust


PADDING_BBOX = 0.1
JITTER_BBOX = 0.1
BBOX_CROP = True
RANDOM_FLIP = True
RANDOM_JITTER = True


class P3DCarDataset(TorchDataset):
    root = DATASETS_PATH
    name = 'p3d_car'
    n_channels = 3

    def __init__(self, split, img_size, **kwargs):
        kwargs = deepcopy(kwargs)
        self.data_path = path_exists(self.root / 'pascal_3d' / 'Images')
        self.split = split
        eff_split = 'train' if split == 'val' else split
        path = self.data_path.parent / 'ucmr_anno' / 'data' / f'car_{eff_split}.mat'
        self.data = scio.loadmat(str(path), struct_as_record=False, squeeze_me=True)['images']
        self.size = len(self.data) if self.split != 'val' else 5

        self.bbox_crop = kwargs.pop('bbox_crop', BBOX_CROP)
        self.resize_mode = kwargs.pop('resize_mode', 'crop')
        assert self.resize_mode in ['crop', 'pad']
        self.padding_mode = kwargs.pop('padding_mode', 'edge')
        if isinstance(img_size, int):
            self.img_size, self.keep_aspect = (img_size, img_size), True
            self.random_crop = kwargs.pop('random_crop', False) and split == 'train'
        else:
            self.img_size, self.keep_aspect = img_size, False
        self.random_flip = kwargs.pop('random_flip', RANDOM_FLIP)
        self.random_jitter = kwargs.pop('random_jitter', RANDOM_JITTER)
        self.padding_box = kwargs.pop('padding_box', PADDING_BBOX)
        self.jitter_box = kwargs.pop('jitter_box', JITTER_BBOX)
        assert len(kwargs) == 0, kwargs

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data = self.data[idx]
        img = Image.open(self.data_path / data.rel_path).convert('RGB')
        if self.bbox_crop:
            bbox = np.array([data.bbox.x1, data.bbox.y1, data.bbox.x2, data.bbox.y2]) - 1
            bw, bh = bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1
            bbox += np.asarray([round(self.padding_box * s) for s in [-bw, -bh, bw, bh]], dtype=np.int64)
            if self.random_jitter and self.split == 'train':
                bw, bh = bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1
                bbox += np.asarray([round(self.jitter_box * s * (1-2*random())) for s in [bw, bh, bw, bh]],
                                   dtype=np.int64)
            bbox = square_bbox(bbox.tolist())
            p_left, p_top = max(0, -bbox[0]), max(0, -bbox[1])
            p_right, p_bottom = max(0, bbox[2] - img.size[0]), max(0, bbox[3] - img.size[1])
            if sum([p_left, p_top, p_right, p_bottom]) > 0:
                img = Fvision.pad(img, (p_left, p_top, p_right, p_bottom), padding_mode=self.padding_mode)
                bbox = bbox + np.asarray([p_left, p_top, p_left, p_top])
            img = img.crop(bbox)

        img = self.transform(img)
        poses = torch.cat([torch.eye(3), torch.Tensor([[0], [0], [2.732]])], dim=1)
        return {'imgs': img, 'masks': torch.empty(1, *self.img_size), 'poses': poses}, -1

    @property
    @lru_cache()
    def transform(self):
        if self.keep_aspect:
            size = self.img_size[0]
            if self.bbox_crop:
                tsfs = [Resize(size), ToTensor()]
            elif self.resize_mode == 'pad':
                tsfs = [ResizeCust(size, fit_inside=True), SquarePad(padding_mode=self.padding_mode), ToTensor()]
            elif self.random_crop:
                tsfs = [Resize(size), RandomCrop(size), ToTensor()]
            else:
                tsfs = [Resize(size), CenterCrop(size), ToTensor()]
        else:
            tsfs = [Resize(self.img_size), ToTensor()]
        if self.random_flip and self.split == 'train':
            tsfs = [RandomHorizontalFlip()] + tsfs
        return Compose(tsfs)
