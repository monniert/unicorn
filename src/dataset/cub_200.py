from copy import deepcopy
from functools import lru_cache
from PIL import Image

import numpy as np
import pandas as pd
from random import random
import torch
from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision.transforms import (ToTensor, Compose, Resize, RandomCrop, CenterCrop, functional as Fvision,
                                    RandomHorizontalFlip)

from utils import path_exists, get_files_from, use_seed
from utils.image import square_bbox
from utils.path import DATASETS_PATH
from .torch_transforms import SquarePad, Resize as ResizeCust


PADDING_BBOX = 0.05
JITTER_BBOX = 0.05
BBOX_CROP = True
RANDOM_FLIP = True
RANDOM_JITTER = True
SPLIT_DATA = True


class CUB200Dataset(TorchDataset):
    root = DATASETS_PATH
    name = 'cub_200'
    n_channels = 3

    def __init__(self, split, img_size, **kwargs):
        kwargs = deepcopy(kwargs)
        self.split = split
        self.data_path = path_exists(DATASETS_PATH / 'cub_200' / 'images')
        self.input_files = get_files_from(self.data_path, ['png', 'jpg'], recursive=True, sort=True)

        root = self.data_path.parent
        filenames = pd.read_csv(root / 'images.txt', sep=' ', index_col=0, header=None)[1].tolist()
        bboxes = pd.read_csv(root / 'bounding_boxes.txt', sep=' ', index_col=0, header=None).astype(int)
        bboxes[3], bboxes[4] = bboxes[1] + bboxes[3], bboxes[2] + bboxes[4]  # XXX bbox format before is [x, y, w, h]
        self.bbox_mapping = {filenames[k]: bboxes.iloc[k].tolist() for k in range(len(filenames))}
        assert len(self.bbox_mapping) == len(self.input_files)

        split_data = kwargs.pop('split_data', SPLIT_DATA)
        if split_data:
            split_labels = pd.read_csv(root / 'train_test_split.txt', sep=' ', index_col=0, header=None)[1].tolist()
            split_label = 0 if split == 'test' else 1
            self.input_files = [self.data_path / f for f, lab in zip(filenames , split_labels) if lab == split_label]
        if self.split in ['val', 'test']:  # XXX images are sorted by model so we shuffle
            with use_seed(123):
                np.random.shuffle(self.input_files)

        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.bbox_crop = kwargs.pop('bbox_crop', True)
        self.resize_mode = kwargs.pop('resize_mode', 'pad')
        assert self.resize_mode in ['crop', 'pad']
        self.padding_mode = kwargs.pop('padding_mode', 'edge')
        self.random_flip = kwargs.pop('random_flip', RANDOM_FLIP)
        self.random_jitter = kwargs.pop('random_jitter', RANDOM_JITTER)
        self.random_crop = kwargs.pop('random_crop', False) and split == 'train'
        assert len(kwargs) == 0, kwargs

    def __len__(self):
        return len(self.input_files) if self.split != 'val' else 5

    def __getitem__(self, idx):
        img = Image.open(self.input_files[idx]).convert('RGB')
        if self.bbox_crop:
            bbox = np.asarray(self.bbox_mapping[str(self.input_files[idx].relative_to(self.data_path))])
            bw, bh = bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1
            bbox += np.asarray([round(PADDING_BBOX * s) for s in [-bw, -bh, bw, bh]], dtype=np.int64)
            if self.random_jitter and self.split == 'train':
                bbox += np.asarray([round(JITTER_BBOX * s * (1-2*random())) for s in [bw, bh, bw, bh]], dtype=np.int64)
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
        size = self.img_size[0]
        if self.bbox_crop:
            tsfs = [Resize(size), ToTensor()]
        elif self.resize_mode == 'pad':
            tsfs = [ResizeCust(size, fit_inside=True), SquarePad(padding_mode=self.padding_mode), ToTensor()]
        elif self.random_crop:
            tsfs = [Resize(size), RandomCrop(size), ToTensor()]
        else:
            tsfs = [Resize(size), CenterCrop(size), ToTensor()]
        if self.random_flip and self.split == 'train':
            tsfs = [RandomHorizontalFlip()] + tsfs
        return Compose(tsfs)
