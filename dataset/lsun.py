import io
from functools import lru_cache
import os
import pickle
from PIL import Image
from random import random
import string

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision.transforms import CenterCrop, Compose, ToTensor, Resize, RandomHorizontalFlip, functional as Fvision

from utils import path_exists
from utils.path import DATASETS_PATH, TMP_PATH
from .torch_transforms import SquarePad, Resize as ResizeCust


PADDING_BBOX = 0.1
JITTER_BBOX = 0.1
RANDOM_FLIP = True
RANDOM_JITTER = True


class LSUNDataset(TorchDataset):
    name = 'lsun'
    n_channels = 3

    def __init__(self, split, tag, img_size, **kwargs):
        super().__init__()
        import lmdb
        try:
            self.data_path = path_exists(DATASETS_PATH / self.name / tag)
            root = DATASETS_PATH
        except FileNotFoundError:
            self.data_path = path_exists(TMP_PATH / 'datasets' / self.name / 'images')
            root = TMP_PATH / 'datasets'
        self.split = split
        self.tag = tag

        self.env = lmdb.open(str(self.data_path), max_readers=1, readonly=True, lock=False, readahead=False,
                             meminit=False)
        with self.env.begin(write=False) as txn:
            self.size = txn.stat()["entries"]

        # Cache files
        rel_path = self.data_path.relative_to(root)
        cache_file = "_cache_" + "".join(c for c in str(rel_path) if c in string.ascii_letters)
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = list(txn.cursor().iternext(keys=True, values=False))
            pickle.dump(self.keys, open(cache_file, "wb"))

        self.cleaned = kwargs.pop('cleaned', False)
        if self.cleaned:
            self.indices = pd.read_csv(self.data_path / 'indices.txt', sep=' ', header=None, index_col=0)[1].to_list()
            self.size = len(self.indices)
        self.max_size = kwargs.pop('max_size', self.size)
        self.chunk_idx = 0

        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.resize_mode = kwargs.pop('resize_mode', 'pad')
        assert self.resize_mode in ['crop', 'pad']
        self.padding_mode = kwargs.pop('padding_mode', 'edge')
        self.random_flip = kwargs.pop('random_flip', RANDOM_FLIP) and self.split == 'train'
        self.random_jitter = kwargs.pop('random_jitter', RANDOM_JITTER) and self.split == 'train'
        assert len(kwargs) == 0, kwargs

    def __len__(self):
        return min(self.size, self.max_size) if self.split != 'val' else 5

    def step(self):
        self.chunk_idx += 1
        if (self.chunk_idx + 1) * self.max_size > self.size:
            self.chunk_idx = 0

    def __getitem__(self, idx):
        if self.split == 'val':
            idx += 17
        env = self.env
        real_idx = self.chunk_idx * self.max_size + idx
        if self.cleaned:
            real_idx = self.indices[real_idx]
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[real_idx])

        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        if self.random_jitter:
            w, h = img.size
            bbox = np.asarray([0, 0, w, h], np.float32)
            # Increase bbox size with borders and jitter the bbox
            bw, bh = bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1
            bbox += np.asarray([PADDING_BBOX * s for s in [-bw, -bh, bw, bh]], dtype=np.float32)
            bbox += np.asarray([JITTER_BBOX * s * (1 - 2 * random()) for s in [bw, bh, bw, bh]], dtype=np.float32)
            bbox = np.asarray([int(round(x)) for x in bbox], dtype=np.uint16)  # convert to int
            # Pad image if bbox is outside the image scope, and adjust bbox to new image size
            p_left, p_top = max(0, -bbox[0]), max(0, -bbox[1])
            p_right, p_bottom = max(0, bbox[2] - img.size[0]), max(0, bbox[3] - img.size[1])
            if sum([p_left, p_top, p_right, p_bottom]) > 0:
                img = Fvision.pad(img, (p_left, p_top, p_right, p_bottom), padding_mode=self.padding_mode)
            bbox += np.asarray([p_left, p_top, p_left, p_top], dtype=np.uint16)
            img = img.crop(bbox)
        img = self.transform(img)
        poses = torch.cat([torch.eye(3), torch.Tensor([[0], [0], [2.732]])], dim=1)
        return {'imgs': img, 'masks': torch.empty(1, *self.img_size), 'poses': poses}, -1

    @property
    @lru_cache()
    def transform(self):
        size = self.img_size[0]
        if self.resize_mode == 'pad':
            tsfs = [ResizeCust(size, fit_inside=True), SquarePad(padding_mode=self.padding_mode), ToTensor()]
        else:
            tsfs = [Resize(size), CenterCrop(size), ToTensor()]
        if self.random_flip:
            tsfs = [RandomHorizontalFlip()] + tsfs
        return Compose(tsfs)
