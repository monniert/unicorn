import io
from functools import lru_cache
import os
import pickle
from PIL import Image
import string

import pandas as pd
import torch
from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision.transforms import CenterCrop, Compose, ToTensor, Resize, RandomHorizontalFlip

from utils import path_exists
from utils.path import DATASETS_PATH
from .torch_transforms import SquarePad, Resize as ResizeCust


RANDOM_FLIP = True


class LSUNDataset(TorchDataset):
    root = DATASETS_PATH
    name = 'lsun'
    n_channels = 3

    def __init__(self, split, tag, img_size, **kwargs):
        super().__init__()
        import lmdb
        self.data_path = path_exists(self.root / self.name / tag)
        self.split = split
        self.tag = tag

        self.env = lmdb.open(str(self.data_path), max_readers=1, readonly=True, lock=False, readahead=False,
                             meminit=False)
        with self.env.begin(write=False) as txn:
            self.size = txn.stat()["entries"]
        # Cache files
        cache_file = "_cache_" + "".join(c for c in str(self.data_path) if c in string.ascii_letters)
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
        self.random_flip = kwargs.pop('random_flip', RANDOM_FLIP)
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
        imgs = self.transform(Image.open(buf).convert("RGB"))
        poses = torch.cat([torch.eye(3), torch.Tensor([[0], [0], [2.732]])], dim=1)
        return {'imgs': imgs, 'masks': torch.empty(1, *self.img_size), 'poses': poses}, -1

    @property
    @lru_cache()
    def transform(self):
        size = self.img_size[0]
        if self.resize_mode == 'pad':
            tsfs = [ResizeCust(size, fit_inside=True), SquarePad(padding_mode=self.padding_mode), ToTensor()]
        else:
            tsfs = [Resize(size), CenterCrop(size), ToTensor()]
        if self.random_flip and self.split == 'train':
            tsfs = [RandomHorizontalFlip()] + tsfs
        return Compose(tsfs)
