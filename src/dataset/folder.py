from copy import deepcopy
from functools import lru_cache
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision.transforms import ToTensor, Compose, Resize, RandomCrop, CenterCrop

from utils import path_exists, get_files_from
from utils.image import IMG_EXTENSIONS
from utils.path import DATASETS_PATH, PROJECT_PATH
from .torch_transforms import SquarePad, Resize as ResizeCust


class AbstractFolderDataset(TorchDataset):
    root = DATASETS_PATH
    name = NotImplementedError
    n_channels = 3

    def __init__(self, split, img_size, **kwargs):
        kwargs = deepcopy(kwargs)
        self.split = split
        try:
            self.data_path = path_exists(DATASETS_PATH / self.name)
        except FileNotFoundError:
            self.data_path = path_exists(PROJECT_PATH / self.name)

        self.input_files = get_files_from(self.data_path, IMG_EXTENSIONS, recursive=True, sort=True)

        self.resize_mode = kwargs.pop('resize_mode', 'pad')
        assert self.resize_mode in ['crop', 'pad']
        self.padding_mode = kwargs.pop('padding_mode', 'edge')
        if isinstance(img_size, int):
            self.img_size, self.keep_aspect = (img_size, img_size), True
            self.random_crop = kwargs.pop('random_crop', False) and split == 'train'
        else:
            self.img_size, self.keep_aspect = img_size, False
        assert len(kwargs) == 0, kwargs

    def __len__(self):
        return len(self.input_files) if self.split != 'val' else 5

    def __getitem__(self, idx):
        imgs = self.transform(Image.open(self.input_files[idx]).convert('RGB'))
        poses = torch.cat([torch.eye(3), torch.Tensor([[0], [0], [2.732]])], dim=1)
        return {'imgs': imgs, 'masks': torch.empty(1, *self.img_size), 'poses': poses}, -1

    @property
    @lru_cache()
    def transform(self):
        if self.keep_aspect:
            size = self.img_size[0]
            if self.resize_mode == 'pad':
                tsfs = [ResizeCust(size, fit_inside=True), SquarePad(padding_mode=self.padding_mode), ToTensor()]
            elif self.random_crop:
                tsfs = [Resize(size), RandomCrop(size), ToTensor()]
            else:
                tsfs = [Resize(size), CenterCrop(size), ToTensor()]
        else:
            tsfs = [Resize(self.img_size), ToTensor()]
        return Compose(tsfs)
