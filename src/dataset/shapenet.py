from copy import deepcopy
from PIL import Image
import yaml

import numpy as np
import torch
from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision.transforms.functional import to_tensor

from utils import path_exists
from utils.path import DATASETS_PATH


class ShapeNetDataset(TorchDataset):
    root = DATASETS_PATH
    name = 'shapenet_nmr'
    n_channels = 3
    img_size = (64, 64)
    n_tot_views = 24
    n_views = 1

    def __init__(self, split, n_views=1, categories=None, **kwargs):
        kwargs = deepcopy(kwargs)
        self.split = split
        self.n_views = n_views
        self.flatten_views = kwargs.pop('flatten_views', True)
        self.include_test = kwargs.pop('include_test', False)
        assert len(kwargs) == 0

        with open(self.data_path / 'metadata.yaml') as fp:
            cfg = yaml.load(fp, Loader=yaml.FullLoader)
        indices = list(cfg.keys())
        cat2idx = {n: k for k in cfg for n in cfg[k]['name'].split(',')}
        if categories is None:
            categories = indices
        else:
            categories = [categories] if isinstance(categories, str) else categories
            categories = list({cat2idx[c] for c in categories})

        self.models = self.get_models(self.split, categories)
        if self.include_test and self.split == 'train':
            self.models += self.get_models('val', categories) + self.get_models('test', categories)
        self.n_models = len(self.models)

        self._R_col_adj = torch.Tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        self._R_row_adj = torch.Tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        self._pc_adj = torch.Tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

    @property
    def data_path(self):
        return path_exists(DATASETS_PATH / self.name)

    def get_models(self, split, categories):
        models = []
        for c in categories:
            with open(self.data_path / c / f'softras_{split}.lst', 'r') as f:
                names = f.read().split('\n')
            names = list(filter(lambda x: len(x) > 0, names))
            models += [{'category': c, 'model': n} for n in names]
        return models

    @property
    def is_sv_train(self):
        return self.split == 'train' and self.n_views == 1

    def __len__(self):
        if not (self.is_sv_train and self.flatten_views):
            return self.n_models
        else:
            return self.n_models * self.n_tot_views

    def __getitem__(self, idx):
        if self.is_sv_train and self.flatten_views:
            # XXX we consider each view as independent samples
            idx, indices = idx % self.n_models, [idx // self.n_models]
        else:
            indices = range(self.n_tot_views)
            if self.n_views < self.n_tot_views:
                indices = np.random.choice(indices, self.n_views, replace=False)

        cat = self.models[idx]['category']
        model = self.models[idx]['model']
        path = self.data_path / cat / model
        cameras = np.load(path / 'cameras.npz')
        pc_npz = np.load(path / 'pointcloud.npz')
        points = torch.Tensor(pc_npz['points']) @ self._pc_adj
        normals = torch.Tensor(pc_npz['normals']) @ self._pc_adj

        imgs, masks, poses = [], [], []
        for i in indices:
            imgs.append(to_tensor(Image.open(path / 'image' / '{}.png'.format(str(i).zfill(4)))))
            masks.append(to_tensor(Image.open(path / 'mask' / '{}.png'.format(str(i).zfill(4))).convert('L')))
            poses.append(self.adjust_extrinsics(torch.Tensor(cameras[f'world_mat_{i}'])))  # 3x4

        if self.n_views > 1:
            return ({'imgs': torch.stack(imgs), 'masks': torch.stack(masks), 'poses': torch.stack(poses)},
                    {'points': points, 'normals': normals})
        else:
            return ({'imgs': imgs[0], 'masks': masks[0], 'poses': poses[0]},
                    {'points': points, 'normals': normals})

    def adjust_extrinsics(self, P):
        R, T = torch.split(P[:-1], [3, 1], dim=1)
        R = self._R_row_adj @ R.T @ self._R_col_adj
        return torch.cat([R, T], dim=1)
