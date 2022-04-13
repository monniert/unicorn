from collections import defaultdict, OrderedDict
import numpy as np
import pandas as pd
from pathlib import Path
from pytorch3d.ops import sample_points_from_meshes as sample_points
from pytorch3d.ops.points_alignment import iterative_closest_point
import torch

from utils.logger import print_warning
from .chamfer import chamfer_distance


# Maximum values for bounding box [-0.5, 0.5]^3
EMPTY_PCL_LIST = [('chamfer-L1', np.sqrt(6)), ('chamfer-L1-ICP', np.sqrt(6)), ('normal-cos', 0), ('normal-cos-ICP', 0)]


class Metrics:
    log_data = True

    def __init__(self, *names, log_file=None, append=False):
        self.names = list(names)
        self.meters = defaultdict(AverageMeter)
        if log_file is not None and self.log_data:
            self.log_file = Path(log_file)
            if not self.log_file.exists() or not append:
                with open(self.log_file, mode='w') as f:
                    f.write("iteration\tepoch\tbatch\t" + "\t".join(self.names) + "\n")
        else:
            self.log_file = None

    def log_and_reset(self, *names, it=None, epoch=None, batch=None):
        self.log(it, epoch, batch)
        self.reset(*names)

    def log(self, it, epoch, batch):
        if self.log_file is not None:
            with open(self.log_file, mode="a") as file:
                file.write(f"{it}\t{epoch}\t{batch}\t" + "\t".join(map("{:.6f}".format, self.values)) + "\n")

    def reset(self, *names):
        if len(names) == 0:
            names = self.names
        for name in names:
            self[name].reset()

    def read_log(self):
        if self.log_file is not None:
            return pd.read_csv(self.log_file, sep='\t', index_col=0)
        else:
            return pd.DataFrame()

    def __getitem__(self, name):
        return self.meters[name]

    def __repr__(self):
        return ', '.join(['{}={:.4f}'.format(name, self[name].avg) for name in self.names])

    def __len__(self):
        return len(self.names)

    @property
    def values(self):
        return [self[name].avg for name in self.names]

    def update(self, *name_val, N=1):
        if len(name_val) == 1:
            d = name_val[0]
            assert isinstance(d, dict)
            for k, v in d.items():
                self.update(k, v, N=N)
        else:
            assert len(name_val) == 2
            name, val = name_val
            if name not in self.names:
                raise KeyError(f'{name} not in current metrics')
            if isinstance(val, (tuple, list)):
                self[name].update(val[0], N=val[1])
            else:
                self[name].update(val, N=N)

    def get_named_values(self, filter_fn=None):
        names, values = self.names, self.values
        if filter_fn is not None:
            zip_fn = lambda k_v: filter_fn(k_v[0])
            names, values = map(list, zip(*filter(zip_fn, zip(names, values))))
        return list(zip(names, values))


class AverageMeter:
    """Compute and store the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0

    def update(self, val, N=1):
        self.val = val
        self.sum += val * N
        self.count += N
        self.avg = self.sum / self.count if self.count != 0 else 0


class MeshEvaluator:
    """
    Mesh evaluation class by computing similarity metrics between meshes.
    Code inspired from https://github.com/autonomousvision/differentiable_volumetric_rendering (see im2mesh/eval.py)
    Args:
        n_points (int): number of points to be used for evaluation
    """

    def __init__(self, names=None, log_file=None, fast_cpu=False, append=False):
        self.names = [n for n, v in EMPTY_PCL_LIST] if names is None else names
        self.metrics = Metrics(*self.names, log_file=log_file, append=append)
        self.fast_cpu = fast_cpu
        self.N = 50000 if fast_cpu else 100000

    def update(self, mesh_pred, labels):
        pc_gt, normal_gt = labels['points'], labels['normals']
        for k in range(len(mesh_pred)):
            res = self.evaluate(mesh_pred[k], pc_gt=pc_gt[k], normal_gt=normal_gt[k], run_icp=True)
            self.metrics.update(res)

    def evaluate(self, mesh_pred=None, mesh_gt=None, pc_gt=None, normal_gt=None, run_icp=False):
        assert mesh_gt is not None or pc_gt is not None

        results = []
        if mesh_gt is not None:
            assert len(mesh_gt) == 1
            scale = mesh_gt.verts_packed().abs().max().item()
            if abs(scale - 0.5) > 0.02:  # mesh not normalized to unit cube [-0.5, 0.5]^3
                mesh_pred, mesh_gt = list(map(lambda m: m.scale_verts(0.5 / scale), [mesh_pred, mesh_gt]))
                print_warning('mesh not normalized to unit_cube')
            pc_gt, normal_gt = map(torch.squeeze, sample_points(mesh_gt, self.N, return_normals=True))
        assert isinstance(pc_gt, torch.Tensor)
        with_norm = normal_gt is not None

        pc_pred, normal_pred = map(torch.squeeze, sample_points(mesh_pred, self.N, return_normals=True))
        if self.N < len(pc_pred):
            idxs = torch.randperm(len(pc_pred))[:self.N]
            pc_pred, normal_pred = pc_pred[idxs], normal_pred[idxs] if with_norm else None
        if self.N < len(pc_gt):
            idxs = torch.randperm(len(pc_gt))[:self.N]
            pc_gt, normal_gt = pc_gt[idxs], normal_gt[idxs] if with_norm else None

        if run_icp:
            max_iter = 10 if self.fast_cpu else 30
            pc_pred_icp = iterative_closest_point(pc_pred[None], pc_gt[None], max_iterations=max_iter)[2][0]
            pc_preds, tags = [pc_pred, pc_pred_icp], ['', '-ICP']
        else:
            pc_preds, tags = [pc_pred], ['']

        if not with_norm:
            normal_pred, normal_gt = None, None
        else:
            normal_pred, normal_gt = normal_pred[None], normal_gt[None]
        for pc, tag in zip(pc_preds, tags):
            chamfer_L1, normal = chamfer_distance(pc_gt[None], pc[None], x_normals=normal_gt, y_normals=normal_pred,
                                                  return_L1=True, return_mean=True)
            results += [('chamfer-L1' + tag, chamfer_L1.item()), ('normal-cos' + tag, 1 - normal.item())]

        results = list(filter(lambda x: x[0] in self.names, results))
        return OrderedDict(results)

    def compute(self):
        return self.metrics.values

    def log_and_reset(self, it, epoch, batch):
        self.metrics.log_and_reset(it=it, epoch=epoch, batch=batch)

    def read_log(self):
        return self.metrics.read_log()
