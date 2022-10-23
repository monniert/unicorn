from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .comp_cars import CompCarsDataset
from .cub_200 import CUB200Dataset
from .folder import AbstractFolderDataset
from .lsun import LSUNDataset
from .p3d_car import P3DCarDataset
from .shapenet import ShapeNetDataset
from utils.logger import print_log


def create_train_val_test_loader(cfg, rank=None, world_size=None):
    kwargs = cfg["dataset"]
    name = kwargs.pop("name")
    train = get_dataset(name)(split="train", **kwargs)
    val = get_dataset(name)(split="val", **kwargs)
    test = get_dataset(name)(split="test", **kwargs)

    bs, nw = cfg["training"]["batch_size"], cfg["training"].get("n_workers", 4)
    if rank is not None:
        sampler = DistributedSampler(train, rank=rank, num_replicas=world_size)
        train_loader = DataLoader(train, batch_size=bs, num_workers=nw, shuffle=False, pin_memory=True, sampler=sampler)
    else:
        train_loader = DataLoader(train, batch_size=bs, num_workers=nw, shuffle=True, pin_memory=True)

    val_loader, test_loader = map(lambda d: DataLoader(d, batch_size=bs, num_workers=nw, pin_memory=True), [val, test])
    ntr, nv, nte = len(train), len(val), len(test)
    print_log(f"Dataset '{name}' init: kwargs={kwargs}, n_train={ntr}, n_val={nv}, n_test={nte}, bs={bs}, n_work={nw}")
    return train_loader, val_loader, test_loader


def get_dataset(dataset_name):
    datasets = {
        'comp_cars': CompCarsDataset,
        'cub_200': CUB200Dataset,
        'lsun': LSUNDataset,
        'p3d_car': P3DCarDataset,
        'shapenet_nmr': ShapeNetDataset,
    }
    if dataset_name not in datasets:
        class FolderDataset(AbstractFolderDataset):
            name = dataset_name
        return FolderDataset
    else:
        return datasets[dataset_name]
