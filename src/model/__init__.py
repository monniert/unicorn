import torch

from .unicorn import Unicorn
from .tools import count_parameters
from utils.logger import print_log
from utils import path_exists


def create_model(cfg, dataset=None):
    kwargs = cfg["model"]
    name = kwargs.pop("name")
    model = get_model(name)(dataset=dataset, **kwargs)
    print_log("Model '{}' init: nb_params={:,}, kwargs={}".format(name, count_parameters(model), kwargs))
    return model


def get_model(name):
    return {
        'unicorn': Unicorn,
    }[name]


def load_model_from_path(model_path, dataset, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(path_exists(model_path), map_location=device.type)
    model = get_model(checkpoint['model_name'])(dataset, **checkpoint['model_kwargs']).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.set_cur_epoch(checkpoint['epoch'])
    return model


class DDPCust(torch.nn.parallel.DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def to(self, device):
        self.module.to(device)
        return self
