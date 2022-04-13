from copy import deepcopy
from torch.optim import SGD, Adam, AdamW, ASGD, Adamax, Adadelta, Adagrad, RMSprop
from utils.logger import print_log


def create_optimizer(cfg, model):
    kwargs = deepcopy(cfg["training"]["optimizer"] or {})
    name = kwargs.pop("name")
    optimizer = get_optimizer(name)(model.parameters(), **kwargs)
    print_log(f"Optimizer '{name}' init: kwargs={kwargs}")
    return optimizer


def get_optimizer(name):
    if name is None:
        name = 'sgd'
    return {
        "sgd": SGD,
        "adam": Adam,
        "adamw": AdamW,
        "asgd": ASGD,
        "adamax": Adamax,
        "adadelta": Adadelta,
        "adagrad": Adagrad,
        "rmsprop": RMSprop,
    }[name]
