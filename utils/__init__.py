from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import time
import yaml

from numpy.random import seed as np_seed
from numpy.random import get_state as np_get_state
from numpy.random import set_state as np_set_state
from random import seed as rand_seed
from random import getstate as rand_get_state
from random import setstate as rand_set_state
import torch
from torch import manual_seed as torch_seed
from torch import get_rng_state as torch_get_state
from torch import set_rng_state as torch_set_state


def path_exists(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError('{} does not exist'.format(path.absolute()))
    return path


def path_mkdir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_files_from(dir_path, valid_extensions=None, recursive=False, sort=False):
    path = path_exists(dir_path)
    if recursive:
        files = [f.absolute() for f in path.glob('**/*') if f.is_file()]
    else:
        files = [f.absolute() for f in path.glob('*') if f.is_file()]

    if valid_extensions is not None:
        valid_extensions = [valid_extensions] if isinstance(valid_extensions, str) else valid_extensions
        valid_extensions = ['.{}'.format(ext) if not ext.startswith('.') else ext for ext in valid_extensions]
        files = list(filter(lambda f: f.suffix in valid_extensions, files))

    return sorted(files) if sort else files


def load_yaml(path):
    path = path_exists(path)
    with open(path) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    return cfg


def dump_yaml(cfg, path):
    path = path_exists(path)
    with open(path, 'w') as f:
        return yaml.safe_dump(cfg, f)


@contextmanager
def timer(name, unit='s'):
    start = time.time()
    yield
    delta = time.time() - start
    if unit == 's':
        pass
    elif unit == 'min':
        delta /= 60
    else:
        raise NotImplementedError
    print('{}: {:.2f}{}'.format(name, delta, unit))


class use_seed:
    def __init__(self, seed=None):
        if seed is not None:
            assert isinstance(seed, int) and seed >= 0
        self.seed = seed

    def __enter__(self):
        if self.seed is not None:
            self.rand_state = rand_get_state()
            self.np_state = np_get_state()
            self.torch_state = torch_get_state()
            self.torch_cudnn_deterministic = torch.backends.cudnn.deterministic
            rand_seed(self.seed)
            np_seed(self.seed)
            torch_seed(self.seed)
            torch.backends.cudnn.deterministic = True
        return self

    def __exit__(self, typ, val, _traceback):
        if self.seed is not None:
            rand_set_state(self.rand_state)
            np_set_state(self.np_state)
            torch_set_state(self.torch_state)
            torch.backends.cudnn.deterministic = self.torch_cudnn_deterministic

    def __call__(self, f):
        @wraps(f)
        def wrapper(*args, **kw):
            seed = self.seed if self.seed is not None else kw.pop('seed', None)
            with use_seed(seed):
                return f(*args, **kw)

        return wrapper
