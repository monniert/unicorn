import argparse

import numpy as np
from torch.utils.data import DataLoader

from dataset import get_dataset
from model import load_model_from_path
from model.renderer import save_mesh_as_gif
from utils import path_mkdir
from utils.path import MODELS_PATH
from utils.logger import print_log
from utils.mesh import save_mesh_as_obj, normalize
from utils.pytorch import get_torch_device


BATCH_SIZE = 32
N_WORKERS = 4
PRINT_ITER = 10
SAVE_GIF = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D reconstruction from single-view images in a folder')
    parser.add_argument('-m', '--model', nargs='?', type=str, required=True, help='Model name to use')
    parser.add_argument('-i', '--input', nargs='?', type=str, required=True, help='Input folder')
    args = parser.parse_args()
    assert args.model is not None and args.input is not None

    device = get_torch_device()
    data = get_dataset(args.input)(split="train", img_size=64)
    loader = DataLoader(data, batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=False)
    print_log(f"Found {len(data)} images in the folder")

    m = load_model_from_path(MODELS_PATH / args.model, data).to(device)
    m.eval()

    print_log("Starting reconstruction...")
    out = path_mkdir(args.input + '_rec')
    n_zeros = int(np.log10(len(data) - 1)) + 1
    for j, (inp, _) in enumerate(loader):
        imgs = inp['imgs'].to(device)
        meshes = m.predict_meshes(m.encoder(imgs))

        B, d, e = len(imgs), m.T_cam[-1], np.mean(m.elev_range)
        for k in range(B):
            nb = j*B + k
            if nb % PRINT_ITER == 0:
                print_log(f"Reconstructed {nb} images...")
            i = str(nb).zfill(n_zeros)
            mcenter = normalize(meshes[k], mode=None, center=True, use_center_mass=True)
            save_mesh_as_obj(mcenter, out / f'{i}_mesh.obj')
            if SAVE_GIF:
                save_mesh_as_gif(mcenter, out / f'{i}_mesh.gif', n_views=100, dist=d, elev=e, renderer=m.renderer)

    print_log("Done!")
