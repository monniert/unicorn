from functools import partial
import imageio
from PIL import Image
from pathlib import Path
import os
import shutil

import numpy as np
import torch

from . import path_exists, path_mkdir, get_files_from
from .logger import print_info, print_warning


os.environ['IMAGEIO_FFMPEG_EXE'] = '/usr/bin/ffmpeg'  # XXX pytorch's ffmpeg install does not support H.264 format
IMG_EXTENSIONS = ['jpeg', 'jpg', 'JPG', 'png', 'ppm', 'JPEG']
MAX_GIF_SIZE = 64


def resize(img, size, keep_aspect_ratio=True, resample=Image.ANTIALIAS, fit_inside=True):
    if isinstance(size, int):
        return resize(img, (size, size), keep_aspect_ratio=keep_aspect_ratio, resample=resample, fit_inside=fit_inside)
    elif keep_aspect_ratio:
        if fit_inside:
            ratio = float(min([s1 / s2 for s1, s2 in zip(size, img.size)]))  # XXX bug with np.float64 and round
        else:
            ratio = float(max([s1 / s2 for s1, s2 in zip(size, img.size)]))  # XXX bug with np.float64 and round
        size = round(ratio * img.size[0]), round(ratio * img.size[1])
    return img.resize(size, resample=resample)


def convert_to_img(arr):
    if isinstance(arr, Image.Image):
        return arr

    if isinstance(arr, torch.Tensor):
        if len(arr.shape) == 4:
            arr = arr.squeeze(0)
        elif len(arr.shape) == 2:
            arr = arr.unsqueeze(0)
        arr = arr.permute(1, 2, 0).detach().cpu().numpy()

    assert isinstance(arr, np.ndarray)
    if len(arr.shape) == 3:
        if arr.shape[0] <= 3:  # XXX CHW to HWC
            arr = arr.transpose(1, 2, 0)
        if arr.shape[2] == 1:  # XXX HW1 to HW
            arr = arr[:, :, 0]
    if np.issubdtype(arr.dtype, np.floating):
        arr = (arr.clip(0, 1) * 255)
    return Image.fromarray(arr.astype(np.uint8)).convert('RGB')


def convert_to_rgba(t):
    assert isinstance(t, (torch.Tensor,)) and len(t.size()) == 3
    return Image.fromarray((t.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)*255).astype(np.uint8), 'RGBA')


def save_gif(imgs_or_path, name, in_ext='jpg', size=None, total_sec=10):
    if isinstance(imgs_or_path, (str, Path)):
        path = path_exists(imgs_or_path)
        files = sorted(get_files_from(path, in_ext), key=lambda p: int(p.stem))
        try:
            # XXX images MUST be converted to adaptive color palette otherwise resulting gif has very bad quality
            imgs = [Image.open(f).convert('P', palette=Image.ADAPTIVE) for f in files]
        except OSError as e:
            print_warning(e)
            return None
    else:
        # XXX images MUST be converted to adaptive color palette otherwise resulting gif has very bad quality
        imgs, path = [convert_to_img(i).convert('P', palette=Image.ADAPTIVE) for i in imgs_or_path], Path('.')

    if len(imgs) > 0:
        if size is not None and size != imgs[0].size:
            imgs = list(map(lambda i: resize(i, size=size), imgs))
        tpf = int(total_sec * 1000 / len(imgs))
        imgs[0].save(path.parent / name, optimize=False, save_all=True, append_images=imgs[1:], duration=tpf, loop=0)


def save_video(path, name, in_ext='jpg', as_gif=False, fps=24, quality=8):
    path = path_exists(path)
    files = sorted(get_files_from(path, in_ext), key=lambda p: int(p.stem))
    imgs = np.stack([np.asarray(Image.open(f)) for f in files])
    imageio.mimwrite(name, imgs, fps=fps, quality=quality)

    if as_gif:
        gname = name.split('.')[0] + '.gif'
        os.system(f'ffmpeg -i {name} -vf "fps={fps},'
                  f'split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 {gname}')
        shutil.move(gname, str(path.parent / gname))

    # XXX output video is huge + has incorrect format/codec, passing it through ffmpeg fixes the issue
    os.system(f'ffmpeg -i {name} tmp_{name}')
    shutil.move(f'tmp_{name}', name)
    shutil.move(name, str(path.parent / name))


def draw_border(img, color, width):
    a = np.asarray(img)
    for k in range(width):
        a[k, :] = color
        a[-k-1, :] = color
        a[:, k] = color
        a[:, -k-1] = color
    return Image.fromarray(a)


def square_bbox(bbox):
    """Converts a bbox to have a square shape by increasing size along non-max dimension."""
    sq_bbox = [int(round(x)) for x in bbox]  # convert to int
    width, height = sq_bbox[2] - sq_bbox[0], sq_bbox[3] - sq_bbox[1]
    maxdim = max(width, height)
    offset_w, offset_h = [int(round((maxdim - s) / 2.0)) for s in [width, height]]

    sq_bbox[0], sq_bbox[1] = sq_bbox[0] - offset_w, sq_bbox[1] - offset_h
    sq_bbox[2], sq_bbox[3] = sq_bbox[0] + maxdim, sq_bbox[1] + maxdim
    return sq_bbox


class ImageResizer:
    """Resize images from a given input directory, keeping aspect ratio or not."""
    def __init__(self, input_dir, output_dir, size, in_ext=IMG_EXTENSIONS, out_ext='jpg', keep_aspect_ratio=True,
                 resample=Image.ANTIALIAS, fit_inside=True, rename=False, verbose=True):
        self.input_dir = path_exists(input_dir)
        self.files = get_files_from(input_dir, valid_extensions=in_ext, recursive=True, sort=True)
        self.output_dir = path_mkdir(output_dir)
        self.out_extension = out_ext
        self.resize_func = partial(resize, size=size, keep_aspect_ratio=keep_aspect_ratio, resample=resample,
                                   fit_inside=fit_inside)
        self.rename = rename
        self.name_size = int(np.ceil(np.log10(len(self.files))))
        self.verbose = verbose

    def run(self):
        for k, filename in enumerate(self.files):
            if self.verbose:
                print_info('Resizing and saving {}'.format(filename))
            img = Image.open(filename).convert('RGB')
            img = self.resize_func(img)
            name = str(k).zfill(self.name_size) if self.rename else filename.stem
            img.save(self.output_dir / '{}.{}'.format(name, self.out_extension))


class ImageLogger:
    log_data = True

    def __init__(self, log_dir, target_images=None, n_images=None, out_ext='jpg'):
        if not self.log_data:
            return None

        self.log_dir = path_mkdir(log_dir)
        if len(target_images) > 1:
            if isinstance(target_images, dict):
                if len(target_images['imgs'].shape) == 5:  # multi-view images
                    target_images = target_images['imgs'][:, 1]
                else:
                    target_images = target_images['imgs']
            elif target_images[0].shape != target_images[1].shape:
                target_images = target_images[0]
        self.n_images = len(target_images) if target_images is not None else n_images
        [path_mkdir(self.log_dir / f'img{k}' / 'evolution') for k in range(self.n_images)]
        if target_images is not None:
            [convert_to_img(im).save(self.log_dir / f'img{k}' / 'input.png') for k, im in enumerate(target_images)]
        self.out_ext = out_ext

    def save(self, images, it=None):
        if not self.log_data:
            return None

        if len(images) > 1 and images[0].shape != images[1].shape:
            images = images[0]
        assert len(images) == self.n_images
        if not hasattr(self, '_img_size'):
            self._img_size = tuple(images.shape[2:])
        for k in range(self.n_images):
            if it is not None:
                convert_to_img(images[k]).save(self.log_dir / f'img{k}' / 'evolution' / f'{it}.{self.out_ext}')
            else:
                convert_to_img(images[k]).save(self.log_dir / f'img{k}' / 'final.png')

    def save_gif(self, rmtree=True):
        if not self.log_data:
            return None

        for k in range(self.n_images):
            save_gif(self.log_dir / f'img{k}' / 'evolution', 'evolution.gif', size=self.gif_size)
            if rmtree:
                shutil.rmtree(str(self.log_dir / f'img{k}' / 'evolution'))

    @property
    def gif_size(self):
        if hasattr(self, '_img_size'):
            return MAX_GIF_SIZE if MAX_GIF_SIZE < max(self._img_size) else self._img_size
        else:
            return MAX_GIF_SIZE
