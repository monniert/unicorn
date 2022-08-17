
from contextlib import contextmanager
from matplotlib import pyplot as plt
import PIL
import os
import socket

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import seaborn as sns

from .logger import print_log


VIZ_HEIGHT = 300
VIZ_WIDTH = 500
VIZ_MAX_IMG_SIZE = 64


class Visualizer:
    def __init__(self, viz_port, run_dir):
        if viz_port is not None:
            import visdom
            os.environ["http_proxy"] = ""  # XXX set to solve proxy issues
            if viz_port == 8888 and socket.gethostname() != 'eole':
                viz_port = 8892
            visualizer = visdom.Visdom(port=viz_port, env=f'{run_dir.parent.name}_{run_dir.name}')
            visualizer.delete_env(visualizer.env)  # Clean env before plotting
            print_log(f"Visualizer initialised at {viz_port}")
        else:
            visualizer = None
            print_log("No visualizer initialized")
        self.visualizer = visualizer

    def upload_images(self, images, title, ncol=None):
        if self.visualizer is None:
            return None
        if max(images.shape[2:]) > VIZ_MAX_IMG_SIZE:
            images = F.interpolate(images, size=VIZ_MAX_IMG_SIZE, mode='bilinear', align_corners=False)
        ncol = ncol or len(images)
        self.visualizer.images(images.clamp(0, 1), win=title, nrow=ncol, opts={'title': title,
                               'store_history': True, 'width': VIZ_WIDTH, 'height': VIZ_HEIGHT})

    def upload_lineplot(self, cur_iter, named_values, title):
        if self.visualizer is None:
            return None
        names, values = map(list, zip(*named_values))
        y, x = [values], [[cur_iter] * len(values)]
        self.visualizer.line(y, x, win=title, update='append', opts={'title': title,
                             'legend': names, 'width': VIZ_WIDTH, 'height': VIZ_HEIGHT})

    def upload_barplot(self, named_values, title):
        if self.visualizer is None:
            return None
        names, values = map(list, zip(*named_values))
        self.visualizer.bar(values, win=title, opts={'title': title, 'width': VIZ_HEIGHT, 'height': VIZ_HEIGHT})


@contextmanager
def seaborn_context(n_colors=10):
    with sns.color_palette('colorblind', n_colors), \
            sns.axes_style('white', {'axes.grid': True, 'legend.frameon': True}):
        yield


def plot_lines(df, columns, title, figsize=(10, 5.625), drop_na=True, style=None, unit_yaxis=False, lw=2):
    if not isinstance(columns, (list, tuple)):
        columns = [columns]

    with seaborn_context(len(columns)):
        fig, ax = plt.subplots(figsize=figsize)
        for y in columns:
            if drop_na:
                s = df[y].dropna()
                if len(s) > 0:
                    s.plot(ax=ax, linewidth=lw, style=style)
            else:
                df[y].plot(ax=ax, linewidth=lw, style=style)
        if unit_yaxis:
            ax.set_ylim((0, 1))
            ax.set_yticks([k/10 for k in range(11)])
        ax.grid(axis="x", which='both', color='0.5', linestyle='--', linewidth=0.5)
        ax.grid(axis='y', which='major', color='0.5', linestyle='-', linewidth=0.5)
        ax.legend(framealpha=1, edgecolor='0.3', fancybox=False)
        ax.set_title(title, fontweight="bold")
        fig.tight_layout()

    return fig


def plot_bar(df, title, figsize=(10, 5.625), unit_yaxis=False):
    assert isinstance(df, pd.Series)
    with seaborn_context(1):
        fig, ax = plt.subplots(figsize=figsize)
        df.plot(kind='bar', ax=ax, edgecolor='k', width=0.8, rot=0, linewidth=1)
        if unit_yaxis:
            ax.set_ylim((0, 1))
            ax.set_yticks([k/10 for k in range(11)])
        ax.grid(axis="x", which='both', color='0.5', linestyle='--', linewidth=0.5)
        ax.grid(axis='y', which='major', color='0.5', linestyle='-', linewidth=0.5)
        ax.set_title(title, fontweight="bold")
        fig.tight_layout()

    return fig


def plot_img_grid(images, nrow=1, ncol=None, scale=4):
    if isinstance(images, (list, tuple)):
        assert len(images) > 0
        if isinstance(images[0], PIL.Image.Image):
            images = list(map(np.asarray, images))
        elif isinstance(images[0], torch.Tensor):
            images = [i.cpu().numpy() for i in images]
        if images[0].shape[-1] > 3:   # XXX images are CHW
            images = [i.permute(1, 2, 0) for i in images]

    elif len(images.shape) == 4 and images.shape[-1] > 3:  # images are BCHW
        if isinstance(images, torch.Tensor):
            images = images.permute(0, 2, 3, 1).clamp(0, 1)
        else:
            images = images.transpose(0, 2, 3, 1).clip(0, 1)

    if ncol is None:
        ncol = (len(images) - 1) // nrow + 1
    gridspec_kw = {"wspace": 0.05, "hspace": 0.05}
    fig, axarr = plt.subplots(nrow, ncol, figsize=(ncol * scale, nrow * scale), gridspec_kw=gridspec_kw)
    for ax, img in zip(axarr.ravel(), images):
        if img.shape[-1] == 1:
            img = img[:, :, 0]
        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        else:
            ax.imshow(img)
        ax.set_axis_off()
