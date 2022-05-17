from copy import deepcopy
from math import log2
import numpy as np
import torch
from torch import nn
from .tools import (Identity, conv1x1, conv3x3, create_mlp, N_UNITS, N_LAYERS, create_upsample_layer,
                    kaiming_weights_init)
from utils.logger import print_log


class GiraffeGenerator(nn.Module):
    """Neural renderer class from https://github.com/autonomousvision/giraffe"""

    def __init__(self, n_features=128, inp_dim=128, img_size=64, out_dim=3, min_features=16, **kwargs):
        super().__init__()
        kwargs = deepcopy(kwargs)
        self.inp_dim = inp_dim
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.n_blocks = int(log2(min(self.img_size)))
        self.feat_w, self.feat_h = [int(s / 2 ** self.n_blocks) for s in self.img_size]
        self.use_rgb_skip = kwargs.pop('use_rgb_skip', True)
        self.use_norm = kwargs.pop('use_norm', False)
        upsample_feat, upsample_rgb = kwargs.pop('upsample_feat', 'nn'), kwargs.pop('upsample_rgb', 'bilinear')
        assert len(kwargs) == 0

        n_ch_fn = lambda i: max(n_features // (2 ** i), min_features)
        n_flat_features = n_features * np.prod([self.feat_w, self.feat_h])
        self.reshape_features = n_features != n_flat_features
        self.conv_in = conv1x1(inp_dim, n_flat_features) if n_flat_features != inp_dim else Identity()
        seq = [conv3x3(n_ch_fn(i + 1), n_ch_fn(i + 2)) for i in range(self.n_blocks - 1)]
        self.conv_layers = nn.ModuleList([conv3x3(n_features, n_features // 2)] + seq)
        self.upsample_feat = create_upsample_layer(upsample_feat)
        self.upsample_rgb = create_upsample_layer(upsample_rgb)

        if self.use_rgb_skip:
            seq = [conv3x3(n_ch_fn(i + 1), out_dim) for i in range(self.n_blocks)]
            self.conv_rgb = nn.ModuleList([conv3x3(n_features, out_dim)] + seq)
        else:
            self.conv_rgb = conv3x3(n_ch_fn(self.n_blocks), out_dim)
        if self.use_norm:
            self.norms = nn.ModuleList([nn.InstanceNorm2d(n_ch_fn(i + 1)) for i in range(self.n_blocks)])
        self.actvn = nn.ReLU(inplace=True)

        [kaiming_weights_init(m) for m in self.modules()]

    def forward(self, inp):
        inp = inp[..., None, None] if len(inp.shape) < 4 else inp
        net = self.conv_in(inp)
        if self.reshape_features:  # XXX in case of non square images, we reshape the features
            net = net.view(len(net), -1, self.feat_h, self.feat_w)

        if self.use_rgb_skip:
            rgb = self.upsample_rgb(self.conv_rgb[0](net))

        for idx, layer in enumerate(self.conv_layers):
            hid = layer(self.upsample_feat(net))
            if self.use_norm:
                hid = self.norms[idx](hid)
            net = self.actvn(hid)

            if self.use_rgb_skip:
                rgb = rgb + self.conv_rgb[idx + 1](net)
                if idx < self.n_blocks - 1:
                    rgb = self.upsample_rgb(rgb)

        if not self.use_rgb_skip:
            rgb = self.conv_rgb(net)
        return torch.sigmoid(rgb)


class ProgressiveGiraffeGenerator(nn.Module):
    def __init__(self, inp_dim, powers, milestones, **kwargs):
        super().__init__()
        self.powers = [powers] if isinstance(powers, int) else powers
        self.n_powers = len(self.powers)
        self.latent_size = self.powers[-1]
        assert all([self.latent_size % p == 0 for p in powers])
        self.repeat_latent = [self.latent_size // p for p in powers]
        n_features = kwargs.pop('n_features', self.latent_size)
        NU, NL = kwargs.pop('n_reg_units', N_UNITS), kwargs.pop('n_reg_layers', N_LAYERS)
        self.regressor = create_mlp(inp_dim, self.latent_size, NU, NL, zero_last_init=True)
        self.generator = GiraffeGenerator(n_features=n_features, inp_dim=self.latent_size, **kwargs)
        self.cur_milestone = 0
        self.set_milestones(milestones)

    def forward(self, x):
        B, C, device = x.size(0), self.latent_size, x.device
        latent_final = self.regressor(x)
        if self.act_idx < self.n_powers:
            p = self.current_code_size
            mask = torch.zeros(B, C, device=device)
            mask[:, :p] = torch.ones(B, p, device=device)
            latent_final = mask * latent_final

        self._latent = latent_final
        return self.generator(latent_final)

    def step(self):
        self.cur_milestone += 1
        while self.act_idx < self.n_powers and self.act_milestones[self.act_idx] <= self.cur_milestone:
            self.activations[self.act_idx] = True
            m, p = self.cur_milestone, self.powers[self.act_idx]
            print_log('Milestone {}, progressive giraffe: power {} activated'.format(m, p))
            self.act_idx += 1

    def set_cur_milestone(self, k):
        self.cur_milestone = k
        while self.act_idx < self.n_powers and self.act_milestones[self.act_idx] <= self.cur_milestone:
            self.activations[self.act_idx] = True
            self.act_idx += 1
        powers, act = self.powers, self.activations
        print_log('progressive giraffe activated powers={}'.format([k for k, a in zip(powers, act) if a]))

    def set_milestones(self, milestones):
        if milestones is not None:
            milestones = [milestones] if isinstance(milestones, int) else milestones
            assert len(milestones) == self.n_powers
            self.act_milestones = milestones
            n_act = (np.asarray(milestones) <= self.cur_milestone).sum()
            self.act_idx = n_act
            self.activations = [True] * n_act + [False] * (self.n_powers - n_act)
            powers, act = self.powers, self.activations
            print_log('progressive giraffe activated powers={}'.format([k for k, a in zip(powers, act) if a]))
        else:
            self.act_milestones = [-1] * self.n_powers
            self.act_idx = self.n_powers
            self.activations = [True] * self.n_powers

    @property
    def is_frozen(self):
        return sum(self.activations) == 0

    @property
    def current_code_size(self):
        return self.powers[self.act_idx - 1] if self.act_idx > 0 else 0
