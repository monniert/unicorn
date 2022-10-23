from copy import deepcopy

import numpy as np
import torch
from torch import nn

from .tools import N_UNITS, N_LAYERS, create_mlp, kaiming_weights_init
from utils.logger import print_log


class Field(nn.Module):
    """Corresponds to a field modeled by a coordinate-based MLPs"""
    def __init__(self, n_units=N_UNITS, n_layers=N_LAYERS, latent_size=None, in_ch=3,
                 out_ch=3, zero_last_init=True, with_norm=False, dropout=False, bias_last=True):
        super().__init__()
        NU, NL = n_units, n_layers
        if latent_size is not None:
            self.linear_x = nn.Linear(in_ch, NU)
            self.linear_z = nn.Linear(latent_size, NU)
            if with_norm:
                self.act = nn.Sequential(nn.BatchNorm1d(NU), nn.ReLU(True))
            else:
                self.act = nn.ReLU(True)
            self.mlp = create_mlp(NU, out_ch, NU, NL - 1, zero_last_init=zero_last_init, with_norm=with_norm,
                                  dropout=dropout, bias_last=bias_last)
            [kaiming_weights_init(m) for m in [self.linear_x, self.linear_z]]
        else:
            self.mlp = create_mlp(in_ch, out_ch, NU, NL, zero_last_init=zero_last_init, with_norm=with_norm,
                                  dropout=dropout, bias_last=bias_last)

    def forward(self, x, latent=None):
        if latent is not None:
            N, B = len(x), len(latent)
            x = x[None] if len(x.shape) == 2 else x
            x = self.act((self.linear_x(x) + self.linear_z(latent[:, None])).view(B * N, -1))
            return self.mlp(x).view(B, N, -1)  # BN3
        else:
            return self.mlp(x)  # BN3


class ProgressiveField(nn.Module):
    def __init__(self, inp_dim, name, powers, milestones=None, **kwargs):
        super().__init__()
        kwargs = deepcopy(kwargs)
        self.powers = [powers] if isinstance(powers, int) else powers
        self.n_powers = len(self.powers)
        self.latent_size = self.powers[-1]
        assert all([self.latent_size % p == 0 for p in powers])
        self.repeat_latent = [self.latent_size // p for p in powers]
        NU, NL = kwargs.pop('n_reg_units', N_UNITS), kwargs.pop('n_reg_layers', N_LAYERS)
        bias_last = kwargs.pop('bias_last', True)
        self.regressor = create_mlp(inp_dim, self.latent_size, NU, NL, zero_last_init=True)
        NU, NL = kwargs.pop('n_field_units', N_UNITS), kwargs.pop('n_field_layers', N_LAYERS)
        self.field = Field(NU, NL, latent_size=self.latent_size, zero_last_init=True, bias_last=bias_last)
        self.cur_milestone = 0
        self.set_milestones(milestones)
        assert len(kwargs) == 0

    def forward(self, x, features):
        B, C, device = features.size(0), self.latent_size, x.device
        latent_final = self.regressor(features)
        if self.act_idx < self.n_powers:
            p = self.current_code_size
            mask = torch.zeros(B, C, device=device)
            mask[:, :p] = torch.ones(B, p, device=device)
            latent_final = mask * latent_final

        self._latent = latent_final
        return self.field(x, latent_final)

    def step(self):
        self.cur_milestone += 1
        while self.act_idx < self.n_powers and self.act_milestones[self.act_idx] <= self.cur_milestone:
            self.activations[self.act_idx] = True
            m, p = self.cur_milestone, self.powers[self.act_idx]
            print_log('Milestone {}, progressive field: {} activated'.format(m, p))
            self.act_idx += 1

    def set_cur_milestone(self, k):
        self.cur_milestone = k
        while self.act_idx < self.n_powers and self.act_milestones[self.act_idx] <= self.cur_milestone:
            self.activations[self.act_idx] = True
            self.act_idx += 1
        powers, activations = self.powers, self.activations
        print_log('progressive field activated powers={}'.format([k for k, a in zip(powers, activations) if a]))

    def set_milestones(self, milestones):
        if milestones is not None:
            milestones = [milestones] if isinstance(milestones, int) else milestones
            assert len(milestones) == self.n_powers
            self.act_milestones = milestones
            n_act = (np.asarray(milestones) <= self.cur_milestone).sum()
            self.act_idx = n_act
            self.activations = [True] * n_act + [False] * (self.n_powers - n_act)
            powers, activations = self.powers, self.activations
            print_log('progressive field activated powers={}'.format([k for k, a in zip(powers, activations) if a]))
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
