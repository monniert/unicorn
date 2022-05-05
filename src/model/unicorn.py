from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from toolz import valfilter

import numpy as np
import torch
import torch.nn as nn
from pytorch3d.loss import mesh_laplacian_smoothing as laplacian_smoothing
from pytorch3d.renderer import TexturesVertex, look_at_view_transform, TexturesUV
from pytorch3d.structures import Meshes

from .encoder import Encoder
from .field import ProgressiveField
from .generator import ProgressiveGiraffeGenerator
from .loss import get_loss
from .renderer import Renderer, save_mesh_as_gif
from .tools import create_mlp, init_rotations, convert_3d_to_uv_coordinates, safe_model_state_dict
from .tools import azim_to_rotation_matrix, elev_to_rotation_matrix, roll_to_rotation_matrix, cpu_angle_between
from utils import path_mkdir, use_seed
from utils.image import convert_to_img
from utils.logger import print_warning
from utils.mesh import save_mesh_as_obj, repeat, get_icosphere, normal_consistency, normalize
from utils.metrics import MeshEvaluator
from utils.pytorch import torch_to


# POSE & SCALE DEFAULT
N_POSES = 6
N_ELEV_AZIM = [1, 6]
SCALE_ELLIPSE = [1, 0.7, 0.7]
PRIOR_TRANSLATION = [0., 0., 2.732]

# SWAP DEFAULT
MIN_ANGLE = 20
N_VPBINS = 4
MEMSIZE = 1024


class Unicorn(nn.Module):
    name = 'unicorn'

    def __init__(self, dataset, **kwargs):
        super().__init__()
        img_size = dataset.img_size
        self.init_kwargs = deepcopy(kwargs)
        self.encoder = Encoder(img_size, **kwargs.get('encoder', {}))
        self._init_meshes(**kwargs.get('mesh', {}))
        self.renderer = Renderer(img_size, **kwargs.get('renderer', {}))
        self._init_rend_predictors(**kwargs.get('rend_predictor', {}))
        self._init_background_model(img_size, **kwargs.get('background', {}))
        self._init_milestones(**kwargs.get('milestones', {}))
        self._init_loss(**kwargs.get('loss', {}))
        self.prop_heads = torch.zeros(self.n_poses)
        self.cur_epoch, self.cur_iter = 0, 0
        self._debug = False

    @property
    def n_features(self):
        return self.encoder.out_ch

    @property
    def tx_code_size(self):
        return self.txt_generator.current_code_size

    @property
    def sh_code_size(self):
        return self.deform_field.current_code_size

    def _init_meshes(self, **kwargs):
        kwargs = deepcopy(kwargs)
        mesh_init = kwargs.pop('init', 'sphere')
        scale = kwargs.pop('scale', 1)
        if 'sphere' in mesh_init or 'ellipse' in mesh_init:
            mesh = get_icosphere(4 if 'hr' in mesh_init else 3)
            if 'ellipse' in mesh_init:
                scale = scale * torch.Tensor([SCALE_ELLIPSE])
        else:
            raise NotImplementedError
        self.mesh_src = mesh.scale_verts(scale)
        self.register_buffer('uvs', convert_3d_to_uv_coordinates(self.mesh_src.get_mesh_verts_faces(0)[0])[None])

        self.use_mean_txt = kwargs.pop('use_mean_txt', kwargs.pop('use_mean_text', False))  # retro-compatibility
        dfield_kwargs = kwargs.pop('deform_fields', {})
        tgen_kwargs = kwargs.pop('texture_uv', {})
        assert len(kwargs) == 0

        self.deform_field = ProgressiveField(inp_dim=self.n_features, name='deformation', **dfield_kwargs)
        self.txt_generator = ProgressiveGiraffeGenerator(inp_dim=self.n_features, **tgen_kwargs)

    def _init_rend_predictors(self, **kwargs):
        kwargs = deepcopy(kwargs)
        self.n_poses = kwargs.pop('n_poses', N_POSES)
        n_elev, n_azim = kwargs.pop('n_elev_azim', N_ELEV_AZIM)
        assert self.n_poses == n_elev * n_azim
        self.alternate_optim = kwargs.pop('alternate_optim', True)
        self.pose_step = True

        NF, NP = self.n_features, self.n_poses

        # Translation
        self.T_regressors = nn.ModuleList([create_mlp(NF, 3, zero_last_init=True) for _ in range(NP)])
        T_range = kwargs.pop('T_range', 1)
        T_range = [T_range] * 3 if isinstance(T_range, (int, float)) else T_range
        self.register_buffer('T_range', torch.Tensor(T_range))
        self.register_buffer('T_init', torch.zeros(3))
        self.register_buffer('T_cam', torch.Tensor(kwargs.pop('prior_translation', PRIOR_TRANSLATION)))

        # Rotation
        self.rot_regressors = nn.ModuleList([create_mlp(NF, 3, zero_last_init=True) for _ in range(NP)])
        a_range, e_range, r_range = kwargs.pop('azim_range'), kwargs.pop('elev_range'), kwargs.pop('roll_range')
        azim, elev, roll = [(e[1] - e[0]) / n for e, n in zip([a_range, e_range, r_range], [n_azim, n_elev, 1])]
        R_init = init_rotations('uniform', n_elev=n_elev, n_azim=n_azim, elev_range=e_range, azim_range=a_range)
        self.register_buffer('R_range', torch.Tensor([azim * 0.52, elev * 0.52, roll * 0.52]))
        self.register_buffer('R_init', R_init)
        self.azim_range, self.elev_range, self.roll_range = a_range, e_range, r_range
        self.register_buffer('R_cam', torch.eye(3))

        # Scale
        self.scale_regressor = create_mlp(NF, 3, zero_last_init=True)
        scale_range = kwargs.pop('scale_range', 0.5)
        scale_range = [scale_range] * 3 if isinstance(scale_range, (int, float)) else scale_range
        self.register_buffer('scale_range', torch.Tensor(scale_range))
        self.register_buffer('scale_init', torch.ones(3))

        # Pose probabilities
        self.proba_regressor = create_mlp(NF, NP)

        assert len(kwargs) == 0, kwargs

    @property
    def n_candidates(self):
        return 1 if self.hard_select else self.n_poses

    @property
    def hard_select(self):
        if self.alternate_optim and not self._debug:
            return False if (self.training and self.pose_step) else True
        else:
            return False

    def _init_background_model(self, img_size, **kwargs):
        if len(kwargs) > 0:
            bkg_kwargs = deepcopy(kwargs)
            self.bkg_generator = ProgressiveGiraffeGenerator(inp_dim=self.n_features, img_size=img_size, **bkg_kwargs)

    def _init_milestones(self, **kwargs):
        kwargs = deepcopy(kwargs)
        self.milestones = {
            'constant_txt': kwargs.pop('constant_txt', kwargs.pop('constant_text', 0)),  # retro-compatibility
            'freeze_T_pred': kwargs.pop('freeze_T_predictor', 0),
            'freeze_R_pred': kwargs.pop('freeze_R_predictor', 0),
            'freeze_s_pred': kwargs.pop('freeze_scale_predictor', 0),
            'freeze_shape': kwargs.pop('freeze_shape', 0),
            'mean_txt': kwargs.pop('mean_txt', kwargs.pop('mean_text', self.use_mean_txt)),  # retro-compatibility
        }
        assert len(kwargs) == 0

    def _init_loss(self, **kwargs):
        kwargs = deepcopy(kwargs)
        loss_weights = {
            'rgb': kwargs.pop('rgb_weight', 1.0),
            'normal': kwargs.pop('normal_weight', 0),
            'laplacian': kwargs.pop('laplacian_weight', 0),
            'perceptual': kwargs.pop('perceptual_weight', 0),
            'uniform': kwargs.pop('uniform_weight', 0),
            'swap': kwargs.pop('swap_weight', 0),
        }
        name = kwargs.pop('name', 'mse')
        perceptual_kwargs = kwargs.pop('perceptual', {})
        self.swap_memsize = kwargs.pop('swap_memsize', MEMSIZE)
        self.swap_n_vpbins = kwargs.pop('swap_n_vpbins', N_VPBINS)
        self.swap_min_angle = kwargs.pop('swap_min_angle', MIN_ANGLE)
        self.swap_memory = {k: torch.empty(0) for k in ['sh', 'tx', 'S', 'R', 'T', 'bg', 'img']}
        assert len(kwargs) == 0, kwargs

        self.loss_weights = valfilter(lambda v: v > 0, loss_weights)
        self.loss_names = [f'loss_{n}' for n in list(self.loss_weights.keys()) + ['total']]
        self.criterion = get_loss(name)(reduction='none')
        if 'perceptual' in self.loss_weights:
            self.perceptual_loss = get_loss('perceptual')(**perceptual_kwargs)

    @property
    def pred_background(self):
        return hasattr(self, 'bkg_generator')

    def is_live(self, name):
        milestone = self.milestones[name]
        if isinstance(milestone, bool):
            return milestone
        else:
            return True if self.cur_epoch < milestone else False

    def to(self, device):
        super().to(device)
        self.mesh_src = self.mesh_src.to(device)
        self.renderer = self.renderer.to(device)
        return self

    def forward(self, inp, debug=False, return_meshes=False):
        # XXX pytorch3d objects are not well handled by DDP so we need to manually move them to GPU
        # self.mesh_src, self.renderer = [t.to(inp['imgs'].device) for t in [self.mesh_src, self.renderer]]
        self._debug = debug

        imgs, K, B = inp['imgs'], self.n_candidates, len(inp['imgs'])
        features = self.encoder(imgs)
        meshes = self.predict_meshes(features)
        R, T = self.predict_poses(features)
        bkgs = self.predict_background(features) if self.pred_background else None

        if self.alternate_optim:
            if self.pose_step:
                meshes, bkgs = meshes.detach(), bkgs.detach() if self.pred_background else None
            else:
                R, T = R.detach(), T.detach()

        posed_meshes, R_cam, T_cam = self.update_with_poses(meshes, R, T)
        fgs, alpha = self.renderer(posed_meshes, R_cam, T_cam).split([3, 1], dim=1)  # (K*B)CHW
        rec = fgs * alpha + (1 - alpha) * bkgs if self.pred_background else fgs
        losses, select_idx = self.compute_losses(meshes, imgs, rec, R, T)

        if debug:
            out = rec.view(K, B, *rec.shape[1:]) if K > 1 else rec[None]
        elif return_meshes:
            # we need to realign mesh to account for pose parametrization mismatch
            verts, faces = posed_meshes.verts_padded(), posed_meshes.faces_padded()
            R_gt, T_gt = map(lambda t: t.squeeze(2), inp['poses'].split([3, 1], dim=2))
            verts = (verts @ R_cam + T_cam[:, None] - T_gt[:, None]) @ R_gt.transpose(1, 2)
            meshes = Meshes(verts=verts, faces=faces, textures=meshes.textures)
            out = losses, meshes
        else:
            rec = rec.view(K, B, *rec.shape[1:])[select_idx, torch.arange(B)] if K > 1 else rec
            out = losses, rec

        self._debug = False
        return out

    def predict_meshes(self, features):
        verts, faces = self.mesh_src.get_mesh_verts_faces(0)
        meshes = self.mesh_src.extend(len(features))  # XXX does a copy
        meshes.offset_verts_(self.predict_disp_verts(verts, features))
        meshes.textures = self.predict_textures(faces, features)
        meshes.scale_verts_(self.predict_scales(features))
        return meshes

    def predict_disp_verts(self, verts, features):
        disp_verts = self.deform_field(verts, features)
        if self.is_live('freeze_shape'):
            disp_verts = disp_verts * 0
        return disp_verts.view(-1, 3)

    def predict_textures(self, faces, features):
        B = len(features)
        perturbed = self.training and np.random.binomial(1, p=0.2)
        maps = self.txt_generator(features)
        if self.is_live('constant_txt'):
            H, W = maps.shape[-2:]
            nb = int(H * W * 0.1)
            idxh, idxw = torch.randperm(H)[:nb], torch.randperm(W)[:nb]
            maps = maps[:, :, idxh, idxw].mean(2)[..., None, None].expand(-1, -1, H, W)
        elif self.training and perturbed and self.use_mean_txt and self.is_live('mean_txt'):
            H, W = maps.shape[-2:]
            nb = int(H * W * 0.1)
            idxh, idxw = torch.randperm(H)[:nb], torch.randperm(W)[:nb]
            maps = maps[:, :, idxh, idxw].mean(2)[..., None, None].expand(-1, -1, H, W)
        return TexturesUV(maps.permute(0, 2, 3, 1), faces[None].expand(B, -1, -1), self.uvs.expand(B, -1, -1))

    def predict_scales(self, features):
        s_pred = self.scale_regressor(features).tanh()
        if self.is_live('freeze_s_pred'):
            s_pred = s_pred * 0
        self._scales = s_pred * self.scale_range + self.scale_init
        return self._scales

    def predict_poses(self, features):
        B = len(features)

        T_pred = torch.stack([p(features) for p in self.T_regressors], dim=0).tanh()
        if self.is_live('freeze_T_pred'):
            T_pred = T_pred * 0
        T = (T_pred * self.T_range + self.T_init).view(-1, 3)

        R_pred = torch.stack([p(features) for p in self.rot_regressors], dim=0).tanh()  # KBC
        R_pred = R_pred[..., [1, 0, 2]]  # XXX for retro-compatibility
        if self.is_live('freeze_R_pred'):
            R_pred = R_pred * 0
        R_pred = (R_pred * self.R_range + self.R_init[:, None]).view(-1, 3)
        azim, elev, roll = map(lambda t: t.squeeze(1), R_pred.split([1, 1, 1], 1))
        R = azim_to_rotation_matrix(azim) @ elev_to_rotation_matrix(elev) @ roll_to_rotation_matrix(roll)

        self._pose_proba = torch.softmax(self.proba_regressor(features.view(B, -1)).permute(1, 0), dim=0)  # KB
        if self.hard_select:
            indices = self._pose_proba.max(0)[1]
            select_fn = lambda t: t.view(self.n_poses, B, *t.shape[1:])[indices, torch.arange(B)]
            R, T = map(select_fn, [R, T])
        return R, T

    def predict_background(self, features):
        res = self.bkg_generator(features)
        return res.repeat(self.n_candidates, 1, 1, 1) if self.n_candidates > 1 else res

    def update_with_poses(self, meshes, R, T):
        K, B = len(T) // len(meshes), len(meshes)
        meshes = repeat(meshes, K)  # XXX returns a copy of meshes
        Nv = meshes.num_verts_per_mesh()[0]
        meshes = Meshes(meshes.verts_padded() @ R, faces=meshes.faces_padded(), textures=meshes.textures)
        meshes.offset_verts_(T[:, None].expand(-1, Nv, -1).reshape(-1, 3))
        R, T = self.R_cam[None].expand(K * B, -1, -1), self.T_cam[None].expand(K * B, -1)
        return meshes, R, T

    def compute_losses(self, meshes, imgs, rec, R, T):
        K, B = self.n_candidates, len(imgs)
        if K > 1:
            imgs = imgs.repeat(K, 1, 1, 1)
        losses = {k: torch.tensor(0.0, device=imgs.device) for k in self.loss_weights}
        update_3d, update_pose = (not self.pose_step, self.pose_step) if self.alternate_optim else (True, True)

        # Standard reconstrution error on RGB values
        if 'rgb' in losses:
            losses['rgb'] = self.loss_weights['rgb'] * self.criterion(rec, imgs).flatten(1).mean(1)

        # Perceptual loss
        if 'perceptual' in losses:
            losses['perceptual'] = self.loss_weights['perceptual'] * self.perceptual_loss(rec, imgs)

        # Mesh regularization
        if update_3d:
            if 'normal' in losses:
                losses['normal'] = self.loss_weights['normal'] * normal_consistency(meshes)
            if 'laplacian' in losses:
                losses['laplacian'] = self.loss_weights['laplacian'] * laplacian_smoothing(meshes, method='uniform')

        # Swap loss
        # XXX when latent spaces are small, codes are similar so there is no need to compute the swap loss
        if update_3d and 'swap' in losses and (self.tx_code_size > 0 and self.sh_code_size > 0):
            B, dev = len(meshes), imgs.device
            verts, faces, textures = meshes.verts_padded(), meshes.faces_padded(), meshes.textures
            scales = self._scales[:, None]
            z_sh, z_tx = [m._latent for m in [self.deform_field, self.txt_generator]]
            z_bg = self.bkg_generator._latent if self.pred_background else torch.empty(B, 1, device=dev)
            for n, t in [('sh', z_sh), ('tx', z_tx), ('bg', z_bg), ('S', scales), ('R', R), ('T', T), ('img', imgs)]:
                self.swap_memory[n] = torch.cat([self.swap_memory[n].to(dev), t.detach()])[-self.swap_memsize:]

            # we compute the nearest neighbors within a random target viewpoint range
            min_angle, nb_vpbins = self.swap_min_angle, self.swap_n_vpbins
            with torch.no_grad():
                sim_sh = (z_sh[None] - self.swap_memory['sh'][:, None]).pow(2).sum(-1)
                sim_tx = (z_tx[None] - self.swap_memory['tx'][:, None]).pow(2).sum(-1)
                angles = cpu_angle_between(self.swap_memory['R'][:, None], R[None]).view(sim_sh.shape)
                angle_bins = torch.randint(0, nb_vpbins, (B,), device=dev).float()
                bin_size = (180. - min_angle) / nb_vpbins
                min_angles, max_angles = [(angle_bins + k) * bin_size + min_angle for k in range(2)]
                mask = (angles < min_angles).float() + (angles > max_angles).float()
                idx_sh, idx_tx = map(lambda t: (t + t.max() * mask).argmin(0), [sim_sh, sim_tx])

            v_src, f_src = self.mesh_src.get_mesh_verts_faces(0)
            swap_list, select = [], lambda n, indices: self.swap_memory[n][indices]
            sh_imgs, tx_imgs = select('img', idx_sh), select('img', idx_tx)

            # Swapped shapes
            with torch.no_grad():
                # we recompute parameters with the current network state
                sh_features = self.encoder(sh_imgs)
                sh_tx = self.predict_textures(f_src, sh_features)
                sh_S = self.predict_scales(sh_features)[:, None]
                sh_R, sh_T = self.predict_poses(sh_features)
                sh_bg = self.predict_background(sh_features) if self.pred_background else None
            sh_mesh = Meshes((verts / scales) * sh_S, faces, sh_tx)
            swap_list.append([sh_mesh, sh_R, sh_T, sh_bg, sh_imgs])

            # Swapped textures
            with torch.no_grad():
                # we recompute parameters with the current network state
                tx_features = self.encoder(tx_imgs)
                tx_verts = v_src + self.predict_disp_verts(v_src, tx_features).view(B, -1, 3)
                tx_S = self.predict_scales(tx_features)[:, None]
                tx_R, tx_T = self.predict_poses(tx_features)
                tx_bg = self.predict_background(tx_features) if self.pred_background else None
            tx_mesh = Meshes(tx_verts * tx_S, faces, textures)
            swap_list.append([tx_mesh, tx_R, tx_T, tx_bg, tx_imgs])

            loss = 0.
            for swap_inp in swap_list:
                swap_mesh, R, T, bkgs, imgs = swap_inp
                posed_meshes, R_cam, T_cam = self.update_with_poses(swap_mesh, R, T)
                rec_sw, alpha_sw = self.renderer(posed_meshes, R_cam, T_cam).split([3, 1], dim=1)
                rec_sw = rec_sw * alpha_sw + (1 - alpha_sw) * bkgs[:B] if bkgs is not None else rec_sw
                if 'rgb' in losses:
                    loss += self.loss_weights['rgb'] * self.criterion(rec_sw, imgs).flatten(1).mean(1)
                if 'perceptual' in losses:
                    loss += self.loss_weights['perceptual'] * self.perceptual_loss(rec_sw, imgs)
            losses['swap'] = self.loss_weights['swap'] * loss

        # Pose priors
        if update_pose and 'uniform' in losses:
            losses['uniform'] = self.loss_weights['uniform'] * (self._pose_proba.mean(1) - 1 / K).abs().mean()

        dist = sum(losses.values())
        if K > 1:
            dist, select_idx = dist.view(K, B), self._pose_proba.max(0)[1]
            dist = (self._pose_proba * dist).sum(0)
            for k, v in losses.items():
                if v.numel() != 1:
                    losses[k] = (self._pose_proba * v.view(K, B)).sum(0).mean()

            # For monitoring purpose only
            pose_proba_d = self._pose_proba.detach().cpu()
            self._prob_heads = pose_proba_d.mean(1).tolist()
            self._prob_max = pose_proba_d.max(0)[0].mean().item()
            self._prob_min = pose_proba_d.min(0)[0].mean().item()
            count = torch.zeros(K, B).scatter(0, select_idx[None].cpu(), 1).sum(1)
            self.prop_heads = count / B

        else:
            select_idx = torch.zeros(B).long()
            for k, v in losses.items():
                if v.numel() != 1:
                    losses[k] = v.mean()

        losses['total'] = dist.mean()
        return losses, select_idx

    def iter_step(self):
        self.cur_iter += 1
        if self.alternate_optim and self.cur_iter % self.alternate_optim == 0:
            self.pose_step = not self.pose_step

    def step(self):
        self.cur_epoch += 1
        self.deform_field.step()
        self.txt_generator.step()
        if self.pred_background:
            self.bkg_generator.step()

    def set_cur_epoch(self, epoch):
        self.cur_epoch = epoch
        self.deform_field.set_cur_milestone(epoch)
        self.txt_generator.set_cur_milestone(epoch)
        if self.pred_background:
            self.bkg_generator.set_cur_milestone(epoch)

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        unloaded_params = []
        state = self.state_dict()
        for name, param in safe_model_state_dict(state_dict).items():
            if name in state:
                try:
                    state[name].copy_(param.data if isinstance(param, nn.Parameter) else param)
                except RuntimeError:
                    print_warning(f'Error load_state_dict param={name}: {list(param.shape)}, {list(state[name].shape)}')
            else:
                unloaded_params.append(name)
        if len(unloaded_params) > 0:
            print_warning(f'load_state_dict: {unloaded_params} not found')

    ########################
    # Visualization utils
    ########################

    def get_synthetic_textures(self, colored=False):
        verts = self.mesh_src.verts_packed()
        if colored:
            colors = (verts - verts.min(0)[0]) / (verts.max(0)[0] - verts.min(0)[0])
        else:
            colors = torch.ones(verts.shape, device=verts.device) * 0.8
        return TexturesVertex(verts_features=colors[None])

    def get_prototype(self):
        verts = self.mesh_src.get_mesh_verts_faces(0)[0]
        latent = torch.zeros(1, self.n_features, device=verts.device)
        meshes = self.mesh_src.offset_verts(self.deform_field(verts, latent).view(-1, 3))
        return meshes

    @use_seed()
    @torch.no_grad()
    def get_random_prototype_views(self, N=10):
        mesh = self.get_prototype()
        if mesh is None:
            return None

        mesh.textures = self.get_synthetic_textures(colored=True)
        azim = torch.randint(*self.azim_range, size=(N,))
        elev = torch.randint(*self.elev_range, size=(N,)) if np.diff(self.elev_range)[0] > 0 else self.elev_range[0]
        R, T = look_at_view_transform(dist=self.T_cam[-1], elev=elev, azim=azim, device=mesh.device)
        return self.renderer(mesh.extend(N), R, T).split([3, 1], dim=1)[0]

    @torch.no_grad()
    def save_prototype(self, path=None):
        mesh = self.get_prototype()
        if mesh is None:
            return None

        path = path_mkdir(path or Path('.'))
        d, elev = self.T_cam[-1], np.mean(self.elev_range)
        mesh.textures = self.get_synthetic_textures()
        save_mesh_as_obj(mesh, path / 'proto.obj')
        save_mesh_as_gif(mesh, path / 'proto_li.gif', dist=d, elev=elev, renderer=self.renderer, eye_light=True)
        mesh.textures = self.get_synthetic_textures(colored=True)
        save_mesh_as_gif(mesh, path / 'proto_uv.gif', dist=d, elev=elev, renderer=self.renderer)

    ########################
    # Evaluation utils
    ########################

    @torch.no_grad()
    def quantitative_eval(self, loader, device):
        self.eval()
        mesh_eval = MeshEvaluator()
        for inp, labels in loader:
            if isinstance(labels, torch.Tensor) and torch.all(labels == -1):
                break

            mesh_pred = self(torch_to(inp, device), return_meshes=True)[1]
            mesh_eval.update(mesh_pred, torch_to(labels, device))
        return OrderedDict(zip(mesh_eval.metrics.names, mesh_eval.metrics.values))

    @torch.no_grad()
    def qualitative_eval(self, loader, device, path=None, N=32):
        path = path or Path('.')
        self.eval()
        self.save_prototype(path / 'model')

        renderer = self.renderer
        n_zeros, NI = int(np.log10(N - 1)) + 1, max(N // loader.batch_size, 1)
        for j, (inp, _) in enumerate(loader):
            if j == NI:
                break
            imgs = inp['imgs'].to(device)
            features = self.encoder(imgs)
            meshes = self.predict_meshes(features)
            R, T = self.predict_poses(features)

            posed_meshes, R_new, T_new = self.update_with_poses(meshes, R, T)
            rec, alpha = renderer(posed_meshes, R_new, T_new).split([3, 1], dim=1)  # (K*B)CHW
            if self.pred_background:
                bkgs = self.predict_background(features)
                rec = rec * alpha + (1 - alpha) * bkgs

            B, NV = len(imgs), 50
            d, e = self.T_cam[-1], np.mean(self.elev_range)
            for k in range(B):
                i = str(j*B+k).zfill(n_zeros)
                convert_to_img(imgs[k]).save(path / f'{i}_inpraw.png')
                convert_to_img(rec[k]).save(path / f'{i}_inprec_full.png')
                if self.pred_background:
                    convert_to_img(bkgs[k]).save(path / f'{i}_inprec_wbkg.png')

                mcenter = normalize(meshes[k], mode=None, center=True, use_center_mass=True)
                save_mesh_as_gif(mcenter, path / f'{i}_meshabs.gif', n_views=NV, dist=d, elev=e, renderer=renderer)
                save_mesh_as_obj(mcenter, path / f'{i}_mesh.obj')
                mcenter.textures = self.get_synthetic_textures(colored=True)
                save_mesh_as_obj(mcenter, path / f'{i}_meshuv.obj')
                save_mesh_as_gif(mcenter, path / f'{i}_meshuv_raw.gif', dist=d, elev=e, renderer=renderer)
