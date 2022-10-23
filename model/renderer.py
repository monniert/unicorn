from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pytorch3d.ops.marching_cubes import marching_cubes_naive
from pytorch3d.renderer import (FoVPerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
                                PointLights, DirectionalLights, Materials, look_at_view_transform,
                                PerspectiveCameras, TexturesVertex)
from pytorch3d.renderer.mesh.shader import SoftPhongShader
from pytorch3d.renderer.mesh.shading import phong_shading, flat_shading, gouraud_shading
from pytorch3d.structures import Meshes

from .pytorch3d_monkey import AmbientLights
from utils.image import save_gif
from utils.pytorch import get_torch_device


LAYERED_SHADER = True
SHADING_TYPE = 'raw'
VIZ_IMG_SIZE = 256


class Renderer(nn.Module):
    def __init__(self, img_size, **kwargs):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self._init_kwargs = deepcopy(kwargs)
        self.init_cameras(**kwargs.get('cameras', {}))
        self.init_lights(**kwargs.get('lights', {}))
        blend_kwargs = {'sigma': kwargs.get('sigma', 1e-4),
                        'background_color': kwargs.get('background_color', (1, 1, 1))}
        n_faces = kwargs.get('faces_per_pixel', 25)
        blend_params = BlendParams(**blend_kwargs)
        s_kwargs = {'cameras': self.cameras, 'lights': self.lights, 'blend_params': blend_params,
                    'debug': kwargs.get('debug', False)}
        if kwargs.get('layered_shader', LAYERED_SHADER):
            shader_cls = LayeredShader
            s_kwargs['clip_inside'] = kwargs.get('clip_inside', True)
            s_kwargs['shading_type'] = kwargs.get('shading_type', SHADING_TYPE)
        else:
            shader_cls = SoftPhongShaderPlus

        # approximative differentiable renderer for training
        raster_settings = RasterizationSettings(image_size=img_size, blur_radius=np.log(1./1e-4-1.)*blend_params.sigma,
                                                faces_per_pixel=n_faces, perspective_correct=False)
        self.renderer = MeshRenderer(MeshRasterizer(cameras=self.cameras, raster_settings=raster_settings),
                                     shader_cls(**s_kwargs))

        # exact anti-aliased rendering for visualization
        viz_size = kwargs.get('viz_size', (VIZ_IMG_SIZE, VIZ_IMG_SIZE))
        s_kwargs['blend_params'] = BlendParams(background_color=blend_kwargs['background_color'], sigma=0)
        raster_settings = RasterizationSettings(image_size=(viz_size[0]*2, viz_size[1]*2), blur_radius=0.0,
                                                faces_per_pixel=1, perspective_correct=False)
        self.viz_renderer = VizMeshRenderer(MeshRasterizer(cameras=self.cameras, raster_settings=raster_settings),
                                            shader_cls(**s_kwargs))

    def init_cameras(self, **kwargs):
        kwargs = deepcopy(kwargs)
        name = kwargs.pop('name', 'fov')
        cam_cls = {'fov': FoVPerspectiveCameras, 'perspective': PerspectiveCameras}[name]
        self.cameras = cam_cls(**kwargs)

    def init_lights(self, **kwargs):
        kwargs = deepcopy(kwargs)
        name = kwargs.pop('name', 'ambient')
        light_cls = {'ambient': AmbientLights, 'directional': DirectionalLights, 'point': PointLights}[name]
        self.lights = light_cls(**kwargs)
        if name == 'directional':
            self.lights._direction = self.lights.direction
            self.lights._ambient_color = self.lights.ambient_color
            self.lights._diffuse_color = self.lights.diffuse_color
            self.lights._specular_color = self.lights.specular_color

    @property
    def init_kwargs(self):
        return deepcopy(self._init_kwargs)

    def forward(self, meshes, R, T, viz_purpose=False):
        return self.viz_renderer(meshes, R=R, T=T) if viz_purpose else self.renderer(meshes, R=R, T=T)  # BCHW

    def to(self, device):
        super().to(device)
        self.renderer = self.renderer.to(device)
        self.viz_renderer = self.viz_renderer.to(device)
        return self

    def update_lights(self, direction=None, ka=None, kd=None, ks=None):
        if direction is not None:
            self.lights.direction = direction
        if ka is not None:
            self.lights.ambient_color = ka
        if kd is not None:
            self.lights.diffuse_color = kd
        if ks is not None:
            self.lights.specular_color = ks

    def reset_default_lights(self):
        self.lights.direction = self.lights._direction
        self.lights.ambient_color = self.lights._ambient_color
        self.lights.diffuse_color = self.lights._diffuse_color
        self.lights.specular_color = self.lights._specular_color

    @torch.no_grad()
    def compute_vertex_visibility(self, meshes, R, T):
        fragments = self.viz_renderer.rasterizer(meshes, R=R, T=T)
        pix_to_face = fragments.pix_to_face
        packed_faces = meshes.faces_packed()
        packed_verts = meshes.verts_packed()
        visibility_map = torch.zeros(packed_verts.shape[0])
        visible_faces = pix_to_face.unique()[1:]  # we remove the -1 index
        visible_verts_idx = packed_faces[visible_faces]
        unique_visible_verts_idx = torch.unique(visible_verts_idx)
        visibility_map[unique_visible_verts_idx] = 1.0
        return visibility_map.view(len(meshes), -1).bool()


class VizMeshRenderer(MeshRenderer):
    """Renderer for visualization, with anti-aliasing"""
    @torch.no_grad()
    def __call__(self, *input, **kwargs):
        res = super().__call__(*input, **kwargs)
        return F.avg_pool2d(res, kernel_size=2, stride=2)


class LayeredShader(nn.Module):
    def __init__(self, device='cpu', cameras=None, lights=None, materials=None, blend_params=None, clip_inside=True,
                 shading_type='phong', debug=False):
        super().__init__()
        self.lights = lights if lights is not None else DirectionalLights(device=device)
        self.materials = (materials if materials is not None else Materials(device=device))
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()
        self.clip_inside = clip_inside
        if shading_type == 'phong':
            shading_fn = phong_shading
        elif shading_type == 'flat':
            shading_fn = flat_shading
        elif shading_type == 'gouraud':
            shading_fn = gouraud_shading
        elif shading_type == 'raw':
            shading_fn = lambda x: x
        else:
            raise NotImplementedError
        self.shading_fn = shading_fn
        self.shading_type = shading_type
        self.debug = debug

    def to(self, device):
        # Manually move to device modules which are not subclasses of nn.Module
        cameras = self.cameras
        if cameras is not None:
            self.cameras = cameras.to(device)
        self.materials = self.materials.to(device)
        self.lights = self.lights.to(device)
        return self

    def forward(self, fragments, meshes, **kwargs):
        blend_params = kwargs.get("blend_params", self.blend_params)
        if self.shading_type == 'raw':
            colors = meshes.sample_textures(fragments)
            if not torch.all(self.lights.ambient_color == 1):
                colors *= self.lights.ambient_color
        else:
            sh_kwargs = {'meshes': meshes, 'fragments': fragments, 'cameras': kwargs.get("cameras", self.cameras),
                         'lights': kwargs.get("lights", self.lights),
                         'materials': kwargs.get("materials", self.materials)}
            if self.shading_type != 'gouraud':
                sh_kwargs['texels'] = meshes.sample_textures(fragments)
            colors = self.shading_fn(**sh_kwargs)
        return layered_rgb_blend(colors, fragments, blend_params, clip_inside=self.clip_inside, debug=self.debug)


def layered_rgb_blend(colors, fragments, blend_params, clip_inside=True, debug=False):
    N, H, W, K = fragments.pix_to_face.shape
    device = fragments.pix_to_face.device
    pixel_colors = torch.ones((N, H, W, 4), dtype=colors.dtype, device=colors.device)
    background_ = blend_params.background_color
    if not isinstance(background_, torch.Tensor):
        background = torch.tensor(background_, dtype=torch.float32, device=device)
    else:
        background = background_.to(device)

    mask = fragments.pix_to_face >= 0  # mask for padded pixels.
    if blend_params.sigma == 0:
        alpha = (fragments.dists <= 0).float() * mask
    elif clip_inside:
        alpha = torch.exp(-fragments.dists.clamp(0) / blend_params.sigma) * mask
    else:
        alpha = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask
    occ_alpha = torch.cumprod(1.0 - alpha, dim=-1)
    occ_alpha = torch.cat([torch.ones(N, H, W, 1, device=device), occ_alpha], dim=-1)
    colors = torch.cat([colors, background[None, None, None, None].expand(N, H, W, 1, -1)], dim=-2)
    alpha = torch.cat([alpha, torch.ones(N, H, W, 1, device=device)], dim=-1)
    pixel_colors[..., :3] = (occ_alpha[..., None] * alpha[..., None] * colors).sum(-2)
    pixel_colors[..., 3] = 1 - occ_alpha[:, :, :, -1]

    if debug:
        return colors, alpha, occ_alpha, pixel_colors.permute(0, 3, 1, 2)
    else:
        return pixel_colors.permute(0, 3, 1, 2)  # BCHW


class SoftPhongShaderPlus(SoftPhongShader):
    """Rewriting to permute output tensor + working `to` method for multi-gpus"""
    def forward(self, fragments, meshes, **kwargs):
        return super().forward(fragments, meshes, **kwargs).permute(0, 3, 1, 2)

    def to(self, device):
        cameras = self.cameras
        if cameras is not None:
            self.cameras = cameras.to(device)
        self.materials = self.materials.to(device)
        self.lights = self.lights.to(device)
        return self


@torch.no_grad()
def render_rotated_views(mesh, img_size=256, n_views=50, elev=30, dist=2.5, R=None, T=None, bkg=None,
                         renderer=None, rend_kwargs=None, eye_light=False, device=None):
    device = get_torch_device() if device is None else device
    rend_kwargs = {} if rend_kwargs is None else rend_kwargs
    renderer = Renderer(img_size, **rend_kwargs) if renderer is None else renderer
    if eye_light:
        if R is not None:
            raise NotImplementedError
        if isinstance(renderer.lights, AmbientLights):
            kwargs = renderer.init_kwargs
            kwargs['lights'] = {'name': 'directional', 'direction': [[0, 0, -1]], 'ambient_color': [[0.7, 0.7, 0.7]],
                                'diffuse_color': [[0.3, 0.3, 0.3]], 'specular_color': [[0., 0., 0.]]}
            kwargs['shading_type'] = 'phong'
            kwargs['faces_per_pixel'] = 1
            renderer = Renderer(img_size, **kwargs)
    elev, dist = 0 if R is not None else elev, 0 if T is not None else dist
    R, T = R if R is not None else torch.eye(3).to(device), T if T is not None else torch.zeros(3).to(device)

    if bkg is not None:
        if bkg.shape[-1] != img_size:
            bkg = F.interpolate(bkg[None], size=(img_size, img_size), mode='bilinear', align_corners=False)[0]
    mesh, renderer = mesh.to(device), renderer.to(device)

    azim = torch.linspace(-180, 180, n_views)
    views, B = [], 10
    for k in range((n_views - 1) // B + 1):
        # we render by batch of B views to avoid OOM
        R_view = look_at_view_transform(dist=1, elev=elev, azim=azim[k*B: (k+1)*B], device=device)[0]
        T_view = torch.Tensor([[0., 0., dist]]).to(device).expand(len(R_view), -1)
        if eye_light:
            d = torch.Tensor([[0, 0, -1]]).to(device) @ R_view.transpose(1, 2)
            renderer.update_lights(direction=d)
        views.append(renderer(mesh.extend(len(R_view)), R_view @ R, T_view + T, viz_purpose=True).clamp(0, 1).cpu())

    rec, alpha = torch.cat(views, dim=0).split([3, 1], dim=1)
    if bkg is not None:
        rec = rec * alpha + (1 - alpha) * bkg.cpu()
    return rec


def save_mesh_as_gif(mesh, filename, img_size=256, n_views=50, elev=30, dist=2.732, R=None, T=None, bkg=None,
                     renderer=None, rend_kwargs=None, eye_light=False):
    imgs = render_rotated_views(mesh, img_size, n_views, elev, dist, R=R, T=T, bkg=bkg, renderer=renderer,
                                rend_kwargs=rend_kwargs, eye_light=eye_light)
    save_gif(imgs, filename)


def save_voxel_as_gif(voxel, filename, **kwargs):
    device = kwargs.get('device', get_torch_device())
    voxel = voxel if len(voxel.shape) == 4 else voxel[None]
    assert voxel.size(0) == 1
    verts, faces = [t[0] for t in marching_cubes_naive(voxel.float().to(device))]
    verts = (verts / verts.abs().max()) * 0.5
    mesh = Meshes(verts[None], faces[None], TexturesVertex(verts_features=torch.ones(verts.shape)[None] * 0.8))
    save_mesh_as_gif(mesh, filename, **kwargs)
