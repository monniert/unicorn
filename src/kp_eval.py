import argparse
from PIL import ImageDraw

import torch
import numpy as np
from tqdm import tqdm
from pytorch3d.ops import knn_points

from dataset import get_dataset
from dataset.cub_200 import KP_COLOR_MAPPING, EVAL_IMG_SIZE
from model import load_model_from_path
from model.renderer import VIZ_IMG_SIZE
from utils import path_exists, path_mkdir, use_seed
from utils.image import convert_to_img, resize
from utils.logger import print_log
from utils.path import RUNS_PATH
from utils.pytorch import get_torch_device
from utils.html import HtmlImagesPageGenerator


N_MAX_VIZ = 100
RADIUS = 7


class KeyPointEvaluator:
    thresholds = [0.05, 0.1, 0.15]
    names = ['pck-5', 'pck-10', 'pck-15']

    def __init__(self, tag, img_size=64, n_pairs=10000, visualize=True):
        self.run_path = path_exists(RUNS_PATH / 'cub_200' / tag)
        if visualize:
            self.out_path = path_mkdir(self.run_path / 'kp_quali')
        self.n_pairs = n_pairs
        self.visualize = visualize
        self.dataset = get_dataset('cub_200')(split='test', img_size=img_size, eval_mode=True)
        N, idx0, idx1 = len(self.dataset), [], []
        while len(idx0) < n_pairs:
            with use_seed(len(idx0) + 123):
                indices = np.random.permutation(np.arange(0, N))
            middle = N // 2
            idx0, idx1 = idx0 + indices[:middle].tolist(), idx1 + indices[-middle:].tolist()
        self.pair_indices = list(zip(*[idx0, idx1]))[:n_pairs]
        self.device = get_torch_device()
        self.size_factor = max(EVAL_IMG_SIZE)

        mkwargs = {'mesh': {'init': 'ellipse_hr'}}  # we use higher mesh resolution for better precision
        self.model = load_model_from_path(self.run_path / 'model.pkl', **mkwargs).to(self.device)
        self.model.eval()
        print_log('Evaluation init')

    def evaluate(self):
        errors, validity_mask = [], []
        for iteration, (idx0, idx1) in tqdm(enumerate(self.pair_indices)):
            imgs = torch.stack([self.dataset[idx0][0]['imgs'], self.dataset[idx1][0]['imgs']], dim=0)
            kps = torch.stack([self.dataset[idx0][1]['kps'], self.dataset[idx1][1]['kps']], dim=0)
            kp0_to_img1, kp1_to_img0, recs = self.predict(imgs.to(self.device), kps.to(self.device))
            kp0_to_img1, kp1_to_img0, recs = map(lambda t: t.cpu(), [kp0_to_img1, kp1_to_img0, recs])

            kps_vis = (kps[0, :, 2] * kps[1, :, 2])[None].expand(2, -1)
            kps_pred = torch.stack([kp1_to_img0, kp0_to_img1], dim=0)
            kps_err = (kps_pred - kps[:, :, :2]).norm(dim=2)

            errors.append(kps_err)
            validity_mask.append(kps_vis)
            if self.visualize and iteration < N_MAX_VIZ:
                self.visualize_keypoints(kp1_to_img0, kps[0], imgs[0], recs[0], kps[1], imgs[1], recs[1], iteration)

        thresh = torch.Tensor(self.thresholds)
        errors, validity_mask = torch.cat(errors), torch.cat(validity_mask)
        n_tot = validity_mask.sum()
        errors, validity_mask = [t[..., None].expand(-1, -1, len(thresh)) for t in [errors, validity_mask]]

        pcks = ((errors <= thresh * max(EVAL_IMG_SIZE)) * validity_mask).sum(dim=[0, 1]) / n_tot
        print_log(', '.join(['{}={:.4f}'.format(name, val) for name, val in zip(self.names, pcks)]))

        with open(self.run_path / 'kp_scores.txt', mode='w') as f:
            f.write('\t'.join(self.names) + '\n')
            f.write('\t'.join(['{:.4f}'.format(val) for val in pcks]) + '\n')
        print_log('Evaluation over')

    @torch.no_grad()
    def predict(self, imgs, kps):
        meshes, (R, T), bkgs = self.model.predict_mesh_pose_bkg(imgs)
        pred_fg, pred_mask = self.model.renderer(meshes, R, T, viz_purpose=True).split([3, 1], dim=1)
        recs = pred_fg * pred_mask
        kp0_to_img1 = self.transfer_keypoints(kps[0], R[0], T[0], meshes[0], R[1], T[1], meshes[1])
        kp1_to_img0 = self.transfer_keypoints(kps[1], R[1], T[1], meshes[1], R[0], T[0], meshes[0])
        return kp0_to_img1, kp1_to_img0, recs

    def transfer_keypoints(self, kp_src, R_src, T_src, mesh_src, R_tgt, T_tgt, mesh_tgt):
        camera, size = self.model.renderer.cameras, EVAL_IMG_SIZE
        R_src, T_src, R_tgt, T_tgt = [t[None] if t.size(0) != 1 else t for t in [R_src, T_src, R_tgt, T_tgt]]

        # Consider visible vertices only
        verts_viz_src = self.model.renderer.compute_vertex_visibility(mesh_src, R_src, T_src).squeeze()
        verts_3d_src = mesh_src.verts_packed()[verts_viz_src]
        # Project vertex 3D positions from source mesh to image space
        verts_2d_src = camera.transform_points_screen(verts_3d_src[None], image_size=size, R=R_src, T=T_src)[:, :, :2]
        # Find index correspondence between keypoints and vertices
        kp2proj_idx = knn_points(kp_src[:, :2][None], verts_2d_src).idx.long().squeeze()

        # Select vertices in target mesh associated to keypoints and project them to image space
        verts_3d_tgt = mesh_tgt.verts_packed()[verts_viz_src][kp2proj_idx]
        kp_out = camera.transform_points_screen(verts_3d_tgt[None], image_size=size, R=R_tgt, T=T_tgt)[0, :, :2]
        return kp_out

    def visualize_keypoints(self, kp_tgt_pred, kp_tgt_gt, img_tgt, rec_tgt, kp_src_gt, img_src, rec_src, iteration):
        kp_viz = kp_tgt_gt[:, 2] * kp_src_gt[:, 2]
        kp_tgt_gt, kp_src_gt = kp_tgt_gt[:, :2], kp_src_gt[:, :2]
        img_tgt1, img_tgt2, img_src = [resize(convert_to_img(i), VIZ_IMG_SIZE) for i in [img_tgt, img_tgt, img_src]]
        rec_tgt, rec_src = map(convert_to_img, [rec_tgt, rec_src])

        img_list = [img_tgt1, img_tgt2, rec_tgt, img_src, rec_src]
        drawers = [ImageDraw.Draw(i) for i in img_list]
        for cnt in range(len(kp_tgt_gt)):
            if kp_viz[cnt] == 0:
                continue

            gt_x, gt_y = map(round, kp_tgt_gt[cnt].tolist())
            drawers[0].ellipse((gt_x - RADIUS, gt_y - RADIUS, gt_x + RADIUS, gt_y + RADIUS), fill=KP_COLOR_MAPPING[cnt])
            gt_x, gt_y = map(round, kp_tgt_pred[cnt].tolist())
            drawers[1].ellipse((gt_x - RADIUS, gt_y - RADIUS, gt_x + RADIUS, gt_y + RADIUS), fill=KP_COLOR_MAPPING[cnt])
            drawers[2].ellipse((gt_x - RADIUS, gt_y - RADIUS, gt_x + RADIUS, gt_y + RADIUS), fill=KP_COLOR_MAPPING[cnt])
            gt_x, gt_y = map(round, kp_src_gt[cnt].tolist())
            drawers[3].ellipse((gt_x - RADIUS, gt_y - RADIUS, gt_x + RADIUS, gt_y + RADIUS), fill=KP_COLOR_MAPPING[cnt])
            drawers[4].ellipse((gt_x - RADIUS, gt_y - RADIUS, gt_x + RADIUS, gt_y + RADIUS), fill=KP_COLOR_MAPPING[cnt])

        prefix = str(iteration).zfill(int(np.ceil(np.log10(N_MAX_VIZ))))
        for img, suffix in zip(img_list, ['gt', 'pred', 'pred_rec', 'src', 'src_rec']):
            img.save(self.out_path / f'{prefix}_{suffix}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline to evaluate a NN model on keypoints')
    parser.add_argument('-t', '--tag', nargs='?', type=str, required=True, help='Run tag of the experiment')
    parser.add_argument('-n', '--n_pairs', nargs='?', default=10000, type=int, help='number of pair to eval')
    args = parser.parse_args()
    assert args.tag is not None

    tester = KeyPointEvaluator(args.tag, img_size=64, n_pairs=args.n_pairs or 10000)
    tester.evaluate()
    html_gen = HtmlImagesPageGenerator(tester.run_path / 'kp_quali', 'kp_report.html', 'KP report', nb_col=5)
    html_gen.run()
