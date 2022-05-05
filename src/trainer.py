import argparse
import os
import shutil
import time
from toolz import merge, valmap, keyfilter
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dataset import create_train_val_test_loader
from model import create_model, DDPCust
from optimizer import create_optimizer
from scheduler import create_scheduler
from utils import use_seed, path_exists, path_mkdir, load_yaml
from utils.image import ImageLogger
from utils.logger import create_logger, print_log, print_warning, Verbose
from utils.metrics import Metrics, MeshEvaluator
from utils.path import CONFIGS_PATH, RUNS_PATH
from utils.plot import plot_lines, Visualizer
from utils.pytorch import get_torch_device, torch_to


LOG_FMT = "Epoch [{}/{}], Iter [{}/{}], {}".format
N_VIZ_SAMPLES = 4
torch.backends.cudnn.benchmark = True  # XXX accelerate training if fixed input size for each layer
warnings.filterwarnings("ignore")


class Trainer:
    """Pipeline to train a model on a particular dataset, both specified by a config cfg."""
    @use_seed()
    def __init__(self, cfg, run_dir, gpu=None, rank=None, world_size=None):
        self.is_master = gpu is None or rank == 0
        if not self.is_master:  # turning off logging and eval
            Metrics.log_data, ImageLogger.log_data, Verbose.mute = False, False, True

        self.run_dir = path_mkdir(run_dir)
        self.device = get_torch_device(gpu, verbose=True)
        self.train_loader, self.val_loader, self.test_loader = create_train_val_test_loader(cfg, rank, world_size)
        self.model = create_model(cfg, self.train_loader.dataset).to(self.device)
        self.optimizer = create_optimizer(cfg, self.model)
        self.scheduler = create_scheduler(cfg, self.optimizer)
        self.epoch_start, self.batch_start = 1, 1
        self.n_epoches, self.n_batches = cfg["training"].get("n_epoches"), len(self.train_loader)
        self.cur_lr = self.scheduler.get_last_lr()[0]
        self.multi_gpu = False
        if gpu is not None:
            self.model = DDPCust(self.model, device_ids=[gpu], output_device=gpu)
            self.multi_gpu = True
        self.load_from(cfg)
        print_log(f"Training state: epoch={self.epoch_start}, batch={self.batch_start}, lr={self.cur_lr}")

        append = self.epoch_start > 1
        self.train_stat_interval = cfg["training"]["train_stat_interval"]
        self.val_stat_interval = cfg["training"]["val_stat_interval"]
        self.save_epoches = cfg["training"].get("save_epoches", [])
        names = self.model.loss_names if hasattr(self.model, 'loss_names') else ['loss']
        names += [f'prop_head{k}' for k in range(len(self.model.prop_heads))]
        self.train_metrics = Metrics(*['time/img'] + names, log_file=self.run_dir / 'train_metrics.tsv', append=append)
        self.val_metrics = Metrics('loss_val', log_file=self.run_dir / 'val_metrics.tsv', append=append)
        self.val_scores = MeshEvaluator(['chamfer-L1', 'chamfer-L1-ICP'], self.run_dir / 'val_scores.tsv',
                                        fast_cpu=True, append=append)
        samples = next(iter(self.val_loader if len(self.val_loader) > 0 else self.train_loader))[0]
        self.viz_samples = valmap(lambda t: t.to(self.device)[:N_VIZ_SAMPLES], samples)
        self.rec_logger = ImageLogger(self.run_dir / 'reconstructions', target_images=self.viz_samples)
        if self.with_training:  # no visualizer if eval only
            viz_port = cfg["training"].get('visualizer_port') if self.is_master else None
            self.visualizer = Visualizer(viz_port, self.run_dir)

    @property
    def with_training(self):
        return self.epoch_start < self.n_epoches

    @property
    def dataset_name(self):
        return self.train_loader.dataset.name

    def load_from(self, cfg):
        pretrained, resume = cfg["training"].get("pretrained"), cfg["training"].get("resume")
        assert not (pretrained is not None and resume is not None)
        tag = pretrained or resume
        if tag is not None:
            path = path_exists(RUNS_PATH / self.dataset_name / tag / 'model.pkl')
            checkpoint = torch.load(path, map_location=self.device)
            if self.multi_gpu:
                self.model.module.load_state_dict(checkpoint["model_state"])
            else:
                self.model.load_state_dict(checkpoint["model_state"])
            if resume is not None:
                if checkpoint["batch"] == self.n_batches:
                    self.epoch_start, self.batch_start = checkpoint["epoch"] + 1, 1
                else:
                    self.epoch_start, self.batch_start = checkpoint["epoch"], checkpoint["batch"] + 1
                self.model.set_cur_epoch(checkpoint["epoch"])
                print_log(f"epoch_start={self.epoch_start}, batch_start={self.batch_start}")
                try:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                except ValueError:
                    print_warning("ValueError: loaded optim state contains parameters that don't match")
                scheduler_state = keyfilter(lambda k: k in ['last_epoch', '_step_count'], checkpoint["scheduler_state"])
                self.scheduler.load_state_dict(scheduler_state)
                self.cur_lr = self.scheduler.get_last_lr()[0]
                print_log(f"scheduler state_dict: {self.scheduler.state_dict()}")
            print_log(f"Checkpoint {tag} loaded")

    @use_seed()
    def run(self):
        cur_iter = (self.epoch_start - 1) * self.n_batches + self.batch_start
        for epoch in range(self.epoch_start, self.n_epoches + 1):
            batch_start = self.batch_start if epoch == self.epoch_start else 1
            for batch, (images, _labels) in enumerate(self.train_loader, start=1):
                if batch < batch_start:
                    continue
                self.run_single_batch_train(images)
                if cur_iter % self.train_stat_interval == 0 and self.is_master:
                    self.log_train_metrics(cur_iter, epoch, batch)

                if cur_iter % self.val_stat_interval == 0 and self.is_master:
                    if len(self.val_loader.dataset) > 10:
                        self.run_val()
                        self.log_val_metrics(cur_iter, epoch, batch)
                    self.log_visualizations(cur_iter)
                    self.save(epoch=epoch, batch=batch)
                cur_iter += 1
            self.step(epoch + 1, batch=1)
            if epoch in self.save_epoches:
                self.save(epoch=epoch, batch=batch, checkpoint=True)

        if self.is_master:
            N, B = (self.n_epoches, self.n_batches) if self.with_training else (self.epoch_start, self.batch_start)
            self.save(epoch=N, batch=B)
            self.save_metric_plots()
            self.evaluate()
        print_log("Training over")

    def run_single_batch_train(self, images):
        start_time = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        loss, pred = self.model(torch_to(images, self.device, non_blocking=self.multi_gpu))
        if isinstance(loss, torch.Tensor):
            loss.mean().backward()  # XXX we need to aggregate in case of DDP outputs
            dict_loss = {'loss': loss.detach().mean().item()}
        else:
            loss['total'].mean().backward()  # XXX we need to aggregate in case of DDP outputs
            dict_loss = {f'loss_{k}': v.detach().mean().item() for k, v in loss.items()}
        self.optimizer.step()
        self.model.iter_step()

        B = len(pred)
        self.train_metrics.update(merge({'time/img': (time.time() - start_time) / B}, dict_loss), N=B)
        self.train_metrics.update({f'prop_head{i}': p for i, p in enumerate(self.model.prop_heads)}, N=B)

    @torch.no_grad()
    def run_val(self):
        self.model.eval()
        model = self.model.module if self.multi_gpu else self.model
        for images, labels in self.val_loader:
            loss, pred = model(torch_to(images, self.device), return_meshes=True)
            loss = (loss if isinstance(loss, torch.Tensor) else loss['total']).item()
            self.val_metrics.update('loss_val', loss, N=len(pred))
            self.val_scores.update(pred, torch_to(labels, self.device))
            break  # XXX we only evaluate on the first batch for fast cpu

    def step(self, epoch, batch):
        self.model.step()
        self.scheduler.step()
        lr = self.scheduler.get_last_lr()[0]
        if lr != self.cur_lr:
            self.cur_lr = lr
            print_log(LOG_FMT(epoch, self.n_epoches, batch, self.n_batches, f'LR update: lr={lr}'))
        if hasattr(self.train_loader.dataset, 'step'):
            self.train_loader.dataset.step()

    def log_train_metrics(self, it, epoch, batch):
        print_log(LOG_FMT(epoch, self.n_epoches, batch, self.n_batches, f'train_metrics: {self.train_metrics}')[:1000])
        metrics = self.train_metrics
        self.visualizer.upload_lineplot(it, metrics.get_named_values(lambda s: 'loss' in s), title='train_losses')
        named_values = metrics.get_named_values(lambda s: 'prop_head' in s)
        proportions = torch.Tensor([v for k, v in named_values])
        if len(proportions) > 1:
            self.visualizer.upload_barplot(named_values, title='head assigned proportions')
        if hasattr(self.model, '_prob_heads'):
            named_values = [(k, v) for k, v in enumerate(self.model._prob_heads)]
            self.visualizer.upload_barplot(named_values, title='avg probability per head')
            named_values = [('max', self.model._prob_max), ('min', self.model._prob_min)]
            self.visualizer.upload_lineplot(it, named_values, title='probability statistics')
        metrics.log_and_reset(it=it, epoch=epoch, batch=batch)

    def log_val_metrics(self, it, epoch, batch):
        metrics = self.val_metrics
        print_log(LOG_FMT(epoch, self.n_epoches, batch, self.n_batches, f'val_metrics: {metrics}'))
        self.visualizer.upload_lineplot(it, metrics.get_named_values(), title='val_metrics')
        names, scores = self.val_scores.names, self.val_scores.compute()
        print_log(LOG_FMT(epoch, self.n_epoches, batch, self.n_batches,
                          "val_scores: " + ", ".join(["{}={:.4f}".format(k, v) for k, v in zip(names, scores)])))
        self.visualizer.upload_lineplot(it, list(zip(names, scores)), title='val_scores')
        self.val_metrics.log_and_reset(it=it, epoch=epoch, batch=batch)
        self.val_scores.log_and_reset(it=it, epoch=epoch, batch=batch)

    @torch.no_grad()
    def log_visualizations(self, cur_iter):
        self.model.eval()
        viz_imgs = self.viz_samples['imgs']
        rec = self.model(self.viz_samples, debug=True).permute(1, 0, 2, 3, 4)
        images = torch.cat([viz_imgs[:, None], rec], dim=1)
        self.visualizer.upload_images(images.reshape(-1, *viz_imgs.shape[1:]), 'candidates', images.shape[1])
        rec = self.model(self.viz_samples)[1]
        images = torch.stack([viz_imgs, rec], dim=1)
        self.visualizer.upload_images(images.reshape(-1, *viz_imgs.shape[1:]), 'recons', images.shape[1])
        self.rec_logger.save(rec, cur_iter)

        images = self.model.get_random_prototype_views(seed=4321)
        if images is not None:
            self.visualizer.upload_images(images, 'prototype views')

    def save(self, epoch, batch, checkpoint=False):
        state = {
            "epoch": epoch, "batch": batch, "model_name": self.model.name, "model_kwargs": self.model.init_kwargs,
            "model_state": self.model.state_dict(), "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }
        name = f'model_{epoch}.pkl' if checkpoint else 'model.pkl'
        torch.save(state, self.run_dir / name)
        print_log(f"Model saved at {self.run_dir / name}")

    @torch.no_grad()
    def save_metric_plots(self):
        self.model.eval()
        df_train, df_val, df_scores = [m.read_log() for m in [self.train_metrics, self.val_metrics, self.val_scores]]
        if len(df_train) == 0:
            print_log('No metrics or plots to save')
            return None

        df = df_train.join(df_val[['loss_val']], how="outer")
        loss_names = list(filter(lambda col: 'loss' in col, df.columns))
        plot_lines(df, loss_names, title="Loss").savefig(self.run_dir / "loss.pdf")
        if len(df_scores) > 0:
            names = list(filter(lambda col: col in df_scores, self.val_scores.names))
            plot_lines(df_scores, names, title="Val scores").savefig(self.run_dir / 'val_scores.pdf')

        names = list(filter(lambda col: col.startswith('prop_head'), df.columns))
        if len(names) > 0 :
            plot_lines(df, names, title='Head proportions').savefig(self.run_dir / 'head_proportions.pdf')
        self.rec_logger.save(self.model(self.viz_samples)[1])
        self.rec_logger.save_gif()
        print_log("Metrics and plots saved")

    def evaluate(self):
        self.model.eval()
        # quantitative
        scores = self.model.quantitative_eval(self.test_loader, self.device)
        print_log('final_scores: ' + ', '.join(["{}={:.5f}".format(k, v) for k, v in scores.items()]))
        with open(self.run_dir / 'final_scores.tsv', mode='w') as f:
            f.write("\t".join(scores.keys()) + "\n")
            f.write("\t".join(map('{:.5f}'.format, scores.values())) + "\n")

        # qualitative
        out = path_mkdir(self.run_dir / 'quali_eval')
        self.model.qualitative_eval(self.test_loader, self.device, path=out, N=32)
        print_log("Evaluation over")


def train_multi(gpu, cfg, run_dir, seed, n_gpus, n_nodes, n_rank):
    rank, world_size = n_rank * n_gpus + gpu, n_gpus * n_nodes
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    trainer = Trainer(cfg, run_dir, seed=seed + rank, gpu=gpu, rank=rank, world_size=world_size)
    trainer.run(seed=seed + rank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline to train a NN model specified by a YML config')
    parser.add_argument('-t', '--tag', nargs='?', type=str, required=True, help='Run tag of the experiment')
    parser.add_argument('-c', '--config', nargs='?', type=str, required=True, help='Config file name')
    parser.add_argument('-nr', '--n_rank', default=0, type=int, help='rank of the node')
    args = parser.parse_args()
    assert args.tag is not None and args.config is not None

    cfg = load_yaml(CONFIGS_PATH / args.config)
    seed, dataset = cfg['training'].get('seed', 4321), cfg['dataset']['name']
    if (RUNS_PATH / dataset / args.tag).exists():
        run_dir = RUNS_PATH / dataset / args.tag
    else:
        run_dir = path_mkdir(RUNS_PATH / dataset / args.tag)
    create_logger(run_dir)
    shutil.copy(str(CONFIGS_PATH / args.config), str(run_dir))

    n_gpus, n_nodes = cfg['training'].get('n_gpus', 1), cfg['training'].get('n_nodes', 1)
    n_gpus = min(torch.cuda.device_count(), n_gpus)
    print_log(f'Trainer init: config_file={args.config}, run_dir={run_dir}, n_gpus={n_gpus}')
    if n_gpus > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        mp.spawn(train_multi, nprocs=n_gpus, args=(cfg, run_dir, seed, n_gpus, n_nodes, args.n_rank))
    else:
        trainer = Trainer(cfg, run_dir, seed=seed)
        trainer.run(seed=seed)
