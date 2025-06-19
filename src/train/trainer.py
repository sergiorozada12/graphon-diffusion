import os
from datetime import datetime
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

from src.sde.equations import VPSDE, VESDE, subVPSDE
from src.models.score_network import ScoreNetwork
from src.train.ema import ExponentialMovingAverage
from src.utils import mask_adjs, node_flags, gen_noise


class DiffusionGraphModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        self.model = ScoreNetwork(**cfg.model)
        self.sde = self._build_sde(cfg.sde)
        self._ema = None

        self.dataset_name = cfg.data.data
        self.loss_eps = cfg.train.eps
        self.reduce_mean = cfg.train.reduce_mean
        self.likelihood_weighting = False

        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.ckpt_dir = os.path.join("checkpoints", self.dataset_name, self.timestamp)
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def _build_sde(self, cfg_sde):
        if cfg_sde.type == 'VP':
            return VPSDE(beta_min=cfg_sde.beta_min, beta_max=cfg_sde.beta_max, N=cfg_sde.num_scales)
        elif cfg_sde.type == 'VE':
            return VESDE(sigma_min=cfg_sde.beta_min, sigma_max=cfg_sde.beta_max, N=cfg_sde.num_scales)
        elif cfg_sde.type == 'subVP':
            return subVPSDE(beta_min=cfg_sde.beta_min, beta_max=cfg_sde.beta_max, N=cfg_sde.num_scales)
        else:
            raise NotImplementedError(f"SDE type {cfg_sde.type} not supported.")

    def on_fit_start(self):
        self._ema = ExponentialMovingAverage(self.model.parameters(), decay=self.cfg.train.ema)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay)
        if self.cfg.train.lr_schedule:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=self.cfg.train.lr_decay)
            return [opt], [scheduler]
        return opt

    def forward(self, x, adj, flags, t):
        score_unnorm = self.model(x, adj, flags)
        std = self.sde.marginal_prob(torch.zeros_like(adj), t)[1]
        score = -score_unnorm / std[:, None, None]
        return score

    def training_step(self, batch, batch_idx):
        # Data
        x, adj = batch
        flags = node_flags(adj)
        device = adj.device

        # Noise
        t = torch.rand(adj.shape[0], device=device) * (self.sde.T - self.loss_eps) + self.loss_eps
        # t = torch.rand(adj.shape[0], device=device).pow(1 / 0.5) * (self.sde.T - self.loss_eps) + self.loss_eps
        perturbed, noise, std = self._perturb_data(adj, flags, t)
        score = self(x, perturbed, flags, t)

        # Loss
        loss = self._loss(adj, t, noise, std, score)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_after_backward(self):
        grad_norm = self.cfg.train.grad_norm
        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_norm)

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False
    ):
        optimizer.step(closure=optimizer_closure)
        self._ema.update(self.model.parameters())

    def validation_step(self, batch, batch_idx):
        # Data
        x, adj = batch
        flags = node_flags(adj)
        device = adj.device
        self._ema.store(self.model.parameters())
        self._ema.copy_to(self.model.parameters())

        # Noise
        t = torch.rand(adj.shape[0], device=device) * (self.sde.T - self.loss_eps) + self.loss_eps
        # t = torch.rand(adj.shape[0], device=device).pow(1 / 0.5) * (self.sde.T - self.loss_eps) + self.loss_eps
        perturbed, noise, std = self._perturb_data(adj, flags, t)
        score = self(x, perturbed, flags, t)

        # Loss
        with torch.no_grad():
            loss = self._loss(adj, t, noise, std, score)
        self._ema.restore(self.model.parameters())
        self.log("val/loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_train_end(self):
        ckpt_path = os.path.join(self.ckpt_dir, "final.ckpt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'ema_state_dict': self._ema.state_dict(),
            'cfg': OmegaConf.to_container(self.cfg, resolve=True),  # <-- fix
        }, ckpt_path)

        symlink_path = os.path.join("checkpoints", self.dataset_name, "last_version.ckpt")
        if os.path.islink(symlink_path) or os.path.exists(symlink_path):
            os.remove(symlink_path)
        os.symlink(os.path.abspath(ckpt_path), symlink_path)

    def _____loss(self, adj, t, noise, std, score):
        reduce_op = torch.mean if self.reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
        if not self.likelihood_weighting:
            losses = torch.square(score * std[:, None, None] + noise)
            losses = losses.view(losses.size(0), -1)
            agg = reduce_op(losses, dim=-1)
        else:
            g2 = self.sde.sde(torch.zeros_like(adj), t)[1] ** 2
            losses = torch.square(score + noise / std[:, None, None])
            losses = losses.view(losses.size(0), -1)
            agg = reduce_op(losses, dim=-1)
            agg *= g2
        return torch.mean(agg)
    
    def _loss(self, adj, t, noise, std, score):
        # Mask: only consider valid node pairs (non-padding)
        flags = node_flags(adj)  # B × N
        mask = (flags[:, :, None] * flags[:, None, :]).float()  # B × N × N

        if not self.likelihood_weighting:
            losses = torch.square(score * std[:, None, None] + noise)
        else:
            g2 = self.sde.sde(torch.zeros_like(adj), t)[1] ** 2
            losses = torch.square(score + noise / std[:, None, None])

        losses = losses * mask  # Apply mask
        num_valid = mask.sum(dim=(1, 2)).clamp(min=1)  # B

        agg = losses.sum(dim=(1, 2)) / num_valid  # normalize per graph
        return agg.mean()  # average over batch

    def _perturb_data(self, adj, flags, t):
        noise = gen_noise(adj, flags)
        mean, std = self.sde.marginal_prob(adj, t)
        perturbed = mean + std[:, None, None] * noise
        perturbed = mask_adjs(perturbed, flags)
        return perturbed, noise, std
