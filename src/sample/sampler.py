import math
import torch
import matplotlib.pyplot as plt
import networkx as nx
import wandb
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger

from src.utils import adjs_to_graphs, quantize
from src.sde.equations import VPSDE, VESDE, subVPSDE
from src.sde.solver import get_pc_solver, get_s4_solver
from src.train.ema import ExponentialMovingAverage
from src.models.score_network import ScoreNetwork


class Sampler:
    def __init__(self, cfg, datamodule, logger=None):
        self.cfg = cfg
        self.device = torch.device(cfg.general.device)
        self.logger = logger
        self.num_nodes = cfg.sampler.num_nodes or cfg.data.max_node_num

        ckpt = torch.load(f"checkpoints/{cfg.data.data}/last_version.ckpt", map_location=self.device)

        self.model = ScoreNetwork(**self.cfg.model).to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])

        if cfg.sampler.use_ema:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=self.cfg.train.ema)
            self.ema.load_state_dict(ckpt['ema_state_dict'])
            self.ema.copy_to(self.model.parameters())
        else:
            self.ema = None

        self.sde = self._get_sde(self.cfg.sde)
        self.sampling_fn = self._get_sampling_fn()

        self.datamodule = datamodule

    def _get_sde(self, sde_cfg):
        if sde_cfg.type == 'VP':
            return VPSDE(beta_min=sde_cfg.beta_min, beta_max=sde_cfg.beta_max, N=sde_cfg.num_scales)
        elif sde_cfg.type == 'VE':
            return VESDE(sigma_min=sde_cfg.beta_min, sigma_max=sde_cfg.beta_max, N=sde_cfg.num_scales)
        elif sde_cfg.type == 'subVP':
            return subVPSDE(beta_min=sde_cfg.beta_min, beta_max=sde_cfg.beta_max, N=sde_cfg.num_scales)
        else:
            raise NotImplementedError(f"SDE type {sde_cfg.type} not supported.")

    def _get_sampling_fn(self):
        shape_x = (self.cfg.data.batch_size, self.num_nodes, self.cfg.data.max_feat_num)
        shape_adj = (self.cfg.data.batch_size, self.num_nodes, self.num_nodes)

        get_sampler = get_s4_solver if self.cfg.sampler.solver == 'S4' else get_pc_solver
        return get_sampler(
            sde_adj=self.sde,
            shape_x=shape_x,
            shape_adj=shape_adj,
            predictor=self.cfg.sampler.predictor,
            corrector=self.cfg.sampler.corrector,
            snr=self.cfg.sampler.snr,
            scale_eps=self.cfg.sampler.scale_eps,
            n_steps=self.cfg.sampler.n_steps,
            probability_flow=self.cfg.sampler.probability_flow,
            continuous=True,
            denoise=self.cfg.sampler.noise_removal,
            eps=self.cfg.sampler.eps,
            device=self.device
        )

    def _make_full_flags(self):
        return torch.ones(self.cfg.data.batch_size, self.num_nodes, dtype=torch.bool, device=self.device)

    def plot_sampled_graphs(self, graph_list, num_graphs=20):
        num_graphs = min(num_graphs, len(graph_list))
        n_cols = 5
        n_rows = math.ceil(num_graphs / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
        axes = axes.flatten()

        for i in range(n_rows * n_cols):
            ax = axes[i]
            ax.axis("off")
            if i < num_graphs:
                G = graph_list[i]
                pos = nx.spring_layout(G, seed=42)
                nx.draw(G, pos, ax=ax, node_size=50, with_labels=False)

        fig.suptitle("Sampled Graphs", fontsize=16)
        plt.tight_layout()

        sample_dir = Path("samples") / self.cfg.data.data
        sample_dir.mkdir(parents=True, exist_ok=True)
        out_path = sample_dir / "sampled_graphs.png"
        plt.savefig(out_path)

        if self.logger and isinstance(self.logger, WandbLogger):
            self.logger.experiment.log({"sampled_graphs": wandb.Image(str(out_path))})

        plt.close(fig)

    def sample(self):
        num_rounds = math.ceil(len(self.datamodule.test_graphs) / self.cfg.data.batch_size)
        generated = []

        for _ in range(num_rounds):
            flags = self._make_full_flags()
            adj, _ = self.sampling_fn(self.model, flags)
            samples = quantize(adj)
            graphs = adjs_to_graphs(samples, is_cuda=self.device.type != 'cpu')
            generated.extend(graphs)

        self.plot_sampled_graphs(generated[:len(self.datamodule.test_graphs)])
        return generated[:len(self.datamodule.test_graphs)]