from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

# from configs.config_extrapolation_tree import MainConfig
# from configs.config_extrapolation_planar import MainConfig
from configs.config_extrapolation_sbm import MainConfig
# from configs.config_community_small import MainConfig
from src.datasets.community_small import GraphDatasetModule
from src.train.trainer import DiffusionGraphModule
from src.sample.sampler import Sampler
from src.evaluation.eval import EvaluationMetrics


def main():
    cfg = OmegaConf.structured(MainConfig())
    logger = WandbLogger(project="graphon-diffusion", config=OmegaConf.to_container(cfg)) if cfg.general.use_wandb else None
    pl.seed_everything(cfg.general.seed)

    datamodule = GraphDatasetModule(cfg)
    datamodule.setup()

    model = DiffusionGraphModule(cfg)
    trainer = Trainer(
        accelerator=cfg.general.device,
        devices=1,
        max_epochs=cfg.train.num_epochs,
        logger=logger,
        log_every_n_steps=1,
        enable_checkpointing=False,
    )
    trainer.fit(model, datamodule=datamodule)

    sampler = Sampler(cfg=cfg, datamodule=datamodule, logger=logger)
    generated_graphs = sampler.sample()

    evaluator = EvaluationMetrics(
        generated_graphs=generated_graphs,
        graph_type=cfg.data.data.split("_")[-1],
        # graph_type="sbm",
        train_graphs=datamodule.train_graphs
    )
    evaluator.run_all_metrics(
        out_folder="eval",
        filename=f"{cfg.data.data}_metrics",
        wandb_logger=logger
    )


if __name__ == "__main__":
    main()
