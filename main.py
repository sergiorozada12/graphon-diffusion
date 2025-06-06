from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from config import MainConfig
from src.datasets.community_small import CommunitySmall
from src.train.trainer import DiffusionGraphModule
from src.sample.sampler import Sampler


def main():
    cfg = OmegaConf.structured(MainConfig())
    logger = WandbLogger(project="graphon-diffusion", config=OmegaConf.to_container(cfg)) if cfg.general.use_wandb else None
    pl.seed_everything(cfg.general.seed)

    datamodule = CommunitySmall(cfg)
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
    _ = sampler.sample()


if __name__ == "__main__":
    main()