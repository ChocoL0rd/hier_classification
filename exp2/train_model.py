import hydra
import hydra.core.hydra_config
from omegaconf import OmegaConf

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import numpy as np
import pandas as pd

import logging

# import my tools
from tools import cfg2model
from tools import ClsTreeTrainer, cfg2datasets
import random

# fix random seeds to make results reproducible.
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    save_path = hydra_cfg.runtime.output_dir
    OmegaConf.resolve(cfg)

    # creating model
    model, preproc, tokenizer = cfg2model(cfg["model_cfg"])

    log.info(f"Number of params: {sum(p.numel() for p in model.parameters())}")
    log.info(f"Number of params, requires grad: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # create datasets
    datasets = cfg2datasets(cfg["dataset_cfg"], preproc, tokenizer)

    datasets["train"].aug_flag = True
    datasets["val"].aug_flag = False

    # initialize trainer
    trainer = ClsTreeTrainer(
        model,
        tokenizer,
        datasets["train"],
        datasets["val"],
        cfg["train_cfg"],
    )

    # train model
    trainer.train(
        cfg["train_cfg"]["epochs"],
        cfg["train_cfg"]["save_path"],
        cfg["train_cfg"]["save_period"],
        cfg["train_cfg"]["val_period"]
    )

    os.mkdir(os.path.join(save_path, "test"))

    stats_df = trainer.test("train_val", datasets["train_val"])
    stats_df.to_csv(os.path.join(save_path, "test", f"train_val.csv"), index=False)

    stats_df = trainer.test("val", datasets["val"])
    stats_df.to_csv(os.path.join(save_path, "test", f"val.csv"), index=False)

    trainer.writer.close()


if __name__ == "__main__":
    my_app()
