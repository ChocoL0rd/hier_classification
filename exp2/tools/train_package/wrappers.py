from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, ReduceLROnPlateau
from torch.optim import SGD, Adam, AdamW

import logging

__all__ = [
    "OptimizerWrapper",
    "SchedulerWrapper"
]

log = logging.getLogger(__name__)


class OptimizerWrapper:
    def __init__(self, cfg, model_params):
        if cfg["name"] == "sgd":
            self.optimizer = SGD(
                params=model_params,
                lr=cfg["lr"],
                weight_decay=cfg["weight_decay"],
            )
        elif cfg["name"] == "adam":
            self.optimizer = Adam(
                params=model_params,
                lr=cfg["lr"],
                betas=(cfg["beta1"], cfg["beta2"]),
                weight_decay=cfg["weight_decay"],
            )
        elif cfg["name"] == "adamw":
            self.optimizer = AdamW(
                params=model_params,
                lr=cfg["lr"],
                betas=(cfg["beta1"], cfg["beta2"]),
                weight_decay=cfg["weight_decay"]
            )
        else:
            msg = f"Optimizer {cfg['name']} is not supported"
            log.critical(msg)
            raise ValueError(msg)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def get_param_groups(self):
        return self.optimizer.param_groups

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def get_weight_decay(self):
        return self.optimizer.param_groups[0]['weight_decay']


class SchedulerWrapper:
    def __init__(self, cfg, optimizer):
        self.scheduler_name = cfg["name"]
        self.warmup_epoch = cfg["warmup_epochs"]
        self.warmup_counter = 0
        self.loss_value_is_needed = False

        if self.scheduler_name == "cosine_annealing_warm_restarts":
            self.scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=cfg["T_0"],
                T_mult=1,
                eta_min=cfg["eta_min"],
                last_epoch=-1
            )

        elif self.scheduler_name == 'step_lr':
            self.scheduler = StepLR(optimizer, step_size=cfg["step_size"], gamma=cfg["factor"])
        elif self.scheduler_name == 'reduce_on_plateau':
            self.scheduler = ReduceLROnPlateau(optimizer, factor=cfg["factor"])
            self.loss_value_is_needed = True
        else:
            msg = f"Scheduler {cfg['name']} is not supported"
            log.critical(msg)
            raise ValueError(msg)

    def step(self, loss_value=None):
        if self.warmup_counter < self.warmup_epoch:
            self.warmup_counter += 1
        else:
            if self.loss_value_is_needed:
                self.scheduler.step(loss_value)
            else:
                self.scheduler.step()
