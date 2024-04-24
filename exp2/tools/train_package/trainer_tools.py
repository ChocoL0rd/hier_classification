import logging
import os
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.optim
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from .wrappers import OptimizerWrapper, SchedulerWrapper
from .checker import ConvergenceChecker
from .dataset_tools import TrainCLSTreeDataset, TestCLSTreeDataset


# one batch training
class ClsTreeTrainer:
    def __init__(
            self,
            model,
            tokenizer,
            train_set: TrainCLSTreeDataset,
            val_set: TestCLSTreeDataset,
            cfg,
    ):
        """
        :param model: model to train.
        :param tokenizer: can convert list of indexes back to names of categories
        :param train_set: returns paths, img_batch, target_batch, mask_batch
        :param val_set: validation dataset. Format of returned look info as train_set
        :param cfg:
        {
            device: cuda or cpu,
            pin_memory: bool,
            num_workers: int,

            epochs: int,
            val_period: int,
            batch_size: int,

            save_path: str,
            save_period: int,

            optimizer: *see optimizer wrapper*,
            scheduler: *see scheduler wrapper*,
            convergence_checker: *see convergence_checker*,

            # loss: *see loss wrapper*,
        }
        """
        if cfg["device"].startswith("cuda"):
            assert torch.cuda.is_available(), f"Cuda is available = {torch.cuda.is_available()}"

        self.device = torch.device(cfg["device"])
        self.pin_memory = cfg["pin_memory"]
        self.num_workers = cfg["num_workers"]

        self.epochs = cfg["epochs"]
        self.val_period = cfg["val_period"]
        self.batch_size = cfg["batch_size"]

        self.save_path = cfg["save_path"]
        self.save_period = cfg["save_period"]

        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        self.optimizer: OptimizerWrapper = OptimizerWrapper(
            cfg["optimizer"],
            self.model.parameters()
        )
        self.scheduler: SchedulerWrapper = SchedulerWrapper(
            cfg["scheduler"],
            self.optimizer.optimizer
        )
        self.convergence_checker = ConvergenceChecker(cfg["convergence_checker"])

        self.loss = nn.CrossEntropyLoss()

        self.train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=False if cfg["weighted_rand_sampler"] else True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers if self.pin_memory else 0,
            sampler=WeightedRandomSampler(
                weights=train_set.get_balancing_weights(),
                num_samples=cfg["n_train_samples"] if cfg["n_train_samples"] else train_set.size,
                replacement=cfg["replacement"],
            ) if cfg["weighted_rand_sampler"] else None
        )

        self.val_loader = DataLoader(
            dataset=val_set,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers if self.pin_memory else 0,
            sampler=RandomSampler(val_set, num_samples=cfg["n_val_samples"]) if cfg["n_val_samples"] else None,
            collate_fn=lambda x: x[0]
        )

        self.writer = SummaryWriter(log_dir=self.save_path)
        self.log = logging.getLogger()

    @torch.no_grad()
    def test(self, log_name, dataset):
        # self.model.eval()

        self.log.info(f"Test {log_name} dataset")
        loader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers if self.pin_memory else 0,
            collate_fn=lambda x: x[0]
        )

        stats = {
            "path": [],
            "target": [],
            "predict": [],
            "main_type_acc": [],
            "attrs_acc": [],
            "is_correct": []
        }

        for img_path, img_batch, target in tqdm(loader):
            pred = np.array(self.model.inference(img_batch.unsqueeze(0).to(self.device)))
            target = np.array(target)

            stats["path"].append(img_path)
            stats["target"].append(target.tolist())
            stats["predict"].append(pred.tolist())
            stats["is_correct"].append(self.tokenizer.is_correct(pred))
            stats["main_type_acc"].append(pred[0] == target[0])
            cur_attrs = [target[i] == pred[i] if i < len(pred) else False for i in range(1, len(target))]
            stats["attrs_acc"].append(cur_attrs.count(True) / len(cur_attrs))

        self.log.info(f"Mean correctness: {np.mean(stats['is_correct'])}")
        self.log.info(f"Mean main type accuracy: {np.mean(stats['main_type_acc'])}")
        self.log.info(f"Mean attribute accuracy: {np.mean(stats['attrs_acc'])}")

        return pd.DataFrame(stats)

    @torch.no_grad()
    def validate(self):
        # don't use eval, it causes issues.
        # self.model.eval()

        self.log.info("Validation")

        # stats - accuracies
        stats = {
            "main_type_acc": [],
            "attrs_acc": [],
            "correctness": [],
        }

        for img_path, img_batch, target in tqdm(self.val_loader):
            # img_path - string
            # img_batch - (3, H, W)
            # target - list of strings
            pred = np.array(self.model.inference(img_batch.unsqueeze(0).to(self.device)))

            target = np.array(target)
            # print(pred)
            stats["main_type_acc"].append(pred[0] == target[0])
            stats["correctness"].append(self.tokenizer.is_correct(pred))
            cur_attrs = [target[i] == pred[i] if i < len(pred) else False for i in range(1, len(target))]
            stats["attrs_acc"].append(cur_attrs.count(True)/len(cur_attrs))

        for key, value in stats.items():
            stats[key] = np.mean(value)

        self.log.info(f"Mean correctness: {stats['correctness']}")
        self.log.info(f"Mean main type accuracy: {stats['main_type_acc']}")
        self.log.info(f"Mean attribute accuracy: {stats['attrs_acc']}")

        return stats

    def train_epoch(self):
        self.model.train()

        self.log.info("Training")
        mean_loss = []
        for img_path, img_batch, sent_batch, target_batch in tqdm(self.train_loader):
            self.optimizer.zero_grad()

            target_batch = target_batch.to(self.device)
            pred = self.model(
                sent_batch.to(self.device),
                img_batch.to(self.device),
            )

            is_pad_mask = self.tokenizer.idxs2is_pad_mask(sent_batch)
            target_batch = target_batch[~is_pad_mask]
            loss = self.loss(pred, target_batch)
            loss.backward()
            self.optimizer.step()

            mean_loss.append(loss.detach().cpu().item())

        mean_loss = np.mean(mean_loss)
        self.log.info(f"Mean loss: {mean_loss}")

        return mean_loss

    def train(self, epochs, save_path, save_period, val_period):
        self.log.info("Train")
        if epochs is None:
            epochs = self.epochs
        if save_path is None:
            save_path = self.save_path
        if save_period is None:
            save_period = self.save_period
        if val_period is None:
            val_period = self.val_period

        stats = {
            "train_loss": [],
            "main_type_acc": [],
            "attrs_acc": [],
            "correctness": [],
        }

        for epoch in range(1, epochs + 1):
            self.log.info(f"Epoch: {epoch}/{epochs}")
            train_loss = self.train_epoch()
            stats["train_loss"].append(train_loss)
            self.writer.add_scalar("Train/Loss", train_loss, epoch)

            if epoch % save_period == 0 or epoch == epochs:
                self.log.info("Saving model ...")
                torch.save(self.model.state_dict(), os.path.join(save_path, "model.pt"))

            if epoch % val_period == 0 or epoch == epochs:
                val_stats = self.validate()

                for key, value in val_stats.items():
                    stats[key].append(value)
                    self.writer.add_scalar(f"Val/{key}", value, epoch)

            self.scheduler.step(train_loss)
            if self.convergence_checker.check(train_loss):
                break

            stats_df = pd.DataFrame(stats)
            stats_df.to_csv(os.path.join(self.save_path, "train_stats.csv"), index=False)
