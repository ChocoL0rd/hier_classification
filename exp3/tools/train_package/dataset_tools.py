import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from logging import getLogger
import random
from collections import defaultdict

log = getLogger(__name__)


class TrainCLSTreeDataset(Dataset):
    def __init__(
            self,
            log_name,
            info,
            preproc,
            tokenizer,
    ):
        """
        :param info: data in format
        {
            img_path: [cls1, cls2, cls3, ...],
            ...,
        }
        :param preproc: image preprocessing, input - RGB numpy array, output - torch Tensor
        :param tokenizer: converts sentence to proper input
        """
        self.log_name = log_name
        self.info = info
        self.img_paths = list(info.keys())
        self.preproc = preproc
        self.tokenizer = tokenizer
        self.size = len(info.keys())
        self.img_path_counts = self._count_img_paths(self.img_paths, self.info)

        log.info(f"TrainCLSTreeDataset {self.log_name} is created. Size {self.size}")

    @staticmethod
    def _count_img_paths(img_paths, info):
        count_dict = defaultdict(int)
        for img_path, classes in info.items():
            classes_tuple = tuple(classes)
            count_dict[classes_tuple] += 1
        count_result = [count_dict.get(tuple(info[path]), 0) for path in img_paths]
        return count_result

    def get_balancing_weights(self):
        return [1.0 / (self.img_path_counts[cls_idx] + 1e-6) for cls_idx in self.img_path_counts]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            log.info(f"Wrong reading {img_path}. Second attempt ...")
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                log.info(f"Second attempt {img_path} is None")
                raise Exception(f"Error reading {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.preproc(img)

        row = self.info[img_path]
        # randomly pick what to predict
        random_mask = np.random.choice([True, False], size=len(row))
        # to guarantee one True in mask
        random_mask[np.random.randint(0, random_mask.shape[0])] = True

        mask_idxs = self.tokenizer.row2mask_idxs(row)
        mask_idxs = self.tokenizer.pad_complement(mask_idxs[random_mask])

        # dirty code, because it complements sequence of output indexes
        # with pad index, which is from input indexes
        # it doesn't affect training because in trainer we use only not padded
        # positions from mask_idxs
        target_idxs = self.tokenizer.rows2idxs(row)
        target_idxs = self.tokenizer.pad_complement(target_idxs[random_mask])

        return img_path, img, torch.from_numpy(mask_idxs).to(torch.long), torch.from_numpy(target_idxs).to(torch.long)


class TestCLSTreeDataset(Dataset):
    def __init__(
            self,
            log_name,
            info,
            preproc,
    ):
        """
        :param info: data in format
        {
            img_path: [cls1, cls2, cls3, ...],
            ...,
        }
        :param preproc: image preprocessing, input - RGB numpy array, output - torch Tensor
        """
        self.log_name = log_name
        self.info = info
        self.img_paths = list(info.keys())
        self.preproc = preproc
        self.size = len(info.keys())

        log.info(f"TestCLSTreeDataset {self.log_name} is created. Size {self.size}")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            log.info(f"Wrong reading {img_path}. Second attempt ...")
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                log.info(f"Second attempt {img_path} is None")
                raise Exception(f"Error reading {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.preproc(img)
        return img_path, img, self.info[img_path]


def cfg2datasets(cfg, preproc, tokenizer):
    with open(cfg["path"]) as f:
        info = json.load(f)

    datasets = {}
    for key, value in info.items():
        if key == "train":
            datasets[key] = TrainCLSTreeDataset(
                key,
                value,
                preproc,
                tokenizer
            )
            datasets["train_val"] = TestCLSTreeDataset(
                key,
                value,
                preproc,
            )
        else:
            datasets[key] = TestCLSTreeDataset(
                key,
                value,
                preproc,
            )

    return datasets
