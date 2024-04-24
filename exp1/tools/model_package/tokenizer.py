import torch
import numpy as np
import copy

from typing import List, Union
import random


class TreeTokenizer:
    def __init__(self, tree):
        """
        :param tree: dictionary of format
        {
            main_class1: [
               [attr_value_1, attr_value_2, attr_value_3],
               [attr_value_1, attr_value_2],
               ...
            ],
            ...
        }
        """

        self.tree = tree
        self.max_sent_length = self._get_tree_height(tree)

        self.mask = "[MASK]"
        self.pad = "[PAD]"

        self.idx2name = self._get_tree_nodes(tree) + [self.mask, self.pad]
        self.name2idx = {name: i for i, name in enumerate(self.idx2name)}

        self.idx_tree = self._get_idx_tree(tree, self.name2idx)
        self.n_tokens = len(self.idx2name)

        self.mask_idx = self.name2idx[self.mask]
        self.pad_idx = self.name2idx[self.pad]
        self.main_idxs = [self.name2idx[main_class] for main_class in tree.keys()]  # indexes of main classes
        self.main_idx2n_attrs = {main_idx: len(attrs_list) for main_idx, attrs_list in self.idx_tree.items()}

        self.init_sent = [self.mask_idx]
        self.rows2idxs = np.vectorize(lambda x: self.name2idx[x])

    def get_n_attrs(self, idx):
        return self.main_idx2n_attrs[idx]

    def is_main_idx(self, idx):
        return idx in self.main_idxs

    @staticmethod
    def _get_idx_tree(tree: dict, name2idx: dict):
        idx_tree = {name2idx[key]: copy.deepcopy(value) for key, value in tree.items()}

        for main_class_idx, attrs_list in idx_tree.items():
            for i, attr_list in enumerate(attrs_list):
                # Update the original list in idx_tree
                idx_tree[main_class_idx][i] = [name2idx[attr_value] for attr_value in attr_list]

        return idx_tree

    @staticmethod
    def _get_tree_height(tree):
        # equal max number of attributes + 1
        return np.max([len(attrs_list) for attrs_list in tree.values()]) + 1

    @staticmethod
    def _get_tree_nodes(tree: dict):
        # returns unique names of classes and attributes
        nodes = []
        for main_class, attrs_list in tree.items():

            # check if main_class is string
            if not isinstance(main_class, str):
                raise TypeError("All nodes of TreeTokenizer must be strings")

            nodes.append(main_class)
            for attr_value_list in attrs_list:

                # check if all attrs are strings
                for attr_value in attr_value_list:
                    if not isinstance(attr_value, str):
                        raise TypeError("All nodes of TreeTokenizer must be strings")

                nodes += attr_value_list

        return list(sorted(set(nodes)))

    def is_correct(self, str_list: list) -> bool:
        if len(str_list) > 0 and str_list[0] in self.tree.keys():
            attrs_list = self.tree[str_list[0]]
            if len(str_list) - 1 != len(attrs_list):
                return False

            for i, attr_name in enumerate(str_list[1:]):
                if attr_name not in attrs_list[i]:
                    return False

            return True

        return False

    def pad_complement(self, row: list):
        """ Complements to the max sentence length with [PAD]. """
        length = len(row)

        if not self.is_correct(row):
            raise ValueError(f"Wrong sequence: {row}")

        return row + [self.pad for _ in range(self.max_sent_length - length)]

    def idxs2is_pad_mask(self, idxs):
        return idxs == self.pad_idx

    def idxs2is_masked_mask(self, idxs):
        return idxs == self.mask_idx

    def idxs2row(self, idxs: np.array, mask=None) -> list:
        """
        :param idxs: array where last dimension is sentence of indexes
        :param mask: if needed to delete something else
        :return: nested list where last dimension is list of strings
        """

        if mask is None:
            mask = idxs != self.pad_idx

        if idxs.ndim > 1:
            result = []
            for i in range(idxs.shape[0]):
                result.append(self.idxs2row(idxs[i], mask[i]))
            return result
        elif idxs.ndim == 1:
            result = np.take(self.idx2name, idxs[mask]).tolist()
            return result
        elif idxs.ndim == 0:
            return self.idx2name[idxs]

    def random_mask_idxs(self, idxs: np.array):
        """
            Generates mask array with randomly chosen element as a True
        :param idxs: 1d array of padded idxs
        :return: masked_idxs with picked mask_idx, target_idx
        """
        pad_num = self.idxs2is_pad_mask(idxs).sum()
        masking_num = random.randint(0, idxs.shape[0] - pad_num - 1)
        masked_idxs = np.copy(idxs)
        masked_idxs[masking_num] = self.mask_idx
        return masked_idxs, idxs[masking_num]


if __name__ == "__main__":
    tokenizer = TreeTokenizer(
        {
            "dress": [
                [
                    "neckline_Collar",
                    "neckline_V-neck",
                    "neckline_Open"
                ],
                [
                    "sleeve_Long",
                    "sleeve_3/4",
                    "sleeve_Sleeveless"
                ]
            ],
            "blazer": [
                [
                    "pattern_Plain",
                    "pattern_Tartan",
                    "pattern_Patterned",
                    "pattern_Floral"
                ],
                [
                    "neckline_V-neck",
                    "neckline_Lapel",
                    "neckline_Collar"
                ],
                [
                    "sleeve_Long",
                    "sleeve_Sleeveless",
                    "sleeve_Short",
                    "sleeve_3/4"
                ],
                [
                    "length_Regular",
                    "length_Long",
                    "length_Cropped"
                ]
            ]
        }
    )

    # print(tokenizer.tree)
    # print(tokenizer.idx_tree)

    rows = [
        ["dress", "neckline_Collar", "sleeve_Long"],
        # ["dress", "neckline_Collar", "neckline_Collar"],
        # ["pattern_Plain", "b", "d"],
        # ["blazer"],
        ["blazer", "pattern_Floral", "neckline_V-neck", "sleeve_Short", "length_Long"],
        # []
    ]

    # for row in rows:
    #     print(tokenizer.is_correct(row))

    # padded_rows = [tokenizer.pad_complement(row) for row in rows]
    # print(padded_rows)
    # print(type(padded_rows))
    #
    # idxs = tokenizer.rows2idxs(padded_rows)
    # print(idxs)
    # print(tokenizer.idxs2is_pad_mask(idxs))
    # print(type(idxs))
    #
    # restored_row = tokenizer.idxs2row(idxs)
    # print(restored_row)

    row = rows[0]
    padded_row = tokenizer.pad_complement(row)
    idxs = tokenizer.rows2idxs(row)
    masked_idx, target = tokenizer.random_mask_idxs(idxs)

    print(f"Masked: {tokenizer.idxs2row(masked_idx)}")
    print(f"Target: {tokenizer.idxs2row(target)}")
    print(f"Row: {tokenizer.idxs2row(idxs)}")

    # masked_sentence, mask, masked_mask, target = tokenizer.random_mask(idx[0])
    # print("------------")
    # print(masked_sentence)
    # print(tokenizer.idx2row(masked_sentence, mask=mask))
    # print(mask)
    # print(masked_mask)
    # print(tokenizer.idx2name[target])
    #
    # print()
