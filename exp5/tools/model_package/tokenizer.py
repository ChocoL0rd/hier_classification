import numpy as np
import copy

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

        self.idx2name = self._get_tree_nodes(tree)
        self.name2idx = {name: i for i, name in enumerate(self.idx2name)}

        self.idx_tree = self._get_idx_tree(tree, self.name2idx)
        self.out_vocab_size = len(self.idx2name)

        self.attrs_mask_tree, self.n_masks = self._get_attrs_mask_tree(self.tree)
        self.pad_idx = self.n_masks

        self.in_vocab_size = self.n_masks + 1

        self.main_idxs = [self.name2idx[main_class] for main_class in tree.keys()]  # indexes of main classes
        self.main_idx2n_attrs = {main_idx: len(attrs_list) for main_idx, attrs_list in self.idx_tree.items()}

        self.pred_mask = self._get_pred_mask()

        self.rows2idxs = np.vectorize(lambda x: self.name2idx[x])

#np.where(self.pred_mask[self.name2idx["cardigan"]] != -np.inf)[0]
# [np.where((self.pred_mask[self.attrs_mask_tree["cardigan"]] != -np.inf)[i])[0] for i in range(6)]
    def _get_pred_mask(self):
        """
        Returns mask where, on each mask_idx row 0 on positions that are predictable for this mask_idx
        and -inf otherwise
        """
        pred_mask = np.full((self.in_vocab_size, self.out_vocab_size), -np.inf)
        pred_mask[0, self.main_idxs] = 0

        for main_name, mask_idxs in self.attrs_mask_tree.items():
            main_idx = self.name2idx[main_name]
            attrs_list = self.idx_tree[main_idx]
            for i, mask_idx in enumerate(mask_idxs):
                attrs = attrs_list[i]
                pred_mask[mask_idx, attrs] = 0

        return pred_mask

    def row2mask_idxs(self, row):
        if not self.is_correct(row):
            return

        mask_idxs = [0]  # start with main class
        mask_idxs += self.attrs_mask_tree[row[0]]

        return np.array(mask_idxs)

    def get_n_attrs(self, idx):
        return self.main_idx2n_attrs[idx]

    def is_main_idx(self, idx):
        return idx in self.main_idxs

    @staticmethod
    def _get_attrs_mask_tree(tree):
        attrs_mask_tree = {main_class: [] for main_class in tree}
        mask_counter = 1  # 0 is for main masks, so start with 1

        for main_class, attrs_list in tree.items():
            for _ in attrs_list:
                attrs_mask_tree[main_class].append(mask_counter)
                mask_counter += 1

        return attrs_mask_tree, mask_counter

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

    def pad_complement(self, idxs: np.array):
        """ Complements 1d array to the max sentence length with [PAD]. """
        return np.concatenate(
            [
                idxs,
                [self.pad_idx for _ in range(self.max_sent_length - idxs.shape[0])]
            ],
            axis=0
        )

    def idxs2is_pad_mask(self, idxs):
        return idxs == self.pad_idx

    def idxs2row(self, idxs: np.array, mask=None) -> list:
        """
        :param idxs: array where last dimension is sentence of indexes
        :param mask: if needed to delete something else
        :return: nested list where last dimension is list of strings
        """

        if mask is None:
            # mask = idxs != self.pad_idx
            mask = np.ones_like(idxs, dtype=bool)

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
                ],
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

    rows = [
        ["dress", "neckline_Collar", "sleeve_Long"],
        ["blazer", "pattern_Floral", "neckline_V-neck", "sleeve_Short", "length_Long"],
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

    # row = rows[0]
    # padded_row = tokenizer.pad_complement(row)
    # idxs = tokenizer.rows2idxs(row)
    #
    # print(f"Row: {tokenizer.idxs2row(idxs)}")

    print(tokenizer.tree)
    print(tokenizer.idx_tree)
    print(tokenizer.mas)
    print(tokenizer.attrs_mask_tree)
    print(tokenizer.row2mask_idxs(rows[0]))
    print(tokenizer.pred_mask)