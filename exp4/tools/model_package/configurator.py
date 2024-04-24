import logging
import torch
import json

from .encoder import dinov2
from .decoder import custom
from .tokenizer import TreeTokenizer
from .vit_cls_tree import ViTCLSTree

log = logging.getLogger()

enc_name2configurator = {
    "dinov2": dinov2.configurator,
}

dec_name2configurator = {
    "custom": custom.configurator,
}


def cfg2model(cfg):
    """ return model, preprocessing and tokenizer"""
    with open(cfg["tree_path"]) as f:
        tree = json.load(f)

    tokenizer = TreeTokenizer(tree)
    encoder, preproc = enc_name2configurator[cfg["enc_cfg"]["name"]](cfg["enc_cfg"])
    decoder = dec_name2configurator[cfg["dec_cfg"]["name"]](
        cfg["dec_cfg"],
        n_in_tokens=tokenizer.in_vocab_size,
        n_out_tokens=tokenizer.out_vocab_size,
        pad_idx=tokenizer.pad_idx
    )
    model = ViTCLSTree(tokenizer, encoder, decoder)

    if cfg["load_pretrained"]:
        log.info(f"Loading pretrained model: {cfg['pretrained_path']} ...")
        model.load_state_dict(torch.load(cfg["pretrained_path"]))

    return model, preproc, tokenizer
