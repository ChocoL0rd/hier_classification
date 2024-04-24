import torch
from torch import nn
from torchvision import transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2


version2dim = {
    "vits14": 384,
    "vitb14": 768,
    "vitl14": 1024,
    "vitg14": 1536,

    "vits14_reg": 384,
    "vitb14_reg": 768,
    "vitl14_reg": 1024,
    "vitg14_reg": 1536,
}


class ModifiedDinoVisionTransformer(nn.Module):
    def __init__(self, original_model):
        super(ModifiedDinoVisionTransformer, self).__init__()
        self.original_model = original_model

    def forward(self, x):
        return self.original_model(x, is_training=True)["x_norm_patchtokens"]


def configurator(cfg):
    """ vits14, vitb14, vitl14, vitg14, """

    model = ModifiedDinoVisionTransformer(
        torch.hub.load('facebookresearch/dinov2', f"dinov2_{cfg['version']}")
    )

    if cfg["freeze"]:
        for param in model.parameters():
            param.requires_grad = False

    preproc = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    return model, lambda x: preproc(image=x)["image"]

