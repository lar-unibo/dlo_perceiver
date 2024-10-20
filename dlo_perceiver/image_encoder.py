import torch
import torch.nn as nn
from torch.nn import functional as F

from dlo_perceiver.backbone import resnet
from dlo_perceiver.backbone.utils import IntermediateLayerGetter


class ResNetEncoder(nn.Module):
    def __init__(self, model_name, latent_dim, pretrained=True):
        super(ResNetEncoder, self).__init__()
        # Load a pre-trained ResNet model
        backbone = resnet.__dict__[model_name](
            pretrained=pretrained, replace_stride_with_dilation=[False, False, True]
        )
        return_layers = {"layer4": "out", "layer1": "low_level"}
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        self.project = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.project_latent = nn.Sequential(
            nn.Conv2d(2096, latent_dim, 1, bias=False),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        z = self.backbone(x)
        low_level_feature = self.project(z["low_level"])
        output_feature = F.interpolate(
            z["out"], size=low_level_feature.shape[2:], mode="bilinear", align_corners=False
        )
        z = self.project_latent(torch.cat([low_level_feature, output_feature], dim=1))
        return z
