from __future__ import annotations
import math
import torch
import torch.nn as nn
try:
    import timm
except ImportError as exc:
    raise ImportError("paper_impl.model requires timm; install paper_impl/requirements.txt") from exc
from .config import ModelConfig
from .dynamic_bifpn import DynamicBiFPN

class SeparableHeadBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
    def forward(self, x):
        return self.pw(self.dw(x))

class EfficientDetHead(nn.Module):
    """Shared EfficientDet-style convolution head with level-specific BN."""
    def __init__(self, channels, repeats, num_levels, out_channels, prior=None):
        super().__init__()
        self.conv_rep = nn.ModuleList([SeparableHeadBlock(channels) for _ in range(repeats)])
        self.bn_rep = nn.ModuleList([
            nn.ModuleList([nn.BatchNorm2d(channels, eps=1e-3, momentum=1e-2) for _ in range(num_levels)])
            for _ in range(repeats)
        ])
        self.act = nn.SiLU(inplace=True)
        self.predict = nn.Conv2d(channels, out_channels, 3, padding=1)
        if prior is not None:
            nn.init.constant_(self.predict.bias, -math.log((1.0 - prior) / prior))

    def forward(self, features):
        out = []
        for level, x in enumerate(features):
            for r, conv in enumerate(self.conv_rep):
                x = self.act(self.bn_rep[r][level](conv(x)))
            out.append(self.predict(x))
        return out

class DynamicEfficientDet(nn.Module):
    """EfficientNet backbone + Dynamic BiFPN + EfficientDet-style heads."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config.resolve()
        self.backbone = timm.create_model(
            self.config.backbone, pretrained=self.config.pretrained_backbone,
            features_only=True, out_indices=(2, 3, 4),
        )
        in_channels = list(self.backbone.feature_info.channels())
        self.fpn = DynamicBiFPN(
            in_channels, self.config.fpn_channels, self.config.fpn_repeats,
            self.config.dynamic.epsilon, self.config.dynamic.alpha_max,
            self.config.dynamic.alpha_init,
        )
        self.class_net = EfficientDetHead(
            self.config.fpn_channels, self.config.head_repeats, 5,
            self.config.num_anchors * self.config.num_classes, self.config.class_prior,
        )
        self.box_net = EfficientDetHead(
            self.config.fpn_channels, self.config.head_repeats, 5,
            self.config.num_anchors * 4, None,
        )

    def forward_pyramid(self, x, return_states=False):
        return self.fpn(self.backbone(x), return_states=return_states)

    def forward(self, x, return_features=False):
        if return_features:
            pyramid, states = self.forward_pyramid(x, True)
        else:
            pyramid = self.forward_pyramid(x, False)
            states = None
        cls = self.class_net(pyramid)
        box = self.box_net(pyramid)
        return (cls, box, {"pyramid": pyramid, "states": states}) if return_features else (cls, box)

    def polarity_regularizer(self):
        return self.fpn.polarity_regularizer()
