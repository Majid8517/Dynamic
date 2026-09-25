from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, act=True):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=1e-2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SeparableRefine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.act = nn.SiLU(inplace=True)
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels, eps=1e-3, momentum=1e-2)
    def forward(self, x):
        return self.bn(self.pw(self.dw(self.act(x))))

def _inverse_sigmoid(p):
    p = min(max(p, 1e-8), 1.0 - 1e-8)
    return math.log(p / (1.0 - p))

class PolarityAwareFusion(nn.Module):
    """Configurable fusion primitive used by the canonical model and ablations.

    Canonical/full path:
        alpha_i = alpha_max * sigmoid(alpha_hat_i)
        z_i(+) = w_i * (1 + alpha_i)
        z_i(-) = w_i * (1 - alpha_i)
        s_i = softplus(z_i)
        a_i = s_i / (epsilon + sum_j s_j)

    Ablation switches can independently disable learnable edge weights, polarity,
    softplus, or positive normalization.  No feature map is ever subtracted.
    """
    def __init__(
        self,
        num_inputs,
        epsilon=1e-4,
        alpha_max=0.95,
        alpha_init=0.02,
        adaptive_weights=True,
        use_polarity=True,
        use_softplus=True,
        use_positive_normalization=True,
    ):
        super().__init__()
        if not (0.0 <= alpha_init < alpha_max < 1.0):
            raise ValueError("Require 0 <= alpha_init < alpha_max < 1")
        self.num_inputs = int(num_inputs)
        self.epsilon = float(epsilon)
        self.alpha_max = float(alpha_max)
        self.adaptive_weights = bool(adaptive_weights)
        self.use_polarity = bool(use_polarity)
        self.use_softplus = bool(use_softplus)
        self.use_positive_normalization = bool(use_positive_normalization)
        self.edge_weights = nn.Parameter(torch.ones(num_inputs))
        self.alpha_hat = nn.Parameter(torch.full((num_inputs,), _inverse_sigmoid(alpha_init / alpha_max)))

    def effective_alpha(self):
        if not self.use_polarity:
            return torch.zeros_like(self.alpha_hat)
        return self.alpha_max * torch.sigmoid(self.alpha_hat)

    def effective_weights(self, polarity=0, dtype=None):
        if polarity not in (-1, 0, 1):
            raise ValueError("polarity must be -1, 0, or +1")
        base = self.edge_weights if self.adaptive_weights else torch.ones_like(self.edge_weights)
        if self.use_polarity and polarity != 0:
            base = base * (1.0 + float(polarity) * self.effective_alpha())
        transformed = F.softplus(base) if self.use_softplus else base
        if self.use_positive_normalization:
            transformed = transformed / (transformed.sum() + self.epsilon)
        return transformed.to(dtype=dtype) if dtype is not None else transformed

    # Backward-compatible name used by existing tests and manuscript traceability.
    def normalized_weights(self, polarity=0, dtype=None):
        return self.effective_weights(polarity=polarity, dtype=dtype)

    def forward(self, features, polarity=0):
        if len(features) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} features, got {len(features)}")
        ref = features[0].shape[-2:]
        if any(f.shape[-2:] != ref for f in features):
            raise ValueError("Features must be spatially aligned before fusion")
        w = self.effective_weights(polarity, dtype=features[0].dtype)
        stacked = torch.stack(list(features), dim=-1)
        return (stacked * w.view(*([1] * (stacked.ndim - 1)), -1)).sum(dim=-1)

@dataclass
class CellState:
    td: List[torch.Tensor]
    bu_plus: List[torch.Tensor]
    bu_minus: List[torch.Tensor]

class DynamicBiFPNCell(nn.Module):
    """Configurable pyramid cell.

    With dual_bottom_up=False it performs TD -> BU+.
    With dual_bottom_up=True it performs TD -> BU+ -> BU-.
    When share_bottom_up=True the exact BU fusion/refinement modules are reused
    in BU+ and BU-, making parameter sharing structural and testable.
    """
    def __init__(
        self, channels, epsilon, alpha_max, alpha_init,
        adaptive_weights=True, use_polarity=True, use_softplus=True,
        use_positive_normalization=True, dual_bottom_up=True,
        share_bottom_up=True,
    ):
        super().__init__()
        self.dual_bottom_up = bool(dual_bottom_up)
        self.share_bottom_up = bool(share_bottom_up)
        fusion_kw = dict(
            epsilon=epsilon, alpha_max=alpha_max, alpha_init=alpha_init,
            adaptive_weights=adaptive_weights, use_polarity=use_polarity,
            use_softplus=use_softplus,
            use_positive_normalization=use_positive_normalization,
        )
        self.td_fusions = nn.ModuleList([PolarityAwareFusion(2, **fusion_kw) for _ in range(4)])
        self.td_refine = nn.ModuleList([SeparableRefine(channels) for _ in range(4)])
        self.bu_fusions = nn.ModuleList([PolarityAwareFusion(3, **fusion_kw) for _ in range(4)])
        self.bu_refine = nn.ModuleList([SeparableRefine(channels) for _ in range(4)])
        if self.dual_bottom_up and not self.share_bottom_up:
            self.bu_minus_fusions = nn.ModuleList([PolarityAwareFusion(3, **fusion_kw) for _ in range(4)])
            self.bu_minus_refine = nn.ModuleList([SeparableRefine(channels) for _ in range(4)])
        else:
            self.bu_minus_fusions = None
            self.bu_minus_refine = None

    @staticmethod
    def _up(x, ref):
        return x if x.shape[-2:] == ref.shape[-2:] else F.interpolate(x, size=ref.shape[-2:], mode="nearest")

    @staticmethod
    def _down(x, ref):
        return x if x.shape[-2:] == ref.shape[-2:] else F.adaptive_max_pool2d(x, ref.shape[-2:])

    def forward(self, p: Sequence[torch.Tensor], return_state=False):
        if len(p) != 5:
            raise ValueError("Expected P3-P7")
        p = list(p)
        td = [None] * 5
        td[4] = p[4]
        for level in range(3, -1, -1):
            fused = self.td_fusions[level]([p[level], self._up(td[level + 1], p[level])], polarity=+1)
            td[level] = self.td_refine[level](fused)

        bu_plus = [None] * 5
        bu_plus[0] = td[0]
        for level in range(1, 5):
            propagated = self._down(bu_plus[level - 1], td[level])
            fused = self.bu_fusions[level - 1]([p[level], td[level], propagated], polarity=+1)
            bu_plus[level] = self.bu_refine[level - 1](fused)

        if not self.dual_bottom_up:
            bu_minus = list(bu_plus)
        else:
            minus_fusions = self.bu_fusions if self.share_bottom_up else self.bu_minus_fusions
            minus_refine = self.bu_refine if self.share_bottom_up else self.bu_minus_refine
            bu_minus = [None] * 5
            bu_minus[0] = bu_plus[0]
            for level in range(1, 5):
                propagated = self._down(bu_minus[level - 1], bu_plus[level])
                fused = minus_fusions[level - 1]([bu_plus[level], td[level], propagated], polarity=-1)
                bu_minus[level] = minus_refine[level - 1](fused)

        state = CellState(td=td, bu_plus=bu_plus, bu_minus=bu_minus)
        return (bu_minus, state) if return_state else bu_minus

    def polarity_values(self):
        modules = list(self.td_fusions) + list(self.bu_fusions)
        if self.bu_minus_fusions is not None:
            modules += list(self.bu_minus_fusions)
        return [m.effective_alpha() for m in modules if m.use_polarity]

class DynamicBiFPN(nn.Module):
    def __init__(
        self, in_channels, out_channels, repeats=3, epsilon=1e-4,
        alpha_max=0.95, alpha_init=0.02, adaptive_weights=True,
        use_polarity=True, use_softplus=True, use_positive_normalization=True,
        dual_bottom_up=True, share_bottom_up=True,
        use_polarity_regularization=True,
    ):
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError("Expected three backbone features for P3-P5")
        self.use_polarity_regularization = bool(use_polarity_regularization)
        self.input_proj = nn.ModuleList([ConvNormAct(c, out_channels, 1, act=False) for c in in_channels])
        self.p6_proj = ConvNormAct(in_channels[-1], out_channels, 1, act=False)
        self.cells = nn.ModuleList([
            DynamicBiFPNCell(
                out_channels, epsilon, alpha_max, alpha_init,
                adaptive_weights=adaptive_weights,
                use_polarity=use_polarity,
                use_softplus=use_softplus,
                use_positive_normalization=use_positive_normalization,
                dual_bottom_up=dual_bottom_up,
                share_bottom_up=share_bottom_up,
            ) for _ in range(repeats)
        ])

    def _prepare_pyramid(self, feats):
        p3, p4, p5 = [proj(x) for proj, x in zip(self.input_proj, feats)]
        p6 = F.max_pool2d(self.p6_proj(feats[-1]), 3, 2, 1)
        p7 = F.max_pool2d(p6, 3, 2, 1)
        return [p3, p4, p5, p6, p7]

    def forward(self, backbone_features, return_states=False):
        pyramid = self._prepare_pyramid(backbone_features)
        states = []
        for cell in self.cells:
            if return_states:
                pyramid, state = cell(pyramid, True)
                states.append(state)
            else:
                pyramid = cell(pyramid, False)
        return (pyramid, states) if return_states else pyramid

    def polarity_regularizer(self):
        if not self.use_polarity_regularization:
            return next(self.parameters()).new_zeros(())
        vals = []
        for cell in self.cells:
            vals.extend(cell.polarity_values())
        if not vals:
            return next(self.parameters()).new_zeros(())
        return torch.cat([v.reshape(-1) for v in vals]).pow(2).mean()
