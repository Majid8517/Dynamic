"""Canonical, paper-aligned implementation of Stability-Aware Dynamic BiFPN.

The core fusion and EMA modules can be imported without optional backbone
packages. DynamicEfficientDet is loaded lazily.
"""
from .config import ExperimentConfig, ModelConfig, DynamicBiFPNConfig, TrainConfig
from .dynamic_bifpn import DynamicBiFPN, DynamicBiFPNCell, PolarityAwareFusion
from .ema import EMATeacher

__all__ = [
    "ExperimentConfig", "ModelConfig", "DynamicBiFPNConfig", "TrainConfig",
    "DynamicBiFPN", "DynamicBiFPNCell", "PolarityAwareFusion",
    "EMATeacher", "DynamicEfficientDet",
]

def __getattr__(name):
    if name == "DynamicEfficientDet":
        from .model import DynamicEfficientDet
        return DynamicEfficientDet
    raise AttributeError(name)
