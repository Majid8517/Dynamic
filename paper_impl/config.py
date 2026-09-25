from __future__ import annotations
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple
import yaml

@dataclass
class DynamicBiFPNConfig:
    """Canonical Dynamic BiFPN switches.

    The defaults reproduce the full paper implementation.  The boolean switches
    are intentionally explicit so each mechanism can be disabled in controlled
    ablation configurations without maintaining divergent model copies.
    """
    epsilon: float = 1e-4
    alpha_max: float = 0.95
    alpha_init: float = 0.02
    distill_mu: float = 0.10
    polarity_lambda: float = 1e-3
    ema_rho: float = 0.999
    distill_intermediate_weight: float = 0.5
    distill_final_weight: float = 0.5

    # Reproducible mechanism switches. Full/canonical defaults are all enabled.
    ablation_id: str = "FULL"
    adaptive_weights: bool = True
    use_polarity: bool = True
    use_softplus: bool = True
    use_positive_normalization: bool = True
    dual_bottom_up: bool = True
    share_bottom_up: bool = True
    use_polarity_regularization: bool = True
    use_ema_distillation: bool = True

    def validate(self) -> None:
        if self.epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        if not (0.0 < self.alpha_max < 1.0):
            raise ValueError("alpha_max must be in (0, 1)")
        if not (0.0 <= self.alpha_init < self.alpha_max):
            raise ValueError("alpha_init must satisfy 0 <= alpha_init < alpha_max")
        if not (0.0 <= self.ema_rho < 1.0):
            raise ValueError("ema_rho must be in [0, 1)")
        if self.distill_mu < 0 or self.polarity_lambda < 0:
            raise ValueError("loss weights must be non-negative")
        s = self.distill_intermediate_weight + self.distill_final_weight
        if abs(s - 1.0) > 1e-6:
            raise ValueError("distillation stage weights must sum to 1")
        if self.share_bottom_up and not self.dual_bottom_up:
            raise ValueError("share_bottom_up is meaningful only when dual_bottom_up=True")
        if self.use_polarity_regularization and not self.use_polarity:
            raise ValueError("polarity regularization requires use_polarity=True")

EFFICIENTDET_SCALES: Dict[str, Dict[str, Any]] = {
    "d0": dict(backbone="tf_efficientnet_b0", image_size=512, fpn_channels=64, fpn_repeats=3, head_repeats=3),
    "d1": dict(backbone="tf_efficientnet_b1", image_size=640, fpn_channels=88, fpn_repeats=4, head_repeats=3),
    "d2": dict(backbone="tf_efficientnet_b2", image_size=768, fpn_channels=112, fpn_repeats=5, head_repeats=3),
    "d3": dict(backbone="tf_efficientnet_b3", image_size=896, fpn_channels=160, fpn_repeats=6, head_repeats=4),
    "d4": dict(backbone="tf_efficientnet_b4", image_size=1024, fpn_channels=224, fpn_repeats=7, head_repeats=4),
    "d5": dict(backbone="tf_efficientnet_b5", image_size=1280, fpn_channels=288, fpn_repeats=7, head_repeats=4),
    "d6": dict(backbone="tf_efficientnet_b6", image_size=1280, fpn_channels=384, fpn_repeats=8, head_repeats=5),
    "d7": dict(backbone="tf_efficientnet_b6", image_size=1536, fpn_channels=384, fpn_repeats=8, head_repeats=5),
}

@dataclass
class ModelConfig:
    variant: str = "d0"
    num_classes: int = 1
    image_size: int | None = None
    backbone: str | None = None
    pretrained_backbone: bool = True
    fpn_channels: int | None = None
    fpn_repeats: int | None = None
    head_repeats: int | None = None
    min_level: int = 3
    max_level: int = 7
    num_scales: int = 3
    aspect_ratios: Tuple[float, ...] = (1.0, 2.0, 0.5)
    anchor_scale: float = 4.0
    class_prior: float = 0.01
    dynamic: DynamicBiFPNConfig = field(default_factory=DynamicBiFPNConfig)

    def resolve(self) -> "ModelConfig":
        key = self.variant.lower().replace("efficientdet-", "").replace("efficientdet_", "")
        if key not in EFFICIENTDET_SCALES:
            raise ValueError(f"Unknown EfficientDet variant: {self.variant}")
        base = EFFICIENTDET_SCALES[key]
        self.variant = key
        self.image_size = int(base["image_size"]) if self.image_size is None else int(self.image_size)
        self.backbone = str(base["backbone"]) if self.backbone is None else self.backbone
        self.fpn_channels = int(base["fpn_channels"]) if self.fpn_channels is None else int(self.fpn_channels)
        self.fpn_repeats = int(base["fpn_repeats"]) if self.fpn_repeats is None else int(self.fpn_repeats)
        self.head_repeats = int(base["head_repeats"]) if self.head_repeats is None else int(self.head_repeats)
        self.dynamic.validate()
        if self.max_level - self.min_level + 1 != 5:
            raise ValueError("This paper implementation expects P3-P7")
        return self

    @property
    def num_anchors(self) -> int:
        return self.num_scales * len(self.aspect_ratios)

@dataclass
class TrainConfig:
    epochs: int = 70
    batch_size: int = 16
    optimizer: str = "adam"
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    grad_clip_norm: float = 10.0
    amp: bool = True
    seed: int = 42
    deterministic: bool = True
    num_workers: int = 4
    save_every: int = 1

@dataclass
class DataConfig:
    kind: str = "voc"
    root: str = ""
    image_dir: str = ""
    annotation_dir: str = ""
    split_file: str = ""
    coco_json: str = ""
    class_names: List[str] = field(default_factory=lambda: ["lesion"])
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    horizontal_flip_prob: float = 0.0

@dataclass
class EvalConfig:
    metric: str = "coco"
    score_threshold: float = 0.25
    nms_threshold: float = 0.60
    max_detections: int = 100

@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output_dir: str = "runs/dynamic_bifpn"

    def resolve(self) -> "ExperimentConfig":
        self.model.resolve()
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        model_raw = dict(raw.get("model", {}))
        dyn = DynamicBiFPNConfig(**model_raw.pop("dynamic", {}))
        return cls(
            model=ModelConfig(dynamic=dyn, **model_raw),
            train=TrainConfig(**raw.get("train", {})),
            data=DataConfig(**raw.get("data", {})),
            eval=EvalConfig(**raw.get("eval", {})),
            output_dir=raw.get("output_dir", "runs/dynamic_bifpn"),
        ).resolve()

    def dump_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
