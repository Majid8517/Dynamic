from __future__ import annotations
import argparse
from collections import OrderedDict
from pathlib import Path
from .config import ExperimentConfig

CONFIG_DIR = Path(__file__).resolve().parent / "configs" / "ablations"

ABLATION_CONFIGS = OrderedDict([
    ("A0", "coco_d0_a0_fixed_fusion_reference.yaml"),
    ("A1", "coco_d0_a1_adaptive_weights.yaml"),
    ("A2", "coco_d0_a2_polarity_modulation.yaml"),
    ("A3", "coco_d0_a3_softplus.yaml"),
    ("A4", "coco_d0_a4_positive_normalization.yaml"),
    ("A5", "coco_d0_a5_dual_bottom_up_unshared.yaml"),
    ("A6", "coco_d0_a6_shared_bottom_up.yaml"),
    ("A7", "coco_d0_a7_polarity_regularization.yaml"),
    ("A8", "coco_d0_a8_full_with_ema.yaml"),
])

FLAG_ORDER = [
    "adaptive_weights",
    "use_polarity",
    "use_softplus",
    "use_positive_normalization",
    "dual_bottom_up",
    "share_bottom_up",
    "use_polarity_regularization",
    "use_ema_distillation",
]

EXPECTED = OrderedDict([
    ("A0", (0,0,0,0,0,0,0,0)),
    ("A1", (1,0,0,0,0,0,0,0)),
    ("A2", (1,1,0,0,0,0,0,0)),
    ("A3", (1,1,1,0,0,0,0,0)),
    ("A4", (1,1,1,1,0,0,0,0)),
    ("A5", (1,1,1,1,1,0,0,0)),
    ("A6", (1,1,1,1,1,1,0,0)),
    ("A7", (1,1,1,1,1,1,1,0)),
    ("A8", (1,1,1,1,1,1,1,1)),
])

DESCRIPTIONS = {
    "A0": "fixed-fusion TD->BU reference",
    "A1": "+ learnable adaptive edge weights",
    "A2": "+ polarity-aware modulation",
    "A3": "+ Softplus-positive re-parameterization",
    "A4": "+ epsilon-stabilized positive normalization",
    "A5": "+ complementary BU- pass, unshared parameters",
    "A6": "+ shared BU+/BU- fusion and refinement parameters",
    "A7": "+ polarity-magnitude regularization",
    "A8": "+ training-only EMA topology-aware distillation (canonical full)",
}

def load_variant(variant: str) -> ExperimentConfig:
    key = variant.upper()
    if key not in ABLATION_CONFIGS:
        raise KeyError(f"Unknown ablation variant {variant}. Expected one of {list(ABLATION_CONFIGS)}")
    return ExperimentConfig.from_yaml(CONFIG_DIR / ABLATION_CONFIGS[key])

def flag_tuple(cfg: ExperimentConfig):
    d = cfg.model.dynamic
    return tuple(int(bool(getattr(d, name))) for name in FLAG_ORDER)

def validate_matrix() -> None:
    previous = None
    for key, expected in EXPECTED.items():
        cfg = load_variant(key)
        observed = flag_tuple(cfg)
        if cfg.model.dynamic.ablation_id != key:
            raise AssertionError(f"{key}: ablation_id={cfg.model.dynamic.ablation_id}")
        if observed != expected:
            raise AssertionError(f"{key}: expected {expected}, got {observed}")
        if previous is not None:
            changed = sum(a != b for a,b in zip(previous, observed))
            if changed != 1:
                raise AssertionError(f"{key}: expected exactly one mechanism change from previous variant, got {changed}")
        previous = observed

def main():
    p = argparse.ArgumentParser(description="Inspect/validate the prospective A0-A8 ablation matrix.")
    p.add_argument("--variant", choices=list(ABLATION_CONFIGS), default=None)
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    if args.validate:
        validate_matrix()
        print("A0-A8 ablation matrix validated: one mechanism changes at each step.")
    keys = [args.variant] if args.variant else list(ABLATION_CONFIGS)
    for key in keys:
        cfg = load_variant(key)
        flags = ", ".join(f"{n}={int(v)}" for n,v in zip(FLAG_ORDER, flag_tuple(cfg)))
        print(f"{key}: {DESCRIPTIONS[key]}")
        print(f"  config={CONFIG_DIR / ABLATION_CONFIGS[key]}")
        print(f"  {flags}")

if __name__ == "__main__":
    main()
