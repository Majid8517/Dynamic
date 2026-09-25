import torch
from paper_impl.config import ModelConfig
from paper_impl.model import DynamicEfficientDet

def test_full_model_forward_without_pretrained_download():
    cfg = ModelConfig(
        variant="d0",
        num_classes=1,
        image_size=128,
        pretrained_backbone=False,
        fpn_channels=32,
        fpn_repeats=1,
        head_repeats=1,
    ).resolve()
    model = DynamicEfficientDet(cfg).eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        cls, box, features = model(x, return_features=True)
    assert len(cls) == 5
    assert len(box) == 5
    assert len(features["pyramid"]) == 5
    assert len(features["states"]) == 1
    assert cls[0].shape[0] == 1
    assert box[0].shape[0] == 1


def test_ablation_a0_full_model_forward_without_pretrained_download():
    from paper_impl.config import DynamicBiFPNConfig
    cfg = ModelConfig(
        variant="d0",
        num_classes=1,
        image_size=128,
        pretrained_backbone=False,
        fpn_channels=32,
        fpn_repeats=1,
        head_repeats=1,
        dynamic=DynamicBiFPNConfig(
            ablation_id="A0",
            adaptive_weights=False,
            use_polarity=False,
            use_softplus=False,
            use_positive_normalization=False,
            dual_bottom_up=False,
            share_bottom_up=False,
            use_polarity_regularization=False,
            use_ema_distillation=False,
        ),
    ).resolve()
    model = DynamicEfficientDet(cfg).eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        cls, box, features = model(x, return_features=True)
    assert len(cls) == 5
    assert len(box) == 5
    assert len(features["states"]) == 1
    assert features["states"][0].bu_minus[0] is features["states"][0].bu_plus[0]
