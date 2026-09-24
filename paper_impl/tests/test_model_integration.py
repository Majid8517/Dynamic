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
