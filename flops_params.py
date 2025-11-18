
import torch
from thop import profile
from effdet.efficientdet import EfficientDet
from effdet.config import get_efficientdet_config

def load_model():
    config = get_efficientdet_config('efficientdet_d0')#efficientdet_d0 
    config.num_classes = 1
    model = EfficientDet(config)
    return model

model = load_model()
dummy_input = torch.randn(1, 3, 512, 512)
flops, params = profile(model, inputs=(dummy_input,))
print(f" FLOPs: {flops / 1e9:.2f} GFLOPs")
print(f" Params: {params / 1e6:.2f} M")
