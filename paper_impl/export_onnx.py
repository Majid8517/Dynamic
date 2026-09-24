from __future__ import annotations
import argparse
import torch
from .config import ExperimentConfig
from .model import DynamicEfficientDet

class RawDetector(torch.nn.Module):
    """Export raw heads only; NMS remains explicit outside the graph."""
    def __init__(self,model):
        super().__init__(); self.model=model
    def forward(self,x):
        cls,box=self.model(x); return tuple(cls+box)

def main():
    p=argparse.ArgumentParser(description="Export Dynamic BiFPN to ONNX")
    p.add_argument("--config",required=True); p.add_argument("--checkpoint",required=True); p.add_argument("--output",default="dynamic_bifpn.onnx"); p.add_argument("--opset",type=int,default=17)
    args=p.parse_args(); cfg=ExperimentConfig.from_yaml(args.config)
    model=DynamicEfficientDet(cfg.model).eval(); ckpt=torch.load(args.checkpoint,map_location="cpu"); model.load_state_dict(ckpt["model"],strict=True)
    wrapper=RawDetector(model); x=torch.zeros(1,3,cfg.model.image_size,cfg.model.image_size)
    names=[f"cls_p{i}" for i in range(3,8)]+[f"box_p{i}" for i in range(3,8)]
    torch.onnx.export(wrapper,x,args.output,opset_version=args.opset,input_names=["image"],output_names=names,dynamic_axes={"image":{0:"batch"}})
    print(f"exported {args.output}")

if __name__=="__main__": main()
