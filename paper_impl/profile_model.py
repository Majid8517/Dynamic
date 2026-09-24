from __future__ import annotations
import argparse, json
import torch
from .config import ExperimentConfig
from .model import DynamicEfficientDet

def main():
    p=argparse.ArgumentParser(description="Report parameter count; optional FLOPs via fvcore")
    p.add_argument("--config",required=True); p.add_argument("--device",default="cpu")
    args=p.parse_args(); cfg=ExperimentConfig.from_yaml(args.config)
    device=torch.device(args.device); model=DynamicEfficientDet(cfg.model).to(device).eval()
    total=sum(p.numel() for p in model.parameters()); trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    result={"parameters":total,"trainable_parameters":trainable,"parameters_M":total/1e6}
    x=torch.zeros(1,3,cfg.model.image_size,cfg.model.image_size,device=device)
    try:
        from fvcore.nn import FlopCountAnalysis
        flops=FlopCountAnalysis(model,x).total()
        result["FLOPs"]=int(flops); result["GFLOPs"]=flops/1e9; result["flop_backend"]="fvcore"
    except Exception as exc:
        result["GFLOPs"]=None; result["flop_note"]="Install fvcore and rerun for FLOP counting; no FLOP value is inferred."
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
