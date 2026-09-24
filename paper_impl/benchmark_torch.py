from __future__ import annotations
import argparse,json,statistics,time
import torch
from .config import ExperimentConfig
from .model import DynamicEfficientDet

def main():
    p=argparse.ArgumentParser(description="Benchmark PyTorch CUDA latency; not TensorRT")
    p.add_argument("--config",required=True); p.add_argument("--checkpoint",required=True); p.add_argument("--warmup",type=int,default=30); p.add_argument("--runs",type=int,default=100); p.add_argument("--fp16",action="store_true")
    args=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU is required")
    cfg=ExperimentConfig.from_yaml(args.config); device=torch.device("cuda")
    model=DynamicEfficientDet(cfg.model).to(device).eval(); ckpt=torch.load(args.checkpoint,map_location="cpu"); model.load_state_dict(ckpt["model"],strict=True)
    dtype=torch.float16 if args.fp16 else torch.float32
    if args.fp16: model.half()
    x=torch.zeros(1,3,cfg.model.image_size,cfg.model.image_size,device=device,dtype=dtype)
    with torch.no_grad():
        for _ in range(args.warmup): model(x)
        torch.cuda.synchronize(); times=[]
        for _ in range(args.runs):
            t0=time.perf_counter(); model(x); torch.cuda.synchronize(); times.append((time.perf_counter()-t0)*1000)
    median=statistics.median(times)
    print(json.dumps({"backend":"PyTorch CUDA","precision":"FP16" if args.fp16 else "FP32","runs":args.runs,"mean_ms":statistics.mean(times),"median_ms":median,"stdev_ms":statistics.pstdev(times),"fps_from_median":1000/median,"note":"Do not report these values as TensorRT latency."},indent=2))

if __name__=="__main__": main()
