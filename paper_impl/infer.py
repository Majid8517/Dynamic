from __future__ import annotations
import argparse
import torch
from PIL import Image,ImageDraw
from torchvision.transforms import functional as TF
from .anchors import generate_anchors
from .config import ExperimentConfig
from .metrics import postprocess_predictions
from .model import DynamicEfficientDet

def main():
    p=argparse.ArgumentParser(description="Single-image Dynamic BiFPN inference")
    p.add_argument("--config",required=True); p.add_argument("--checkpoint",required=True); p.add_argument("--image",required=True)
    p.add_argument("--output",default="prediction.png"); p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args=p.parse_args(); cfg=ExperimentConfig.from_yaml(args.config); device=torch.device(args.device)
    model=DynamicEfficientDet(cfg.model).to(device).eval(); ckpt=torch.load(args.checkpoint,map_location="cpu"); model.load_state_dict(ckpt["model"],strict=True)
    image=Image.open(args.image).convert("RGB"); orig=image.copy(); ow,oh=image.size
    x=TF.resize(image,[cfg.model.image_size,cfg.model.image_size],antialias=True)
    x=TF.normalize(TF.to_tensor(x),cfg.data.mean,cfg.data.std)[None].to(device)
    with torch.no_grad(): cls,box=model(x)
    shapes=[(o.shape[2],o.shape[3]) for o in cls]
    anchors=torch.cat(generate_anchors((cfg.model.image_size,cfg.model.image_size),shapes,cfg.model.num_scales,cfg.model.aspect_ratios,cfg.model.anchor_scale,device),0)
    pred=postprocess_predictions(cls,box,anchors,cfg.model.num_classes,cfg.eval.score_threshold,cfg.eval.nms_threshold,cfg.eval.max_detections)[0]
    draw=ImageDraw.Draw(orig); sx=ow/cfg.model.image_size; sy=oh/cfg.model.image_size
    for b,s,l in zip(pred["boxes"].cpu(),pred["scores"].cpu(),pred["labels"].cpu()):
        x1,y1,x2,y2=float(b[0])*sx,float(b[1])*sy,float(b[2])*sx,float(b[3])*sy
        draw.rectangle([x1,y1,x2,y2],width=2); draw.text((x1,y1),f"{int(l)}:{float(s):.3f}")
    orig.save(args.output); print(f"saved {args.output}")

if __name__=="__main__": main()
