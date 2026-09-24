from __future__ import annotations
import argparse, json
import torch
from torch.utils.data import DataLoader
from .config import ExperimentConfig
from .datasets import DetectionResizeNormalize,VocXmlDetectionDataset,CocoJsonDetectionDataset,detection_collate
from .engine import predict_dataset,set_reproducible
from .metrics import evaluate_coco,evaluate_voc50
from .model import DynamicEfficientDet

def build_dataset(cfg):
    transform=DetectionResizeNormalize(cfg.model.image_size,cfg.data.mean,cfg.data.std,training=False)
    if cfg.data.kind.lower()=="voc":
        return VocXmlDetectionDataset(cfg.data.image_dir,cfg.data.annotation_dir,cfg.data.class_names,transform,cfg.data.split_file or None)
    return CocoJsonDetectionDataset(cfg.data.image_dir,cfg.data.coco_json,transform)

def main():
    p=argparse.ArgumentParser(description="Evaluate with explicit metric semantics")
    p.add_argument("--config",required=True); p.add_argument("--checkpoint",required=True)
    p.add_argument("--metric",choices=["coco","voc50"],default=None)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args=p.parse_args(); cfg=ExperimentConfig.from_yaml(args.config)
    if args.metric: cfg.eval.metric=args.metric
    set_reproducible(cfg.train.seed,cfg.train.deterministic); device=torch.device(args.device)
    ds=build_dataset(cfg)
    loader=DataLoader(ds,batch_size=max(1,min(cfg.train.batch_size,8)),shuffle=False,num_workers=cfg.train.num_workers,collate_fn=detection_collate)
    model=DynamicEfficientDet(cfg.model).to(device)
    ckpt=torch.load(args.checkpoint,map_location="cpu"); model.load_state_dict(ckpt["model"],strict=True)
    preds,targets=predict_dataset(model,loader,cfg,device)
    metrics=evaluate_voc50(preds,targets,cfg.model.num_classes) if cfg.eval.metric=="voc50" else evaluate_coco(preds,targets,cfg.model.image_size,cfg.model.num_classes)
    print(json.dumps(metrics,indent=2))

if __name__=="__main__": main()
