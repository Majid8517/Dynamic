from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from .config import ExperimentConfig
from .datasets import CocoJsonDetectionDataset, DetectionResizeNormalize, VocXmlDetectionDataset, detection_collate
from .ema import EMATeacher
from .engine import append_jsonl, build_optimizer, build_scheduler, save_checkpoint, set_reproducible, train_one_epoch
from .model import DynamicEfficientDet

def build_dataset(cfg, training=True):
    transform=DetectionResizeNormalize(cfg.model.image_size,cfg.data.mean,cfg.data.std,cfg.data.horizontal_flip_prob,training)
    if cfg.data.kind.lower()=="voc":
        return VocXmlDetectionDataset(cfg.data.image_dir,cfg.data.annotation_dir,cfg.data.class_names,transform,cfg.data.split_file or None)
    if cfg.data.kind.lower()=="coco":
        return CocoJsonDetectionDataset(cfg.data.image_dir,cfg.data.coco_json,transform)
    raise ValueError("data.kind must be voc or coco")

def main():
    p=argparse.ArgumentParser(description="Train paper-aligned Stability-Aware Dynamic BiFPN")
    p.add_argument("--config",required=True); p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args=p.parse_args(); cfg=ExperimentConfig.from_yaml(args.config)
    set_reproducible(cfg.train.seed,cfg.train.deterministic); device=torch.device(args.device)
    out=Path(cfg.output_dir); out.mkdir(parents=True,exist_ok=True); cfg.dump_yaml(out/"resolved_config.yaml")
    ds=build_dataset(cfg,True)
    loader=DataLoader(ds,batch_size=cfg.train.batch_size,shuffle=True,num_workers=cfg.train.num_workers,pin_memory=device.type=="cuda",collate_fn=detection_collate)
    model=DynamicEfficientDet(cfg.model).to(device)
    teacher=EMATeacher(model,rho=cfg.model.dynamic.ema_rho).to(device)
    opt=build_optimizer(model,cfg.train)
    sched=build_scheduler(opt,cfg.train.epochs,cfg.train.warmup_epochs,cfg.train.lr,cfg.train.min_lr)
    scaler=GradScaler(enabled=bool(cfg.train.amp and device.type=="cuda"))
    for epoch in range(cfg.train.epochs):
        metrics=train_one_epoch(model,teacher,loader,opt,scaler,cfg,device,epoch)
        sched.step(); rec={"epoch":epoch+1,"lr":opt.param_groups[0]["lr"],**metrics}
        print(rec); append_jsonl(out/"train_log.jsonl",rec)
        if (epoch+1)%cfg.train.save_every==0 or epoch+1==cfg.train.epochs:
            save_checkpoint(out/f"checkpoint_epoch_{epoch+1:03d}.pth",model,teacher,opt,sched,scaler,epoch+1,cfg,metrics)

if __name__=="__main__": main()
