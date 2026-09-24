from __future__ import annotations
import json, math, random, subprocess
from pathlib import Path
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from .anchors import generate_anchors
from .losses import detection_loss, topology_distillation_loss
from .metrics import postprocess_predictions

def set_reproducible(seed, deterministic=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
        try: torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError: pass

def git_commit_or_unknown():
    try: return subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except Exception: return "unknown"

def build_optimizer(model,cfg):
    if cfg.optimizer.lower()=="adam": return torch.optim.Adam(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    if cfg.optimizer.lower()=="adamw": return torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    raise ValueError(f"Unsupported optimizer {cfg.optimizer}")

def build_scheduler(optimizer,epochs,warmup_epochs,base_lr,min_lr):
    def fn(epoch):
        if warmup_epochs>0 and epoch<warmup_epochs: return max((epoch+1)/warmup_epochs,1e-6)
        p=(epoch-warmup_epochs)/max(1,epochs-warmup_epochs)
        cosine=.5*(1+math.cos(math.pi*min(max(p,0.0),1.0)))
        return (min_lr+(base_lr-min_lr)*cosine)/base_lr
    return torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=fn)

def train_one_epoch(model,teacher,loader,optimizer,scaler,exp_cfg,device,epoch):
    model.train(); totals={k:0.0 for k in ["loss","detection","classification","box","distill","polarity"]}; n=0
    for images,targets in loader:
        images=images.to(device,non_blocking=True)
        targets=[{k:(v.to(device) if torch.is_tensor(v) else v) for k,v in t.items()} for t in targets]
        optimizer.zero_grad(set_to_none=True)
        amp_enabled=bool(exp_cfg.train.amp and device.type=="cuda")
        with autocast(enabled=amp_enabled):
            cls_out,box_out,sfeat=model(images,return_features=True)
            det=detection_loss(cls_out,box_out,targets,exp_cfg.model.image_size,exp_cfg.model.num_classes,exp_cfg.model.num_scales,exp_cfg.model.aspect_ratios,exp_cfg.model.anchor_scale)
            with torch.no_grad():
                _,tstates=teacher.model.forward_pyramid(images,return_states=True)
            dist=topology_distillation_loss(sfeat["states"],tstates,exp_cfg.model.dynamic.distill_intermediate_weight,exp_cfg.model.dynamic.distill_final_weight)
            pol=model.polarity_regularizer()
            total=det["detection"]+exp_cfg.model.dynamic.distill_mu*dist+exp_cfg.model.dynamic.polarity_lambda*pol
        scaler.scale(total).backward(); scaler.unscale_(optimizer)
        if exp_cfg.train.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(),exp_cfg.train.grad_clip_norm)
        scaler.step(optimizer); scaler.update(); teacher.update(model)
        vals={"loss":total,"detection":det["detection"],"classification":det["classification"],"box":det["box"],"distill":dist,"polarity":pol}
        for k,v in vals.items(): totals[k]+=float(v.detach())
        n+=1
    return {k:v/max(1,n) for k,v in totals.items()}

@torch.no_grad()
def predict_dataset(model,loader,exp_cfg,device):
    model.eval(); preds=[]; targets_out=[]
    for images,targets in loader:
        images=images.to(device,non_blocking=True); cls_out,box_out=model(images)
        shapes=[(o.shape[2],o.shape[3]) for o in cls_out]
        anchors=torch.cat(generate_anchors((exp_cfg.model.image_size,exp_cfg.model.image_size),shapes,exp_cfg.model.num_scales,exp_cfg.model.aspect_ratios,exp_cfg.model.anchor_scale,device),0)
        batch=postprocess_predictions(cls_out,box_out,anchors,exp_cfg.model.num_classes,exp_cfg.eval.score_threshold,exp_cfg.eval.nms_threshold,exp_cfg.eval.max_detections)
        preds.extend([{k:v.cpu() for k,v in p.items()} for p in batch])
        targets_out.extend([{k:(v.cpu() if torch.is_tensor(v) else v) for k,v in t.items()} for t in targets])
    return preds,targets_out

def save_checkpoint(path,model,teacher,optimizer,scheduler,scaler,epoch,exp_cfg,metrics=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"epoch":epoch,"model":model.state_dict(),"teacher":teacher.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict() if scheduler else None,"scaler":scaler.state_dict() if scaler else None,"config":exp_cfg.to_dict(),"git_commit":git_commit_or_unknown(),"metrics":metrics or {},"torch_version":torch.__version__},path)

def append_jsonl(path,record):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8") as f: f.write(json.dumps(record,sort_keys=True)+"\n")
