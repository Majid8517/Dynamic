from __future__ import annotations
import torch
import torch.nn.functional as F
from .anchors import assign_targets, generate_anchors

def _flatten_head(outputs, outputs_per_anchor):
    flat=[]
    for out in outputs:
        b,c,h,w=out.shape
        A=c//outputs_per_anchor
        flat.append(out.view(b,A,outputs_per_anchor,h,w).permute(0,3,4,1,2).reshape(b,-1,outputs_per_anchor))
    return flat

def focal_loss_with_ignore(logits, labels, num_classes, alpha=0.25, gamma=1.5):
    valid=labels>=0; pos=labels>0
    target=logits.new_zeros(logits.shape)
    if pos.any(): target[pos,labels[pos]-1]=1.0
    ce=F.binary_cross_entropy_with_logits(logits,target,reduction="none")
    prob=torch.sigmoid(logits); pt=prob*target+(1-prob)*(1-target)
    at=alpha*target+(1-alpha)*(1-target)
    return (at*(1-pt).pow(gamma)*ce*valid[:,None]).sum()/pos.sum().clamp(min=1).float()

def smooth_l1_box_loss(pred,target,positive,beta=0.1):
    if not positive.any(): return pred.sum()*0.0
    d=(pred[positive]-target[positive]).abs()
    l=torch.where(d<beta,0.5*d.pow(2)/beta,d-0.5*beta)
    return l.sum()/positive.sum().clamp(min=1).float()

def topology_distillation_loss(student_states,teacher_states,intermediate_weight=0.5,final_weight=0.5):
    s=student_states[-1]; t=teacher_states[-1]
    def stage(sf,tf):
        vals=[]
        for a,b in zip(sf,tf):
            av=F.normalize(a.flatten(1),p=2,dim=1,eps=1e-8)
            bv=F.normalize(b.detach().flatten(1),p=2,dim=1,eps=1e-8)
            vals.append(F.mse_loss(av,bv))
        return torch.stack(vals).mean()
    return intermediate_weight*stage(s.bu_plus,t.bu_plus)+final_weight*stage(s.bu_minus,t.bu_minus)

def detection_loss(cls_outputs,box_outputs,targets,image_size,num_classes,num_scales,aspect_ratios,anchor_scale,alpha=0.25,gamma=1.5,box_beta=0.1,box_weight=50.0):
    cls_l=_flatten_head(cls_outputs,num_classes); box_l=_flatten_head(box_outputs,4)
    shapes=[(o.shape[2],o.shape[3]) for o in cls_outputs]
    anchors=torch.cat(generate_anchors((image_size,image_size),shapes,num_scales,aspect_ratios,anchor_scale,cls_outputs[0].device),0)
    cp=torch.cat(cls_l,1); bp=torch.cat(box_l,1)
    cl=[]; bl=[]
    for b,t in enumerate(targets):
        gb=t["boxes"].to(anchors.device,dtype=anchors.dtype); gl=t["labels"].to(anchors.device,dtype=torch.long)
        labels,bt=assign_targets(anchors,gb,gl)
        cl.append(focal_loss_with_ignore(cp[b],labels,num_classes,alpha,gamma))
        bl.append(smooth_l1_box_loss(bp[b],bt,labels>0,box_beta))
    cls_loss=torch.stack(cl).mean(); box_loss=torch.stack(bl).mean()
    return {"detection":cls_loss+box_weight*box_loss,"classification":cls_loss,"box":box_loss,"anchors":anchors}
