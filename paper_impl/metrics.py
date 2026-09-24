from __future__ import annotations
import numpy as np
import torch
from torchvision.ops import batched_nms
from .anchors import box_iou, decode_boxes

def postprocess_predictions(cls_outputs, box_outputs, anchors, num_classes, score_threshold=0.25, nms_threshold=0.60, max_detections=100):
    bsz = cls_outputs[0].shape[0]
    cls_flat, box_flat = [], []
    for c, b in zip(cls_outputs, box_outputs):
        B, cc, h, w = c.shape
        A = cc // num_classes
        cls_flat.append(c.view(B,A,num_classes,h,w).permute(0,3,4,1,2).reshape(B,-1,num_classes))
        box_flat.append(b.view(B,A,4,h,w).permute(0,3,4,1,2).reshape(B,-1,4))
    cls_flat = torch.cat(cls_flat,1); box_flat = torch.cat(box_flat,1)
    outputs = []
    for i in range(bsz):
        probs = cls_flat[i].sigmoid()
        scores, labels0 = probs.max(1)
        keep = scores >= score_threshold
        if not keep.any():
            outputs.append({"boxes":anchors.new_zeros((0,4)),"scores":anchors.new_zeros((0,)),"labels":torch.zeros(0,dtype=torch.long,device=anchors.device)})
            continue
        scores=scores[keep]; labels=labels0[keep]+1
        boxes=decode_boxes(box_flat[i][keep],anchors[keep])
        k=batched_nms(boxes,scores,labels,nms_threshold)[:max_detections]
        outputs.append({"boxes":boxes[k],"scores":scores[k],"labels":labels[k]})
    return outputs

def voc_ap(rec, prec):
    mrec=np.concatenate(([0.0],rec,[1.0])); mpre=np.concatenate(([0.0],prec,[0.0]))
    for i in range(mpre.size-1,0,-1): mpre[i-1]=max(mpre[i-1],mpre[i])
    idx=np.where(mrec[1:]!=mrec[:-1])[0]
    return float(np.sum((mrec[idx+1]-mrec[idx])*mpre[idx+1]))

def evaluate_voc50(predictions, targets, num_classes):
    """Single-threshold AP@0.50. Deliberately not labeled COCO mAP."""
    aps=[]
    for cls in range(1,num_classes+1):
        gt_by_image={}; npos=0
        for idx,t in enumerate(targets):
            mask=t["labels"]==cls; boxes=t["boxes"][mask].cpu()
            gt_by_image[idx]={"boxes":boxes,"used":torch.zeros(len(boxes),dtype=torch.bool)}; npos+=len(boxes)
        dets=[]
        for idx,p in enumerate(predictions):
            mask=p["labels"]==cls
            for box,score in zip(p["boxes"][mask].cpu(),p["scores"][mask].cpu()):
                dets.append((float(score),idx,box))
        dets.sort(key=lambda x:x[0],reverse=True)
        tp=np.zeros(len(dets)); fp=np.zeros(len(dets))
        for j,(_,idx,box) in enumerate(dets):
            gt=gt_by_image[idx]
            if len(gt["boxes"])==0: fp[j]=1; continue
            ious=box_iou(box[None],gt["boxes"])[0]; k=int(torch.argmax(ious))
            if float(ious[k])>=0.5 and not bool(gt["used"][k]):
                tp[j]=1; gt["used"][k]=True
            else: fp[j]=1
        if npos==0: continue
        tp=np.cumsum(tp); fp=np.cumsum(fp)
        rec=tp/max(npos,1); prec=tp/np.maximum(tp+fp,np.finfo(np.float64).eps)
        aps.append(voc_ap(rec,prec))
    return {"metric_basis":"VOC-style AP@0.50","AP@0.50":float(np.mean(aps) if aps else 0.0)}

def evaluate_coco(predictions, targets, image_size, num_classes):
    """COCO AP@[0.50:0.95] using pycocotools with explicit metric labels."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise ImportError("COCO evaluation requires pycocotools") from exc
    images=[]; anns=[]; cats=[{"id":c,"name":str(c)} for c in range(1,num_classes+1)]; results=[]; ann_id=1
    for idx,(p,t) in enumerate(zip(predictions,targets)):
        image_id=int(t.get("image_id",torch.tensor(idx)).item())
        images.append({"id":image_id,"width":image_size,"height":image_size})
        for box,label in zip(t["boxes"].cpu(),t["labels"].cpu()):
            x1,y1,x2,y2=[float(v) for v in box]
            anns.append({"id":ann_id,"image_id":image_id,"category_id":int(label),"bbox":[x1,y1,x2-x1,y2-y1],"area":max(0,x2-x1)*max(0,y2-y1),"iscrowd":0}); ann_id+=1
        for box,score,label in zip(p["boxes"].cpu(),p["scores"].cpu(),p["labels"].cpu()):
            x1,y1,x2,y2=[float(v) for v in box]
            results.append({"image_id":image_id,"category_id":int(label),"bbox":[x1,y1,x2-x1,y2-y1],"score":float(score)})
    gt=COCO(); gt.dataset={"images":images,"annotations":anns,"categories":cats}; gt.createIndex()
    dt=gt.loadRes(results)
    ev=COCOeval(gt,dt,"bbox"); ev.params.imgIds=[x["id"] for x in images]
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return {"metric_basis":"COCO-style AP@[0.50:0.95]","mAP@[0.50:0.95]":float(ev.stats[0]),"AP@0.50":float(ev.stats[1]),"AP@0.75":float(ev.stats[2])}
