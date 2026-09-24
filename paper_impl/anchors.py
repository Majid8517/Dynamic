from __future__ import annotations
import math
import torch

def box_iou(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    a1 = (boxes1[:, 2]-boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3]-boxes1[:, 1]).clamp(min=0)
    a2 = (boxes2[:, 2]-boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3]-boxes2[:, 1]).clamp(min=0)
    return inter / (a1[:, None] + a2[None, :] - inter).clamp(min=1e-8)

def generate_anchors(image_size, feature_shapes, num_scales=3, aspect_ratios=(1.0,2.0,0.5), anchor_scale=4.0, device=None):
    H, W = image_size
    out = []
    for h, w in feature_shapes:
        sy, sx = H/float(h), W/float(w)
        cy = (torch.arange(h, device=device, dtype=torch.float32)+0.5)*sy
        cx = (torch.arange(w, device=device, dtype=torch.float32)+0.5)*sx
        yy, xx = torch.meshgrid(cy, cx, indexing="ij")
        centers = torch.stack([xx, yy], -1).reshape(-1,2)
        base = math.sqrt(sx*sy)*anchor_scale
        sizes = []
        for s in range(num_scales):
            scale = 2.0 ** (s/num_scales)
            for ratio in aspect_ratios:
                rw = math.sqrt(ratio); rh = 1.0/rw
                sizes.append((base*scale*rw, base*scale*rh))
        half = centers.new_tensor([[aw/2, ah/2] for aw,ah in sizes])
        c = centers[:,None,:]
        level = torch.cat([c-half[None,:,:], c+half[None,:,:]], -1)
        out.append(level.reshape(-1,4))
    return out

def encode_boxes(gt, anchors):
    ax=(anchors[:,0]+anchors[:,2])*.5; ay=(anchors[:,1]+anchors[:,3])*.5
    aw=(anchors[:,2]-anchors[:,0]).clamp(min=1e-6); ah=(anchors[:,3]-anchors[:,1]).clamp(min=1e-6)
    gx=(gt[:,0]+gt[:,2])*.5; gy=(gt[:,1]+gt[:,3])*.5
    gw=(gt[:,2]-gt[:,0]).clamp(min=1e-6); gh=(gt[:,3]-gt[:,1]).clamp(min=1e-6)
    return torch.stack([(gx-ax)/aw,(gy-ay)/ah,torch.log(gw/aw),torch.log(gh/ah)],1)

def decode_boxes(delta, anchors):
    ax=(anchors[:,0]+anchors[:,2])*.5; ay=(anchors[:,1]+anchors[:,3])*.5
    aw=(anchors[:,2]-anchors[:,0]).clamp(min=1e-6); ah=(anchors[:,3]-anchors[:,1]).clamp(min=1e-6)
    gx=delta[:,0]*aw+ax; gy=delta[:,1]*ah+ay
    gw=torch.exp(delta[:,2].clamp(max=8))*aw; gh=torch.exp(delta[:,3].clamp(max=8))*ah
    return torch.stack([gx-gw/2,gy-gh/2,gx+gw/2,gy+gh/2],1)

def assign_targets(anchors, gt_boxes, gt_labels, positive_iou=0.5, negative_iou=0.4):
    labels=torch.zeros(len(anchors),dtype=torch.long,device=anchors.device)
    bt=torch.zeros((len(anchors),4),dtype=anchors.dtype,device=anchors.device)
    if gt_boxes.numel()==0: return labels,bt
    ious=box_iou(anchors,gt_boxes); max_iou,match=ious.max(1)
    labels[(max_iou>=negative_iou)&(max_iou<positive_iou)]=-1
    pos=max_iou>=positive_iou
    best=ious.argmax(0); pos[best]=True; match[best]=torch.arange(len(gt_boxes),device=anchors.device)
    labels[pos]=gt_labels[match[pos]]
    bt[pos]=encode_boxes(gt_boxes[match[pos]],anchors[pos])
    return labels,bt
