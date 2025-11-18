import torch
from torchvision.ops import nms

def postprocess(boxes, scores, score_thresh=0.3, iou_thresh=0.5):
    scores, labels = torch.max(scores, dim=-1)
    mask = scores > score_thresh

    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    keep = nms(boxes, scores, iou_thresh)
    return boxes[keep], scores[keep], labels[keep]
