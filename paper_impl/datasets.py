from __future__ import annotations

import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class DetectionResizeNormalize:
    """Traceable detector-side preprocessing only.

    The implementation performs RGB conversion, deterministic resize, tensor
    conversion, and configurable normalization. It does not reconstruct or
    claim undocumented upstream MRI preprocessing. Optional horizontal flipping
    is disabled by default and must be enabled explicitly.
    """

    def __init__(self, size: int, mean, std, hflip_prob: float = 0.0, training: bool = False):
        self.size = int(size)
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.hflip_prob = float(hflip_prob)
        self.training = bool(training)

    def __call__(self, image: Image.Image, target: Dict[str, torch.Tensor]):
        image = image.convert("RGB")
        ow, oh = image.size
        image = TF.resize(image, [self.size, self.size], antialias=True)
        boxes = target["boxes"].clone().float()
        if boxes.numel():
            boxes[:, [0, 2]] *= self.size / float(ow)
            boxes[:, [1, 3]] *= self.size / float(oh)
        if self.training and self.hflip_prob > 0 and random.random() < self.hflip_prob:
            image = TF.hflip(image)
            if boxes.numel():
                x1 = boxes[:, 0].clone()
                x2 = boxes[:, 2].clone()
                boxes[:, 0] = self.size - x2
                boxes[:, 2] = self.size - x1
        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.std)
        target = dict(target)
        target["boxes"] = boxes
        target["orig_size"] = torch.tensor([oh, ow], dtype=torch.long)
        target["size"] = torch.tensor([self.size, self.size], dtype=torch.long)
        return image, target


class VocXmlDetectionDataset(Dataset):
    """Prepared-image + Pascal/VOC XML dataset for detector artifacts."""

    def __init__(self, image_dir, annotation_dir, class_names: Sequence[str], transform=None, split_file=None):
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.class_to_id = {name: i + 1 for i, name in enumerate(class_names)}
        self.transform = transform
        if split_file:
            self.ids = [x.strip() for x in Path(split_file).read_text(encoding="utf-8").splitlines() if x.strip()]
        else:
            self.ids = sorted(p.stem for p in self.annotation_dir.glob("*.xml"))
        if not self.ids:
            raise RuntimeError("No annotation IDs found")

    def __len__(self):
        return len(self.ids)

    def _image_path(self, stem: str) -> Path:
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            p = self.image_dir / f"{stem}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"Image for {stem} not found in {self.image_dir}")

    def __getitem__(self, index: int):
        stem = self.ids[index]
        root = ET.parse(self.annotation_dir / f"{stem}.xml").getroot()
        boxes: List[List[float]] = []
        labels: List[int] = []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            if name not in self.class_to_id:
                continue
            b = obj.find("bndbox")
            if b is None:
                continue
            xmin = float(b.findtext("xmin"))
            ymin = float(b.findtext("ymin"))
            xmax = float(b.findtext("xmax"))
            ymax = float(b.findtext("ymax"))
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_id[name])
        image = Image.open(self._image_path(stem))
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.long),
            "image_id": torch.tensor(index, dtype=torch.long),
            "stem": stem,
        }
        if self.transform:
            image, target = self.transform(image, target)
        return image, target


class CocoJsonDetectionDataset(Dataset):
    """Minimal COCO detection reader with no hidden augmentation."""

    def __init__(self, image_dir, annotation_json, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        data = json.loads(Path(annotation_json).read_text(encoding="utf-8"))
        self.images = {int(x["id"]): x for x in data["images"]}
        self.ids = sorted(self.images)
        self.anns = {i: [] for i in self.ids}
        for ann in data.get("annotations", []):
            if int(ann.get("iscrowd", 0)):
                continue
            self.anns.setdefault(int(ann["image_id"]), []).append(ann)
        cat_ids = sorted(int(c["id"]) for c in data.get("categories", []))
        self.cat_to_contiguous = {cid: i + 1 for i, cid in enumerate(cat_ids)}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index: int):
        image_id = self.ids[index]
        info = self.images[image_id]
        image = Image.open(self.image_dir / info["file_name"])
        boxes, labels = [], []
        for ann in self.anns.get(image_id, []):
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_to_contiguous[int(ann["category_id"])])
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.long),
            "image_id": torch.tensor(image_id, dtype=torch.long),
        }
        if self.transform:
            image, target = self.transform(image, target)
        return image, target


def detection_collate(batch):
    images, targets = zip(*batch)
    return torch.stack(list(images), dim=0), list(targets)
