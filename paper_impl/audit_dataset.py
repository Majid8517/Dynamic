from __future__ import annotations
import argparse, json, xml.etree.ElementTree as ET
from pathlib import Path

def read_ids(path):
    return [x.strip() for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]

def count_boxes(annotation_dir, ids):
    total=0; missing=[]
    for stem in ids:
        p=Path(annotation_dir)/f"{stem}.xml"
        if not p.exists():
            missing.append(stem); continue
        root=ET.parse(p).getroot()
        for obj in root.findall("object"):
            b=obj.find("bndbox")
            if b is None: continue
            xmin=float(b.findtext("xmin")); ymin=float(b.findtext("ymin")); xmax=float(b.findtext("xmax")); ymax=float(b.findtext("ymax"))
            if xmax>xmin and ymax>ymin: total+=1
    return total,missing

def main():
    p=argparse.ArgumentParser(description="Audit explicit image-level train/eval split provenance")
    p.add_argument("--train-ids",required=True); p.add_argument("--eval-ids",required=True); p.add_argument("--annotation-dir",required=True)
    args=p.parse_args()
    train=read_ids(args.train_ids); ev=read_ids(args.eval_ids)
    overlap=sorted(set(train)&set(ev))
    tb,tm=count_boxes(args.annotation_dir,train); eb,em=count_boxes(args.annotation_dir,ev)
    result={"train_images":len(train),"eval_images":len(ev),"unique_train":len(set(train)),"unique_eval":len(set(ev)),"overlap_count":len(overlap),"overlap_ids":overlap[:50],"train_boxes":tb,"eval_boxes":eb,"missing_train_xml":tm[:50],"missing_eval_xml":em[:50],"split_basis":"image-level IDs only; patient independence is not inferred"}
    print(json.dumps(result,indent=2))
    if overlap or tm or em: raise SystemExit(2)

if __name__=="__main__": main()
