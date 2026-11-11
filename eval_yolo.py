"""Evaluate YOLOv8n URPC with the project's pycocotools pipeline
(conf 0.05, NMS 0.5, top-100 per image) for strict comparability with R2/C1/D1/R3.
Uses last.pt (final epoch) + imgsz 416 to match the project convention."""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/htdet")

from htdet.engine.evaluator import coco_eval
from ultralytics import YOLO

CKPT = "/root/autodl-tmp/work_dirs/yolov8n_urpc_416_50ep/weights/last.pt"
ANN = "/root/autodl-tmp/data/annotations/test_clean.json"
IMG_DIR = "/root/autodl-tmp/data/urpc_coco/test2018/image"

model = YOLO(CKPT)

with open(ANN) as f:
    data = json.load(f)
name2id = {im["file_name"]: im["id"] for im in data["images"]}

files = sorted(name2id.keys())
detections = []
for i in range(0, len(files), 64):
    batch = [os.path.join(IMG_DIR, f) for f in files[i:i + 64]]
    outs = model.predict(batch, imgsz=416, conf=0.05, iou=0.5, verbose=False)
    for f, out in zip(files[i:i + 64], outs):
        if out.boxes is None or len(out.boxes) == 0:
            continue
        boxes = out.boxes.xyxy.cpu().numpy()
        scores = out.boxes.conf.cpu().numpy()
        labels = out.boxes.cls.cpu().numpy().astype(np.int64)
        keep = np.argsort(scores)[::-1][:100]
        detections.append((name2id[f], torch.tensor(boxes[keep]).float(),
                           torch.tensor(scores[keep]).float(),
                           torch.tensor(labels[keep]).long()))

ev = coco_eval(detections, ANN)
print(json.dumps(ev, indent=2))
with open("/root/autodl-tmp/work_dirs/yolov8n_urpc_416_50ep/eval_pycoco.json", "w") as f:
    json.dump(ev, f, indent=2)
