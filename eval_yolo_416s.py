"""Evaluate YOLOv8s (416 hard-resize recipe) on test2018.

Images are pre-stretched to 416x416; predictions are rescaled back to the
original image coordinates before pycocotools evaluation, so the protocol
matches the project (conf 0.05, NMS 0.5, top-100).
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/htdet")

from htdet.engine.evaluator import coco_eval
from ultralytics import YOLO

CKPT = "/root/autodl-tmp/work_dirs/yolov8s_urpc_paper_imnet_decay/weights/last.pt"
ANN = "/root/autodl-tmp/data/annotations/test_clean.json"
IMG_DIR = "/root/autodl-tmp/data/yolo_urpc_416s/test/images"   # 416x416 distorted

model = YOLO(CKPT)

with open(ANN) as f:
    data = json.load(f)
meta = {im["file_name"]: (im["id"], im["width"], im["height"]) for im in data["images"]}

files = sorted(meta.keys())
detections = []
for i in range(0, len(files), 64):
    batch = [os.path.join(IMG_DIR, f) for f in files[i:i + 64]]
    outs = model.predict(batch, imgsz=416, conf=0.05, iou=0.5, verbose=False)
    for f, out in zip(files[i:i + 64], outs):
        img_id, W, H = meta[f]
        if out.boxes is None or len(out.boxes) == 0:
            continue
        boxes = out.boxes.xyxy.cpu().numpy()
        scores = out.boxes.conf.cpu().numpy()
        labels = out.boxes.cls.cpu().numpy().astype(np.int64)
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / 416.0 * W
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / 416.0 * H
        keep = np.argsort(scores)[::-1][:100]
        detections.append((img_id, torch.tensor(boxes[keep]).float(),
                           torch.tensor(scores[keep]).float(),
                           torch.tensor(labels[keep]).long()))

ev = coco_eval(detections, ANN)
print(json.dumps(ev, indent=2))
with open("/root/autodl-tmp/work_dirs/yolov8s_urpc_paper_imnet_decay/eval_pycoco.json", "w") as f:
    json.dump(ev, f, indent=2)
