"""Evaluate trained RF-DETR-Nano on test2018 with the project's own
pycocotools evaluator (score_thr 0.05, top-100 per image) for a strictly
comparable AP/AP50/AP75/APs/APm/APl vs R2/C1/D1/R3."""
import json
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/htdet")

from htdet.engine.evaluator import coco_eval

CKPT = "/root/autodl-tmp/work_dirs/rfdetr_nano_urpc/checkpoint_best_total.pth"
ANN = "/root/autodl-tmp/data/annotations/test_clean.json"
IMG_DIR = "/root/autodl-tmp/data/urpc_coco/test2018/image"

from rfdetr import RFDETR

model = RFDETR.from_checkpoint(CKPT)

with open(ANN) as f:
    data = json.load(f)
name2id = {im["file_name"]: im["id"] for im in data["images"]}

files = sorted(name2id.keys())
detections = []
BS = 16
for i in range(0, len(files), BS):
    batch = [os.path.join(IMG_DIR, f) for f in files[i:i + BS]]
    outs = model.predict(batch, threshold=0.05)
    if not isinstance(outs, list):
        outs = [outs]
    for f, det in zip(files[i:i + BS], outs):
        if len(det) == 0:
            continue
        keep = np.argsort(det.confidence)[::-1][:100]
        boxes = det.xyxy[keep]
        scores = det.confidence[keep]
        labels = det.class_id[keep]
        detections.append((name2id[f], torch.tensor(boxes).float(),
                           torch.tensor(scores).float(), torch.tensor(labels).long()))

ev = coco_eval(detections, ANN)
print(json.dumps(ev, indent=2))
with open("/root/autodl-tmp/work_dirs/rfdetr_nano_urpc/eval_pycoco.json", "w") as f:
    json.dump(ev, f, indent=2)
