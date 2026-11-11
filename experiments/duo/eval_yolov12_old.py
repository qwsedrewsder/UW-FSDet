"""Eval YOLOv12n on URPC test2018 with project protocol (conf 0.05/NMS 0.5/top100).
Loads the RAW weights (ckpt["model"]), NOT the EMA weights, per user request.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/yolov12")
sys.path.insert(0, "/root/autodl-tmp/htdet")

from ultralytics import YOLO
from htdet.engine.evaluator import coco_eval

CKPT = "/root/autodl-tmp/work_dirs/yolov12n_urpc/weights/last.pt"
ANN = "/root/autodl-tmp/data/annotations/test_clean.json"
IMG_DIR = "/root/autodl-tmp/data/urpc_coco/test2018/image"
OUT = "/root/autodl-tmp/work_dirs/yolov12n_urpc/eval_pycoco_raw.json"


def main():
    m = YOLO(CKPT)  # ultralytics loads EMA by default
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    raw = ck.get("model") or ck.get("ema")
    if isinstance(raw, torch.nn.Module):
        raw = raw.state_dict()
    m.model.load_state_dict(raw)  # raw weights (EMA disabled run)
    m.model.cuda().eval()
    print("loaded RAW weights from", CKPT, "| epoch:", ck.get("epoch"))

    data = json.load(open(ANN))
    name2id = {im["file_name"]: im["id"] for im in data["images"]}
    files = sorted(name2id.keys())

    detections = []
    for i in range(0, len(files), 64):
        batch = [os.path.join(IMG_DIR, f) for f in files[i:i + 64]]
        outs = m.predict(batch, imgsz=416, conf=0.05, iou=0.5, verbose=False)
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
    json.dump(ev, open(OUT, "w"), indent=2)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
