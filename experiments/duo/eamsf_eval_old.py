"""Eval EAMSF-DETR (URPC 416, no-scale) on test2018 with the project protocol:
conf 0.05 / NMS 0.5 / top-100 per image, pycocotools via htdet.engine.evaluator.coco_eval.

Defaults to testing the FINAL checkpoint (last.pt) with EMA weights (official convention).
Set EAMSF_RAW=1 to eval raw weights; EAMSF_CKPT / EAMSF_OUT to override paths;
EAMSF_LIMIT for a quick smoke test.
"""
import functools
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/EAMSF-DETR")
sys.path.insert(0, "/root/autodl-tmp/htdet")

torch.load = functools.partial(torch.load, weights_only=False)  # fork predates torch 2.6 default

from ultralytics import RTDETR
from htdet.engine.evaluator import coco_eval

CKPT = os.environ.get("EAMSF_CKPT", "/root/autodl-tmp/work_dirs/eamsf_detr_urpc_416_noscale/weights/last.pt")
ANN = "/root/autodl-tmp/data/annotations/test_clean.json"
IMG_DIR = "/root/autodl-tmp/data/urpc_coco/test2018/image"
OUT = os.environ.get("EAMSF_OUT", "/root/autodl-tmp/work_dirs/eamsf_detr_urpc_416_noscale/eval_pycoco_last_ema.json")
RAW = os.environ.get("EAMSF_RAW", "0") == "1"
LIMIT = int(os.environ.get("EAMSF_LIMIT", "0"))


def main():
    m = RTDETR(CKPT)
    ck = torch.load(CKPT, map_location="cpu")
    if RAW:
        sd = ck.get("model")
        if isinstance(sd, torch.nn.Module):
            sd = sd.state_dict()
        m.model.load_state_dict(sd)
        print("loaded RAW weights | epoch:", ck.get("epoch"))
    else:
        print("loaded EMA weights (default) | epoch:", ck.get("epoch"))
    m.model.cuda().eval()

    data = json.load(open(ANN))
    name2id = {im["file_name"]: im["id"] for im in data["images"]}
    files = sorted(name2id.keys())
    if LIMIT:
        files = files[:LIMIT]

    detections = []
    for i in range(0, len(files), 32):
        batch = [os.path.join(IMG_DIR, f) for f in files[i:i + 32]]
        outs = m.predict(batch, imgsz=416, conf=0.05, iou=0.5, verbose=False)
        for f, out in zip(files[i:i + 32], outs):
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
