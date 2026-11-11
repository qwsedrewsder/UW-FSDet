"""Eval HyUOD (URPC 416, no-scale, from scratch) on test2018 with project protocol:
conf 0.05 / NMS 0.5 / top-100 per image, pycocotools via htdet.engine.evaluator.coco_eval.

Uses the HyUOD-bundled ultralytics validator (loads 9-ch RGB+t+a inputs via its dataset),
monkey-patches pred_to_json to capture native-coordinate predictions (bypasses COCO id remap).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/autodl-tmp/HyUOD")
sys.path.insert(0, "/root/autodl-tmp/htdet")

from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from htdet.engine.evaluator import coco_eval

CKPT = os.environ.get("HYUOD_CKPT", "/root/autodl-tmp/work_dirs/hyuod_urpc_416_noscale/weights/last.pt")
ANN = "/root/autodl-tmp/data/annotations/test_clean.json"
DATA = "/root/autodl-tmp/data/hyuod_urpc/hyuod_urpc.yaml"
OUT = os.environ.get("HYUOD_OUT", "/root/autodl-tmp/work_dirs/hyuod_urpc_416_noscale/eval_pycoco_last.json")
IMGSZ = int(os.environ.get("HYUOD_IMGSZ", "416"))

captured = []


def patched(self, predn, filename):
    captured.append((Path(filename).stem, predn.detach().cpu()))


DetectionValidator.pred_to_json = patched


def main():
    m = YOLO(CKPT)
    print("evaluating", CKPT)
    m.val(data=DATA, split="test", imgsz=IMGSZ, conf=0.05, iou=0.5, max_det=100, batch=32,
          save_json=True, plots=False, project="/root/autodl-tmp/work_dirs",
          name="hyuod_eval_tmp", exist_ok=True)

    data = json.load(open(ANN))
    stem2id = {os.path.splitext(im["file_name"])[0]: im["id"] for im in data["images"]}

    detections = []
    for stem, pred in captured:
        if stem not in stem2id or pred is None or len(pred) == 0:
            continue
        order = np.argsort(-pred[:, 4].numpy())[:100]
        detections.append((stem2id[stem],
                           pred[order, :4].float(),
                           pred[order, 4].float(),
                           pred[order, 5].long()))

    print("images with dets:", len(detections), "/", len(stem2id))
    ev = coco_eval(detections, ANN)
    print(json.dumps(ev, indent=2))
    json.dump(ev, open(OUT, "w"), indent=2)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
