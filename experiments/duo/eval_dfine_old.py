"""Eval dumped D-FINE predictions with project pycocotools pipeline.
Usage: python eval_dfine_pred.py [--pred preds.json] --out out.json
"""
import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/htdet")

from htdet.engine.evaluator import coco_eval

ap = argparse.ArgumentParser()
ap.add_argument("--pred", default="/root/autodl-tmp/work_dirs/dfine_test_preds.json")
ap.add_argument("--out", required=True)
args = ap.parse_args()

preds = json.load(open(args.pred))
detections = []
for p in preds:
    boxes = np.array(p["boxes"], dtype=np.float32)
    scores = np.array(p["scores"], dtype=np.float32)
    labels = np.array(p["labels"], dtype=np.int64)
    if len(boxes) == 0:
        continue
    m = scores >= 0.05
    boxes, scores, labels = boxes[m], scores[m], labels[m]
    keep = np.argsort(scores)[::-1][:100]
    detections.append((p["image_id"], torch.tensor(boxes[keep]).float(),
                       torch.tensor(scores[keep]).float(), torch.tensor(labels[keep]).long()))

ev = coco_eval(detections, "/root/autodl-tmp/data/annotations/test_clean.json")
print(json.dumps(ev, indent=2))
json.dump(ev, open(args.out, "w"), indent=2)
