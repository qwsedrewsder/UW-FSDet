"""EAMSF-DETR on DUO using the existing 416-pixel URPC recipe."""

import functools
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/EAMSF-DETR")
sys.path.insert(0, "/root/autodl-tmp/htdet")
torch.load = functools.partial(torch.load, weights_only=False)
from ultralytics import RTDETR
from htdet.engine.evaluator import coco_eval

ROOT = Path("/root/duo_exp")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    run = ROOT / "runs" / f"eamsf_seed{args.seed}"
    assert (ROOT / "data/manifest.json").exists()
    os.chdir("/root/autodl-tmp/EAMSF-DETR")
    model = RTDETR("/root/autodl-tmp/EAMSF-DETR/ultralytics/cfg/models/myconfig/eamsf-detr.yaml")
    model.train(data=str(ROOT / "data/yolo/data.yaml"), cache=False,
                imgsz=416, epochs=300, batch=32, workers=4, device="0",
                scale=0.0, project=str(run.parent), name=run.name,
                seed=args.seed, exist_ok=True, save_period=-1)
    checkpoint = run / "weights/last.pt"
    assert checkpoint.exists(), checkpoint
    detector = RTDETR(str(checkpoint))
    detector.model.cuda().eval()
    ann = ROOT / "data/annotations/instances_test.json"
    image_dir = ROOT / "data/raw/DUO/images/test"
    data = json.loads(ann.read_text())
    files = sorted(image["file_name"] for image in data["images"])
    image_id = {image["file_name"]: image["id"] for image in data["images"]}
    detections = []
    for start in range(0, len(files), 32):
        current = files[start:start + 32]
        outputs = detector.predict([str(image_dir / f) for f in current], imgsz=416,
                                   conf=0.05, iou=0.5, verbose=False)
        for name, output in zip(current, outputs):
            if output.boxes is None or len(output.boxes) == 0:
                continue
            boxes = output.boxes.xyxy.cpu().numpy()
            scores = output.boxes.conf.cpu().numpy()
            labels = output.boxes.cls.cpu().numpy().astype(np.int64)
            keep = np.argsort(scores)[::-1][:100]
            detections.append((image_id[name], torch.tensor(boxes[keep]).float(),
                               torch.tensor(scores[keep]).float(),
                               torch.tensor(labels[keep]).long()))
    metrics = coco_eval(detections, str(ann))
    (run / "test_eval.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
