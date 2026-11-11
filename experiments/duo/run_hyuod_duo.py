"""HyUOD on DUO, preserving its 9-channel TA input and 416 recipe."""

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

ROOT = Path("/root/duo_exp")
CAPTURED = []


def capture(self, predn, filename):
    CAPTURED.append((Path(filename).stem, predn.detach().cpu()))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    os.chdir("/root/autodl-tmp/HyUOD")
    run = ROOT / "runs" / f"hyuod_seed{args.seed}"
    model = YOLO("train_yaml/hyuod.yaml")
    model.train(data=str(ROOT / "data/hyuod/data.yaml"),
                project=str(run.parent), name=run.name, epochs=400, batch=16,
                optimizer="SGD", pretrained=False, imgsz=416, scale=0.0,
                iou=0.4, seed=args.seed, exist_ok=True, save_period=-1)
    checkpoint = run / "weights/last.pt"
    assert checkpoint.exists(), checkpoint
    DetectionValidator.pred_to_json = capture
    detector = YOLO(str(checkpoint))
    detector.val(data=str(ROOT / "data/hyuod/data.yaml"), split="test",
                 imgsz=416, conf=0.05, iou=0.5, max_det=100, batch=16,
                 save_json=True, plots=False, project=str(run.parent),
                 name=run.name + "_test", exist_ok=True)
    ann = ROOT / "data/annotations/instances_test.json"
    data = json.loads(ann.read_text())
    stem_to_id = {Path(im["file_name"]).stem: im["id"] for im in data["images"]}
    detections = []
    for stem, pred in CAPTURED:
        if stem not in stem_to_id or pred is None or len(pred) == 0:
            continue
        order = np.argsort(-pred[:, 4].numpy())[:100]
        detections.append((stem_to_id[stem], pred[order, :4].float(),
                           pred[order, 4].float(), pred[order, 5].long()))
    metrics = coco_eval(detections, str(ann))
    (run / "test_eval.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
