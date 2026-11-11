"""Evaluate RT-DETR checkpoint on DUO test using COCO boxes and shared score cut."""

import argparse
import json
import sys
import numpy as np
import torch
sys.path.insert(0, "/root/autodl-tmp/RT-DETR/rtdetr_pytorch")
sys.path.insert(0, "/root/autodl-tmp/htdet")
from src.core import YAMLConfig
from src.misc import dist
from htdet.engine.evaluator import coco_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--ann", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    dist.init_distributed()
    cfg = YAMLConfig(args.config)
    model = cfg.model
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ema = state.get("ema")
    sd = ema["module"] if isinstance(ema, dict) and "module" in ema else state["model"]
    model.load_state_dict(sd)
    model.cuda().eval()
    post = cfg.postprocessor
    detections = []
    with torch.no_grad():
        for samples, targets in cfg.val_dataloader:
            outputs = model(samples.cuda())
            sizes = torch.stack([t["orig_size"] for t in targets]).cuda()
            results = post(outputs, sizes)
            for target, result in zip(targets, results):
                boxes = result["boxes"].cpu()
                scores = result["scores"].cpu()
                labels = result["labels"].cpu()
                keep = scores >= 0.05
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                if len(scores):
                    order = torch.argsort(scores, descending=True)[:100]
                    detections.append((int(target["image_id"].item()), boxes[order],
                                       scores[order], labels[order]))
    metrics = coco_eval(detections, args.ann)
    with open(args.out, "w") as stream:
        json.dump(metrics, stream, indent=2)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
