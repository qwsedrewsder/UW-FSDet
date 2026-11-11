"""Evaluate a trained DINO checkpoint on the untouched DUO test split."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/root/duo_exp")
sys.path.insert(0, "/root/autodl-tmp/DINO")
sys.path.insert(0, "/root/autodl-tmp/htdet")
from main import get_args_parser, build_model_main
from util.slconfig import SLConfig
from datasets.coco import build as build_coco_dataset
from htdet.engine.evaluator import coco_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    args_cli = parser.parse_args()
    run = ROOT / "runs" / f"dino_seed{args_cli.seed}"
    sys.argv = ["eval", "-c", "config/DINO/urpc_4scale_416.py", "--coco_path",
                str(ROOT / "data/dino_test")]
    args = get_args_parser().parse_args()
    cfg = SLConfig.fromfile(args.config_file)
    for key, value in cfg._cfg_dict.to_dict().items():
        if key not in vars(args):
            setattr(args, key, value)
    args.use_ema = False
    model, _, postprocessors = build_model_main(args)
    checkpoint = torch.load(run / "checkpoint.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.cuda().eval()
    dataset = build_coco_dataset("val", args)
    detections = []
    with torch.no_grad():
        for index in range(len(dataset)):
            image, target = dataset[index]
            output = model(image[None].cuda())
            orig_size = target["orig_size"].unsqueeze(0).cuda()
            result = postprocessors["bbox"](output, orig_size)[0]
            boxes, scores, labels = result["boxes"].cpu(), result["scores"].cpu(), result["labels"].cpu()
            keep = scores >= 0.05
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            if len(scores):
                order = torch.argsort(scores, descending=True)[:100]
                detections.append((int(target["image_id"]), boxes[order], scores[order], labels[order]))
    ann = ROOT / "data/annotations/instances_test.json"
    metrics = coco_eval(detections, str(ann))
    (run / "test_eval.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
