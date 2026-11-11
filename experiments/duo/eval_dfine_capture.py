"""Capture native-coordinate D-FINE predictions for the DUO test evaluator."""

import argparse
import json
import os
import sys

sys.path.insert(0, "/root/autodl-tmp/D-FINE")
from src.core import YAMLConfig, yaml_utils
from src.misc import dist_utils
from src.solver import TASKS

DUMP = []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume", required=True)
    parser.add_argument("--pred-out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-method", default="builtin")
    parser.add_argument("--print-rank", type=int, default=0)
    parser.add_argument("--local-rank", type=int)
    args = parser.parse_args()
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)
    updates = yaml_utils.parse_cli([])
    updates.update({k: v for k, v in vars(args).items() if k not in ("pred_out", "config", "seed", "print_method", "print_rank", "local_rank")})
    cfg = YAMLConfig(args.config, **updates)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    from src.data.dataset.coco_eval import CocoEvaluator
    original = CocoEvaluator.update

    def update(self, results):
        for image_id, out in results.items():
            DUMP.append({"image_id": int(image_id), "boxes": out["boxes"].cpu().tolist(),
                         "scores": out["scores"].cpu().tolist(), "labels": out["labels"].cpu().tolist()})
        return original(self, results)

    CocoEvaluator.update = update
    TASKS[cfg.yaml_cfg["task"]](cfg).val()
    if dist_utils.get_rank() == 0:
        with open(args.pred_out, "w") as stream:
            json.dump(DUMP, stream)
    dist_utils.cleanup()


if __name__ == "__main__":
    main()
