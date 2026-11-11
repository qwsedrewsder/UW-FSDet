"""Run D-FINE --test-only on test2018, hooking CocoEvaluator.update to dump raw
predictions (original coords), then evaluate with the project pycocotools pipeline
(conf 0.05, top-100) for strict comparability.

Usage:
  torchrun --nproc_per_node=1 eval_dfine.py -c configs/dfine/custom/dfine_hgnetv2_s_urpc.yml \
      -r /root/autodl-tmp/work_dirs/dfine_s_urpc/last.pth
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse

from src.core import YAMLConfig, yaml_utils
from src.misc import dist_utils
from src.solver import TASKS

DUMP = []


def main(args):
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)
    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update(
        {
            k: v
            for k, v in args.__dict__.items()
            if k not in ["update"] and v is not None
        }
    )
    cfg = YAMLConfig(args.config, **update_dict)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    from src.data.dataset.coco_eval import CocoEvaluator

    orig_update = CocoEvaluator.update

    def hook(self, res):
        for img_id, out in res.items():
            DUMP.append({
                "image_id": int(img_id),
                "boxes": out["boxes"].cpu().tolist(),
                "scores": out["scores"].cpu().tolist(),
                "labels": out["labels"].cpu().tolist(),
            })
        return orig_update(self, res)

    CocoEvaluator.update = hook

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.val()

    if dist_utils.get_rank() == 0:
        with open("/root/autodl-tmp/work_dirs/dfine_test_preds.json", "w") as f:
            json.dump(DUMP, f)
        print(f"dumped {len(DUMP)} images -> dfine_test_preds.json")
    dist_utils.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, required=True)
    parser.add_argument("-u", "--update", nargs="+", help="update yaml config")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-method", type=str, default="builtin")
    parser.add_argument("--print-rank", type=int, default=0)
    parser.add_argument("--local-rank", type=int, help="local rank id")
    args = parser.parse_args()
    main(args)
