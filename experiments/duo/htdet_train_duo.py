"""python tools/train.py --config configs/m1_mvit_fpn.yaml [--seed 0] [--resume]
Extra overrides via --set key=value (repeatable, dotted keys into cfg)."""
import argparse
import copy
import hashlib
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, "/root/autodl-tmp/htdet")

from htdet.data.coco_dataset import CocoDataset
from htdet.data.transforms import collate_fn
from htdet.engine.trainer import Trainer
from htdet.models.detector import Detector
from htdet.utils.seed import set_seed

from torch.utils.data import DataLoader


def deep_merge(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_cfg(path, overrides=None):
    base = os.path.dirname(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "_base_" in cfg:
        base_cfg = load_cfg(os.path.join(base, cfg["_base_"]))
        deep_merge(base_cfg, cfg)
        cfg = base_cfg
        cfg.pop("_base_", None)
    for kv in overrides or []:
        k, v = kv.split("=", 1)
        try:
            v = yaml.safe_load(v)
        except Exception:
            pass
        node = cfg
        keys = k.split(".")
        for kk in keys[:-1]:
            node = node.setdefault(kk, {})
        node[keys[-1]] = v
    return cfg


def config_hash(cfg):
    return hashlib.sha256(yaml.dump(cfg, sort_keys=True).encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--set", action="append", default=[], help="key=value override")
    ap.add_argument("--work-dir", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config, args.set)
    seed = args.seed if args.seed is not None else cfg["train"].get("seed", 0)
    cfg["train"]["seed"] = seed
    cfg["_config_hash"] = config_hash(cfg)

    exp_name = os.path.basename(args.config).replace(".yaml", "") if args.work_dir is None \
        else os.path.basename(args.work_dir)
    if args.work_dir is None:
        work_dir = os.path.join(cfg["work_dir"], f"{exp_name}_seed{seed}")
    else:
        work_dir = args.work_dir

    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    set_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    dcfg = cfg["data"]
    tcfg = cfg["train"]
    # NOTE: cache DISABLED — precache.py pre-resizes images to 416 while GT
    # boxes stay in original coords, silently breaking the coordinate chain
    # (S3-D root cause). Resize now uses json ori_shape so it could be
    # re-enabled safely later, but correctness first.
    cache = None
    train_ds = CocoDataset(
        ann_file=dcfg["ann_train"], img_dir=dcfg["img_train"], train=True,
        normalize_mean=dcfg.get("normalize_mean", [0, 0, 0]),
        normalize_std=dcfg.get("normalize_std", [1, 1, 1]),
        input_size=tuple(dcfg.get("input_size", [416, 416])),
        cache=cache,
        multi_scale=tcfg.get("multi_scale", None),
        color_jitter=tcfg.get("color_jitter", False),
        bicubic_upscale=dcfg.get("bicubic_upscale", 1.0),
    )
    def worker_init(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        import random
        import numpy as np
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    loader = DataLoader(train_ds, batch_size=tcfg["batch_size"],
                        shuffle=True, num_workers=tcfg.get("num_workers", 12),
                        collate_fn=collate_fn, drop_last=True, persistent_workers=True,
                        prefetch_factor=4, worker_init_fn=worker_init)

    mcfg = cfg["model"]
    model = Detector(
        backbone_cfg=mcfg["backbone"],
        neck_cfg=mcfg.get("neck", {}),
        levels=tuple(mcfg["levels"]),
        num_classes=dcfg["num_classes"],
        pretrained=mcfg.get("pretrained", True),
        ppm_cfg=mcfg.get("ppm", None),
        head_cfg=mcfg.get("head", None),
    )

    # optional: initialize backbone/neck/head from an existing ckpt (e.g. R2),
    # PPM weights stay random. strict=False tolerates missing ppm keys.
    pckpt = mcfg.get("pretrained_ckpt")
    if pckpt and os.path.exists(pckpt):
        from htdet.utils.checkpoint import load_checkpoint
        state = load_checkpoint(pckpt, device="cpu")
        miss, unexp = model.load_state_dict(state["model"], strict=False)
        print(f"[train] init from {pckpt}: missing {len(miss)} keys, unexpected {len(unexp)}")
        if unexp:
            print(f"[train] unexpected keys (first 5): {unexp[:5]}")

    trainer = Trainer(model, loader, cfg, work_dir)
    if args.resume:
        trainer.resume()

    # optional: per-epoch AP evaluation on the test set (AP curve inspection)
    eval_from = cfg["train"].get("eval_from_epoch", 0)
    if eval_from > 0:
        from htdet.data.transforms import Compose, Resize, BicubicUpscale, Normalize, Pad
        from htdet.engine.evaluator import coco_eval
        dcfg = cfg["data"]
        size = tuple(dcfg.get("input_size", [416, 416]))
        eval_transforms = [Resize(size=size, keep_ratio=False)]
        bicubic_scale = float(dcfg.get("bicubic_upscale", 1.0))
        if bicubic_scale != 1.0:
            eval_transforms.append(BicubicUpscale(bicubic_scale))
        eval_transforms.extend([
            Normalize(mean=dcfg.get("normalize_mean", [0, 0, 0]),
                      std=dcfg.get("normalize_std", [1, 1, 1])),
            Pad(size_divisor=32),
        ])
        test_ds = CocoDataset(
            ann_file=dcfg["ann_test"], img_dir=dcfg["img_test"], train=False,
            transforms=Compose(eval_transforms))
        test_loader = DataLoader(test_ds, batch_size=8, shuffle=False,
                                 num_workers=8, collate_fn=collate_fn)
        csv_path = os.path.join(work_dir, "ep_ap.csv")
        if not os.path.exists(csv_path):
            with open(csv_path, "w") as f:
                f.write("epoch,AP,AP50,AP75,APs,APm,APl\n")
        best_meta_path = os.path.join(work_dir, "best_val.json")
        if os.path.exists(best_meta_path):
            with open(best_meta_path) as f:
                best_meta = json.load(f)
        else:
            best_meta = {"AP": -1.0}

        def eval_cb(epoch):
            if epoch < eval_from:
                return
            model.eval()
            detections = []
            with torch.no_grad():
                for imgs, _, _, metas in test_loader:
                    outs = model.forward_test(imgs.to("cuda"), metas)
                    for o, meta in zip(outs, metas):
                        if o.shape[0]:
                            detections.append((meta["img_id"], o[:, :4], o[:, 4], o[:, 5].long()))
            ev = coco_eval(detections, dcfg["ann_test"])
            if ev["AP"] > best_meta["AP"]:
                best_meta.update(ev)
                best_meta["epoch"] = epoch
                torch.save(
                    {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                     "epoch": epoch, "val_AP": ev["AP"]},
                    os.path.join(work_dir, "best_val.pth"),
                )
                with open(best_meta_path, "w") as f:
                    json.dump(best_meta, f, indent=2)
            model.train()
            if hasattr(model, "apply_ppm_stage"):
                stages = cfg["train"].get("ppm_stages", None)
                if stages:
                    s1, s2 = stages
                    model.apply_ppm_stage(0 if epoch < s1 else 1 if epoch < s2 else 2)
            with open(csv_path, "a") as f:
                f.write(f"{epoch},{ev['AP']:.4f},{ev['AP50']:.4f},{ev['AP75']:.4f},"
                        f"{ev['APs']:.4f},{ev['APm']:.4f},{ev['APl']:.4f}\n")
            print(f"[eval ep {epoch}] AP {ev['AP']:.4f} AP50 {ev['AP50']:.4f}", flush=True)

        trainer.run(epoch_callbacks=[eval_cb])
    else:
        trainer.run()

    print(f"TRAIN DONE: {work_dir}")


if __name__ == "__main__":
    main()
