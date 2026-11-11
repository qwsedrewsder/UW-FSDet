"""S2 gate: overfit 8 training images, 300 iters, AP50 > 0.9 on those 8 images.
Random flip disabled, no dropout of structure. Also used to catch assigner/coder/loss bugs."""
import os
import sys

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from htdet.data.coco_dataset import CocoDataset
from htdet.data.transforms import Compose, Resize, Normalize, Pad, collate_fn
from htdet.engine.evaluator import coco_eval
from htdet.models.detector import Detector
from htdet.utils.seed import set_seed


def main():
    seed = 0
    set_seed(seed)
    cfg = {
        "data": {
            "ann_train": "/root/autodl-tmp/data/annotations/train_clean.json",
            "img_train": "/root/autodl-tmp/data/urpc_coco/train2018/image",
            "ann_test": "/root/autodl-tmp/data/annotations/train_clean.json",
            "num_classes": 4,
        },
        "train": {"batch_size": 8, "num_workers": 4, "amp": False, "lr": 0.01},
    }
    transforms = Compose([
        Resize(size=(416, 416), keep_ratio=False),
        Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        Pad(size_divisor=32),
    ])
    ds = CocoDataset(ann_file=cfg["data"]["ann_train"], img_dir=cfg["data"]["img_train"],
                     train=False, transforms=transforms)
    idx = list(range(8))
    small = Subset(ds, idx)
    loader = DataLoader(small, batch_size=8, shuffle=True, num_workers=4, collate_fn=collate_fn)

    model = Detector(backbone_cfg={"name": "mobilevit", "variant": "xs", "pretrained": True},
                     neck_cfg={"name": "fgfpn", "fuse_conv": "pairwise_cross"},
                     levels=(4, 8, 16, 32), num_classes=4).cuda()
    backbone_ids = set(id(p) for p in model.backbone.parameters())
    opt = torch.optim.SGD([
        {"params": [p for p in model.parameters() if id(p) not in backbone_ids], "lr": 0.01},
        {"params": model.backbone.parameters(), "lr": 0.005},
    ], momentum=0.9, weight_decay=1e-4)

    model.train()
    for it in range(1500):
        for imgs, gtb, gtl, metas in loader:
            imgs = imgs.cuda()
            gtb = [g.cuda() for g in gtb]
            gtl = [l.cuda() for l in gtl]
            losses = model.forward_train(imgs, gtb, gtl, metas)
            loss = losses["loss_cls"] + losses["loss_reg"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 35)
            opt.step()
        if it % 200 == 0:
            print(f"iter {it}: loss {loss.item():.4f} cls {losses['loss_cls'].item():.4f} "
                  f"reg {losses['loss_reg'].item():.4f} pos {losses['num_pos']}")

    # evaluate on the same 8 images (subset ann file so no FN from other 2893 imgs)
    import json as _json
    import tempfile
    from pycocotools.coco import COCO

    full = COCO(cfg["data"]["ann_test"])
    sub_imgs = [full.imgs[i] for i in idx]
    sub_ids = set(idx)
    sub_anns = [a for a in full.anns.values() if a["image_id"] in sub_ids]
    sub_cats = full.cats
    sub_ann_file = os.path.join(tempfile.mkdtemp(), "sanity_ann.json")
    with open(sub_ann_file, "w") as f:
        _json.dump({"images": sub_imgs, "annotations": sub_anns, "categories": list(sub_cats.values())}, f)

    model.eval()
    detections = []
    with torch.no_grad():
        for imgs, _, _, metas in loader:
            outs = model.forward_test(imgs.cuda(), metas)
            for o, meta in zip(outs, metas):
                detections.append((meta["img_id"], o[:, :4], o[:, 4], o[:, 5].long()))
    ev = coco_eval(detections, sub_ann_file)
    print(f"SANITY: AP50={ev['AP50']:.3f} AP={ev['AP']:.3f}")
    assert ev["AP50"] > 0.9, f"GATE FAILED: AP50={ev['AP50']} <= 0.9"
    print("SANITY OVERFIT GATE PASSED")


if __name__ == "__main__":
    main()
