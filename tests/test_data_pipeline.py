"""P1: data pipeline regression tests (S3-D). Run: python tests/test_data_pipeline.py"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from htdet.data.coco_dataset import CocoDataset
from htdet.data.transforms import Compose, Resize, Normalize, Pad

ANN = "/root/autodl-tmp/data/annotations/train_clean.json"
IMG = "/root/autodl-tmp/data/urpc_coco/train2018/image"


def make_ds(cache=None):
    tr = Compose([Resize(size=(416, 416)), Normalize(), Pad(size_divisor=32)])
    return CocoDataset(ann_file=ANN, img_dir=IMG, train=False, transforms=tr, cache=cache)


def test_D1_roundtrip():
    """Boxes from dataset, inverted via scale_factor, must match json bbox (<1px)."""
    print("\n=== D1: coordinate roundtrip ===")
    ds = make_ds()
    with open(ANN) as f:
        data = json.load(f)
    id2img = {i["id"]: i for i in data["images"]}
    ann_by_img = {}
    for a in data["annotations"]:
        ann_by_img.setdefault(a["image_id"], []).append(a)

    worst = 0.0
    n_checked = 0
    for idx in np.random.RandomState(0).choice(len(ds), 50, replace=False):
        img, boxes, labels, meta = ds[idx]
        info = id2img[ds.imgs[idx]["id"]]
        sfx, sfy = meta["scale_factor"]
        back = boxes.clone()
        back[:, [0, 2]] /= sfx
        back[:, [1, 3]] /= sfy
        anns = ann_by_img.get(info["id"], [])
        assert len(anns) == len(boxes), f"idx {idx}: {len(anns)} anns vs {len(boxes)} boxes"
        for a, b in zip(anns, back):
            x, y, w, h = a["bbox"]
            err = max(abs(b[0].item() - x), abs(b[1].item() - y),
                      abs(b[2].item() - (x + w)), abs(b[3].item() - (y + h)))
            worst = max(worst, err)
            n_checked += 1
    print(f"checked {n_checked} boxes, worst roundtrip error = {worst:.4f} px")
    assert worst < 1.0, f"D1 FAILED: worst err {worst}"
    print("D1 PASSED")


def test_D2_cache_equivalence():
    """cache=None vs cache=npy must give identical boxes & scale_factor."""
    print("\n=== D2: cache vs no-cache equivalence ===")
    cache_dir = "/root/autodl-tmp/data/cache"
    npy = os.path.join(cache_dir, "train_416.npy")
    if not os.path.exists(npy):
        print("D2 SKIPPED: cache files removed (disabled by S3-D)")
        return
    arr = np.load(npy, mmap_mode="r")
    files = json.load(open(os.path.join(cache_dir, "train_files.json")))
    ds1 = make_ds()
    ds2 = make_ds(cache={"arr": arr, "files": files})
    for idx in range(30):
        _, b1, _, m1 = ds1[idx]
        _, b2, _, m2 = ds2[idx]
        bdiff = (b1 - b2).abs().max().item()
        sdiff = max(abs(m1["scale_factor"][0] - m2["scale_factor"][0]),
                    abs(m1["scale_factor"][1] - m2["scale_factor"][1]))
        print(f"idx {idx}: box_maxdiff={bdiff:.4f} scale_diff={sdiff:.6f}")
        assert bdiff < 1.0, f"D2 FAILED at {idx}: box diff {bdiff}"
        assert sdiff < 1e-6, f"D2 FAILED at {idx}: scale diff {sdiff}"
    print("D2 PASSED")


def test_D3_boxes_in_image():
    """All GT boxes must be inside [0,416]x[0,416] after resize."""
    print("\n=== D3: GT boxes inside image ===")
    ds = make_ds()
    n_out = 0
    n_total = 0
    for idx in range(200):
        _, boxes, _, _ = ds[idx]
        n_total += len(boxes)
        if len(boxes) == 0:
            continue
        if (boxes[:, 0] < -1).any() or (boxes[:, 1] < -1).any() or \
           (boxes[:, 2] > 417).any() or (boxes[:, 3] > 417).any():
            n_out += 1
    print(f"checked {n_total} boxes across 200 imgs, out-of-bounds imgs: {n_out}")
    assert n_out == 0, f"D3 FAILED: {n_out} imgs have OOB boxes"
    print("D3 PASSED")


def test_D4_match_rate_floor():
    """H-A match rate must be >85% (anchor bug fixed)."""
    print("\n=== D4: match rate floor (H-A, >85%) ===")
    import torchvision
    from htdet.models.detector import Detector
    ds = make_ds()
    m = Detector(backbone_cfg={"name": "mobilevit", "variant": "xs", "pretrained": False},
                 neck_cfg={"name": "fpn"}, levels=(4, 8, 16, 32), num_classes=4).cuda().eval()
    sizes = []
    match = 0
    total = 0
    with torch.no_grad():
        for idx in range(200):
            img, boxes, labels, meta = ds[idx]
            if len(boxes) == 0:
                continue
            feats = m._get_feats(img.unsqueeze(0).cuda())
            anchors = m.head.get_anchors([f.shape[-2:] for f in feats], "cuda")
            all_a = torch.cat(anchors)
            iou = torchvision.ops.box_iou(all_a, boxes.cuda())
            per_gt = iou.max(dim=0).values
            match += (per_gt >= 0.5).sum().item()
            total += len(boxes)
            sizes.extend((boxes[:, 2] - boxes[:, 0]).numpy().tolist())
    sizes = np.array(sizes)
    print(f"416-space GT width: min={sizes.min():.1f} med={np.median(sizes):.1f} "
          f"p75={np.percentile(sizes, 75):.1f} max={sizes.max():.1f}")
    print(f"match rate: {match}/{total} ({match/total*100:.1f}%)")
    assert match / total > 0.85, f"D4 FAILED: {match/total:.2f}"
    print("D4 PASSED")


def test_D5_monotonicity():
    """match_rate(subset) <= match_rate(superset)."""
    print("\n=== D5: monotonicity ===")
    import torchvision
    from htdet.models.detector import Detector
    ds = make_ds()

    def rate(levels):
        m = Detector(backbone_cfg={"name": "mobilevit", "variant": "xs", "pretrained": False},
                     neck_cfg={"name": "fpn"}, levels=levels, num_classes=4).cuda().eval()
        match = 0
        total = 0
        with torch.no_grad():
            for idx in range(50):
                img, boxes, labels, meta = ds[idx]
                if len(boxes) == 0:
                    continue
                feats = m._get_feats(img.unsqueeze(0).cuda())
                anchors = m.head.get_anchors([f.shape[-2:] for f in feats], "cuda")
                all_a = torch.cat(anchors)
                iou = torchvision.ops.box_iou(all_a, boxes.cuda())
                match += (iou.max(dim=0).values >= 0.5).sum().item()
                total += len(boxes)
        return match / max(total, 1)

    r1 = rate((8, 16, 32))
    r2 = rate((4, 8, 16, 32))
    r3 = rate((2, 4, 8, 16, 32))
    print(f"S1(L3-L5)={r1*100:.1f}%  S2(L2-L5)={r2*100:.1f}%  S3(L1-L5)={r3*100:.1f}%")
    assert r1 <= r2 + 1e-9 and r2 <= r3 + 1e-9, f"D5 FAILED: {r1} {r2} {r3}"
    print("D5 PASSED")


if __name__ == "__main__":
    test_D1_roundtrip()
    test_D2_cache_equivalence()
    test_D3_boxes_in_image()
    test_D4_match_rate_floor()
    test_D5_monotonicity()
    print("\nALL DATA PIPELINE TESTS PASSED")
