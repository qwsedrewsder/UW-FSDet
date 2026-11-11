"""P0: anchor geometry unit tests (S3-C).
Run: python tests/test_anchor_geometry.py
All tests must print actual tables, not just pass/fail.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from htdet.core.anchor import AnchorGenerator
from htdet.data.coco_dataset import CocoDataset
from htdet.data.transforms import Compose, Resize, Normalize, Pad
from htdet.models.detector import Detector


def test_A1_center_coverage():
    """Anchor center ranges per level must cover [0,416]."""
    gen = AnchorGenerator(strides=[4, 8, 16, 32])
    featmap_sizes = [(104, 104), (52, 52), (26, 26), (13, 13)]
    print("\n=== A1: anchor center coverage ===")
    for size, stride in zip(featmap_sizes, [4, 8, 16, 32]):
        h, w = size
        shift_x = (torch.arange(w) + 0.5) * stride
        shift_y = (torch.arange(h) + 0.5) * stride
        print(f"stride={stride:3d} feat={w}x{h:3d} center_x range=[{shift_x.min().item():.1f}, {shift_x.max().item():.1f}] "
              f"center_y range=[{shift_y.min().item():.1f}, {shift_y.max().item():.1f}]  (expect max~416)")


def test_A2_counts_and_stride_source():
    """Per-level anchor count == H*W*9; strides must match real feature strides."""
    print("\n=== A2: anchor counts and stride source ===")
    m = Detector(backbone_cfg={"name": "mobilevit", "variant": "xs", "pretrained": False},
                 neck_cfg={"name": "fpn"}, levels=(4, 8, 16, 32), num_classes=4).cuda().eval()
    with torch.no_grad():
        feats = m._get_feats(torch.zeros(1, 3, 416, 416).cuda())
    anchors = m.head.get_anchors([f.shape[-2:] for f in feats], "cuda")
    for i, (f, a, lvl) in enumerate(zip(feats, anchors, m.levels)):
        H, W = f.shape[-2:]
        expect = H * W * 9
        print(f"level={lvl:3d} (config stride) feat={W}x{H} anchors={a.shape[0]} expected={expect} match={a.shape[0]==expect}")
    print(f"head.anchor_gen.strides = {m.head.anchor_gen.strides}")
    print(f"config levels = {m.levels}")


def test_A3_anchor_sizes():
    """Per-level anchor sizes must be stride*4*{1,2^(1/3),2^(2/3)} * ratios."""
    print("\n=== A3: anchor sizes ===")
    gen = AnchorGenerator(strides=[4, 8, 16, 32])
    for lvl, stride in enumerate([4, 8, 16, 32]):
        base = gen._anchors_per_pos()
        ws = base[:, 2] - base[:, 0]
        hs = base[:, 3] - base[:, 1]
        theory_base = stride * 4
        print(f"stride={stride:3d} theory_base={theory_base:3d} "
              f"actual_w={ws[:3].tolist()} (theory {theory_base}x{1}, {theory_base}x{2**(1/3):.2f}, {theory_base}x{2**(2/3):.2f})")


def test_A4_monotonicity():
    """match_rate(S1) <= match_rate(S2) <= match_rate(S3) for S1⊂S2⊂S3."""
    print("\n=== A4: monotonicity of match rate (subset must be <= superset) ===")
    tr = Compose([Resize(size=(416, 416)), Normalize(), Pad(size_divisor=32)])
    ds = CocoDataset(ann_file="/root/autodl-tmp/data/annotations/train_clean.json",
                     img_dir="/root/autodl-tmp/data/urpc_coco/train2018/image",
                     train=False, transforms=tr)
    import torchvision

    def match_rate(levels):
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
        return match, total

    r1 = match_rate((8, 16, 32))
    r2 = match_rate((4, 8, 16, 32))
    r3 = match_rate((2, 4, 8, 16, 32))
    print(f"S1=L3-L5: {r1[0]}/{r1[1]} ({r1[0]/r1[1]*100:.1f}%)")
    print(f"S2=L2-L5: {r2[0]}/{r2[1]} ({r2[0]/r2[1]*100:.1f}%)")
    print(f"S3=L1-L5: {r3[0]}/{r3[1]} ({r3[0]/r3[1]*100:.1f}%)")
    assert r1[0] / r1[1] <= r2[0] / r2[1] + 1e-9, "VIOLATION: rate(S1) > rate(S2)"
    assert r2[0] / r2[1] <= r3[0] / r3[1] + 1e-9, "VIOLATION: rate(S2) > rate(S3)"


def test_A5_match_rate_floor():
    """H-A match rate should be >85%; print size-bucketed breakdown."""
    print("\n=== A5: match rate floor (H-A, expect >85%) ===")
    import torchvision
    tr = Compose([Resize(size=(416, 416)), Normalize(), Pad(size_divisor=32)])
    ds = CocoDataset(ann_file="/root/autodl-tmp/data/annotations/train_clean.json",
                     img_dir="/root/autodl-tmp/data/urpc_coco/train2018/image",
                     train=False, transforms=tr)
    m = Detector(backbone_cfg={"name": "mobilevit", "variant": "xs", "pretrained": False},
                 neck_cfg={"name": "fpn"}, levels=(4, 8, 16, 32), num_classes=4).cuda().eval()
    buckets = {"<16": [0, 0], "16-32": [0, 0], "32-64": [0, 0], ">64": [0, 0]}
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
            w = (boxes[:, 2] - boxes[:, 0])
            for g in range(len(boxes)):
                sz = w[g].item()
                key = "<16" if sz < 16 else ("16-32" if sz < 32 else ("32-64" if sz < 64 else ">64"))
                buckets[key][1] += 1
                if per_gt[g] >= 0.5:
                    buckets[key][0] += 1
    tot_m = sum(v[0] for v in buckets.values())
    tot = sum(v[1] for v in buckets.values())
    for k, (match, total) in buckets.items():
        print(f"GT size {k:5s}: match {match}/{total} ({match/max(total,1)*100:.1f}%)")
    print(f"TOTAL: {tot_m}/{tot} ({tot_m/max(tot,1)*100:.1f}%)")
    assert tot_m / tot > 0.85, f"match rate {tot_m/tot:.2f} < 0.85"


if __name__ == "__main__":
    test_A1_center_coverage()
    test_A2_counts_and_stride_source()
    test_A3_anchor_sizes()
    test_A4_monotonicity()
    test_A5_match_rate_floor()
    print("\nALL P0 TESTS DONE")
