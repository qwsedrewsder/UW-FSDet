"""S1: per-module parameter count for M0/M1/M2. Gate: M0~36.2M, M1~6.94M, M2~7.68M (+-5%)."""
import argparse
import sys

import torch

sys.path.insert(0, "/root/autodl-tmp/htdet")

from htdet.models.detector import Detector
from htdet.models.mobilevit import count_backbone_params


def count_model(name, variant="xs", fuse_conv="pairwise_cross", pretrained=True, levels=(4, 8, 16, 32), ppm_cfg=None):
    if name == "m0":
        cfg_b = {"name": "resnet50", "pretrained": pretrained}
        cfg_n = {"name": "none"}
        lv = (8, 16, 32, 64, 128)
    elif name == "m1":
        cfg_b = {"name": "mobilevit", "variant": variant, "pretrained": pretrained}
        cfg_n = {"name": "fpn"}
        lv = levels
    elif name == "m2":
        cfg_b = {"name": "mobilevit", "variant": variant, "pretrained": pretrained}
        cfg_n = {"name": "fgfpn", "fuse_conv": fuse_conv, "out_conv": False}
        lv = levels
    elif name == "d9":
        cfg_b = {"name": "mobilevit", "variant": variant, "pretrained": pretrained}
        cfg_n = {"name": "fpn"}
        lv = levels
    else:
        raise ValueError(name)

    model = Detector(backbone_cfg=cfg_b, neck_cfg=cfg_n, levels=lv, num_classes=4,
                     pretrained=pretrained, ppm_cfg=ppm_cfg)
    total = 0
    print(f"== {name} (variant={variant}, fuse_conv={fuse_conv}, levels={lv}, ppm={ppm_cfg is not None}) ==")
    for mname, mod in model.named_children():
        n = sum(p.numel() for p in mod.parameters())
        total += n
        print(f"  {mname:12s} {n/1e6:8.3f} M")
    print(f"  TOTAL        {total/1e6:8.3f} M")
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="xs")
    ap.add_argument("--pretrained", action="store_true", default=False, help="use pretrained weights (params same)")
    ap.add_argument("--fuse-conv", default="pairwise_cross")
    args = ap.parse_args()

    print(f"MobileViT-{args.variant} backbone params: {count_backbone_params(args.variant):.4f} M")

    count_model("m0")
    count_model("m1", variant=args.variant)
    count_model("m2", variant=args.variant, fuse_conv=args.fuse_conv)
    count_model("d9", variant=args.variant, ppm_cfg={"to_rgb": True, "use_perc": False})
    count_model("d9-32ch", variant=args.variant, ppm_cfg={"to_rgb": False, "use_perc": False})
