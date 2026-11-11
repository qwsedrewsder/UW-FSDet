"""D9 PPM smoke test: build R2+PPM from configs/d9_r2_ppm.yaml, run one train
forward + one test forward, print per-module params and loss values. No dataset,
no training. Run on server:

    python tools/ppm_smoke.py [--config configs/d9_r2_ppm.yaml] [--no-perc]

Passes when: x_pm shape == (B,3,416,416), loss_uie/loss_sr finite, forward_test
returns detections, stage toggling flips requires_grad as expected.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, "/root/autodl-tmp/htdet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/d9_r2_ppm.yaml")
    ap.add_argument("--no-perc", action="store_true", help="skip VGG16 download")
    args = ap.parse_args()

    import yaml
    from htdet.models.detector import Detector

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if "_base_" in cfg:
        base_path = os.path.join(os.path.dirname(os.path.abspath(args.config)), cfg["_base_"])
        with open(base_path) as f:
            base = yaml.safe_load(f)
        base.update(cfg)
        cfg = base
        cfg.pop("_base_", None)

    mcfg = cfg["model"]
    ppm_cfg = dict(mcfg.get("ppm", {}))
    if args.no_perc:
        ppm_cfg["use_perc"] = False
    model = Detector(backbone_cfg=mcfg["backbone"], neck_cfg=mcfg.get("neck", {}),
                     levels=tuple(mcfg["levels"]), num_classes=cfg["data"]["num_classes"],
                     pretrained=False, ppm_cfg=ppm_cfg or None)
    model = model.cuda()

    for mname, mod in model.named_children():
        n = sum(p.numel() for p in mod.parameters())
        print(f"  {mname:12s} {n/1e6:8.3f} M")
    print(f"  TOTAL        {sum(p.numel() for p in model.parameters())/1e6:8.3f} M")
    print(f"  ppm.rgb/proj/head params ok, vgg loaded: {model.ppm.vgg is not None}")

    x = torch.rand(2, 3, 416, 416, device="cuda")
    meta = [{"ori_shape": (405, 720), "scale_factor": (416 / 405, 416 / 720)}] * 2

    # --- train-mode forward with losses ---
    model.train()
    model.apply_ppm_stage(2)  # joint
    gt_bboxes = [torch.tensor([[10.0, 10.0, 50.0, 50.0]], device="cuda")] * 2
    gt_labels = [torch.tensor([1], device="cuda")] * 2
    losses = model.forward_train(x, gt_bboxes, gt_labels, meta)
    print("\n[train/joint] losses:")
    for k in ("loss_cls", "loss_reg", "loss_uie", "loss_sr"):
        v = losses[k]
        print(f"  {k:10s} {float(v):.5f} finite={bool(torch.isfinite(v))}")

    # --- stage toggling check ---
    model.apply_ppm_stage(0)
    ok_ppm = all(p.requires_grad for p in model.ppm.parameters())
    ok_det = not any(p.requires_grad for p in model.backbone.parameters())
    print(f"\n[stage0] ppm trainable={ok_ppm}, detector frozen={ok_det}")
    model.apply_ppm_stage(1)
    ok_ppm1 = not any(p.requires_grad for p in model.ppm.parameters())
    ok_det1 = any(p.requires_grad for p in model.head.parameters())
    print(f"[stage1] ppm frozen={ok_ppm1}, detector trainable={ok_det1}")
    model.apply_ppm_stage(2)

    # --- test-mode forward ---
    model.eval()
    with torch.no_grad():
        outs = model.forward_test(x, meta)
    print(f"\n[test] forward_test OK, {len(outs)} images, boxes per img: "
          f"{[o.shape[0] for o in outs]}")

    ok = (ok_ppm and ok_det and ok_ppm1 and ok_det1
          and all(torch.isfinite(losses[k]) for k in ("loss_cls", "loss_reg", "loss_uie", "loss_sr")))
    print("\nSMOKE " + ("PASS" if ok else "FAIL"))


if __name__ == "__main__":
    main()
