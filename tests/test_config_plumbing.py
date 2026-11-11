"""P2: config plumbing tests — every config leaf must be observable in the model.
Catches "config read but not used" bugs (this is the 3rd occurrence).
Run: python tests/test_config_plumbing.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from htdet.models.detector import Detector


def test_anchor_params_plumbed():
    """octave_base_scale / scales_per_octave / ratios must reach the head."""
    m = Detector(backbone_cfg={"name": "mobilevit", "variant": "xs", "pretrained": False},
                 neck_cfg={"name": "fpn", "octave_base_scale": 2, "scales_per_octave": 2,
                           "ratios": [0.5, 1.0]},
                 levels=(4, 8, 16, 32), num_classes=4)
    gen = m.head.anchor_gen
    assert gen.base_sizes == [8, 16, 32, 64], f"base_sizes wrong: {gen.base_sizes}"
    assert gen.num_anchors == 4, f"num_anchors wrong: {gen.num_anchors}"  # 2 scales x 2 ratios
    assert len(gen.ratios) == 2
    # observable difference: anchor widths for scale=2 vs default 4
    w = gen._anchors_per_pos(0)[:, 2] - gen._anchors_per_pos(0)[:, 0]
    assert abs(w.max().item() - 2 * 4 * (2 ** 0.5)) < 1e-4, f"widths wrong: {w}"
    print("anchor params plumbed OK")


def test_assigner_params_plumbed():
    m = Detector(backbone_cfg={"name": "mobilevit", "variant": "xs", "pretrained": False},
                 neck_cfg={"name": "fpn", "pos_iou_thr": 0.4, "neg_iou_thr": 0.3},
                 levels=(4, 8, 16, 32), num_classes=4)
    assert m.head.pos_iou_thr == 0.4
    assert m.head.neg_iou_thr == 0.3
    print("assigner params plumbed OK")


def test_fgfpn_params_plumbed():
    m = Detector(backbone_cfg={"name": "mobilevit", "variant": "xs", "pretrained": False},
                 neck_cfg={"name": "fgfpn", "fuse_conv": "pairwise", "topdown": False,
                           "remove_p1": True, "out_conv": True, "delta_init_ones": True},
                 levels=(4, 8, 16, 32), num_classes=4)
    neck = m.neck
    assert neck.fuse_conv == "pairwise"
    assert neck.topdown is False
    assert neck.remove_p1 is True
    assert neck.out_conv is True
    assert len(neck.smooths) == 5  # out_conv built
    assert neck.deltas.detach()[0, 0].item() == 1.0  # delta_init_ones
    assert neck.deltas.detach()[1, 0].item() == 1.0
    print("fgfpn params plumbed OK")


def test_extra_outs_levels():
    """H-B levels must build extra convs (P6/P7)."""
    m = Detector(backbone_cfg={"name": "mobilevit", "variant": "xs", "pretrained": False},
                 neck_cfg={"name": "fpn"}, levels=(8, 16, 32, 64, 128), num_classes=4)
    assert len(m.neck.extra) == 2, f"extra convs: {len(m.neck.extra)}"
    assert m.detect_idxs == [2, 3, 4, 5, 6], f"detect_idxs: {m.detect_idxs}"
    print("extra_outs plumbed OK")


if __name__ == "__main__":
    test_anchor_params_plumbed()
    test_assigner_params_plumbed()
    test_fgfpn_params_plumbed()
    test_extra_outs_levels()
    print("\nALL CONFIG PLUMBING TESTS PASSED")
