"""Build HyUOD's required transmission/airlight inputs for the fixed DUO split."""

import sys
from pathlib import Path

ROOT = Path("/root/duo_exp/data")
HYUOD = Path("/root/autodl-tmp/HyUOD")
sys.path.insert(0, str(HYUOD))
from ta_generate import dehaze_V2


def main():
    base = ROOT / "hyuod"
    for split in ("train", "val", "test"):
        images = base / "images" / split
        images.mkdir(parents=True, exist_ok=True)
        source = ROOT / "yolo/images" / split
        labels = ROOT / "yolo/labels" / split
        target_labels = base / "labels" / split
        target_labels.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            link = images / item.name
            if not link.exists():
                link.symlink_to(item)
        for item in labels.iterdir():
            link = target_labels / item.name
            if not link.exists():
                link.symlink_to(item)

        transmission = base / "t" / split
        airlight = base / "a" / split
        transmission.mkdir(parents=True, exist_ok=True)
        airlight.mkdir(parents=True, exist_ok=True)
        count = len(list(images.iterdir()))
        if len(list(transmission.iterdir())) != count or len(list(airlight.iterdir())) != count:
            print(f"processing {split}: {count} images", flush=True)
            dehaze_V2(str(images), str(transmission), str(airlight))
        assert len(list(transmission.iterdir())) == count
        assert len(list(airlight.iterdir())) == count

    (base / "data.yaml").write_text(
        f"path: {base}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "nc: 4\nnames:\n  0: holothurian\n  1: echinus\n  2: scallop\n  3: starfish\n"
    )
    print("HyUOD inputs ready", flush=True)


if __name__ == "__main__":
    main()
