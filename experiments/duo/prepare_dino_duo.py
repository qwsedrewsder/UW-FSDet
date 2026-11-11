"""Create DINO's expected COCO folder aliases without copying the images."""

from pathlib import Path

ROOT = Path("/root/duo_exp/data")


def link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.symlink_to(source, target_is_directory=source.is_dir())


def make(root: Path, val_split: str) -> None:
    link(ROOT / "yolo/images/train", root / "train2017")
    link(ROOT / f"yolo/images/{val_split}", root / "val2017")
    link(ROOT / "annotations/instances_train.json", root / "annotations/instances_train2017.json")
    link(ROOT / f"annotations/instances_{val_split}.json", root / "annotations/instances_val2017.json")


def main() -> None:
    make(ROOT / "dino_train", "val")
    make(ROOT / "dino_test", "test")
    print("DINO aliases ready", flush=True)


if __name__ == "__main__":
    main()
