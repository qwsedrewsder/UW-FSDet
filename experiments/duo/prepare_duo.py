"""Prepare the authors' DUO archive for the shared 14-model experiment."""

import argparse
import hashlib
import json
import random
import shutil
import zipfile
from collections import Counter
from pathlib import Path


CLASSES = ("holothurian", "echinus", "scallop", "starfish")
SPLIT_SEED = 20261103
VALIDATION_IMAGES = 667


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    archive = args.archive.resolve()
    out.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as source:
        source_names = set(source.namelist())
        original = {}
        for split, expected in (("train", 6671), ("test", 1111)):
            original[split] = json.loads(
                source.read(f"DUO/annotations/instances_{split}.json")
            )
            data = original[split]
            assert len(data["images"]) == expected, (split, len(data["images"]))
            assert [c["name"] for c in data["categories"]] == list(CLASSES)
            assert [c["id"] for c in data["categories"]] == [1, 2, 3, 4]
            for image in data["images"]:
                member = f'DUO/images/{split}/{image["file_name"]}'
                assert member in source_names, member

        train_ids = sorted(int(im["id"]) for im in original["train"]["images"])
        assert len(train_ids) == len(set(train_ids))
        random.Random(SPLIT_SEED).shuffle(train_ids)
        val_ids = set(train_ids[:VALIDATION_IMAGES])
        split_ids = {
            "train": set(train_ids[VALIDATION_IMAGES:]),
            "val": val_ids,
            "test": {int(im["id"]) for im in original["test"]["images"]},
        }
        assert not split_ids["train"] & split_ids["val"]

        for name in source.namelist():
            if not name.startswith("DUO/images/") or name.endswith("/"):
                continue
            if Path(name).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            target = out / "raw" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with source.open(name) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)

    counts = {}
    yolo_root = out / "yolo"
    for split in ("train", "val", "test"):
        original_split = "test" if split == "test" else "train"
        source = original[original_split]
        ids = split_ids[split]
        images = [im for im in source["images"] if int(im["id"]) in ids]
        annotations = [ann for ann in source["annotations"] if int(ann["image_id"]) in ids]
        annotations = [{**ann, "category_id": int(ann["category_id"]) - 1} for ann in annotations]
        image_by_id = {int(im["id"]): im for im in images}
        by_image = {}
        for ann in annotations:
            by_image.setdefault(int(ann["image_id"]), []).append(ann)

        coco = {
            "info": {"description": "DUO fixed train/validation/test split for UW-FSDet study"},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": [{"id": i, "name": name} for i, name in enumerate(CLASSES)],
        }
        write_json(out / "annotations" / f"instances_{split}.json", coco)

        image_dir = yolo_root / "images" / split
        label_dir = yolo_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for im in images:
            original_file = out / "raw" / "DUO" / "images" / original_split / im["file_name"]
            linked_file = image_dir / im["file_name"]
            if not linked_file.exists():
                linked_file.symlink_to(original_file)
            width, height = int(im["width"]), int(im["height"])
            label_lines = []
            for ann in by_image.get(int(im["id"]), []):
                x, y, w, h = map(float, ann["bbox"])
                assert 0 <= x <= width and 0 <= y <= height
                assert w > 0 and h > 0 and x + w <= width + 1 and y + h <= height + 1
                values = (int(ann["category_id"]), (x + w / 2) / width,
                          (y + h / 2) / height, w / width, h / height)
                label_lines.append(f"{values[0]} {values[1]:.9f} {values[2]:.9f} "
                                   f"{values[3]:.9f} {values[4]:.9f}")
            (label_dir / (Path(im["file_name"]).stem + ".txt")).write_text(
                "\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8"
            )

        counts[split] = {
            "images": len(images),
            "annotations": len(annotations),
            "classes": dict(Counter(CLASSES[a["category_id"]] for a in annotations)),
        }

    yaml = (
        f"path: {yolo_root}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASSES))
    )
    (yolo_root / "data.yaml").write_text(yaml, encoding="utf-8")
    assert counts["train"]["images"] == 6004
    assert counts["val"]["images"] == 667
    assert counts["test"]["images"] == 1111
    assert sum(row["annotations"] for row in counts.values()) == 74515
    manifest = {
        "source_archive": str(archive),
        "source_sha256": sha256(archive),
        "split_seed": SPLIT_SEED,
        "class_names": CLASSES,
        "counts": counts,
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
