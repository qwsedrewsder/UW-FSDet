"""Run one DUO seed for a model implemented in the existing htdet project."""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path("/root/duo_exp")
DATA = ROOT / "data"
RUNS = ROOT / "runs"
PYTHON = "/root/miniconda3/bin/python"
TRAIN = ROOT / "htdet_train_duo.py"
TEST = "/root/autodl-tmp/htdet/tools/test.py"
SOURCES = {
    "e2": "/root/autodl-tmp/htdet/configs/e2_mvit_pan_yolo.yaml",
    "uwfsdet": "/root/autodl-tmp/htdet/configs/d94_e2_ppm_36ep.yaml",
    "baseline": "/root/autodl-tmp/work_dirs/m1_s4_s0/config.yaml",
    "retinanet": "/root/autodl-tmp/work_dirs/m0_s3_s0/config.yaml",
}


def status(path: Path, phase: str, **extra: object) -> None:
    value = {"phase": phase, "updated_utc": datetime.now(timezone.utc).isoformat(), **extra}
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def execute(command: list[str], log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(command) + "\n")
        log.flush()
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                       cwd="/root/autodl-tmp/htdet")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=SOURCES, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    model, seed = args.model, args.seed
    assert (DATA / "manifest.json").exists(), "DUO data preparation must complete first"
    run_dir = RUNS / f"{model}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    log_path = run_dir / "console.log"
    val_json = DATA / "annotations" / "instances_val.json"
    test_json = DATA / "annotations" / "instances_test.json"
    train_json = DATA / "annotations" / "instances_train.json"
    image_train = DATA / "raw" / "DUO" / "images" / "train"
    image_test = DATA / "raw" / "DUO" / "images" / "test"
    for path in (val_json, test_json, train_json, image_train, image_test):
        assert path.exists(), path
    command = [PYTHON, "-u", str(TRAIN), "--config", SOURCES[model],
               "--seed", str(seed), "--work-dir", str(run_dir)]
    overrides = {
        "data.root": str(DATA / "raw" / "DUO"),
        "data.ann_train": str(train_json),
        "data.ann_test": str(val_json),
        "data.img_train": str(image_train),
        "data.img_test": str(image_train),
        "train.eval_from_epoch": 1,
        "train.checkpoint_keep": 1,
        "train.num_workers": 8,
    }
    if model == "uwfsdet":
        e2_checkpoint = RUNS / f"e2_seed{seed}" / "best_val.pth"
        assert e2_checkpoint.exists(), f"DUO E2 checkpoint missing: {e2_checkpoint}"
        overrides["model.pretrained_ckpt"] = str(e2_checkpoint)
    for key, value in overrides.items():
        command.extend(("--set", f"{key}={value}"))

    try:
        status(status_path, "training", model=model, seed=seed)
        execute(command, log_path)
        best = run_dir / "best_val.pth"
        assert best.exists(), f"validation checkpoint missing: {best}"
        if model == "e2":
            status(status_path, "complete", model=model, seed=seed,
                   best_checkpoint=str(best))
            return
        with (run_dir / "config.yaml").open() as stream:
            test_cfg = yaml.safe_load(stream)
        test_cfg["data"]["ann_test"] = str(test_json)
        test_cfg["data"]["img_test"] = str(image_test)
        test_config_path = run_dir / "test_config.yaml"
        with test_config_path.open("w") as stream:
            yaml.safe_dump(test_cfg, stream, sort_keys=False)
        status(status_path, "testing", model=model, seed=seed,
               best_checkpoint=str(best))
        test_result = run_dir / "test_eval.json"
        execute([PYTHON, "-u", TEST, "--config", str(test_config_path),
                 "--ckpt", str(best), "--out", str(test_result)], log_path)
        with test_result.open() as stream:
            metrics = json.load(stream)
        status(status_path, "complete", model=model, seed=seed,
               best_checkpoint=str(best), metrics=metrics)
    except Exception as exc:
        status(status_path, "failed", model=model, seed=seed, error=str(exc))
        raise


if __name__ == "__main__":
    main()
