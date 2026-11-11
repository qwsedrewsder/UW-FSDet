"""Durable sequential GPU queue for the DUO experiment."""

import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/duo_exp")
PYTHON = "/root/miniconda3/bin/python"
SCHEDULE = ROOT / "schedule.json"
STATE = ROOT / "queue_state.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(value: dict) -> None:
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(STATE)


def completion(job: dict) -> bool:
    run = ROOT / "runs" / job["name"]
    if job["kind"] == "htdet":
        path = run / "status.json"
        if not path.exists():
            return False
        state = json.loads(path.read_text())
        return state["phase"] == "complete"
    return (run / "test_eval.json").exists()


def command(job: dict) -> list[str]:
    if job["kind"] == "htdet":
        return [PYTHON, "-u", str(ROOT / "run_htdet_duo.py"),
                "--model", job["model"], "--seed", str(job["seed"])]
    if job["kind"] == "yolov12":
        return [PYTHON, "-u", str(ROOT / "run_yolov12_duo.py"),
                "--size", job["model"], "--seed", str(job["seed"])]
    if job["kind"] == "script":
        return [PYTHON, "-u", str(ROOT / job["script"]),
                "--seed", str(job["seed"])]
    raise ValueError(job)


def main() -> None:
    lock_path = ROOT / "queue.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(STATE.read_text()) if STATE.exists() else {"done": []}
        while True:
            schedule = json.loads(SCHEDULE.read_text())
            pending = [j for j in schedule["jobs"] if j["name"] not in state["done"]]
            if not pending:
                state.update(phase="waiting_for_schedule", updated=now())
                save(state)
                time.sleep(30)
                continue
            job = pending[0]
            if completion(job):
                state["done"].append(job["name"])
                state.update(phase="ready", updated=now())
                save(state)
                continue
            if job["name"] == "e2_seed0":
                run_status = ROOT / "runs/e2_seed0/status.json"
                if run_status.exists():
                    phase = json.loads(run_status.read_text())["phase"]
                    if phase == "failed":
                        raise RuntimeError("Standalone E2 seed0 failed")
                    if phase == "training":
                        state.update(phase="waiting_e2_seed0", updated=now())
                        save(state)
                        time.sleep(30)
                        continue
            cmd = command(job)
            state.update(phase="running", current=job["name"], updated=now())
            save(state)
            log_path = ROOT / "logs" / (job["name"] + ".queue.log")
            with log_path.open("a") as log:
                log.write(f"\n{now()} COMMAND {' '.join(cmd)}\n")
                log.flush()
                try:
                    subprocess.run(cmd, cwd=str(ROOT), stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
                except subprocess.CalledProcessError as exc:
                    state.update(phase="failed", current=job["name"],
                                 returncode=exc.returncode, updated=now())
                    save(state)
                    raise
            if not completion(job):
                raise RuntimeError(f"Job exited without test results: {job['name']}")
            state["done"].append(job["name"])
            state.update(phase="ready", current=None, updated=now())
            save(state)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
