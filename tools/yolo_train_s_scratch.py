"""YOLOv8s FROM SCRATCH on URPC2018, paper-aligned recipe (option B).

- weights: yolov8s.yaml (architecture only, zero pretraining)
- input: 416x416 direct stretch (images pre-resized in yolo_urpc_416s)
- SGD lr0=0.05 momentum 0.9 wd 1e-4; 24 epochs; lr x0.1 at epoch 16/22
- warmup 2500 iters (paper absolute value)
- augment: horizontal flip only (mosaic/hsv/scale/translate/erasing all off)
"""
from ultralytics import YOLO


def step_decay(trainer):
    if trainer.epoch in (16, 22):
        for g in trainer.optimizer.param_groups:
            g["lr"] *= 0.1
        sched = getattr(trainer, "lr_scheduler", None)
        if sched is not None and hasattr(sched, "base_lrs"):
            sched.base_lrs = [b * 0.1 for b in sched.base_lrs]
        print(f"[step-decay] lr x0.1 at epoch {trainer.epoch}", flush=True)


def main():
    model = YOLO("yolov8s.yaml")
    model.add_callback("on_train_epoch_start", step_decay)
    model.train(
        data="/root/autodl-tmp/data/yolo_urpc_416s/data.yaml",
        epochs=24,
        imgsz=416,
        batch=16,
        optimizer="SGD",
        lr0=0.05,
        momentum=0.9,
        weight_decay=0.0001,
        cos_lr=False,
        lrf=1.0,
        warmup_iters=2500,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        augment=False,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        scale=0.0,
        translate=0.0,
        erasing=0.0,
        fliplr=0.5,
        close_mosaic=0,
        device=0,
        project="/root/autodl-tmp/work_dirs",
        name="yolov8s_urpc_paper_scratch",
        seed=0,
        workers=12,
        patience=0,
        save_period=8,
    )


if __name__ == "__main__":
    main()
