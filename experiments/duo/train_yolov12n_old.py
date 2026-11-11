"""YOLOv12n on URPC2018: mosaic + HSV + flip + translate (NO scale/mixup/copy_paste).
Same as train_yolov12n_urpc.py except scale=0 (official n had scale=0.5);
mosaic=1.0 with close_mosaic=10; HSV defaults; translate=0.1; fliplr=0.5.
Other: 416, batch=64, epochs=200, COCO init, SGD (0.01/0.937/5e-4), EMA disabled, no val.
"""
from ultralytics import YOLO


def main():
    model = YOLO("/root/autodl-tmp/yolov12/yolov12n.pt")  # COCO turbo weights
    model.train(
        data="/root/autodl-tmp/data/yolo_urpc/data.yaml",
        epochs=200,
        imgsz=416,
        batch=64,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        # ---- augmentation: mosaic + HSV + flip + translate ----
        mosaic=1.0,
        close_mosaic=10,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        fliplr=0.5,
        scale=0.0,
        mixup=0.0,
        copy_paste=0.0,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        erasing=0.0,
        auto_augment=None,
        device=0,
        workers=8,
        project="/root/autodl-tmp/work_dirs",
        name="yolov12n_urpc_mosaic_hsv",
        exist_ok=True,
        seed=0,
        patience=0,
        save_period=20,
    )
    print("YOLOV12N_MOSAIC_HSV TRAIN DONE")


if __name__ == "__main__":
    main()
