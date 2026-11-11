"""Fine-tune YOLOv8n on URPC2018 with official Ultralytics defaults.

- weights: yolov8n.pt (COCO pretrained, ~6.2MB)
- imgsz 416 (HTDet paper convention), epochs 50, batch 16
- default augmentations (mosaic, hsv, flip) — standard YOLOv8 recipe
- project: /root/autodl-tmp/work_dirs, name: yolov8n_urpc_416_50ep
"""
from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data="/root/autodl-tmp/data/yolo_urpc/data.yaml",
        epochs=50,
        imgsz=416,
        batch=16,
        device=0,
        project="/root/autodl-tmp/work_dirs",
        name="yolov8n_urpc_416_50ep",
        seed=0,
        workers=12,
        patience=0,          # no early stopping, train full 50 epochs
        save_period=10,
    )


if __name__ == "__main__":
    main()
