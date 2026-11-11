from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('train_yaml/hyuod.yaml')
    model.train(data='/root/autodl-tmp/data/hyuod_urpc/hyuod_urpc.yaml',
                project='/root/autodl-tmp/work_dirs',
                name='hyuod_urpc_416_noscale',
                epochs=400,
                batch=16,
                optimizer='SGD',
                pretrained=False,
                imgsz=416,
                scale=0.0,
                iou=0.4,
                seed=0,
                )
