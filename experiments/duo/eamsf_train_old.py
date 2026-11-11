import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/myconfig/eamsf-detr.yaml')
    model.train(data='/root/autodl-tmp/data/yolo_urpc/data.yaml',
                cache=False,
                imgsz=416,
                epochs=300,
                batch=32,
                workers=4,
                device='0',
                scale=0.0,
                project='/root/autodl-tmp/work_dirs',
                name='eamsf_detr_urpc_416_noscale',
                seed=0,
                )
