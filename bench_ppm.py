import torch
import time

torch.backends.cudnn.benchmark = True
from htdet.models.ppm import PPM
from htdet.models.detector import Detector

m = PPM(3, 16, 16, to_rgb=True).cuda()
x = torch.rand(2, 3, 416, 416, device="cuda")

m.train()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    for _ in range(3):
        m(x)
torch.cuda.synchronize()
t0 = time.time()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    for _ in range(10):
        m(x)
torch.cuda.synchronize()
print("PPM full (incl VGG x2): %.3f s/iter" % ((time.time() - t0) / 10))

m.use_perc = False
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    for _ in range(3):
        m(x)
torch.cuda.synchronize()
t0 = time.time()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    for _ in range(10):
        m(x)
torch.cuda.synchronize()
print("PPM no-perc: %.3f s/iter" % ((time.time() - t0) / 10))

t0 = time.time()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    for _ in range(10):
        m.vgg((x - 0.5) / 0.25)
torch.cuda.synchronize()
print("VGG x1: %.3f s" % ((time.time() - t0) / 10))

# baseline R2 (no ppm) full forward_train
det = Detector(backbone_cfg={"name": "mobilevit", "variant": "xs", "pretrained": False},
               neck_cfg={"name": "fpn"}, levels=(4, 8, 16, 32), num_classes=4).cuda()
meta = [{"ori_shape": (405, 720), "scale_factor": (1.0, 1.0)}] * 2
gbb = [torch.tensor([[10.0, 10.0, 50.0, 50.0]], device="cuda")] * 2
glb = [torch.tensor([1], device="cuda")] * 2
for _ in range(3):
    det.forward_train(x, gbb, glb, meta)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    det.forward_train(x, gbb, glb, meta)
torch.cuda.synchronize()
print("R2 forward_train (no ppm, no backward): %.3f s/iter" % ((time.time() - t0) / 10))
