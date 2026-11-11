import torch
import time

torch.backends.cudnn.benchmark = True
from htdet.models.ppm import PPM

m = PPM(3, 16, 16, to_rgb=True).cuda()
x = torch.rand(2, 3, 416, 416, device="cuda")

m.train()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    for _ in range(3):
        xp, ls = m(x)
        (ls["loss_uie"] + ls["loss_sr"]).backward()
torch.cuda.synchronize()
t0 = time.time()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    for _ in range(10):
        xp, ls = m(x)
        (ls["loss_uie"] + ls["loss_sr"]).backward()
torch.cuda.synchronize()
print("PPM fwd+bwd: %.3f s/iter" % ((time.time() - t0) / 10))

# breakdown: no-perc backward
m.use_perc = False
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    for _ in range(3):
        xp, ls = m(x)
        (ls["loss_uie"] + ls["loss_sr"]).backward()
torch.cuda.synchronize()
t0 = time.time()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    for _ in range(10):
        xp, ls = m(x)
        (ls["loss_uie"] + ls["loss_sr"]).backward()
torch.cuda.synchronize()
print("PPM fwd+bwd no-perc: %.3f s/iter" % ((time.time() - t0) / 10))

# SR branch alone fwd+bwd
sr = m.sr
t0 = time.time()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    for _ in range(10):
        f, y = sr(x)
        torch.nn.functional.mse_loss(y, x).backward()
torch.cuda.synchronize()
print("SR fwd+bwd(MSE): %.3f s/iter" % ((time.time() - t0) / 10))

# UIE branch alone fwd+bwd (TV only)
uie = m.uie
t0 = time.time()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    for _ in range(10):
        f, y = uie(x)
        (m.uie_tv_w * m._tv_loss(y)).backward()
torch.cuda.synchronize()
print("UIE fwd+bwd(TV): %.3f s/iter" % ((time.time() - t0) / 10))
