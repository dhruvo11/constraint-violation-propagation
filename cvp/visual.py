"""Visual branch h_theta: ResNet-18 encoder + U-Net decoder over RGB (+) ELA."""
import io
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "visual_resunet.pt")
MEAN = torch.tensor([0.485, 0.456, 0.406, 0.1])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225, 0.15])[:, None, None]


def ela(arr, q=90, scale=20.0):
    """Error Level Analysis: |I - JPEG_q(I)|, averaged over channels, scaled to [0, 1]."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=q)
    re = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), np.float32)
    return np.clip(np.abs(arr.astype(np.float32) - re).mean(2) / scale, 0, 1)


def to_input(arr):
    x = np.concatenate([arr.astype(np.float32) / 255.0, ela(arr)[..., None]], 2)
    return (torch.from_numpy(x).permute(2, 0, 1) - MEAN) / STD


class Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(cin + cskip, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True),
                                  nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class ResUNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None)
        conv1 = nn.Conv2d(4, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            conv1.weight[:, :3] = r.conv1.weight
            conv1.weight[:, 3:] = r.conv1.weight.mean(1, keepdim=True)
        self.stem = nn.Sequential(conv1, r.bn1, r.relu)          # /2, 64
        self.pool = r.maxpool
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4  # /4 .. /32
        self.full = nn.Sequential(nn.Conv2d(4, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(True))
        self.u4, self.u3 = Up(512, 256, 256), Up(256, 128, 128)
        self.u2, self.u1 = Up(128, 64, 64), Up(64, 64, 32)
        self.u0 = Up(32, 16, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x, return_embedding=False):
        s0 = self.full(x)
        s1 = self.stem(x)
        e1 = self.l1(self.pool(s1))
        e2 = self.l2(e1)
        e3 = self.l3(e2)
        e4 = self.l4(e3)
        d = self.u4(e4, e3)
        d = self.u3(d, e2)
        d = self.u2(d, e1)
        d = self.u1(d, s1)
        d = self.u0(d, s0)
        logits = self.head(d)
        if return_embedding:
            return logits, e4.mean((2, 3))
        return logits


def load(device="cpu"):
    m = ResUNet(pretrained=False)
    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    return m.eval().to(device)


@torch.no_grad()
def heatmap(model, arrs, device):
    """Full-page inference. Returns (H in [0,1] as B x H x W numpy, 512-d embeddings)."""
    x = torch.stack([to_input(a) for a in arrs]).to(device, dtype=next(model.parameters()).dtype)
    logits, emb = model(x, return_embedding=True)
    return torch.sigmoid(logits.float())[:, 0].cpu().numpy(), emb.float().cpu().numpy()


@torch.no_grad()
def embed(model, arrs, device):
    """Encoder-only pass: the 512-d global embedding used by the concatenation baseline."""
    x = torch.stack([to_input(a) for a in arrs]).to(device, dtype=next(model.parameters()).dtype)
    e = model.l4(model.l3(model.l2(model.l1(model.pool(model.stem(x))))))
    return e.mean((2, 3)).float().cpu().numpy()
