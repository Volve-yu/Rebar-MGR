"""Documented reconstruction: unpublished dimensions remain assumptions."""
import torch
from torch import nn
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect import DetectionTrainer


class GlobalContext(nn.Module):
    def __init__(self, channels, ratio=0.25):
        super().__init__()
        hidden = max(1, int(channels*ratio))
        self.mask = nn.Conv2d(channels, 1, 1)
        self.transform = nn.Sequential(nn.Conv2d(channels, hidden, 1),
                                       nn.ReLU(), nn.Conv2d(hidden, channels, 1))

    def forward(self, x):
        weights = self.mask(x).flatten(2).softmax(-1)
        context = torch.bmm(x.flatten(2), weights.transpose(1, 2))
        return x+self.transform(context.unsqueeze(-1))


class LiteBlockDG(nn.Module):
    def __init__(self, c1, c2, expansion=0.5, context_ratio=0.25):
        super().__init__()
        hidden = max(1, int(c2*expansion))
        self.body = nn.Sequential(
            nn.Conv2d(c1, hidden, 1, bias=False),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.Conv2d(hidden, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(),
            GlobalContext(hidden, context_ratio),
            nn.Conv2d(hidden, c2, 1, bias=False), nn.BatchNorm2d(c2), nn.ReLU())
        self.skip = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1, bias=False)

    def forward(self, x):
        return self.body(x)+self.skip(x)


def reconstruct(model):
    # Figure 2 retains the final backbone C3k2 (index 8). The description
    # says replace backbone/neck C3k2; follow the diagram and disclose this.
    for index in (2, 4, 6, 13, 16, 19, 22):
        old = model.model[index]
        if old.__class__.__name__ != "C3k2":
            raise ValueError(f"Unexpected YOLO11 graph at layer {index}: {type(old)}")
        c1, c2 = old.cv1.conv.in_channels, old.cv2.conv.out_channels
        repeats = len(old.m)
        block = nn.Sequential(*[LiteBlockDG(c1 if i == 0 else c2, c2)
                                for i in range(repeats)])
        for key in ("i", "f", "type"):
            setattr(block, key, getattr(old, key))
        block.np = sum(p.numel() for p in block.parameters())
        model.model[index] = block
    return model


class RebarLightTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = reconstruct(DetectionModel(cfg, nc=self.data["nc"], verbose=verbose))
        if weights:
            raise ValueError("Use pretrained=False for the reconstruction; no matched pretrained weights supplied")
        return model
