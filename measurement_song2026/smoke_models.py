"""Synthetic training/checkpoint smoke test; never paper experiment results."""
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from models import RebarLightTrainer


def main():
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="song_smoke_") as temp:
        root = Path(temp)
        for split in ("train", "val"):
            for folder in ("images", "labels"):
                (root/folder/split).mkdir(parents=True)
            for i in range(2):
                image = np.zeros((64, 64, 3), dtype=np.uint8)
                image[:, 20:30] = 200
                cv2.imwrite(str(root/"images"/split/(str(i)+".png")), image)
                (root/"labels"/split/(str(i)+".txt")).write_text("0 0.390625 0.5 0.15625 1.0\n")
        cfg = root/"data.yaml"
        cfg.write_text("path: "+json.dumps(str(root))+"\ntrain: images/train\nval: images/val\nnames:\n  0: vertical_rebar\n")
        for name in ("yolov8n", "yolo11n", "rebar_light"):
            model = YOLO("yolov8n.yaml" if name == "yolov8n" else "yolo11n.yaml")
            options = dict(data=str(cfg), epochs=1, batch=2, imgsz=64, device="cpu",
                           workers=0, amp=False, pretrained=False, plots=False,
                           optimizer="SGD", lr0=.01, lrf=.01, patience=0,
                           project=str(root/"runs"), name=name, verbose=False)
            if name == "rebar_light":
                options["trainer"] = RebarLightTrainer
            model.train(**options)
            restored = YOLO(str(root/"runs"/name/"weights"/"best.pt"))
            restored.predict(np.zeros((64, 64, 3), dtype=np.uint8), imgsz=64, verbose=False)
            print("PASS synthetic train/save/reload/predict:", name, flush=True)


if __name__ == "__main__":
    main()
