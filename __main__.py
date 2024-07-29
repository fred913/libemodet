# Initialize the detection engine
# assert cuda is available
import time

import torch

assert torch.cuda.is_available(), "CUDA is not available. Please check your installation."
print("Available CUDA devices: ", torch.cuda.device_count())

from pathlib import Path

import cv2

from libemodet import DetectionEngine

engine = DetectionEngine(model_path=Path(__file__).parent / "weights" / "yolov7-tiny.pt", device='0', img_size=512)

NUM = 200

st = time.time()
for _ in range(NUM):
    res = engine.detect(cv2.imread("PixPin_2024-07-29_16-51-59.png"))

cost = time.time() - st
print(f"total {NUM} images, cost total {cost:.2f}s, average {cost/NUM:.2f}s per image")
print(res)
