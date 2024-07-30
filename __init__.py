from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

from libemodet.emotion import detect_emotion, init

from .models.experimental import attempt_load
from .utils.datasets import LoadImages, letterbox
from .utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from .utils.torch_utils import select_device


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class DetectionResult:
    label: str
    face_confidence: float
    emotion_confidence: float
    box: Box


class DetectionEngine:

    def __init__(self, model_path: str | Path, device: Optional[str] = None, img_size: int = 512) -> None:
        set_logging()
        self.device = select_device(device or "cpu")
        init(self.device)
        self.model_path = model_path
        self.model = attempt_load(self.model_path, map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.img_size = check_img_size(img_size, s=self.stride)
        self.model.to(self.device)
        if self.device.type != 'cpu':
            self.model.half()  # to FP16

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(self.model.parameters())))  # run once

    def img_preprocess(self, imgsrc: str | Path | np.ndarray) -> np.ndarray:
        if isinstance(imgsrc, str) or isinstance(imgsrc, Path):
            img = cv2.imread(str(imgsrc))
        else:
            img = imgsrc

        img: np.ndarray

        # Padded resize
        img = letterbox(img, self.img_size, stride=self.stride, auto=False)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img

    def detect_faces(self, src: np.ndarray, conf_thres: float = 0.5, iou_thres: float = 0.45) -> List[DetectionResult]:
        half = self.device.type != 'cpu'  # half precision only supported on CUDA
        im0s = src.copy()
        img: torch.Tensor = torch.from_numpy(self.img_preprocess(src)).to(self.device)
        img = img.half() if half else img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        det_results: List[DetectionResult] = []
        for det in pred:
            im0 = im0s.copy()
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2, *_ = xyxy
                    label = self.model.module.names[int(cls)] if hasattr(self.model, 'module') else self.model.names[int(cls)]
                    det_results.append(DetectionResult(label=label, face_confidence=conf.item(), box=Box(x1.item(), y1.item(), x2.item(), y2.item()),
                                                       emotion_confidence=0.0))  # Placeholder for emotion confidence

        return det_results

    def detect_emotions(self, target_faces: List[np.ndarray]) -> List[tuple]:
        if len(target_faces) > 0:
            return detect_emotion(target_faces, True)
        return []

    def detect(self, src: np.ndarray, conf_thres: float = 0.5, iou_thres: float = 0.45) -> List[DetectionResult]:
        face_results = self.detect_faces(src, conf_thres, iou_thres)
        target_faces = [src[int(res.box.y1):int(res.box.y2), int(res.box.x1):int(res.box.x2)] for res in face_results]

        emotions = self.detect_emotions(target_faces)

        for i, res in enumerate(face_results):
            if i < len(emotions):
                label: str = emotions[i][0]
                label_str, label_confidence = label.strip(")").split("(")
                label_str = label_str.strip().lower()
                label_confidence = float(label_confidence)

                res.label = label_str
                res.emotion_confidence = label_confidence

        return face_results


def quick_create_engine():
    return DetectionEngine(model_path=Path(__file__).parent / "weights" / "yolov7-tiny.pt", device='0', img_size=512)
