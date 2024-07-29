import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from .emotion import detect_emotion, init
from .models.experimental import attempt_load
from .utils.datasets import LoadImages, LoadStreams
from .utils.general import check_img_size, check_imshow, check_requirements, create_folder, non_max_suppression, scale_coords, set_logging
from .utils.plots import plot_one_box
from .utils.torch_utils import select_device, time_synchronized


def detect(source,
           img_size=512,
           conf_thres=0.5,
           iou_thres=0.45,
           device='cuda',
           hide_img=False,
           output_path="output.png",
           agnostic_nms=False,
           augment=False,
           line_thickness=2,
           hide_conf=False,
           show_fps=False):
    with torch.no_grad():
        view_img, imgsz, show_conf, save_path = not hide_img, img_size, not hide_conf, output_path
        # Directories
        create_folder(save_path)

        # Initialize
        set_logging()
        device = select_device(device)
        init(device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(Path(__file__).parent / "weights" / "yolov7-tiny.pt", map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            model.half()  # to FP16

        dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = ((0, 52, 255), (121, 3, 195), (176, 34, 118), (87, 217, 255), (69, 199, 79), (233, 219, 155), (203, 139, 77), (214, 246, 255))

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        t0 = time.time()
        for src_path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=agnostic_nms)
            t2 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                s, im0, frame = '', im0s.copy(), getattr(dataset, 'frame', 0)

                assert isinstance(im0, np.ndarray), 'im0 is not a numpy array'

                src_path = Path(src_path)  # to Path
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    images = []

                    for *xyxy, conf, cls in reversed(det):
                        x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                        images.append(im0.astype(np.uint8)[int(y1):int(y2), int(x1):int(x2)])

                    if images:
                        emotions = detect_emotion(images, show_conf)
                    # Write results
                    i = 0
                    for *xyxy, conf, cls in reversed(det):
                        if view_img:
                            # Add bbox to image with emotions on
                            label = emotions[i][0]
                            colour = colors[emotions[i][1]]
                            i += 1
                            plot_one_box(xyxy, im0, label=label, color=colour, line_thickness=line_thickness)

                # show results
                if view_img:
                    display_img = cv2.resize(im0, (im0.shape[1] * 2, im0.shape[0] * 2))
                    cv2.imshow("Emotion Detection", display_img)
                    cv2.waitKey(1)  # 1 millisecond

                cv2.imwrite(save_path, im0)


def main():
    detect("a.png")


if __name__ == '__main__':
    main()
