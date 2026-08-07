"""
PyTorch Vehicle Detection Wrapper with Optimised Input Downscaling
Uses SSDLite-MobileNetV3 for fast, accurate CPU vehicle detection.
Fixes: NMS to prevent duplicate boxes on same vehicle; higher conf threshold.
"""
import os
import cv2
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.ops import nms as torch_nms
import numpy as np

# Patch Windows OpenMP issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

COCO_VEHICLE_CLASSES = {
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
}


class VehicleDetector:
    def __init__(self, confidence_threshold=0.42, device="cpu"):
        self.conf_threshold = confidence_threshold
        self.device = torch.device(device)
        print(f"[VehicleDetector] Loading SSDLite MobileNetV3 on {self.device}...")
        self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
            weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        )
        self.model.to(self.device)
        self.model.eval()
        print("[VehicleDetector] Model ready.")

    def set_confidence(self, conf):
        self.conf_threshold = float(conf)

    def detect(self, cv2_bgr_frame):
        """
        Input:  cv2 BGR image (H, W, 3)
        Output: numpy array [[x1, y1, x2, y2, score], ...], list of class_names
        """
        orig_h, orig_w = cv2_bgr_frame.shape[:2]

        # Downscale for fast inference
        small_frame = cv2.resize(cv2_bgr_frame, (320, 320))
        rgb_frame = small_frame[:, :, ::-1].copy()  # BGR -> RGB

        tensor_img = F.to_tensor(rgb_frame).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.model(tensor_img)[0]

        boxes  = predictions["boxes"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()

        dets = []
        cls_names = []

        scale_x = orig_w / 320.0
        scale_y = orig_h / 320.0

        for box, score, label in zip(boxes, scores, labels):
            if score >= self.conf_threshold and label in COCO_VEHICLE_CLASSES:
                x1 = box[0] * scale_x
                y1 = box[1] * scale_y
                x2 = box[2] * scale_x
                y2 = box[3] * scale_y
                dets.append([x1, y1, x2, y2, score])
                cls_names.append(COCO_VEHICLE_CLASSES[label])

        if len(dets) == 0:
            return np.empty((0, 5)), []

        # ── NMS: suppress overlapping boxes from the same physical vehicle ──
        # Without this, one car can produce 2 detections → 2 track IDs → double count.
        if len(dets) > 1:
            boxes_t  = torch.tensor([[d[0], d[1], d[2], d[3]] for d in dets], dtype=torch.float32)
            scores_t = torch.tensor([d[4] for d in dets], dtype=torch.float32)
            keep     = torch_nms(boxes_t, scores_t, iou_threshold=0.40).tolist()
            dets      = [dets[i]      for i in keep]
            cls_names = [cls_names[i] for i in keep]

        return np.array(dets), cls_names
