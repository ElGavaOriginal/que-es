# ¿Qué es? — Real-Time Spanish Vocabulary Recognition
# Author: Ruben A. Salido
# Northeastern University | CS5330: Pattern Recognition and Computer Vision
# Originally developed: Spring 2026


"""
 This function defines the ObjectDetector base class and two concrete implementations:
   - YOLODetector    (YOLOv8 via ultralytics)
   - MobileNetDetector (MobileNet SSD v2 via torchvision)

 Both subclasses return detections in the same format:
   [
       {
           "label":      str,   # COCO class name, e.g. "dog"
           "confidence": float, # 0.0 - 1.0
           "box":        (x1, y1, x2, y2)  # pixel coordinates
       },
       ...
    ]
 Only labels present in TARGET_OBJECTS (config.py) are returned.
"""

from abc import ABC, abstractmethod

import cv2
import torch
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
)
from ultralytics import YOLO

from config import CONFIDENCE_THRESHOLD, TARGET_OBJECTS


# ---------------------------------------------------------------------------
# Abstract base class is included within detector.py for simplicity, 
# since there are only two detector implementations. This was a conscious 
# design choice to keep all detection-related code in one file. 
# ---------------------------------------------------------------------------

class ObjectDetector(ABC):
    """
    Abstract base class for object detectors.
    All subclasses must implement the detect() method.
    """

    @abstractmethod
    def detect(self, frame) -> list[dict]:
        """
        Run object detection on a single BGR frame.

        Args:
            frame: A BGR numpy array (H x W x 3) from OpenCV.

        Returns:
            A list of detection dicts, each containing:
                "label":      COCO class name (str)
                "confidence": score between 0.0 and 1.0 (float)
                "box":        (x1, y1, x2, y2) pixel coords (tuple of ints)
        """
        pass


    def _filter_targets(self, detections: list[dict]) -> list[dict]:
        """
        Filters detections to only include labels in TARGET_OBJECTS
        and above CONFIDENCE_THRESHOLD.
        """
        return [
            d for d in detections
            if d["label"] in TARGET_OBJECTS
            and d["confidence"] >= CONFIDENCE_THRESHOLD
        ]


    def count_detections(self, detections: list[dict]) -> dict:
        """
        Counts how many instances of each label appear in a detection list.
        This is used to generate the summary text in the HUD overlay. Making it 
        easier to compare YOLO and MobileNet performance in real-time.

        Args:
            detections: List of detection dicts from detect().

        Returns:
            Dict of {label: count}, e.g. {"dog": 2, "book": 1}.
        """
        counts = {}
        for d in detections:
            counts[d["label"]] = counts.get(d["label"], 0) + 1
        return counts


# ---------------------------------------------------------------------------
# YOLO detector
# ---------------------------------------------------------------------------

class YOLODetector(ObjectDetector):
    """
    Object detector using YOLOv8 (nano) via the ultralytics library.
    The model file is downloaded automatically on first use if not present.
    """

    # COCO class names in YOLO index order
    COCO_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",
    ]


    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Args:
            model_path: Path to the YOLOv8 weights file.
                        Defaults to "yolov8n.pt" (nano — fast, lightweight).
                        ultralytics will auto-download it if not found.
        """
        self.model = YOLO(model_path)


    def detect(self, frame) -> list[dict]:
        """
        Runs YOLOv8 inference on a single BGR frame.

        Returns filtered detections as a list of dicts.
        """
        results = self.model(frame, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                class_id   = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                label = (
                    self.COCO_CLASSES[class_id]
                    if class_id < len(self.COCO_CLASSES)
                    else str(class_id)
                )

                detections.append({
                    "label":      label,
                    "confidence": confidence,
                    "box":        (x1, y1, x2, y2),
                })

        return self._filter_targets(detections)


# ---------------------------------------------------------------------------
# MobileNet detector
# ---------------------------------------------------------------------------

class MobileNetDetector(ObjectDetector):
    """
    Object detector using SSDLite320 + MobileNetV3-Large via torchvision.
    Pretrained on COCO. Runs on CPU by default; uses MPS on Apple Silicon
    if available.
    """

    # COCO class names in torchvision index order (index 0 = background)
    COCO_CLASSES = [
        "__background__", "person", "bicycle", "car", "motorcycle",
        "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "N/A", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
        "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "N/A",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
        "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
        "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
        "N/A", "dining table", "N/A", "N/A", "toilet", "N/A", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock",
        "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    ]


    def __init__(self):
        # Use MPS (Apple Silicon GPU) if available, otherwise CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.model = ssdlite320_mobilenet_v3_large(weights=weights)
        self.model.to(self.device)
        self.model.eval()

        # Standard ImageNet normalization transform
        self.transform = weights.transforms()


    def detect(self, frame) -> list[dict]:
        """
        Runs MobileNet SSD inference on a single BGR frame.

        Converts the OpenCV BGR frame to an RGB tensor before inference.
        Returns filtered detections as a list of dictions. 
        """
        # OpenCV uses BGR — convert to RGB for torchvision
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to tensor and apply model transforms
        from torchvision.transforms.functional import to_tensor
        img_tensor = to_tensor(rgb_frame).to(self.device)
        img_transformed = self.transform(img_tensor).unsqueeze(0)

        with torch.no_grad():
            predictions = self.model(img_transformed)

        detections = []
        pred = predictions[0]

        for box, label_idx, score in zip(
            pred["boxes"], pred["labels"], pred["scores"]
        ):
            confidence = float(score)
            class_id = int(label_idx)

            label = (
                self.COCO_CLASSES[class_id]
                if class_id < len(self.COCO_CLASSES)
                else str(class_id)
            )

            if label == "N/A":
                continue

            x1, y1, x2, y2 = map(int, box.tolist())

            detections.append({
                "label": label,
                "confidence": confidence,
                "box": (x1, y1, x2, y2),
            })

        return self._filter_targets(detections)
    