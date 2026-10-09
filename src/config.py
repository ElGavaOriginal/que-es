# ¿Qué es? — Real-Time Spanish Vocabulary Recognition
# Author: Ruben A. Salido
# Northeastern University | CS5330: Pattern Recognition and Computer Vision
# Originally developed: Spring 2026


# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------

# YOLOv8 nano — small and fast, good for webcam use
YOLO_MODEL_PATH = "models/yolov8n.pt"

# ---------------------------------------------------------------------------
# Detection settings
# ---------------------------------------------------------------------------

# Minimum confidence score to accept a detection (0.0 - 1.0)
CONFIDENCE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Target object labels. Matches Common Objects in Context (COCO) objects in 
# 'Senderos' textbook vocabulary for Spanish language learning. Only objects 
# in this list will be displayed by the app.
# ---------------------------------------------------------------------------

TARGET_OBJECTS = [
    "refrigerator",
    "oven",
    "sink",
    "couch",
    "chair",
    "potted plant",
    "book",
    "dining table",
    "cup",
    "bowl",
    "fork",
    "knife",
    "spoon",
    "laptop",
    "keyboard",
    "mouse",
    "scissors",
    "cell phone",
    "backpack",
    "car",
]

# ---------------------------------------------------------------------------
# Display settings (OpenCV)
# ---------------------------------------------------------------------------

# Bounding box color in BGR format
BOX_COLOR = (0, 200, 100)       # green

# Text color for Spanish labels
LABEL_COLOR = (255, 255, 255)   # white

# Font scale and thickness for overlay text
FONT_SCALE = 0.7
FONT_THICKNESS = 2

# ---------------------------------------------------------------------------
# Camera settings
# ---------------------------------------------------------------------------

# 0 = default webcam. Change to 1, 2, etc. for external cameras.
CAMERA_INDEX = 0

FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

# ---------------------------------------------------------------------------
# Evaluation settings
# ---------------------------------------------------------------------------

# Number of frames to use when benchmarking models
EVAL_FRAME_COUNT = 200
