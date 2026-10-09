# ¿Qué es? — Real-Time Spanish Vocabulary Recognition
# Author: Ruben A. Salido
# Northeastern University | CS5330: Pattern Recognition and Computer Vision
# Originally developed: Spring 2026

"""
This is the main entry point for ¿Qué es? - A Spanish Vocabulary CV App. 
It initializes the camera, loads the object detection models, and runs the main loop to 
capture frames, run inference, and displays the results with Spanish labels. 

This file can benchmark the performance with either YOLO(v8) or MobileNetDetector 
implementations.  It records accuracy with a confidence score, speed with FPS,
and detection counts for each target object. The results are printed on the terminal and saved
to a CSV file for later analysis. 

CONTROLS: 
     S      -Saves a snapshot as a numbered PNG file
     Q/Esc  - Quit the application

To run on Mac with MPS fallback (for Apple Silicon): 
    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python src/main.py --model yolo
    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python src/main.py --model mobilenet
    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python src/main.py --model compare --frames 100

"""

import argparse
import time
from pathlib import Path
 
from camera import CameraStream
from detector import YOLODetector, MobileNetDetector
from spanish_mapper import SpanishLabelMapper
from display import DisplayManager
from evaluator import ModelEvaluator


 
SNAPSHOT_DIR = Path("snapshots")
 
def get_next_snapshot_path() -> Path:
    """
    Scans the snapshots folder for existing cvSpanishImage_XX.png files
    and returns the next sequential path.
 
    Ex. If cvSpanishImage_09.png exists, returns 'snapshots/cvSpanishImage_10.png'
 
    Returns:
        A Path object for the next snapshot file.
    """
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(SNAPSHOT_DIR.glob("cvSpanishImage_*.png"))
 
    if not existing:
        next_number = 1
    else:
        last_file   = existing[-1].stem
        last_number = int(last_file.split("_")[-1])
        next_number = last_number + 1
 
    return SNAPSHOT_DIR / f"cvSpanishImage_{next_number:02d}.png"


def parse_args():
    """
    Parses command-line arguments.
 
    Returns:
        Namespace with a 'model' attribute: "yolo" or "mobilenet".
    """
    parser = argparse.ArgumentParser(
        description="¿Qué es? — Real-time Spanish Vocabulary CV App"
    )

    parser.add_argument(
        "--model",
        type = str,
        choices = ["yolo", "mobilenet", "compare"],
        default = "yolo",
        help = "yolo | mobilenet | compare",
    )

    parser.add_argument(
        "--frames",
        type = int,
        default = 50,
        help="Number of frames for comparison mode",
    )
    
    return parser.parse_args()


def build_detector(model_name: str):
    """
    Initializes and returns the specified object detector.
 
    Args:
        model_name: "yolo" or "mobilenet"
 
    Returns:
        An instance of YOLODetector or MobileNetDetector.
    """
    if model_name == "mobilenet":
        print("Loading MobileNet SSD...")
        return MobileNetDetector()
    else:
        print("Loading YOLOv8...")
        return YOLODetector()


def run_live(model_name: str):
    """
    Runs the live webcam mode with the specified model. Captures frames, runs inference,
    and displays results with Spanish labels and a HUD overlay.
    Args:
        model_name: "yolo" or "mobilenet"
    Returns:
        None
    """
    detector = build_detector(model_name)
    mapper = SpanishLabelMapper()
    display = DisplayManager()
    model_label = "YOLO" if model_name == "yolo" else "MobileNet"
    fps = 0.0
    frame_time = time.perf_counter()

    print(f"Running LIVE mode with [model_label]")
    print(f"Press S to save, Q/Esc to quit.")

    with CameraStream() as cam:
        while True:
            frame = cam.read()
            if frame is None:
                continue

            detections = detector.detect(frame)
            counts = detector.count_detections(detections)
            spanish_labels = mapper.get_all_labels(counts)
            annotated = display.draw(frame, detections, spanish_labels)
            annotated = display.draw_hud(
            annotated, model_label, len(detections), fps
            )

            display.show(annotated)

            now = time.perf_counter()
            fps = 1.0 / (now - frame_time) if (now - frame_time) > 0 else 0
            frame_time = now

            key = display.get_key()

            if key in (ord("s"), ord("S")):
                path = get_next_snapshot_path()
                display.save_snapshot(annotated, path)
                print(f"saved: {path}")
            elif key in (ord("q"), ord("Q"), 27):
                break
    display.close()
    print("Thank you, have a nice day!")


def run_comparison(frame_count: int):
    print(f"Running COMPARISON mode with {frame_count} frames...")
    evaluator = ModelEvaluator(frame_count = frame_count)
    evaluator.run()


def main():
    args = parse_args()

    if args.model == "compare":
        run_comparison(args.frames)
    elif args.model == "mobilenet":
        run_live("mobilenet")
    else:
        run_live("yolo")

 
if __name__ == "__main__":
    main()
 