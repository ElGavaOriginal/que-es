# ¿Qué es? — Real-Time Spanish Vocabulary Recognition
# Author: Ruben A. Salido
# Northeastern University | CS5330: Pattern Recognition and Computer Vision
# Originally developed: Spring 2026
"""
This file benchmarks the performance of the YOLO(v8) and MobileNetDetector implementations 
on a set of captured frames. It records accuracy with a confidence score, speed with FPS,
and detection counts for each target object. The results are printed on the terminal and saved
to a CSV file for later analysis. 

note to self: to run directly: .venv/bin/python src/evaluator.py

"""

import csv
import time
from pathlib import Path
from camera import CameraStream
from detector import YOLODetector, MobileNetDetector


class ModelEvaluator:
    """
    Benchmarks two object detectors side by side on live webcam frames. 
    The metrics collected per model include: 
       - Average inference time per frame level
       - Frames Per Second
       - Total detections across all frames
       - Average confidence score across all detections
       - Per-label detection counts

    Usage: 
       - evaluator = ModelEvaluator()
       - evaluator.run()
    """


    def __init__(self, frame_count: int = 50, output_path: str = "benchmark_results.csv"):  # not sure about frame_count: int = 50, yet.
        """
        Args:
            frame_count: Number of frames to capture and benchmark on.
            output_path: Path to save the CSV results.
        """
        self.frame_count = frame_count
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents = True, exist_ok = True)    


    def run(self):
        print(f"Captruting {self.frame_count} frames from webcam for benchmarking...")
        frames = self._capture_frames()
        if not frames:
            print("Error: no frames captured. Check your webcam.")
            return
        
        print (f"Running YOLO...")
        yolo_results = self._benchmark(YOLODetector(), "YOLO", frames)
        print (f"Running MobileNet...")
        mobilenet_results = self._benchmark(MobileNetDetector(), "MobileNet", frames)

        self._save_csv([yolo_results, mobilenet_results])
        self._print_comparison(yolo_results, mobilenet_results)


    def _capture_frames(self):
        frames = []
        with CameraStream() as cam:
            while len(frames) < self.frame_count:
                frame = cam.read()
                if frame is not None:
                    frames.append(frame)
        return frames


    def _benchmark(self, detector, model_name, frames):
        total_time = 0.0
        total_detections = 0
        total_confidence = 0.0
        label_counts = {}

        for i, frame in enumerate(frames):
            start = time.perf_counter()
            detections = detector.detect(frame)
            elapsed = time.perf_counter() - start
            total_time += elapsed
            total_detections += len(detections)

            for d in detections:
                total_confidence += d["confidence"]
                label = d["label"]
                label_counts[label] = label_counts.get(label, 0) + 1

            # Indicates progress every 50 frames. This allows user to see that 
            # the benchmarking is still running, especially for MobileNet which is much slower.
            if (i + 1) % 50 == 0:
                print(f" {i + 1}/{len(frames)} frames processed...")

        avg_ms = (total_time / len(frames)) * 1000
        fps = 1000 / avg_ms if avg_ms > 0 else 0
        avg_confidence = (total_confidence / total_detections
                         if total_detections > 0 else 0.00)

        results = {
            "model_name": model_name,
            "avg_ms": round(avg_ms, 2),
            "fps": round(fps, 2),
            "total_detections": total_detections,
            "avg_confidence": round(avg_confidence, 4),
            "label_counts": label_counts,
        }

        self._print_results(results)
        return results


    def _print_results(self, results: dict):
        """
        Prints a summary of benchmark results to the terminal
        """
        print(f"\n  --- {results['model_name']} Results ---")
        print(f"  Avg inference time : {results['avg_ms']} ms")
        print(f"  FPS                : {results['fps']}")
        print(f"  Total detections   : {results['total_detections']}")
        print(f"  Avg confidence     : {results['avg_confidence']:.1%}")
        print(f"  Labels detected    : {results['label_counts']}")

    def _print_comparison(self, yolo: dict, mobilenet: dict):
        """
        Prints a side-by-side comparions of both YOLO and MobileNet results to the terminal.
        """
        print("\n" + "=" * 50)
        print("  MODEL COMPARISON SUMMARY: YOLO vs MobileNet")
        print("=" * 50)
        print(f"  {'Metric':<25} {'YOLO':>10} {'MobileNet':>12}")
        print("-" * 50)
        print(f"  {'Avg inference (ms)':<25} {yolo['avg_ms']:>10} {mobilenet['avg_ms']:>12}")
        print(f"  {'FPS':<25} {yolo['fps']:>10} {mobilenet['fps']:>12}")
        print(f"  {'Total detections':<25} {yolo['total_detections']:>10} {mobilenet['total_detections']:>12}")
        print(f"  {'Avg confidence':<25} {yolo['avg_confidence']:>10.1%} {mobilenet['avg_confidence']:>12.1%}")
        print("=" * 50)
        print(f"\n  Results saved to: {self.output_path}")

    def _save_csv(self, results: list[dict]):
        """
        Saves the benchmark results to a CSV file for later analysis. 
        """
        fieldnames = ["model_name", "avg_ms", "fps", "total_detections", "avg_confidence", "label_counts"]

        with open(self.output_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({k: r[k] for k in fieldnames})

        print(f"\n CSV saved to {self.output_path}")


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run()
