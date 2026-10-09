# ¿Qué es? — Real-Time Spanish Vocabulary Recognition
# Author: Ruben A. Salido
# Northeastern University | CS5330: Pattern Recognition and Computer Vision
# Originally developed: Spring 2026

"""
Manages webcam capture using OpenCV, CameraStream handles opening reading, and 
releasing the camera cleanly. Provides context manager support for safe resource
management. Designed for use in the main application loop to capture frames for
object detection and display.
"""


import cv2
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT


class CameraStream:
    """
    Wraps OpenCV VideoCapture for webcam access.

    Usage:
        cam = CameraStream()
        cam.open()
        frame = cam.read()
        cam.release()

    Or use as a context manager:
        with CameraStream() as cam:
            frame = cam.read()
    """

    def __init__(self):
        self.cap = None


    def open(self):
        """
        Opens the webcam specified by CAMERA_INDEX in config.py.
        Raises RuntimeError if the camera cannot be opened.
        """
        self.cap = cv2.VideoCapture(CAMERA_INDEX)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera at index {CAMERA_INDEX}. "
                "Check that your webcam is connected and not in use."
            )

        # Apply resolution from config
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)


    def read(self):
        """
        Reads a single frame from the webcam.

        Returns:
            A BGR numpy array (H x W x 3) if successful.
            None if the frame could not be read.
        """
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Camera is not open. Call open() first.")

        success, frame = self.cap.read()

        if not success:
            return None

        return frame


    def release(self):
        """
        Releases the webcam and frees the resource.
        Safe to call even if the camera was never opened.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None


    def is_open(self) -> bool:
        """Returns True if the camera is currently open."""
        return self.cap is not None and self.cap.isOpened()


    # ------------------------------------------------------------------
    # Context manager support — allows: with CameraStream() as cam:
    # ------------------------------------------------------------------

    def __enter__(self):
        self.open()
        return self


    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.release()
        return False
