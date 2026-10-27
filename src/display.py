# ¿Qué es? — Real-Time Spanish Vocabulary Recognition
# Author: Ruben A. Salido
# Northeastern University | CS5330: Pattern Recognition and Computer Vision
# Originally developed: Spring 2026

"""
 This file handles all OpenCV rendering for the Spanish Vocabulary CV app.
 DisplayManager draws bounding boxes, Spanish label overlays, and a 
 heads-up display (HUD) on each frame.
"""

import cv2
from config import (
    BOX_COLOR,
    LABEL_COLOR,
    FONT_SCALE,
    FONT_THICKNESS,
)


class DisplayManager:
    """
    Renders detection results onto an OpenCV frame instead of returning raw
    detections. This keeps the main loop cleaner and allows us to easily add a
    HUD overlay with model and performance info.

    Usage:
        display = DisplayManager()
        annotated_frame = display.draw(frame, detections, spanish_labels)
        display.show(annotated_frame)
        if display.get_key() in (ord("q")), ord ("Q"), 27):
            break
    """

    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, window_name: str = "¿Qué es? — Spanish Vocabulary"):
        """
        Args:
            window_name: Title of the OpenCV display window.
        """
        self.window_name = window_name


    def draw(self, frame, detections: list[dict],
             spanish_labels: dict) -> object:
        """
        Draws bounding boxes and Spanish labels onto the frame. 
        By doing this in a separate method, we can keep the main loop cleaner and
        easily add a HUD overlay with model and performance info.

        Args:
            frame:          BGR numpy array from CameraStream.
            detections:     List of detection dicts from ObjectDetector.
            spanish_labels: Dict of {coco_label: spanish_phrase}
                            from SpanishLabelMapper.

        Returns:
            Annotated BGR frame.
        """
        for detection in detections:
            label = detection["label"]
            confidence = detection["confidence"]
            x1, y1, x2, y2 = detection["box"]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

            # Build display text: Spanish phrase + confidence score
            spanish_phrase = spanish_labels.get(label, label)
            display_text = f"{spanish_phrase}  ({confidence:.0%})"

            # Draw label background for readability
            (text_w, text_h), baseline = cv2.getTextSize(
                display_text, self.FONT, FONT_SCALE, FONT_THICKNESS
            )
            label_y = max(y1 - 10, text_h + 10)
            cv2.rectangle(
                frame,
                (x1, label_y - text_h - baseline),
                (x1 + text_w + 6, label_y + baseline),
                BOX_COLOR,
                thickness=cv2.FILLED,
            )

            # Draw Spanish label text
            cv2.putText(
                frame,
                display_text,
                (x1 + 3, label_y),
                self.FONT,
                FONT_SCALE,
                LABEL_COLOR,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )

        return frame


    def draw_hud(self, frame, model_name: str, fps: float) -> object:
        """
        Draws a HUD in the top-left corner that shows the active model,
        model name, and current FPS. This makes it easier to compare 
        YOLO and MobileNet performance in real-time.

        Args:
            frame:           BGR numpy array.
            model_name:      "YOLO" or "MobileNet".
            fps:             Current frames per second.

        Returns:
            Frame with HUD overlay added.
        """
        hud_lines = [
            f"Model: {model_name}",
            f"FPS: {fps:.1f}",
            "A: toggle article | S: snapshot | Q: quit",
        ]

        y_start = 30
        line_gap = 28
        hud_scale = 0.6
        hud_weight = 1

        widths = [
            cv2.getTextSize(line, self.FONT, hud_scale, hud_weight)[0][0]
            for line in hud_lines
        ]
        panel_w = max(widths) + 36
        panel_h = y_start + line_gap * (len(hud_lines))

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0),  cv2.FILLED)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for i, line in enumerate(hud_lines):
            y = y_start + i * line_gap
            #White text on dark panel
            cv2.putText(
                frame, line, (10, y),
                self.FONT, hud_scale, (255, 255, 255), hud_weight + 1, cv2.LINE_AA
            )

        return frame

    def show(self, frame):
        """
        Displays the annotated frame in the OpenCV window and handles user input. 
        This makes it easy to call display.show() once per frame in the main loop.

        Args:
            frame: Annotated BGR numpy array to display.
        """
        cv2.imshow(self.window_name, frame)


    def close(self):
        """
        Closes the OpenCV display window.
        Safe to call even if the window was never opened.
        """
        cv2.destroyAllWindows()


    def get_key(self) -> int:
        """
        Reads a key press from the OpenCV window. Returns the 
        key code, or -1 if no key was pressed. This allows the main 
        loop to check for user input without blocking. Call this 
        once per frame after .show().
        """
        return cv2.waitKey(1) & 0xFF
    

    def save_snapshot(self, frame, path):
        """
        Saves the annotated frame as a PNG file. This is useful for capturing 
        detections and sharing results. 

        Args:
            frame: Annotated BGR numpy array to save.
            path: A path object or string for the output file.
        """
        cv2.imwrite(str(path), frame)
