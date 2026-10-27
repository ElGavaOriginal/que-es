# ¿Qué es? — Real-Time Spanish Vocabulary Recognition
# Author: Ruben A. Salido
# Northeastern University | CS5330: Pattern Recognition and Computer Vision
# Originally developed: Spring 2026

# Unit tests for config.py
# test_config.py
# Unit tests for config.py
# Run with: .venv/bin/python -m unittest test_config.py -v

import unittest
import sys
sys.path.insert(0, "./src")
from config import (
    CONFIDENCE_THRESHOLD,
    TARGET_OBJECTS,
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    FONT_SCALE,
    FONT_THICKNESS,
    EVAL_FRAME_COUNT,
)

class TestConfig(unittest.TestCase):

    def test_confidence_threshold_is_float(self):
        """CONFIDENCE_THRESHOLD should be a float."""
        self.assertIsInstance(CONFIDENCE_THRESHOLD, float)

    def test_confidence_threshold_in_valid_range(self):
        """CONFIDENCE_THRESHOLD should be between 0.0 and 1.0."""
        self.assertGreater(CONFIDENCE_THRESHOLD, 0.0)
        self.assertLess(CONFIDENCE_THRESHOLD, 1.0)

    def test_target_objects_is_not_empty(self):
        """TARGET_OBJECTS should contain at least one item."""
        self.assertGreater(len(TARGET_OBJECTS), 0)

    def test_target_objects_are_all_strings(self):
        """Every item in TARGET_OBJECTS should be a string."""
        for obj in TARGET_OBJECTS:
            self.assertIsInstance(obj, str)

    def test_target_objects_has_no_duplicates(self):
        """TARGET_OBJECTS should not contain duplicate labels."""
        self.assertEqual(len(TARGET_OBJECTS), len(set(TARGET_OBJECTS)))

    def test_camera_index_is_non_negative(self):
        """CAMERA_INDEX should be 0 or a positive integer."""
        self.assertIsInstance(CAMERA_INDEX, int)
        self.assertGreaterEqual(CAMERA_INDEX, 0)

    def test_frame_dimensions_are_positive(self):
        """FRAME_WIDTH and FRAME_HEIGHT should be positive integers."""
        self.assertGreater(FRAME_WIDTH, 0)
        self.assertGreater(FRAME_HEIGHT, 0)

    def test_font_scale_is_positive(self):
        """FONT_SCALE should be a positive number."""
        self.assertGreater(FONT_SCALE, 0)

    def test_font_thickness_is_positive_int(self):
        """FONT_THICKNESS should be a positive integer."""
        self.assertIsInstance(FONT_THICKNESS, int)
        self.assertGreater(FONT_THICKNESS, 0)

    def test_eval_frame_count_is_positive(self):
        """EVAL_FRAME_COUNT should be a positive integer."""
        self.assertIsInstance(EVAL_FRAME_COUNT, int)
        self.assertGreater(EVAL_FRAME_COUNT, 0)

# End of test cases for config.py 

if __name__ == "__main__":
    unittest.main(verbosity=2)
