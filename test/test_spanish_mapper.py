# ¿Qué es? — Real-Time Spanish Vocabulary Recognition
# Author: Ruben A. Salido
# Northeastern University | CS5330: Pattern Recognition and Computer Vision
# Originally developed: Spring 2026

# Unit tests for spanish_mapper.py
# spanish_mapper.py
# Unit tests for spanish_mapper.py
# Run with: .venv/bin/python -m unittest test_spanish_mapper.py -v

import unittest
from spanish_mapper import SpanishLabelMapper
from config import TARGET_OBJECTS

class TestSpanishLabelMapper(unittest.TestCase):

    def setUp(self):
        self.mapper = SpanishLabelMapper()

    def test_singular_masculine_indefinite(self):
        self.assertEqual(self.mapper.get_label("book", 1, article_type="indefinite"), "un libro")

    def test_singular_feminine_indefinite(self):
        self.assertEqual(self.mapper.get_label("chair", 1, article_type="indefinite"), "una silla")

    def test_plural_masculine_indefinite(self):
        self.assertEqual(self.mapper.get_label("book", 2, article_type="indefinite"), "unos libros")

    def test_plural_feminine_indefinite(self):
        self.assertEqual(self.mapper.get_label("chair", 2, article_type="indefinite"), "unas sillas")

    def test_singular_masculine_definite(self):
        self.assertEqual(self.mapper.get_label("book", 1, article_type="definite"), "el libro")

    def test_singular_feminine_definite(self):
        self.assertEqual(self.mapper.get_label("chair", 1, article_type="definite"), "la silla")

    def test_plural_masculine_definite(self):
        self.assertEqual(self.mapper.get_label("book", 2, article_type="definite"), "los libros")

    def test_plural_feminine_definite(self):
        self.assertEqual(self.mapper.get_label("chair", 2, article_type="definite"), "las sillas")

    # irregular verbs
    def test_irregular_singular_masculine_indefinite(self):
        self.assertEqual(self.mapper.get_label("bowl", 1, article_type="indefinite"), "un tazón")

    def test_irregular_plural_masculine_indefinite(self):
        self.assertEqual(self.mapper.get_label("bowl", 2, article_type="indefinite"), "unos tazones")

    def test_irregular_singular_masculine_definite(self):
        self.assertEqual(self.mapper.get_label("mouse", 1, article_type="definite"), "el ratón")

    def test_irregular_plural_masculine_definite(self):
        self.assertEqual(self.mapper.get_label("mouse", 2, article_type="definite"), "los ratones")

    # edge case: scissors
    def test_singular_feminine_definite_edge(self):
        self.assertEqual(self.mapper.get_label("scissors", 1, article_type="definite"), "las tijeras")

    def test_plural_feminine_definite_edge(self):
        self.assertEqual(self.mapper.get_label("scissors", 2, article_type="definite"), "las tijeras")

    def test_singular_feminine_indefinite_edge(self):
        self.assertEqual(self.mapper.get_label("scissors", 1, article_type="indefinite"), "unas tijeras")

    def test_plural_feminine_indefinite_edge(self):
        self.assertEqual(self.mapper.get_label("scissors", 2, article_type="indefinite"), "unas tijeras")

    # error messages triggered
    def test_count_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.mapper.get_label("chair", 0)

    def test_negative_count_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.mapper.get_label("chair", -1)

    def test_invalid_article_type_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.mapper.get_label("chair", 1, article_type="plural")

    def test_invalid_noun_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.mapper.get_label("boat", 0, article_type="definite")

    def test_unmapped_label_returns_raw_label(self):
        self.assertEqual(self.mapper.get_label("boat", 1, article_type="definite"), "boat")

    # not sure if this test is necessary, since invalid counts already raise ValueError for mapped labels
    # def test_unmapped_label_invalid_count_returns_raw_label(self):
    #     self.assertEqual(self.mapper.get_label("boat", 0, article_type="definite"), "boat")

    def test_label_map_matches_target_objects(self):
        self.assertSetEqual(set(SpanishLabelMapper.LABEL_MAP), set(TARGET_OBJECTS))

    def test_get_all_labels_skips_non_target_objects(self):
        result = self.mapper.get_all_labels({"chair": 2, "boat": 1})
        self.assertEqual(result, {"chair": "unas sillas"})

    def test_get_all_labels_definite(self):
        result = self.mapper.get_all_labels({"chair": 2, "book": 1}, "definite")
        self.assertEqual(result, {"chair": "las sillas", "book": "el libro"})


if __name__ == "__main__":
    unittest.main()
    