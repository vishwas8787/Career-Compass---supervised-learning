"""
Unit tests for the CareerCompass career recommendation engine.
"""

import unittest
import numpy as np
import pandas as pd
from src.recommend import (
    load_resources,
    validate_ratings,
    calculate_interest_scores,
    calculate_skill_scores,
    recommend_careers,
)


class TestRecommendationEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profiles, cls.model, cls.features, cls.threshold = load_resources()

    def test_load_resources(self):
        self.assertIsInstance(self.profiles, pd.DataFrame)
        self.assertFalse(self.profiles.empty)
        self.assertGreater(len(self.features), 0)
        self.assertIsNotNone(self.model)

    def test_validate_ratings_valid(self):
        valid_input = {"realistic": 4.5, "investigative": 2}
        validated = validate_ratings(valid_input, "Interest")
        self.assertEqual(validated["realistic"], 4.5)
        self.assertEqual(validated["investigative"], 2.0)

    def test_validate_ratings_out_of_bounds(self):
        with self.assertRaises(ValueError):
            validate_ratings({"realistic": 6.0}, "Interest")
        with self.assertRaises(ValueError):
            validate_ratings({"realistic": -1.0}, "Interest")

    def test_validate_ratings_non_numeric(self):
        with self.assertRaises(ValueError):
            validate_ratings({"realistic": "high"}, "Interest")

    def test_calculate_interest_scores(self):
        interests = {
            "realistic": 0.0,
            "investigative": 5.0,
            "artistic": 0.0,
            "social": 0.0,
            "enterprising": 0.0,
            "conventional": 0.0,
        }
        scores = calculate_interest_scores(self.profiles, interests)
        self.assertEqual(len(scores), len(self.profiles))
        self.assertTrue((scores >= 0).all() and (scores <= 100).all())

    def test_calculate_skill_scores(self):
        skills = {"Mathematics": 3.0, "Critical Thinking": 4.0}
        scores = calculate_skill_scores(self.profiles, skills)
        self.assertEqual(len(scores), len(self.profiles))
        self.assertTrue((scores >= 0).all() and (scores <= 100).all())

    def test_recommend_careers_pipeline(self):
        interests = {
            "realistic": 1.0,
            "investigative": 5.0,
            "artistic": 1.0,
            "social": 1.0,
            "enterprising": 1.0,
            "conventional": 3.0,
        }
        skills = {"Mathematics": 3.0, "Critical Thinking": 3.0}
        recs = recommend_careers(interests, skills, top_n=5)

        self.assertEqual(len(recs), 5)
        self.assertIn("Title", recs.columns)
        self.assertIn("Match Score (%)", recs.columns)
        self.assertIn("Predicted Job Zone", recs.columns)
        self.assertTrue(recs["Match Score (%)"].is_monotonic_decreasing)


if __name__ == "__main__":
    unittest.main()
