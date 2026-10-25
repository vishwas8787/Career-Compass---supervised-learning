"""
Unit tests for the Supervised Learning Job Zone classifier.
"""

import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = PROJECT_ROOT / "models" / "job_zone_model.joblib"
FEATURES_FILE = PROJECT_ROOT / "data" / "processed" / "occupation_features.csv"


class TestJobZoneModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved_dict = joblib.load(MODEL_FILE)
        cls.model = cls.saved_dict["model"]
        cls.features = cls.saved_dict["feature_columns"]
        cls.threshold = cls.saved_dict.get("zone3_threshold", 1.01)
        cls.data = pd.read_csv(FEATURES_FILE)

    def test_model_artifact_structure(self):
        self.assertIn("model", self.saved_dict)
        self.assertIn("feature_columns", self.saved_dict)
        self.assertIn("zone3_threshold", self.saved_dict)
        self.assertGreater(len(self.features), 20)

    def test_model_classes(self):
        # Classes should cover Job Zones 2, 3, 4, 5
        classes = list(self.model.classes_)
        for expected_zone in [2, 3, 4, 5]:
            self.assertIn(expected_zone, classes)

    def test_feature_columns_exist_in_data(self):
        for col in self.features:
            self.assertIn(col, self.data.columns, f"Feature {col} missing from features dataset")

    def test_prediction_output(self):
        sample_x = self.data[self.features].iloc[:5]
        probs = self.model.predict_proba(sample_x)
        self.assertEqual(probs.shape[0], 5)
        self.assertEqual(probs.shape[1], len(self.model.classes_))
        # Probabilities sum to 1.0 per row
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(5), rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
