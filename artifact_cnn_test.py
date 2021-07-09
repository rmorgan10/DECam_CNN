"""Unit tests for artifact_cnn.py"""

import unittest

import numpy as np
import torch

import artifact_cnn
import cnn_utils

class StampEvaluatorTest(unittest.TestCase):
    """Tests for the StampEvaluator class."""

    def setUp(self):
        self.stamps = np.random.uniform(size=(10, 3, 51, 51))
        self.obj = artifact_cnn.StampEvaluator(self.stamps)
        self.dataset = self.obj.make_dataset()

    def test_stored_attributes(self):
        expected_attributes = (
            'cnn1', 'cnn2', 'flux_threshold', 'snr_threshold',
            'cnn1_threshold', 'stamps')
        for attribute in expected_attributes:
            self.assertTrue(hasattr(self.obj, attribute))

    def test_run(self):
        scores = self.obj.run()
        self.assertEqual(len(scores), len(self.stamps))

    def test_load_cnns(self):
        self.assertIsInstance(self.obj.cnn1, cnn_utils.CNN)
        self.assertIsInstance(self.obj.cnn2, cnn_utils.CNN)

    def test_make_dataset(self):
        self.assertIsInstance(self.dataset, cnn_utils.ArtifactDataset)

    def test_run_cnn(self):
        for cnn_name in ('cnn1', 'cnn2'):
            scores = self.obj.run_cnn(self.dataset, cnn_name)
            self.assertIsInstance(scores, np.ndarray)
            self.assertEqual(len(scores), len(self.stamps))
            self.assertLessEqual(scores.max(), 1.0)
            self.assertGreaterEqual(scores.min(), 0.0)

    def test_score_stamps(self):
        preprocess_mask = np.array([0, 1, 1, 1], dtype=bool)
        cnn1_scores = np.array([0.9, 0.4, 0.9, 0.9])
        cnn2_scores = np.array([0.8, 0.8, 0.8, 0.02])
        output = self.obj.score_stamps(
            preprocess_mask, cnn1_scores, cnn2_scores)
        expected_output = np.array([0.0, 0.04, 0.8, 0.1])
        self.assertTrue(np.array_equal(output, expected_output))


if __name__ == '__main__':
    unittest.main()
