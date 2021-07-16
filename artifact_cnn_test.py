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
            'cnn1_threshold', 'stamps', 'flux', 'fluxerr', 'psf',
            'masking_box_width')
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
            
    def test_scale_scores(self):
        # CNN 1
        scores = np.array([-60., -50., -30., -10., -0.001])
        scaled_scores = self.obj._scale_scores(scores, 'cnn1')
        expected_scaled_scores = np.array([0.0, 0.0, 0.95225653, 1.0, 1.0])
        self.assertTrue(np.allclose(scaled_scores, expected_scaled_scores))

        # CNN 2
        scores = np.array([-10., -5., -1., -0.001])
        scaled_scores =	self.obj._scale_scores(scores, 'cnn2')
        expected_scaled_scores = np.array([0., 0.23331225, 0.8468432, 1.])
        self.assertTrue(np.allclose(scaled_scores, expected_scaled_scores))

    def test_score_stamps(self):
        preprocess_mask = np.array([0, 1, 1, 1], dtype=bool)
        cnn1_scores = np.array([0.9, 0.4, 0.9, 0.9])
        cnn2_scores = np.array([0.8, 0.8, 0.8, 0.02])
        output = self.obj.score_stamps(
            preprocess_mask, cnn1_scores, cnn2_scores)
        expected_output = np.array([0.0, 0.04, 0.8, 0.1])
        self.assertTrue(np.array_equal(output, expected_output))

    def test_real_data(self):
        stamp_array = np.load('test_stamps.npy', allow_pickle=True)
        stamp_md = np.load('test_stamp_metadata.npy', allow_pickle=True)
        stamp_evaluator = artifact_cnn.StampEvaluator(
            stamp_array, stamp_md[0], stamp_md[1], stamp_md[2])
        scores = stamp_evaluator.run()

        allowed_ranges = np.load('test_stamp_ranges.npy', allow_pickle=True)
        total = 0
        for score, allowed_range in zip(scores, allowed_ranges):
            total += (score > allowed_range[0] and score < allowed_range[1])
        
        self.assertGreaterEqual(total, len(stamp_array) // 2)


if __name__ == '__main__':
    unittest.main()
