"""DECam CNN-based Difference Imaging Artifact Detection.

Written by Robert Morgan and Adam Shandonay.

See https://arxiv.org/abs/2106.11315v1 for details.
"""

import numpy as np
import torch

import cnn_utils 


class StampEvaluator:
    """An object to score difference imaging stamps.

    Operates on an array of stamps with dimensions (N, 3, 51, 51) where the
    first dimension is the index of the detection, the second axis is (srch,
    temp, diff), the third axis is height, and the final axis is width.

    Usage:
    >>> evaluator = StampEvaluator(stamp_array)
    >>> scores = evaluator.run()
    """

    def __init__(self, stamps: np.ndarray):
        """Instnatiates a StampEvaluator object.

        Stores stamps, thresholds, and trained CNNs as class attributes.

        Args:
          stamps (np.ndarray): A 4-dimensional array of stamps with axes
            (index, stamp type, height, width).
        """
        self.stamps = stamps
        self.snr_threshold = 3.76
        self.flux_threshold = 13.86
        self.cnn1_threshold = 0.5
        self.load_cnns()

    def run(self):
        """Returns stamp scores.

        Converts the stamps to a PyTorch Dataset, applies preprocessing
        filters to remove easy artifacts, applies 2 CNNs to the stamps, and 
        produces a final output probability for each detection.
        """
        dataset = self.make_dataset()
        preprocess_mask = self.preprocess()
        cnn1_scores = self.run_cnn(dataset, 'cnn1')
        cnn2_scores = self.run_cnn(dataset, 'cnn2')
        return self.score_stamps(preprocess_mask, cnn1_scores, cnn2_scores)

    def load_cnns(self):
        """Instantiates CNNs and stores them as class attributes."""
        self.cnn1 = cnn_utils.CNN()
        self.cnn1.load_state_dict(torch.load('model1.pt'))
        self.cnn1.eval()
        self.cnn2 = cnn_utils.CNN()
        self.cnn2.load_state_dict(torch.load('model2.pt'))
        self.cnn2.eval()

    def make_dataset(self) -> torch.utils.data.Dataset:
        """Convert input stamp array to PyTorch Dataset.

        Returns:
          An ArtifactDataset object made from the stamps."""
        transform = cnn_utils.ToTensor()
        return cnn_utils.ArtifactDataset(self.stamps, transform)

    def _snr_preprocessing(self):
        """Apply a SNR threshold to the images."""
        return np.ones(len(self.stamps), dtype=bool)

    def _flux_preprocessing(self):
        """Apply a flux threshold to the images."""

        def gaussian2D(distance_to_center, sigma):
            return (1 / (sigma ** 2 * 2 * np.pi) *
                    np.exp(-0.5 * (distance_to_center / sigma) ** 2))
        
        return np.ones(len(self.stamps), dtype=bool)
        
    def preprocess(self):
        """Run stamps through preprocessing functions.

        Args:
          dataset (torch.utils.data.Dataset): The output of make_dataset().
        
        Returns:
          Boolean array True where a detection passes preprocessing.
        """
        snr_mask = self._snr_preprocessing()
        flux_mask = self._flux_preprocessing()
        return snr_mask & flux_mask
        
    def run_cnn(
            self,
            dataset: torch.utils.data.Dataset,
            cnn_attribute_name: str) -> np.ndarray:
        """Obtain scores from a CNN.

        Args:
          dataset (torch.utils.data.Dataset): The output of make_dataset().
          cnn_attribute_name (str): The name of the CNN to use.

        Returns:
          A numpy array of scores from the CNN.
        """
        cnn = getattr(self, cnn_attribute_name)
        return torch.max(cnn(dataset[:]['image']), 1)[1].data.numpy()

    def score_stamps(
            self,
            preprocess_mask: np.ndarray,
            cnn1_scores: np.ndarray,
            cnn2_scores: np.ndarray) -> np.ndarray:
        """Combine preprocessing and CNN scores into one score.

        Detections that do not pass preprocessing are given a score of 0.
        Detections that pass preprocessing but not cnn1 are given a score
        between 0 and 0.1 based on their raw cnn1 score. Detections passing
        preprocessing and cnn1 are given their cnn2 score, but a floor is
        applied at 0.1.

        Args:
        
        Returns:

        """
        scaled_cnn1_scores = cnn1_scores / 10.0
        floored_cnn2_scores = np.where(cnn2_scores < 0.1, 0.1, cnn2_scores)
        return np.where(
            preprocess_mask,
            np.where(
                cnn1_scores < self.cnn1_threshold,
                scaled_cnn1_scores,
                floored_cnn2_scores),
            0.0)
