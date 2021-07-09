"""Unit tests for cnn_utils.py."""

import unittest

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import cnn_utils

class ArtifactDatasetTest(unittest.TestCase):
    """Tests the ArtifactDataset and ToTensor class."""

    def setUp(self):
        images = np.random.uniform(size=(10, 3, 51, 51))
        transform = cnn_utils.ToTensor()
        self.dataset = cnn_utils.ArtifactDataset(images, transform)

    def test_subclassing(self):
        self.assertIsInstance(self.dataset, Dataset)

    def test_example_type(self):
        example = self.dataset[3]['image']
        self.assertIsInstance(example, torch.Tensor)

    def test_example_shape(self):
        example = self.dataset[3]['image']
        expected_shape = (3, 51, 51)
        self.assertSequenceEqual(example.shape, expected_shape)

    def test_transform_scaling(self):
        images = self.dataset[:]['image'].data.numpy()

        # Check that the average of each detection is 0.5
        avgs = np.mean(images, axis=(-1, -2, -3))
        expected_avgs = np.ones(avgs.shape) * 0.5
        self.assertTrue(np.allclose(avgs, expected_avgs))

        # Check that the std of each detection is 0.1
        stds = np.std(images, axis=(-1, -2, -3))
        expected_stds = np.ones(stds.shape) * 0.1
        self.assertTrue(np.allclose(stds, expected_stds))

        
class CNNTest(unittest.TestCase):
    """Tests for the CNN class."""

    def setUp(self):
        self.cnn = cnn_utils.CNN()
        transform = cnn_utils.ToTensor()
        images = np.random.uniform(size=(10, 3, 51, 51))
        self.dataset = cnn_utils.ArtifactDataset(images, transform)

    def test_subclassing(self):
        self.assertIsInstance(self.cnn, torch.nn.Module)

    def test_ordered_layers(self):
        layers = list(self.cnn.state_dict().keys())
        expected_layers = [
            'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias',
            'conv3.weight', 'conv3.bias', 'fc1.weight', 'fc1.bias',
            'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias']
        self.assertSequenceEqual(layers, expected_layers)

    def test_forward_pass(self):
        batch_size = 5
        dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        batch = next(iter(dataloader))
        output = self.cnn(batch['image']).data.numpy()
        expected_shape = (batch_size, 2)
        self.assertSequenceEqual(output.shape, expected_shape)
        
if __name__ == '__main__':
    unittest.main()
