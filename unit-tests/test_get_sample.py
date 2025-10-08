import unittest
from unittest.mock import patch
import torch

from verifier.abstract_constraint_solver.base_verifier import BaseVerifier

class TestGetSampleCenterAndLabel(unittest.TestCase):
    def setUp(self):
        # Patch BaseVerifier.__init__ to avoid dependencies
        patcher = patch.object(BaseVerifier, "__init__", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)
        
        # Create instance with minimal required attributes
        self.verifier = BaseVerifier()
        self.verifier.input_center = torch.tensor([1.0, 2.0])
        
        # Create dummy dataset with explicit length and indexing behavior
        class DummyDataset:
            def __init__(self):
                self.data = [(torch.tensor([1.0, 2.0]), torch.tensor(1))]
                self.labels = [torch.tensor(1)]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                if idx >= len(self.data):
                    raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self.data)}")
                return self.data[idx]
                
        self.verifier.dataset = DummyDataset()

    def test_extract_sample_input_and_label(self):
        # Test normal case
        center, label = self.verifier.extract_sample_input_and_label(sample_index=0)
        self.assertEqual(label, 1)
        self.assertTrue(torch.equal(center, torch.tensor([[1.0, 2.0]])))
        
    def test_extract_sample_input_and_label_invalid_index(self):
        # Test invalid index case
        with self.assertRaises(IndexError):
            self.verifier.extract_sample_input_and_label(sample_index=1)
            
    def test_extract_sample_input_and_label_shape(self):
        # Test output tensor shape
        center, _ = self.verifier.extract_sample_input_and_label(sample_index=0)
        self.assertEqual(len(center.shape), 2)  # Should be 2D tensor
        self.assertEqual(center.shape[0], 1)    # Batch dimension should be 1
        self.assertEqual(center.shape[1], 2)    # Feature dimension should match input

if __name__ == "__main__":
    unittest.main()