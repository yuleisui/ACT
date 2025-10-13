#!/usr/bin/env python3
"""
Unit tests for act.interval.bounds_propagation APIs.
Tests the REAL production BoundsPropagate class using shared configurations.

This module tests:
- Individual layer handler APIs (_handle_linear, _handle_conv2d, etc.)
- Main propagate_bounds API end-to-end
- Mathematical properties and correctness
- API contracts and error handling
- Integration with shared test configurations

Uses shared test configurations from test_configs.py to provide
comprehensive test coverage with various models and input specifications.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np

# Import REAL production APIs
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from act.interval.bounds_propagation import BoundsPropagate
from act.interval.util.bounds import Bounds
from act.interval.bounds_prop_helper import TrackingMode

# Import shared test configurations
from test_configs import MockFactory, get_unit_test_configs


class TestBoundsPropagateLayers(unittest.TestCase):
    """Test individual layer handler APIs using shared configs."""
    
    def setUp(self):
        """Initialize with REAL BoundsPropagate."""
        torch.manual_seed(42)
        self.propagator = BoundsPropagate(mode=TrackingMode.PERFORMANCE)
    
    def test_handle_linear_correctness(self):
        """Test _handle_linear API with various configurations."""
        test_configs = get_unit_test_configs("layer_tests")
        linear_tests = [c for c in test_configs if c["focus"] == "linear_layers"]
        
        for config in linear_tests:
            with self.subTest(config=config):
                model = MockFactory.create_model(config["model"])
                lb, ub = MockFactory.create_data(config["data"])
                
                # Find first linear layer
                linear_layer = None
                for layer in model.children():
                    if isinstance(layer, nn.Linear):
                        linear_layer = layer
                        break
                
                if linear_layer is None:
                    self.skipTest(f"No linear layer in {config['model']}")
                
                # Adjust input to match layer input size
                if lb.numel() != linear_layer.in_features:
                    lb = lb.view(-1)[:linear_layer.in_features]
                    ub = ub.view(-1)[:linear_layer.in_features]
                
                bounds = Bounds(lb, ub)
                
                # Test REAL API
                result = self.propagator._handle_linear(linear_layer, bounds, idx=0)
                
                # Validate API contract
                self.assertIsInstance(result, Bounds)
                self.assertEqual(result.shape, (linear_layer.out_features,))
                self.assertTrue(torch.all(result.lb <= result.ub))
    
    def test_handle_conv2d_correctness(self):
        """Test _handle_conv2d API with various configurations."""
        test_configs = get_unit_test_configs("layer_tests")
        conv_tests = [c for c in test_configs if c["focus"] == "conv_layers"]
        
        for config in conv_tests:
            with self.subTest(config=config):
                model = MockFactory.create_model(config["model"])
                lb, ub = MockFactory.create_data(config["data"])
                
                # Find first conv layer
                conv_layer = None
                for layer in model.children():
                    if isinstance(layer, nn.Conv2d):
                        conv_layer = layer
                        break
                
                if conv_layer is None:
                    self.skipTest(f"No conv layer in {config['model']}")
                
                bounds = Bounds(lb, ub)
                
                # Test REAL API
                result = self.propagator._handle_conv2d(conv_layer, bounds, idx=0)
                
                # Validate API contract
                self.assertIsInstance(result, Bounds)
                self.assertTrue(torch.all(result.lb <= result.ub))


class TestBoundsPropagatePipeline(unittest.TestCase):
    """Test main propagate_bounds API using shared configs."""
    
    def setUp(self):
        torch.manual_seed(42)
        self.propagator = BoundsPropagate(mode=TrackingMode.PERFORMANCE)
    
    def test_propagate_bounds_configurations(self):
        """Test main API with various model/data combinations."""
        test_configs = get_unit_test_configs("end_to_end_tests")
        
        for config in test_configs:
            with self.subTest(config=config):
                model = MockFactory.create_model(config["model"])
                lb, ub = MockFactory.create_data(config["data"])
                input_bounds = Bounds(lb, ub, _internal=True)
                
                # Test REAL API
                result_bounds = self.propagator.propagate_bounds(model, input_bounds)
                
                # Validate API contract
                self.assertIsInstance(result_bounds, Bounds)
                self.assertTrue(torch.all(result_bounds.lb <= result_bounds.ub))


class TestBoundsPropagatePoperties(unittest.TestCase):
    """Test mathematical properties using shared configs."""
    
    def setUp(self):
        torch.manual_seed(42)
        self.propagator = BoundsPropagate(mode=TrackingMode.PERFORMANCE)
    
    def test_mathematical_properties(self):
        """Test mathematical properties with various configurations."""
        test_configs = get_unit_test_configs("property_tests")
        
        for config in test_configs:
            with self.subTest(config=config):
                model = MockFactory.create_model(config["model"])
                lb, ub = MockFactory.create_data(config["data"])
                
                if config["property"] == "monotonicity":
                    self._test_monotonicity(model, lb, ub)
                elif config["property"] == "soundness":
                    self._test_soundness(model, lb, ub)
                elif config["property"] == "bounds_ordering":
                    self._test_bounds_ordering(model, lb, ub)
    
    def _test_monotonicity(self, model, lb, ub):
        """Test interval inclusion property."""
        # Create wider bounds
        lb2 = lb - 0.1
        ub2 = ub + 0.1
        
        input_bounds1 = Bounds(lb, ub, _internal=True)
        input_bounds2 = Bounds(lb2, ub2, _internal=True)
        
        result1 = self.propagator.propagate_bounds(model, input_bounds1)
        result2 = self.propagator.propagate_bounds(model, input_bounds2)
        
        # Wider input should produce wider output
        self.assertTrue(torch.all(result2.lb <= result1.lb + 1e-6))  # Small tolerance
        self.assertTrue(torch.all(result1.ub <= result2.ub + 1e-6))
    
    def _test_soundness(self, model, lb, ub):
        """Test that actual outputs are within computed bounds."""
        input_bounds = Bounds(lb, ub, _internal=True)
        result_bounds = self.propagator.propagate_bounds(model, input_bounds)
        model.eval()
        
        # Sample random inputs within bounds
        for _ in range(10):  # Reduce samples for speed
            alpha = torch.rand_like(lb)
            test_input = lb + alpha * (ub - lb)
            
            with torch.no_grad():
                actual_output = model(test_input)
            
            # Check soundness with tolerance
            self.assertTrue(torch.all(actual_output >= result_bounds.lb - 1e-6))
            self.assertTrue(torch.all(actual_output <= result_bounds.ub + 1e-6))
    
    def _test_bounds_ordering(self, model, lb, ub):
        """Test that bounds ordering is preserved."""
        input_bounds = Bounds(lb, ub, _internal=True)
        result_bounds = self.propagator.propagate_bounds(model, input_bounds)
        self.assertTrue(torch.all(result_bounds.lb <= result_bounds.ub))


if __name__ == '__main__':
    unittest.main()