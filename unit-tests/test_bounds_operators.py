#!/usr/bin/env python3
"""
Unit tests for Bounds class operator arithmetic methods.
Tests the new operator methods added to the Bounds class.
"""

import unittest
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from act.interval.util.bounds import Bounds


class TestBoundsOperators(unittest.TestCase):
    """Test arithmetic operator methods in Bounds class."""
    
    def setUp(self):
        """Initialize test bounds."""
        torch.manual_seed(42)
        self.lb = torch.tensor([1.0, -2.0, 3.0])
        self.ub = torch.tensor([2.0, -1.0, 4.0])
        self.bounds = Bounds(self.lb, self.ub)
    
    def test_add_constant(self):
        """Test apply_operator with Add operation."""
        result = self.bounds.apply_operator("Add", 5.0)
        expected_lb = torch.tensor([6.0, 3.0, 8.0])
        expected_ub = torch.tensor([7.0, 4.0, 9.0])
        
        self.assertTrue(torch.allclose(result.lb, expected_lb))
        self.assertTrue(torch.allclose(result.ub, expected_ub))
    
    def test_subtract_constant(self):
        """Test apply_operator with Sub operation."""
        result = self.bounds.apply_operator("Sub", 1.0)
        expected_lb = torch.tensor([0.0, -3.0, 2.0])
        expected_ub = torch.tensor([1.0, -2.0, 3.0])
        
        self.assertTrue(torch.allclose(result.lb, expected_lb))
        self.assertTrue(torch.allclose(result.ub, expected_ub))
    
    def test_multiply_constant_positive(self):
        """Test apply_operator with Mul and positive constant."""
        result = self.bounds.apply_operator("Mul", 2.0)
        expected_lb = torch.tensor([2.0, -4.0, 6.0])
        expected_ub = torch.tensor([4.0, -2.0, 8.0])
        
        self.assertTrue(torch.allclose(result.lb, expected_lb))
        self.assertTrue(torch.allclose(result.ub, expected_ub))
    
    def test_multiply_constant_negative(self):
        """Test apply_operator with Mul and negative constant (should flip bounds)."""
        result = self.bounds.apply_operator("Mul", -2.0)
        # When multiplying by negative, bounds should flip
        expected_lb = torch.tensor([-4.0, 2.0, -8.0])  # min of (-2*2, -2*1), etc.
        expected_ub = torch.tensor([-2.0, 4.0, -6.0])  # max of (-2*2, -2*1), etc.
        
        self.assertTrue(torch.allclose(result.lb, expected_lb))
        self.assertTrue(torch.allclose(result.ub, expected_ub))
    
    def test_divide_constant_positive(self):
        """Test apply_operator with Div and positive constant."""
        result = self.bounds.apply_operator("Div", 2.0)
        expected_lb = torch.tensor([0.5, -1.0, 1.5])
        expected_ub = torch.tensor([1.0, -0.5, 2.0])
        
        self.assertTrue(torch.allclose(result.lb, expected_lb))
        self.assertTrue(torch.allclose(result.ub, expected_ub))
    
    def test_divide_constant_negative(self):
        """Test apply_operator with Div and negative constant (should flip bounds)."""
        result = self.bounds.apply_operator("Div", -2.0)
        # When dividing by negative, bounds should flip
        expected_lb = torch.tensor([-1.0, 0.5, -2.0])  # min of (-2/-2, -1/-2), etc.
        expected_ub = torch.tensor([-0.5, 1.0, -1.5])  # max of (-2/-2, -1/-2), etc.
        
        self.assertTrue(torch.allclose(result.lb, expected_lb))
        self.assertTrue(torch.allclose(result.ub, expected_ub))
    
    def test_divide_by_zero_raises_error(self):
        """Test that division by zero raises ValueError."""
        with self.assertRaises(ValueError):
            self.bounds.apply_operator("Div", 0.0)
    
    def test_apply_operator_add(self):
        """Test apply_operator with Add operation."""
        result = self.bounds.apply_operator("Add", 3.0)
        expected_lb = torch.tensor([4.0, 1.0, 6.0])
        expected_ub = torch.tensor([5.0, 2.0, 7.0])
        
        self.assertTrue(torch.allclose(result.lb, expected_lb))
        self.assertTrue(torch.allclose(result.ub, expected_ub))
    
    def test_apply_operator_sub(self):
        """Test apply_operator with Sub operation."""
        result = self.bounds.apply_operator("Sub", 1.5)
        expected_lb = torch.tensor([-0.5, -3.5, 1.5])
        expected_ub = torch.tensor([0.5, -2.5, 2.5])
        
        self.assertTrue(torch.allclose(result.lb, expected_lb))
        self.assertTrue(torch.allclose(result.ub, expected_ub))
    
    def test_apply_operator_mul(self):
        """Test apply_operator with Mul operation."""
        result = self.bounds.apply_operator("Mul", -0.5)
        # Multiplication by negative should flip bounds
        expected_lb = torch.tensor([-1.0, 0.5, -2.0])
        expected_ub = torch.tensor([-0.5, 1.0, -1.5])
        
        self.assertTrue(torch.allclose(result.lb, expected_lb))
        self.assertTrue(torch.allclose(result.ub, expected_ub))
    
    def test_apply_operator_div(self):
        """Test apply_operator with Div operation."""
        result = self.bounds.apply_operator("Div", 4.0)
        expected_lb = torch.tensor([0.25, -0.5, 0.75])
        expected_ub = torch.tensor([0.5, -0.25, 1.0])
        
        self.assertTrue(torch.allclose(result.lb, expected_lb))
        self.assertTrue(torch.allclose(result.ub, expected_ub))
    
    def test_apply_operator_unsupported(self):
        """Test apply_operator with unsupported operation."""
        with self.assertRaises(ValueError):
            self.bounds.apply_operator("Pow", 2.0)
    
    def test_tensor_operations(self):
        """Test operations with tensor operands."""
        operand = torch.tensor([2.0, -1.0, 0.5])
        
        # Test addition with tensor
        result = self.bounds.apply_operator("Add", operand)
        expected_lb = torch.tensor([3.0, -3.0, 3.5])
        expected_ub = torch.tensor([4.0, -2.0, 4.5])
        
        self.assertTrue(torch.allclose(result.lb, expected_lb))
        self.assertTrue(torch.allclose(result.ub, expected_ub))


if __name__ == '__main__':
    unittest.main()