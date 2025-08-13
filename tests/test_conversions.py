import numpy as np
import pytest

# We need to import the functions we're testing
try:
    from cutqc2.cutqc.helper_functions.conversions import (
        nearest_probability_distribution,
        naive_probability_distribution,
        quasi_to_real
    )
except ImportError:
    # Fallback for testing during development
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from cutqc2.cutqc.helper_functions.conversions import (
        nearest_probability_distribution,
        naive_probability_distribution,
        quasi_to_real
    )


class TestNearestProbabilityDistribution:
    """Test the nearest_probability_distribution function"""

    def test_already_valid_distribution(self):
        """Test with an already valid probability distribution"""
        quasiprob = [0.25, 0.25, 0.25, 0.25]
        result = nearest_probability_distribution(quasiprob)
        
        assert np.allclose(result, [0.25, 0.25, 0.25, 0.25])
        assert abs(np.sum(result) - 1.0) < 1e-10
        assert all(p >= 0 for p in result)

    def test_simple_negative_case(self):
        """Test with simple negative probabilities"""
        quasiprob = [0.6, 0.6, -0.1, -0.1]
        result = nearest_probability_distribution(quasiprob)
        
        expected = [0.5, 0.5, 0.0, 0.0]
        assert np.allclose(result, expected)
        assert abs(np.sum(result) - 1.0) < 1e-10
        assert all(p >= 0 for p in result)

    def test_mostly_negative(self):
        """Test with mostly negative values"""
        quasiprob = [1.5, -0.2, -0.2, -0.1]
        result = nearest_probability_distribution(quasiprob)
        
        expected = [1.0, 0.0, 0.0, 0.0]
        assert np.allclose(result, expected)
        assert abs(np.sum(result) - 1.0) < 1e-10
        assert all(p >= 0 for p in result)

    def test_mixed_values(self):
        """Test with mixed positive and negative values"""
        quasiprob = [0.3, 0.4, 0.5, -0.2]
        result = nearest_probability_distribution(quasiprob)
        
        # Should redistribute the negative mass
        assert abs(np.sum(result) - 1.0) < 1e-10
        assert all(p >= 0 for p in result)
        assert result[3] == 0.0  # The negative element should become 0

    def test_edge_case_small_values(self):
        """Test with very small values that might cause numerical issues"""
        quasiprob = [0.001, -0.0005, 0.0003, 0.9996]
        result = nearest_probability_distribution(quasiprob)
        
        # The result should be a valid probability distribution
        assert abs(np.sum(result) - 1.0) < 1e-10
        assert all(p >= -1e-10 for p in result)  # Allow tiny numerical errors

    def test_all_negative_except_one(self):
        """Test when only one element is positive"""
        quasiprob = [2.0, -0.3, -0.3, -0.4]
        result = nearest_probability_distribution(quasiprob)
        
        expected = [1.0, 0.0, 0.0, 0.0]
        assert np.allclose(result, expected)
        assert abs(np.sum(result) - 1.0) < 1e-10
        assert all(p >= 0 for p in result)

    def test_numpy_array_input(self):
        """Test that the function works with numpy arrays"""
        quasiprob = np.array([0.6, 0.6, -0.1, -0.1])
        result = nearest_probability_distribution(quasiprob)
        
        expected = [0.5, 0.5, 0.0, 0.0]
        assert np.allclose(result, expected)
        assert abs(np.sum(result) - 1.0) < 1e-10

    def test_empty_input(self):
        """Test behavior with edge cases"""
        with pytest.raises((ValueError, IndexError, ZeroDivisionError)):
            nearest_probability_distribution([])

    def test_single_element(self):
        """Test with single element"""
        quasiprob = [0.5]
        result = nearest_probability_distribution(quasiprob)
        
        assert np.allclose(result, [1.0])
        assert abs(np.sum(result) - 1.0) < 1e-10

    def test_two_elements(self):
        """Test with two elements"""
        quasiprob = [0.3, 0.7]
        result = nearest_probability_distribution(quasiprob)
        
        assert np.allclose(result, [0.3, 0.7])
        assert abs(np.sum(result) - 1.0) < 1e-10

    def test_consistency_with_random_inputs(self):
        """Test consistency with random inputs"""
        np.random.seed(42)
        for _ in range(10):
            # Generate random quasiprobability
            quasiprob = np.random.uniform(-0.5, 1.5, size=5)
            # Normalize to sum to 1 (so we start with a quasi-probability that sums to 1)
            quasiprob = quasiprob / np.sum(quasiprob)
            
            result = nearest_probability_distribution(quasiprob)
            
            # Check validity
            assert abs(np.sum(result) - 1.0) < 1e-10, f"Sum not 1 for input {quasiprob}"
            assert all(p >= -1e-10 for p in result), f"Negative values for input {quasiprob}"


class TestNaiveProbabilityDistribution:
    """Test the naive_probability_distribution function"""

    def test_simple_case(self):
        """Test simple case with negatives"""
        quasiprob = [0.6, 0.6, -0.1, -0.1]
        result = naive_probability_distribution(quasiprob)
        
        # Should set negatives to 0 and normalize
        expected = [0.5, 0.5, 0.0, 0.0]
        assert np.allclose(result, expected)
        assert abs(np.sum(result) - 1.0) < 1e-10

    def test_all_positive(self):
        """Test with all positive values"""
        quasiprob = [0.2, 0.3, 0.4, 0.6]
        result = naive_probability_distribution(quasiprob)
        
        # Should just normalize
        expected = np.array(quasiprob) / np.sum(quasiprob)
        assert np.allclose(result, expected)
        assert abs(np.sum(result) - 1.0) < 1e-10


class TestQuasiToReal:
    """Test the quasi_to_real function"""

    def test_nearest_mode(self):
        """Test using nearest mode"""
        quasiprob = [0.6, 0.6, -0.1, -0.1]
        result = quasi_to_real(quasiprob, mode="nearest")
        
        expected = [0.5, 0.5, 0.0, 0.0]
        assert np.allclose(result, expected)

    def test_naive_mode(self):
        """Test using naive mode"""
        quasiprob = [0.6, 0.6, -0.1, -0.1]
        result = quasi_to_real(quasiprob, mode="naive")
        
        expected = [0.5, 0.5, 0.0, 0.0]
        assert np.allclose(result, expected)

    def test_invalid_mode(self):
        """Test with invalid mode"""
        quasiprob = [0.6, 0.6, -0.1, -0.1]
        with pytest.raises(NotImplementedError):
            quasi_to_real(quasiprob, mode="invalid")


if __name__ == "__main__":
    # Run tests directly
    test_instance = TestNearestProbabilityDistribution()
    test_instance.test_already_valid_distribution()
    test_instance.test_simple_negative_case()
    test_instance.test_mostly_negative()
    test_instance.test_mixed_values()
    test_instance.test_edge_case_small_values()
    test_instance.test_all_negative_except_one()
    
    print("✓ All manual tests passed!")