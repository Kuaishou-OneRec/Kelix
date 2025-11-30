#!/usr/bin/env python3
"""
Test to verify the DerivedSeries slicing bug fix for seconds_per_step logging issue.

This test documents the bug that was fixed in DerivedSeries._compute() where
slicing was applied after caching instead of before, causing incorrect results
when chaining operations like diff()[1:].avg()[::step].
"""

import pytest
from muse.utils.metrics import Metrics, Series


class TestDerivedSeriesSlicingFix:
    """Test that DerivedSeries slicing works correctly after bug fix."""
    
    def test_slice_after_diff_removes_none(self):
        """Test that slicing [1:] after diff() correctly removes the None value."""
        series = Series("float")
        for i in range(20):
            series.append(float(i))
        
        # diff() produces None as first element
        diff_series = series.diff()
        assert diff_series[0] is None
        assert diff_series[1] == 1.0
        
        # Slicing [1:] should remove the None
        sliced = diff_series[1:]
        values = list(sliced)
        
        # CRITICAL: No None values should remain
        assert None not in values
        assert len(values) == 19
        assert values[0] == 1.0
        assert values[-1] == 1.0
    
    def test_seconds_per_step_pattern(self):
        """Test the exact pattern used for seconds_per_step in training."""
        acc_steps = 4
        logging_per_step = 100
        
        metrics = Metrics()
        metrics.new("step_time", dtype="timestamp", initial_value=lambda: 0.0)
        metrics.initialize()
        
        # Simulate training
        for i in range(500):
            metrics.step_time.append(float(i + 1))
            metrics.step()
        
        # The pattern from initialize_metrics
        seconds_per_step = metrics.step_time[1:][::acc_steps].diff()[1:]
        
        # Should not contain None values
        values = list(seconds_per_step)
        assert None not in values, f"seconds_per_step contains None: {values[:10]}"
        
        # Apply avg and logging slice
        logged = seconds_per_step.avg(window=logging_per_step)[::logging_per_step]
        logged_values = list(logged)
        
        # CRITICAL: Final logged values should not contain None
        assert None not in logged_values, f"Logged values contain None: {logged_values}"
        assert all(v == 4.0 for v in logged_values), f"Expected all 4.0, got {logged_values}"
    
    def test_chained_slicing_operations(self):
        """Test that multiple chained slicing operations work correctly."""
        series = Series("float")
        for i in range(100):
            series.append(float(i))
        
        # Multiple chained operations
        result = series[::2]  # Every other element
        result = result.diff()  # Compute differences
        result = result[1:]  # Skip first None
        result = result.avg(window=5)  # Moving average
        result = result[::3]  # Every 3rd element
        
        values = list(result)
        
        # Should not contain any None values
        assert None not in values
        assert len(values) > 0
        assert all(isinstance(v, (int, float)) for v in values)
    
    def test_derived_series_slice_caching(self):
        """Test that slicing doesn't break caching behavior."""
        series = Series("float")
        for i in range(50):
            series.append(float(i))
        
        diff_series = series.diff()
        sliced_once = diff_series[1:]
        
        # Access twice to test caching
        values1 = list(sliced_once)
        values2 = list(sliced_once)
        
        assert values1 == values2
        assert None not in values1
        assert len(values1) == 49


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
