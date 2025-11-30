#!/usr/bin/env python3
"""Comprehensive tests for Derived Series indexing and slicing operations."""

import pytest
import time

from muse.utils.metrics import Series, DerivedSeries, Metrics


class TestDerivedSeriesIndexingAndSlicing:
    """Test suite for derived series indexing and slicing operations."""
    
    # ========================================================================
    # Basic Indexing Tests
    # ========================================================================
    
    def test_derived_series_positive_indexing(self):
        """Test positive integer indexing on derived series."""
        series = Series("float")
        for i in range(10):
            series.append(float(i))
        
        # Test on various derived operations
        avg = series.avg(window=3)
        assert avg[0] is not None or avg[0] == 0.0  # First value
        assert avg[5] == pytest.approx((3.0 + 4.0 + 5.0) / 3)
        assert avg[9] == pytest.approx((7.0 + 8.0 + 9.0) / 3)
        
        cumsum = series.cumsum()
        assert cumsum[0] == 0.0
        assert cumsum[5] == pytest.approx(15.0)  # 0+1+2+3+4+5
        assert cumsum[9] == pytest.approx(45.0)  # sum(0..9)
    
    def test_derived_series_negative_indexing(self):
        """Test negative integer indexing on derived series."""
        series = Series("float")
        for i in range(10):
            series.append(float(i))
        
        avg = series.avg(window=3)
        assert avg[-1] == pytest.approx((7.0 + 8.0 + 9.0) / 3)
        assert avg[-2] == pytest.approx((6.0 + 7.0 + 8.0) / 3)
        
        cumsum = series.cumsum()
        assert cumsum[-1] == pytest.approx(45.0)
        assert cumsum[-5] == pytest.approx(15.0)  # cumsum at index 5
    
    # ========================================================================
    # Basic Slicing Tests
    # ========================================================================
    
    def test_derived_series_basic_slicing(self):
        """Test basic slicing [start:stop] on derived series."""
        series = Series("float")
        for i in range(10):
            series.append(float(i))
        
        cumsum = series.cumsum()
        sliced = cumsum[2:7]  # Should get indices 2,3,4,5,6
        
        assert isinstance(sliced, DerivedSeries)
        assert len(sliced) == 5
        assert sliced[0] == pytest.approx(3.0)  # cumsum at original index 2
        assert sliced[-1] == pytest.approx(21.0)  # cumsum at original index 6
    
    def test_derived_series_step_slicing(self):
        """Test slicing with step [::step]."""
        series = Series("float")
        for i in range(10):
            series.append(float(i))  # [0,1,2,3,4,5,6,7,8,9]
        
        cumsum = series.cumsum()  # [0,1,3,6,10,15,21,28,36,45]
        sliced = cumsum[::2]  # Indices 0,2,4,6,8
        
        assert isinstance(sliced, DerivedSeries)
        assert len(sliced) == 5
        assert list(sliced) == pytest.approx([0.0, 3.0, 10.0, 21.0, 36.0])
    
    def test_derived_series_negative_step_slicing(self):
        """Test slicing with negative step (reverse)."""
        series = Series("float")
        for i in range(5):
            series.append(float(i))
        
        cumsum = series.cumsum()  # [0, 1, 3, 6, 10]
        reversed_slice = cumsum[::-1]  # [10, 6, 3, 1, 0]
        
        assert isinstance(reversed_slice, DerivedSeries)
        assert list(reversed_slice) == pytest.approx([10.0, 6.0, 3.0, 1.0, 0.0])
    
    # ========================================================================
    # Multi-level Slicing Tests
    # ========================================================================
    
    def test_derived_series_slice_of_slice(self):
        """Test slicing a derived series that's already sliced."""
        series = Series("float")
        for i in range(20):
            series.append(float(i))
        
        # First slice: every 2nd element
        slice1 = series[::2]  # [0,2,4,6,8,10,12,14,16,18]
        
        # Cumsum on slice
        cumsum = slice1.cumsum()  # [0,2,6,12,20,30,42,56,72,90]
        
        # Second slice: every 3rd element of cumsum
        slice2 = cumsum[::3]  # Indices 0,3,6,9 → [0,12,42,90]
        
        assert isinstance(slice2, DerivedSeries)
        assert len(slice2) == 4
        assert list(slice2) == pytest.approx([0.0, 12.0, 42.0, 90.0])
        
        # Access by index
        assert slice2[0] == pytest.approx(0.0)
        assert slice2[-1] == pytest.approx(90.0)
    
    def test_derived_series_complex_slice_chain(self):
        """Test complex slicing chain: [::4][1:][::2]."""
        series = Series("float")
        for i in range(40):
            series.append(float(i))
        
        # [::4] → every 4th: [0,4,8,12,16,20,24,28,32,36]
        s1 = series[::4]
        
        # [1:] → skip first: [4,8,12,16,20,24,28,32,36]
        s2 = s1[1:]
        
        # [::2] → every 2nd: [4,12,20,28,36]
        s3 = s2[::2]
        
        # Apply operation
        cumsum = s3.cumsum()
        expected = [4.0, 16.0, 36.0, 64.0, 100.0]
        assert list(cumsum) == pytest.approx(expected)
    
    # ========================================================================
    # Operations After Slicing
    # ========================================================================
    
    def test_derived_series_operations_after_slicing(self):
        """Test that operations work correctly on sliced derived series."""
        series = Series("float")
        for i in range(20):
            series.append(float(i))
        
        # Slice then apply operations
        sliced = series[::2]  # [0,2,4,6,8,10,12,14,16,18]
        
        # Test avg
        avg = sliced.avg(window=3)
        assert avg[2] == pytest.approx((0.0 + 2.0 + 4.0) / 3)
        assert avg[-1] == pytest.approx((14.0 + 16.0 + 18.0) / 3)
        
        # Test cumsum
        cumsum = sliced.cumsum()
        assert cumsum[5] == pytest.approx(30.0)  # 0+2+4+6+8+10
        
        # Test arithmetic
        doubled = sliced * 2
        assert doubled[0] == pytest.approx(0.0)
        assert doubled[5] == pytest.approx(20.0)
        
        # Test shift
        shifted = sliced.shift(1)
        assert shifted[0] is None
        assert shifted[1] == pytest.approx(0.0)
        assert shifted[-1] == pytest.approx(16.0)
    
    def test_derived_series_slice_skip_first(self):
        """Test [1:] pattern commonly used to skip initial sentinel value."""
        metrics = Metrics()
        metrics.new("loss", dtype="float", reduce="mean")
        metrics.new("step_time", dtype="timestamp", initial_value=lambda: 0.0)
        metrics.initialize()
        
        # Add some values
        for i in range(10):
            metrics.loss.append(float(i))
            metrics.step_time.append(float(i))
            metrics.step()
        
        # Pattern from define_metrics: [::4][1:]
        every_4th = metrics.loss[::4]  # Indices 0,4,8
        skip_first = every_4th[1:]  # Skip sentinel, indices 4,8
        
        assert len(skip_first) == 2
        # Test that values are accessible
        assert skip_first[0] is not None
        assert skip_first[-1] is not None
    
    # ========================================================================
    # Edge Cases
    # ========================================================================
    
    def test_derived_series_empty_slice(self):
        """Test slicing that results in empty series."""
        series = Series("float")
        for i in range(5):
            series.append(float(i))
        
        cumsum = series.cumsum()
        empty = cumsum[10:20]  # Out of range
        
        assert len(empty) == 0
    
    def test_derived_series_single_element_slice(self):
        """Test slicing that results in single element."""
        series = Series("float")
        for i in range(10):
            series.append(float(i))
        
        cumsum = series.cumsum()
        single = cumsum[5:6]  # Single element at index 5
        
        assert len(single) == 1
        assert single[0] == pytest.approx(15.0)
        assert single[-1] == pytest.approx(15.0)
    
    # ========================================================================
    # Real-world Pattern Tests
    # ========================================================================
    
    def test_derived_series_logger_pattern(self):
        """Test the exact pattern used in define_metrics for logging."""
        metrics = Metrics()
        metrics.new("loss", dtype="float", reduce="mean")
        metrics.initialize()
        
        acc_steps = 4
        logging_per_step = 2
        
        # Simulate training loop
        for i in range(40):
            metrics.loss.append(float(i))
            metrics.step()
        
        # Pattern: avg(window=acc_steps)[::acc_steps][1:].avg(window=logging_per_step)[::logging_per_step]
        avg_loss = metrics.loss.avg(window=acc_steps)
        global_loss = avg_loss[::acc_steps][1:]  # Global steps, skip sentinel
        logged_loss = global_loss.avg(window=logging_per_step)[::logging_per_step]
        
        # Verify we can get values
        assert len(logged_loss) > 0
        
        # Test indexing
        if len(logged_loss) > 0:
            val_first = logged_loss[0]
            val_last = logged_loss[-1]
            
            # Print for debugging
            print(f"\nLogger pattern test:")
            print(f"  Total length: {len(logged_loss)}")
            print(f"  First value: {val_first}")
            print(f"  Last value: {val_last}")
            print(f"  All values: {list(logged_loss)}")
            
            assert val_first is not None
            assert val_last is not None
            
            # Verify values are reasonable
            assert val_first is None or (isinstance(val_first, (int, float)) and val_first >= 0)
            assert val_last is None or (isinstance(val_last, (int, float)) and val_last >= 0)
    
    def test_derived_series_perf_metrics_pattern(self):
        """Test pattern for performance metrics: cumsum[::acc_steps][::logging_per_step]."""
        metrics = Metrics()
        metrics.new("tokens", dtype="int", reduce="sum", initial_value=0)
        metrics.initialize()
        
        acc_steps = 4
        logging_per_step = 2
        
        # Simulate training
        for i in range(40):
            metrics.tokens.append(100 * (i + 1))
            metrics.step()
        
        # Pattern: cumsum[::acc_steps][::logging_per_step]
        total_tokens = metrics.tokens.cumsum()
        global_tokens = total_tokens[::acc_steps]
        logged_tokens = global_tokens[::logging_per_step]
        
        assert len(logged_tokens) > 0
        
        # Print for debugging
        print(f"\nPerf metrics pattern test:")
        print(f"  Total length: {len(logged_tokens)}")
        print(f"  All values: {list(logged_tokens)}")
        
        # All values should be accessible and valid
        for i in range(len(logged_tokens)):
            val = logged_tokens[i]
            assert val is not None
            assert val >= 0
        
        # Test negative indexing
        assert logged_tokens[-1] is not None
    
    def test_derived_series_full_training_simulation(self):
        """Test full training simulation with all patterns from define_metrics."""
        metrics = Metrics()
        metrics.new("loss", dtype="float", reduce="mean")
        metrics.new("grad_norm", dtype="float", reduce="mean")
        metrics.new("learning_rate", dtype="float")
        metrics.new("step_time", dtype="timestamp", initial_value=lambda: time.time())
        metrics.new("tokens", dtype="int", reduce="sum", initial_value=0)
        metrics.initialize()
        
        acc_steps = 4
        logging_per_step = 2
        
        # Simulate 40 micro steps
        for i in range(40):
            metrics.loss.append(float(i) * 0.1)
            metrics.grad_norm.append(1.0 + float(i) * 0.01)
            metrics.learning_rate.append(0.001)
            metrics.step_time.append(time.time())
            metrics.tokens.append(100)
            metrics.step()
        
        # All the patterns from define_metrics
        total_tokens = metrics.tokens.cumsum()
        
        # Global-step metrics, skip the first step
        avg_loss = metrics.loss.avg(window=acc_steps)[::acc_steps][1:]
        avg_grad_norm = metrics.grad_norm[::acc_steps][1:]
        learning_rate = metrics.learning_rate[::acc_steps][1:]
        seconds_per_step = metrics.step_time[::acc_steps].diff()[1:]
        
        # Logging metrics, avg over the last logging_per_step steps
        logged_loss = avg_loss.avg(window=logging_per_step)[::logging_per_step]
        logged_grad_norm = avg_grad_norm.avg(window=logging_per_step)[::logging_per_step]
        logged_lr = learning_rate.avg(window=logging_per_step)[::logging_per_step]
        logged_step_time = seconds_per_step.avg(window=logging_per_step)[::logging_per_step]
        logged_tokens = total_tokens[::acc_steps][::logging_per_step]
        
        # Verify all series have values
        print(f"\nFull training simulation:")
        print(f"  logged_loss length: {len(logged_loss)}")
        print(f"  logged_grad_norm length: {len(logged_grad_norm)}")
        print(f"  logged_lr length: {len(logged_lr)}")
        print(f"  logged_step_time length: {len(logged_step_time)}")
        print(f"  logged_tokens length: {len(logged_tokens)}")
        
        assert len(logged_loss) > 0
        assert len(logged_grad_norm) > 0
        assert len(logged_lr) > 0
        assert len(logged_tokens) > 0
        
        # Test that we can access last values
        if len(logged_loss) > 0:
            print(f"  logged_loss[-1]: {logged_loss[-1]}")
            assert logged_loss[-1] is not None
        
        if len(logged_grad_norm) > 0:
            print(f"  logged_grad_norm[-1]: {logged_grad_norm[-1]}")
            assert logged_grad_norm[-1] is not None
        
        if len(logged_lr) > 0:
            print(f"  logged_lr[-1]: {logged_lr[-1]}")
            assert logged_lr[-1] is not None
        
        if len(logged_tokens) > 0:
            print(f"  logged_tokens[-1]: {logged_tokens[-1]}")
            assert logged_tokens[-1] is not None
