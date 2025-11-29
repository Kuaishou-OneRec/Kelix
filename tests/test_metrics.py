#!/usr/bin/env python3
"""Unit tests for Metrics module."""

import pytest
import time
import numpy as np

from muse.utils.metrics import (
    Scalar, Series, DerivedSeries, Metrics,
    Logger, LoggerProxy, LoggerBackend,
    TensorBoardBackend, WandbBackend, CSVBackend, StdoutBackend
)



class TestScalar:
    """Test suite for Scalar class."""

    def test_scalar_initialization_int(self):
        """Test int type initialization."""
        scalar = Scalar(42, "int")
        assert scalar.dtype == "int"
        assert scalar.value == 42
        assert isinstance(scalar.value, int)

    def test_scalar_initialization_float(self):
        """Test float type initialization."""
        scalar = Scalar(3.14, "float")
        assert scalar.dtype == "float"
        assert scalar.value == pytest.approx(3.14)
        assert isinstance(scalar.value, float)

    def test_scalar_initialization_timestamp(self):
        """Test timestamp type initialization."""
        ts = time.time()
        scalar = Scalar(ts, "timestamp")
        assert scalar.dtype == "timestamp"
        assert scalar.value == pytest.approx(ts)
        assert isinstance(scalar.value, float)

    def test_scalar_value_property(self):
        """Test value property returns correct type."""
        # Int type
        scalar_int = Scalar(10, "int")
        assert scalar_int.value == 10
        assert type(scalar_int.value) == int

        # Float type
        scalar_float = Scalar(10.5, "float")
        assert scalar_float.value == pytest.approx(10.5)
        assert type(scalar_float.value) == float

    def test_scalar_type_conversion(self):
        """Test __float__() and __int__() conversion."""
        scalar = Scalar(42.7, "float")
        assert float(scalar) == pytest.approx(42.7)
        assert int(scalar) == 42

        scalar_int = Scalar(10, "int")
        assert float(scalar_int) == pytest.approx(10.0)
        assert int(scalar_int) == 10

    def test_scalar_repr(self):
        """Test string representation."""
        scalar = Scalar(3.14, "float")
        repr_str = repr(scalar)
        assert "Scalar" in repr_str
        assert "3.14" in repr_str
        assert "float" in repr_str

    def test_scalar_reduce_single_machine_mean(self):
        """Test mean reduce in single machine mode."""
        scalar = Scalar(10.0, "float")
        reduced = scalar.reduce(reduction="mean")
        assert reduced.value == pytest.approx(10.0)
        assert reduced.dtype == "float"

    def test_scalar_reduce_single_machine_sum(self):
        """Test sum reduce in single machine mode."""
        scalar = Scalar(10.0, "float")
        reduced = scalar.reduce(reduction="sum")
        assert reduced.value == pytest.approx(10.0)
        assert reduced.dtype == "float"

    def test_scalar_reduce_invalid_reduction(self):
        """Test invalid reduction parameter."""
        scalar = Scalar(10.0, "float")
        with pytest.raises(ValueError, match="Unsupported reduction"):
            scalar.reduce(reduction="invalid")

    def test_scalar_with_zero(self):
        """Test zero value."""
        scalar = Scalar(0, "int")
        assert scalar.value == 0

        scalar_float = Scalar(0.0, "float")
        assert scalar_float.value == pytest.approx(0.0)

    def test_scalar_with_negative(self):
        """Test negative value."""
        scalar = Scalar(-42, "int")
        assert scalar.value == -42

        scalar_float = Scalar(-3.14, "float")
        assert scalar_float.value == pytest.approx(-3.14)

    def test_scalar_with_large_values(self):
        """Test large values."""
        large_int = 2**31 - 1
        scalar = Scalar(large_int, "int")
        assert scalar.value == large_int

        large_float = 1e10
        scalar_float = Scalar(large_float, "float")
        assert scalar_float.value == pytest.approx(large_float)


class TestSeries:
    """Test suite for Series class."""

    def test_series_initialization(self):
        """Test initialization."""
        series = Series("float")
        assert series.dtype == "float"
        assert len(series) == 0
        assert isinstance(series, list)

    def test_series_append_basic(self):
        """Test basic append operation."""
        series = Series("float")
        series.append(1.5)
        series.append(2.5)
        assert len(series) == 2
        assert series[0] == pytest.approx(1.5)
        assert series[1] == pytest.approx(2.5)

    def test_series_append_multiple_values(self):
        """Test adding multiple values."""
        series = Series("int")
        for i in range(10):
            series.append(i)
        assert len(series) == 10
        assert series[0] == 0
        assert series[9] == 9

    def test_series_list_operations(self):
        """Test list inherited operations (indexing, slicing, len)."""
        series = Series("float")
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            series.append(v)

        # Test indexing
        assert series[0] == pytest.approx(1.0)
        assert series[-1] == pytest.approx(5.0)

        # Test slicing - should return Series
        slice_result = series[1:3]
        assert isinstance(slice_result, Series)
        assert slice_result.dtype == "float"
        assert len(slice_result) == 2
        assert slice_result[0] == pytest.approx(2.0)
        assert slice_result[1] == pytest.approx(3.0)

        # Test len
        assert len(series) == 5

    def test_series_slicing_returns_series(self):
        """Test that slicing returns Series object, not list."""
        series = Series("int")
        for i in range(10):
            series.append(i)

        # Various slice operations
        slice1 = series[2:5]
        assert isinstance(slice1, Series)
        assert list(slice1) == [2, 3, 4]
        assert slice1.dtype == "int"

        slice2 = series[:3]
        assert isinstance(slice2, Series)
        assert list(slice2) == [0, 1, 2]

        slice3 = series[7:]
        assert isinstance(slice3, Series)
        assert list(slice3) == [7, 8, 9]

        slice4 = series[::2]
        assert isinstance(slice4, Series)
        assert list(slice4) == [0, 2, 4, 6, 8]

    def test_series_sliced_series_operations(self):
        """Test that sliced Series can use statistical operations."""
        series = Series("float")
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        for v in values:
            series.append(v)

        # Slice and compute statistics
        sliced = series[3:8]  # [4.0, 5.0, 6.0, 7.0, 8.0]
        assert isinstance(sliced, Series)
        
        # avg, sum, cumsum now return DerivedSeries
        avg = sliced.avg()
        assert isinstance(avg, DerivedSeries)
        assert avg[2] == pytest.approx(6.0)  # Middle value (4+5+6)/3 = 5, expanding avg
        
        total = sliced.sum()
        assert isinstance(total, DerivedSeries)
        # sum without window is expanding sum
        assert total[4] == pytest.approx(30.0)  # 4+5+6+7+8

    def test_series_negative_indexing(self):
        """Test negative indexing and slicing."""
        series = Series("int")
        for i in range(5):
            series.append(i)

        # Negative index
        assert series[-1] == 4
        assert series[-2] == 3

        # Negative slice
        sliced = series[-3:]
        assert isinstance(sliced, Series)
        assert list(sliced) == [2, 3, 4]

    def test_series_append_scalar_object(self):
        """Test adding Scalar object."""
        series = Series("float")
        scalar = Scalar(3.14, "float")
        series.append(scalar)
        assert len(series) == 1
        assert series[0] == pytest.approx(3.14)

    def test_series_avg_all(self):
        """Test computing average with expanding window."""
        series = Series("float")
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            series.append(v)

        avg = series.avg()  # Returns DerivedSeries with expanding window
        assert isinstance(avg, DerivedSeries)
        # Expanding average: [1, 1.5, 2, 2.5, 3]
        assert avg[0] == pytest.approx(1.0)
        assert avg[1] == pytest.approx(1.5)
        assert avg[2] == pytest.approx(2.0)
        assert avg[3] == pytest.approx(2.5)
        assert avg[4] == pytest.approx(3.0)

    def test_series_avg_with_window(self):
        """Test computing average with sliding window."""
        series = Series("float")
        for i in range(10):
            series.append(float(i))

        avg = series.avg(window=3)  # Sliding window of 3
        assert isinstance(avg, DerivedSeries)
        # Window 3: [0], [0,1], [0,1,2], [1,2,3], [2,3,4], ...
        assert avg[0] == pytest.approx(0.0)  # 0/1
        assert avg[1] == pytest.approx(0.5)  # (0+1)/2
        assert avg[2] == pytest.approx(1.0)  # (0+1+2)/3
        assert avg[3] == pytest.approx(2.0)  # (1+2+3)/3
        assert avg[4] == pytest.approx(3.0)  # (2+3+4)/3

    def test_series_sum_all(self):
        """Test computing sum with expanding window."""
        series = Series("float")
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            series.append(v)

        total = series.sum()  # Returns DerivedSeries with expanding window
        assert isinstance(total, DerivedSeries)
        # Expanding sum: [1, 3, 6, 10, 15]
        assert total[0] == pytest.approx(1.0)
        assert total[1] == pytest.approx(3.0)
        assert total[2] == pytest.approx(6.0)
        assert total[3] == pytest.approx(10.0)
        assert total[4] == pytest.approx(15.0)

    def test_series_sum_with_window(self):
        """Test computing sum with sliding window."""
        series = Series("int")
        for i in range(10):
            series.append(i)

        total = series.sum(window=3)  # Sliding window of 3
        assert isinstance(total, DerivedSeries)
        # Window 3: [0], [0,1], [0,1,2], [1,2,3], [2,3,4], ...
        assert total[0] == 0  # 0
        assert total[1] == 1  # 0+1
        assert total[2] == 3  # 0+1+2
        assert total[3] == 6  # 1+2+3
        assert total[4] == 9  # 2+3+4

    def test_series_sum_empty(self):
        """Test sum on empty series."""
        series = Series("float")
        # Empty series returns empty DerivedSeries
        total = series.sum()
        assert isinstance(total, DerivedSeries)
        assert len(total) == 0

    def test_series_cumsum_basic(self):
        """Test basic cumsum functionality."""
        series = Series("int")
        values = [1, 2, 3, 4, 5]
        for v in values:
            series.append(v)

        cumsum = series.cumsum()
        assert isinstance(cumsum, DerivedSeries)
        assert len(cumsum) == 5
        assert list(cumsum) == [1, 3, 6, 10, 15]

    def test_series_cumsum_float(self):
        """Test cumsum with float values."""
        series = Series("float")
        values = [1.5, 2.5, 3.0, 4.0]
        for v in values:
            series.append(v)

        cumsum = series.cumsum()
        assert isinstance(cumsum, DerivedSeries)
        assert len(cumsum) == 4
        assert cumsum[0] == pytest.approx(1.5)
        assert cumsum[1] == pytest.approx(4.0)
        assert cumsum[2] == pytest.approx(7.0)
        assert cumsum[3] == pytest.approx(11.0)

    def test_series_cumsum_empty(self):
        """Test cumsum on empty series."""
        series = Series("int")
        cumsum = series.cumsum()
        assert isinstance(cumsum, DerivedSeries)
        assert len(cumsum) == 0

    def test_series_cumsum_single(self):
        """Test cumsum with single element."""
        series = Series("float")
        series.append(5.0)
        cumsum = series.cumsum()
        assert isinstance(cumsum, DerivedSeries)
        assert len(cumsum) == 1
        assert cumsum[0] == pytest.approx(5.0)

    def test_series_type_validation(self):
        """Test type conversion and validation."""
        # Int series
        series_int = Series("int")
        series_int.append(3.7)
        assert series_int[0] == 3  # Should be converted to int

        # Float series
        series_float = Series("float")
        series_float.append(5)
        assert series_float[0] == pytest.approx(5.0)

    def test_series_timestamp_none(self):
        """Test timestamp type with None value."""
        series = Series("timestamp")
        before = time.time()
        series.append(None)
        after = time.time()

        # Value should be between before and after
        assert before <= series[0] <= after

    def test_series_timestamp_explicit(self):
        """Test timestamp type with explicit value."""
        series = Series("timestamp")
        ts = 1234567890.5
        series.append(ts)
        assert series[0] == pytest.approx(ts)

    def test_series_tick_method(self):
        """Test tick() method for timestamp Series."""
        series = Series("timestamp")
        
        before = time.time()
        ts = series.tick()
        after = time.time()
        
        # Returned timestamp should be between before and after
        assert before <= ts <= after
        
        # Value should be added to series
        assert len(series) == 1
        assert series[0] == pytest.approx(ts)
        
        # Multiple ticks
        time.sleep(0.01)  # Small delay to ensure different timestamps
        ts2 = series.tick()
        time.sleep(0.01)
        ts3 = series.tick()
        assert len(series) == 3
        assert ts2 >= ts  # Use >= instead of > to handle edge cases
        assert ts3 >= ts2

    def test_series_tick_non_timestamp_error(self):
        """Test tick() raises error for non-timestamp Series."""
        series_float = Series("float")
        with pytest.raises(TypeError, match="only available for timestamp"):
            series_float.tick()
        
        series_int = Series("int")
        with pytest.raises(TypeError, match="only available for timestamp"):
            series_int.tick()

    def test_series_append_scalar_dtype_mismatch(self):
        """Test appending Scalar with mismatched dtype."""
        series = Series("float")
        scalar_int = Scalar(10, "int")
        with pytest.raises(ValueError, match="dtype.*does not match"):
            series.append(scalar_int)


class TestMetrics:
    """Test suite for Metrics class with flattened structure and index."""

    def test_metrics_initialization(self):
        """Test initialization."""
        metrics = Metrics()
        assert len(metrics._series) == 0
        assert len(metrics._index) == 0
        assert metrics._current_index == 0
        assert metrics._index_name == "step"

    def test_metrics_custom_index_name(self):
        """Test custom index name."""
        metrics = Metrics(index_name="iteration")
        assert metrics._index_name == "iteration"

    def test_metrics_new_creates_series(self):
        """Test creating a series."""
        metrics = Metrics()
        metrics.new("loss", dtype="float")
        assert "loss" in metrics._series
        assert isinstance(metrics._series["loss"], Series)

    def test_metrics_step_increments_index(self):
        """Test that step() increments the index."""
        metrics = Metrics()
        assert len(metrics._index) == 0
        
        metrics.step()
        assert len(metrics._index) == 1
        assert metrics._index[0] == 0
        assert metrics._current_index == 1
        
        metrics.step()
        assert len(metrics._index) == 2
        assert metrics._index[1] == 1
        assert metrics._current_index == 2

    def test_metrics_step_custom_value(self):
        """Test step() with custom value."""
        metrics = Metrics()
        metrics.step(100)
        assert metrics._index[0] == 100
        assert metrics._current_index == 101

    def test_metrics_series_access(self):
        """Test accessing series via attribute."""
        metrics = Metrics()
        metrics.new("loss", dtype="float")
        
        series = metrics.loss
        assert isinstance(series, Series)
        assert series.dtype == "float"

    def test_metrics_access_nonexistent(self):
        """Test accessing non-existent series raises AttributeError."""
        metrics = Metrics()
        with pytest.raises(AttributeError, match="not found"):
            _ = metrics.nonexistent

    def test_metrics_index_alignment(self):
        """Test that series aligns with index."""
        metrics = Metrics()
        metrics.new("loss", dtype="float")
        
        metrics.step()
        metrics.loss.append(0.5)
        
        metrics.step()
        metrics.loss.append(0.3)
        
        assert len(metrics.loss) == 2
        assert metrics.loss[0] == 0.5
        assert metrics.loss[1] == 0.3

    def test_metrics_missing_values(self):
        """Test that missing values are filled correctly when steps are skipped."""
        metrics = Metrics()
        metrics.new("loss", dtype="float")
        metrics.new("tokens", dtype="int")
        
        # Step 0: append loss only
        metrics.step()
        metrics.loss.append(0.5)
        
        # Step 1: append loss only
        metrics.step()
        metrics.loss.append(0.3)
        
        # Step 2: skip loss, append tokens only
        metrics.step()
        metrics.tokens.append(100)
        
        # Check values
        assert len(metrics.loss) == 2  # Only 2 values appended
        # tokens should have missing values for steps 0 and 1, then 100 at step 2
        assert len(metrics.tokens) == 3
        assert metrics.tokens[0] == 0  # Missing value for int is 0
        assert metrics.tokens[1] == 0  # Missing value for int is 0
        assert metrics.tokens[2] == 100

    def test_metrics_append_without_step_raises_error(self):
        """Test that appending without calling step() raises RuntimeError."""
        metrics = Metrics()
        metrics.new("loss", dtype="float")
        
        # Try to append without calling step()
        with pytest.raises(RuntimeError, match="no index step available"):
            metrics.loss.append(0.5)

    def test_metrics_append_twice_same_step_raises_error(self):
        """Test that appending twice to same step raises RuntimeError."""
        metrics = Metrics()
        metrics.new("loss", dtype="float")
        
        metrics.step()
        metrics.loss.append(0.5)
        
        # Try to append again without advancing step
        with pytest.raises(RuntimeError, match="already has a value"):
            metrics.loss.append(0.3)

    def test_metrics_append_exceeds_index_raises_error(self):
        """Test that appending more values than index steps raises RuntimeError."""
        metrics = Metrics()
        metrics.new("loss", dtype="float")
        
        metrics.step()
        metrics.loss.append(0.5)
        
        # This should fail - we've already appended for the only available index
        with pytest.raises(RuntimeError, match="already appended"):
            metrics.loss.append(0.3)

    def test_metrics_workflow(self):
        """Test complete workflow with index."""
        metrics = Metrics()
        metrics.new("loss", dtype="float")
        metrics.new("tokens", dtype="int")
        
        # Simulate training steps
        for i in range(5):
            metrics.step()
            metrics.loss.append(1.0 / (i + 1))
            metrics.tokens.append((i + 1) * 100)
        
        # Check lengths
        assert len(metrics.loss) == 5
        assert len(metrics.tokens) == 5
        
        # Compute derived metrics
        total_tokens = metrics.tokens.cumsum()
        assert total_tokens[4] == 1500  # 100+200+300+400+500

    def test_metrics_logger_proxy(self):
        """Test logger proxy access."""
        metrics = Metrics()
        assert isinstance(metrics.logger, LoggerProxy)

    def test_metrics_add_logger(self):
        """Test adding loggers."""
        metrics = Metrics()
        logger = Logger("test", [StdoutBackend()])
        metrics.add_logger(logger)
        assert len(metrics._loggers) == 1
        assert metrics._loggers[0] is logger


class TestLogger:
    """Test suite for Logger and backends."""

    def test_logger_initialization(self):
        """Test logger initialization."""
        backend = StdoutBackend()
        logger = Logger("test", [backend])
        assert logger.name == "test"
        assert len(logger.backends) == 1

    def test_logger_track(self):
        """Test tracking series."""
        metrics = Metrics()
        metrics.new("loss", dtype="float")
        
        logger = Logger("test", [StdoutBackend()])
        logger.track(metrics.loss, name="loss", group="training")
        
        assert "loss" in logger._tracked_series
        assert logger._tracked_series["loss"][0] is metrics.loss
        assert logger._tracked_series["loss"][1] == "training"

    def test_logger_write(self, capsys):
        """Test writing to stdout backend."""
        metrics = Metrics()
        metrics.new("loss", dtype="float")
        
        metrics.step()
        metrics.loss.append(0.5)
        
        logger = Logger("test", [StdoutBackend()])
        logger.track(metrics.loss, name="loss", group="training")
        logger.write(step=0)
        
        captured = capsys.readouterr()
        assert "Step 0" in captured.out
        assert "training/loss" in captured.out

    def test_logger_proxy_track(self):
        """Test logger proxy broadcasting track()."""
        metrics = Metrics()
        metrics.new("loss", dtype="float")
        
        logger1 = Logger("logger1", [StdoutBackend()])
        logger2 = Logger("logger2", [StdoutBackend()])
        metrics.add_logger(logger1)
        metrics.add_logger(logger2)
        
        # Track via proxy should broadcast to all loggers
        metrics.logger.track(metrics.loss, name="loss", group="training")
        
        assert "loss" in logger1._tracked_series
        assert "loss" in logger2._tracked_series

    def test_metrics_write_logs(self, capsys):
        """Test metrics.write_logs() integration."""
        metrics = Metrics()
        metrics.new("loss", dtype="float")
        
        logger = Logger("test", [StdoutBackend()])
        metrics.add_logger(logger)
        logger.track(metrics.loss, name="loss", group="training")
        
        metrics.step()
        metrics.loss.append(0.5)
        metrics.write_logs()
        
        captured = capsys.readouterr()
        assert "Step 0" in captured.out

    def test_tensorboard_backend(self, tmp_path):
        """Test TensorBoard backend."""
        import os
        log_dir = str(tmp_path / "tb_logs")
        backend = TensorBoardBackend(log_dir)
        
        values = {"training": {"loss": 0.5}}
        backend.write(step=0, values=values)
        backend.close()
        
        # Check that log directory was created
        assert os.path.exists(log_dir)

    def test_csv_backend(self, tmp_path):
        """Test CSV backend."""
        import csv
        csv_path = str(tmp_path / "metrics.csv")
        backend = CSVBackend(csv_path)
        
        values = {"training": {"loss": 0.5, "accuracy": 0.9}}
        backend.write(step=0, values=values)
        backend.write(step=1, values={"training": {"loss": 0.4, "accuracy": 0.92}})
        backend.close()
        
        # Read and verify CSV
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2
            assert rows[0]['step'] == '0'
            assert rows[0]['training/loss'] == '0.5'
            assert rows[1]['step'] == '1'
            assert rows[1]['training/loss'] == '0.4'


class TestReduce:
    """Test suite for reduce() operation."""

    def test_reduce_returns_derived_series(self):
        """Test that reduce() returns DerivedSeries."""
        series = Series("float", metrics=None)
        for i in [1.0, 2.0, 3.0]:
            series.append(i)
        
        reduced = series.reduce("mean")
        assert isinstance(reduced, DerivedSeries)

    def test_reduce_identity_operation(self):
        """Test that reduce uses identity operation."""
        series = Series("float", metrics=None)
        for i in [1.0, 2.0, 3.0]:
            series.append(i)
        
        reduced = series.reduce("mean")
        assert reduced.operation == "identity"
        assert reduced.reduce_op == "mean"

    def test_reduce_on_derived_series(self):
        """Test reduce on DerivedSeries."""
        series = Series("float", metrics=None)
        for i in range(10):
            series.append(float(i))
        
        avg = series.avg(window=3)
        reduced_avg = avg.reduce("mean")
        
        assert isinstance(reduced_avg, DerivedSeries)
        assert reduced_avg.operation == "identity"


class TestIntegration:
    """Integration tests."""

    def test_end_to_end_training_simulation(self):
        """Simulate a training scenario with new structure."""
        metrics = Metrics()

        # Setup metrics (no groups!)
        metrics.new("loss", dtype="float")
        metrics.new("accuracy", dtype="float")
        metrics.new("tokens", dtype="int")
        metrics.new("step_time", dtype="timestamp")

        # Setup logger
        import io
        import sys
        logger = Logger("training", [StdoutBackend()])
        metrics.add_logger(logger)
        
        # Track some metrics
        logger.track(metrics.loss.avg(window=2), name="avg_loss", group="training")

        # Simulate training steps
        for i in range(5):
            metrics.step()
            metrics.loss.append(1.0 / (i + 1))
            metrics.accuracy.append(0.5 + i * 0.1)
            metrics.tokens.append((i + 1) * 100)
            metrics.step_time.tick()

        # Verify data
        assert len(metrics.loss) == 5
        assert len(metrics.accuracy) == 5
        assert len(metrics.tokens) == 5
        
        # Compute derived metrics
        total_tokens = metrics.tokens.cumsum()
        assert total_tokens[4] == 1500

    def test_common_py_pattern(self):
        """Test the pattern used in common.py."""
        metrics = Metrics()
        
        # Micro-step metrics
        metrics.new("loss", dtype="float")
        metrics.new("tokens", dtype="int")
        metrics.new("step_time", dtype="timestamp")
        
        # Simulate steps
        for i in range(10):
            metrics.step()
            metrics.loss.append(1.0 - i * 0.05)
            metrics.tokens.append(100)
            metrics.step_time.tick()
        
        # Derived metrics (like in common.py)
        total_tokens = metrics.tokens.cumsum()
        tokens_rate = total_tokens.diff() / metrics.step_time.diff()
        
        # This should work without errors
        assert len(total_tokens) == 10
        
        # Test reduce
        reduced_loss = metrics.loss.reduce("mean")
        # Indexing should return Scalar
        scalar_result = reduced_loss[0]
        assert isinstance(scalar_result, Scalar)


        # Simulate training epochs
        for epoch in range(5):
            # Training step
            train_loss = 1.0 / (epoch + 1)
            train_acc = 0.5 + epoch * 0.1
            metrics.train.loss.append(train_loss)
            metrics.train.accuracy.append(train_acc)
            best_train_loss = min(best_train_loss, train_loss)

            # Validation step
            val_loss = train_loss * 1.1
            metrics.val.loss.append(val_loss)
            best_val_loss = min(best_val_loss, val_loss)

            # Metadata
            metrics.step.append(epoch)
            metrics.epoch_time.append(None)  # Current timestamp

        # Verify results
        assert len(metrics.train.loss) == 5
        assert len(metrics.val.loss) == 5
        assert len(metrics.step) == 5
        assert len(metrics.epoch_time) == 5

        # Check statistics
        avg_train_loss = metrics.train.loss.avg()
        assert isinstance(avg_train_loss, DerivedSeries)
        assert avg_train_loss[4] > 0  # Last value should be positive

        # Best loss should be tracked externally
        assert best_train_loss == pytest.approx(0.2)  # 1.0 / 5

        # Check last 3 epochs average with window
        recent_avg = metrics.train.loss.avg(window=3)
        assert isinstance(recent_avg, DerivedSeries)
        assert recent_avg[4] > 0  # Last value should be positive

    def test_multiple_dtype_series(self):
        """Test different dtype Series in same Metrics."""
        metrics = Metrics()

        metrics.new("int_metric", dtype="int")
        metrics.new("float_metric", dtype="float")
        metrics.new("timestamp_metric", dtype="timestamp")

        # Add values
        metrics.int_metric.append(42)
        metrics.float_metric.append(3.14)
        metrics.timestamp_metric.append(time.time())

        # Verify types
        assert isinstance(metrics.int_metric[0], int)
        assert isinstance(metrics.float_metric[0], float)
        assert isinstance(metrics.timestamp_metric[0], float)

        # Test statistics return correct DerivedSeries types
        metrics.int_metric.append(10)
        metrics.int_metric.append(20)
        avg = metrics.int_metric.avg()
        assert isinstance(avg, DerivedSeries)
        assert avg.dtype == "int"

    def test_scalar_reduce_chain(self):
        """Test chaining reduce operations."""
        metrics = Metrics()
        metrics.new("loss", dtype="float")

        for i in range(10):
            metrics.loss.append(float(i))

        # Compute average - now returns DerivedSeries
        avg = metrics.loss.avg()
        assert isinstance(avg, DerivedSeries)
        # Expanding average at last position
        assert avg[9] == pytest.approx(4.5)

    def test_series_operations_with_derived_series(self):
        """Test Series operations that return DerivedSeries."""
        metrics = Metrics()
        metrics.new("values", dtype="float")

        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        for v in values:
            metrics.values.append(v)

        # All operations should return DerivedSeries
        avg = metrics.values.avg()
        sum_series = metrics.values.sum()
        cumsum_series = metrics.values.cumsum()

        assert all(isinstance(x, DerivedSeries) for x in [avg, sum_series, cumsum_series])

        # Verify values
        assert avg[9] == pytest.approx(5.5)  # Expanding average at position 9
        assert sum_series[9] == pytest.approx(55.0)  # Expanding sum at position 9
        assert cumsum_series[9] == pytest.approx(55.0)  # Cumulative sum at position 9


# New tests for DerivedSeries and computation graph
class TestDerivedSeries:
    """Test suite for DerivedSeries class and computation graph."""

    def test_derived_series_lazy_evaluation(self):
        """Test that DerivedSeries evaluates lazily."""
        series = Series("float")
        for i in range(5):
            series.append(float(i))

        # Create derived series - should not compute yet
        avg_series = series.avg(window=3)
        assert isinstance(avg_series, DerivedSeries)
        assert avg_series._cache is None or not avg_series._cache_valid

        # Access triggers computation
        value = avg_series[3]
        assert value == pytest.approx(2.0)  # (1+2+3)/3
        assert avg_series._cache is not None
        assert avg_series._cache_valid

    def test_rolling_window_avg(self):
        """Test sliding window average calculation."""
        series = Series("float")
        for i in range(10):
            series.append(float(i))

        avg_series = series.avg(window=3)
        
        # Window positions:
        # [0]: avg([0]) = 0
        # [1]: avg([0,1]) = 0.5
        # [2]: avg([0,1,2]) = 1.0
        # [3]: avg([1,2,3]) = 2.0
        # [4]: avg([2,3,4]) = 3.0
        assert avg_series[0] == pytest.approx(0.0)
        assert avg_series[1] == pytest.approx(0.5)
        assert avg_series[2] == pytest.approx(1.0)
        assert avg_series[3] == pytest.approx(2.0)
        assert avg_series[4] == pytest.approx(3.0)

    def test_rolling_window_sum(self):
        """Test sliding window sum calculation."""
        series = Series("int")
        for i in range(10):
            series.append(i)

        sum_series = series.sum(window=3)
        
        # Window positions:
        # [0]: sum([0]) = 0
        # [1]: sum([0,1]) = 1
        # [2]: sum([0,1,2]) = 3
        # [3]: sum([1,2,3]) = 6
        # [4]: sum([2,3,4]) = 9
        assert sum_series[0] == 0
        assert sum_series[1] == 1
        assert sum_series[2] == 3
        assert sum_series[3] == 6
        assert sum_series[4] == 9

    def test_cumsum_no_window(self):
        """Test cumulative sum (no window parameter)."""
        series = Series("int")
        for i in [1, 2, 3, 4, 5]:
            series.append(i)

        cumsum_series = series.cumsum()
        assert isinstance(cumsum_series, DerivedSeries)
        assert list(cumsum_series) == [1, 3, 6, 10, 15]
        
        # Verify cumsum doesn't accept window parameter
        with pytest.raises(ValueError, match="cumsum does not support window"):
            DerivedSeries(source=series, operation="cumsum", window=3)

    def test_nested_dependencies(self):
        """Test multi-layer dependencies (s3 depends on s2, s2 depends on s1)."""
        series = Series("float")
        for i in range(10):
            series.append(float(i))

        # First level: average with window
        avg_series = series.avg(window=3)
        
        # Second level: sum of averages
        sum_of_avg = avg_series.sum(window=2)
        
        assert isinstance(avg_series, DerivedSeries)
        assert isinstance(sum_of_avg, DerivedSeries)
        
        # Verify computation works
        value = sum_of_avg[4]
        # avg[3] = 2.0, avg[4] = 3.0
        # sum([avg[3], avg[4]]) = 5.0
        assert value == pytest.approx(5.0)

    def test_auto_update_on_append(self):
        """Test automatic cache invalidation and update when source data changes."""
        series = Series("float")
        for i in range(5):
            series.append(float(i))

        avg_series = series.avg(window=3)
        
        # Access to build cache
        initial_value = avg_series[4]  # (2+3+4)/3 = 3.0
        assert initial_value == pytest.approx(3.0)
        assert avg_series._cache_valid
        
        # Append to source - should invalidate cache
        series.append(5.0)
        assert not avg_series._cache_valid
        
        # Access should trigger recomputation
        new_value = avg_series[5]  # (3+4+5)/3 = 4.0
        assert new_value == pytest.approx(4.0)
        assert avg_series._cache_valid

    def test_derived_series_slicing(self):
        """Test slicing of DerivedSeries."""
        series = Series("float")
        for i in range(10):
            series.append(float(i))

        avg_series = series.avg(window=3)
        
        # Slice the derived series
        sliced_avg = avg_series[3:6]
        assert isinstance(sliced_avg, DerivedSeries)
        
        # Values should match original derived series
        for i, idx in enumerate(range(3, 6)):
            assert sliced_avg[i] == pytest.approx(avg_series[idx])

    def test_derived_series_iteration(self):
        """Test iterating over DerivedSeries."""
        series = Series("int")
        for i in [1, 2, 3, 4, 5]:
            series.append(i)

        cumsum_series = series.cumsum()
        
        # Iterate and collect values
        values = list(cumsum_series)
        assert values == [1, 3, 6, 10, 15]

    def test_derived_series_length(self):
        """Test __len__ on DerivedSeries."""
        series = Series("float")
        for i in range(7):
            series.append(float(i))

        avg_series = series.avg(window=3)
        assert len(avg_series) == 7  # Same length as source

    def test_expanding_window(self):
        """Test expanding window (window=None)."""
        series = Series("float")
        for i in range(5):
            series.append(float(i + 1))  # [1, 2, 3, 4, 5]

        # Expanding average (no window specified)
        avg_series = series.avg()
        
        # Should compute average of all values up to each position
        assert avg_series[0] == pytest.approx(1.0)  # avg([1])
        assert avg_series[1] == pytest.approx(1.5)  # avg([1,2])
        assert avg_series[2] == pytest.approx(2.0)  # avg([1,2,3])
        assert avg_series[3] == pytest.approx(2.5)  # avg([1,2,3,4])
        assert avg_series[4] == pytest.approx(3.0)  # avg([1,2,3,4,5])

    def test_cache_invalidation_propagation(self):
        """Test cache invalidation propagates through nested dependencies."""
        series = Series("float")
        for i in range(5):
            series.append(float(i))

        # Create nested dependencies
        avg1 = series.avg(window=2)
        avg2 = avg1.avg(window=2)
        
        # Build caches
        _ = avg1[4]
        _ = avg2[4]
        assert avg1._cache_valid
        assert avg2._cache_valid
        
        # Append to original series
        series.append(5.0)
        
        # All caches should be invalidated
        assert not avg1._cache_valid
        assert not avg2._cache_valid

    def test_derived_from_derived(self):
        """Test creating DerivedSeries from another DerivedSeries."""
        series = Series("int")
        for i in [2, 4, 6, 8, 10]:
            series.append(i)

        # First derived: cumsum
        cumsum_series = series.cumsum()  # [2, 6, 12, 20, 30]
        
        # Second derived: avg of cumsum
        avg_cumsum = cumsum_series.avg(window=2)
        
        assert isinstance(avg_cumsum, DerivedSeries)
        # avg_cumsum[2] = avg([cumsum[1], cumsum[2]]) = avg([6, 12]) = 9
        assert avg_cumsum[2] == pytest.approx(9.0)

    def test_empty_series_derived(self):
        """Test DerivedSeries with empty source."""
        series = Series("float")
        
        avg_series = series.avg()
        assert len(avg_series) == 0
        assert list(avg_series) == []

    def test_single_value_derived(self):
        """Test DerivedSeries with single value."""
        series = Series("float")
        series.append(5.0)
        
        avg_series = series.avg(window=3)
        assert len(avg_series) == 1
        assert avg_series[0] == pytest.approx(5.0)

    def test_cumsum_optimization_for_sliding_window(self):
        """Test that sliding window operations use cumsum optimization."""
        series = Series("int")
        for i in range(100):
            series.append(i)

        # Create derived series with window
        sum_series = series.sum(window=10)
        
        # Trigger computation
        _ = list(sum_series)
        
        # Verify cumsum cache was created for optimization
        assert sum_series._cumsum_cache is not None
        assert sum_series._cumsum_cache_valid


# New tests for arithmetic operations
class TestArithmeticOperations:
    """Test suite for Series arithmetic operations."""

    def test_series_add_scalar(self):
        """Test adding a scalar to Series."""
        series = Series("float")
        for i in [1.0, 2.0, 3.0]:
            series.append(i)
        
        result = series + 10
        assert isinstance(result, DerivedSeries)
        assert list(result) == [11.0, 12.0, 13.0]
    
    def test_series_radd_scalar(self):
        """Test reverse add (scalar + Series)."""
        series = Series("float")
        for i in [1.0, 2.0, 3.0]:
            series.append(i)
        
        result = 10 + series
        assert isinstance(result, DerivedSeries)
        assert list(result) == [11.0, 12.0, 13.0]
    
    def test_series_add_series(self):
        """Test element-wise addition of two Series."""
        s1 = Series("float")
        s2 = Series("float")
        for i in [1.0, 2.0, 3.0]:
            s1.append(i)
            s2.append(i * 2)
        
        result = s1 + s2
        assert isinstance(result, DerivedSeries)
        assert list(result) == [3.0, 6.0, 9.0]
    
    def test_series_sub_scalar(self):
        """Test subtracting a scalar from Series."""
        series = Series("float")
        for i in [10.0, 20.0, 30.0]:
            series.append(i)
        
        result = series - 5
        assert isinstance(result, DerivedSeries)
        assert list(result) == [5.0, 15.0, 25.0]
    
    def test_series_rsub_scalar(self):
        """Test reverse subtract (scalar - Series)."""
        series = Series("float")
        for i in [1.0, 2.0, 3.0]:
            series.append(i)
        
        result = 10 - series
        assert isinstance(result, DerivedSeries)
        assert list(result) == [9.0, 8.0, 7.0]
    
    def test_series_sub_series(self):
        """Test element-wise subtraction of two Series."""
        s1 = Series("float")
        s2 = Series("float")
        for i in [10.0, 20.0, 30.0]:
            s1.append(i)
        for i in [1.0, 5.0, 10.0]:
            s2.append(i)
        
        result = s1 - s2
        assert isinstance(result, DerivedSeries)
        assert list(result) == [9.0, 15.0, 20.0]
    
    def test_series_mul_scalar(self):
        """Test multiplying Series by scalar."""
        series = Series("float")
        for i in [2.0, 3.0, 4.0]:
            series.append(i)
        
        result = series * 3
        assert isinstance(result, DerivedSeries)
        assert list(result) == [6.0, 9.0, 12.0]
    
    def test_series_rmul_scalar(self):
        """Test reverse multiply (scalar * Series)."""
        series = Series("float")
        for i in [2.0, 3.0, 4.0]:
            series.append(i)
        
        result = 3 * series
        assert isinstance(result, DerivedSeries)
        assert list(result) == [6.0, 9.0, 12.0]
    
    def test_series_mul_series(self):
        """Test element-wise multiplication of two Series."""
        s1 = Series("float")
        s2 = Series("float")
        for i in [2.0, 3.0, 4.0]:
            s1.append(i)
            s2.append(i)
        
        result = s1 * s2
        assert isinstance(result, DerivedSeries)
        assert list(result) == [4.0, 9.0, 16.0]
    
    def test_series_div_scalar(self):
        """Test dividing Series by scalar."""
        series = Series("float")
        for i in [10.0, 20.0, 30.0]:
            series.append(i)
        
        result = series / 2
        assert isinstance(result, DerivedSeries)
        assert list(result) == [5.0, 10.0, 15.0]
    
    def test_series_rdiv_scalar(self):
        """Test reverse divide (scalar / Series)."""
        series = Series("float")
        for i in [2.0, 4.0, 5.0]:
            series.append(i)
        
        result = 20 / series
        assert isinstance(result, DerivedSeries)
        assert list(result) == pytest.approx([10.0, 5.0, 4.0])
    
    def test_series_div_series(self):
        """Test element-wise division of two Series."""
        s1 = Series("float")
        s2 = Series("float")
        for i in [10.0, 20.0, 30.0]:
            s1.append(i)
        for i in [2.0, 4.0, 5.0]:
            s2.append(i)
        
        result = s1 / s2
        assert isinstance(result, DerivedSeries)
        assert list(result) == [5.0, 5.0, 6.0]
    
    def test_division_by_zero_error(self):
        """Test that division by zero raises appropriate error."""
        series = Series("float")
        for i in [1.0, 0.0, 3.0]:
            series.append(i)
        
        result = series / 0
        with pytest.raises(ZeroDivisionError):
            _ = list(result)
        
        # Also test element-wise division by zero
        s2 = Series("float")
        for i in [1.0, 0.0, 1.0]:
            s2.append(i)
        
        result2 = series / s2
        with pytest.raises(ZeroDivisionError):
            _ = list(result2)
    
    def test_length_mismatch_error(self):
        """Test that length mismatch in element-wise ops raises error."""
        s1 = Series("float")
        s2 = Series("float")
        for i in [1.0, 2.0, 3.0]:
            s1.append(i)
        for i in [1.0, 2.0]:
            s2.append(i)
        
        result = s1 + s2
        with pytest.raises(ValueError, match="Length mismatch"):
            _ = list(result)
    
    def test_arithmetic_auto_update(self):
        """Test that arithmetic operations auto-update when source changes."""
        series = Series("float")
        for i in [1.0, 2.0, 3.0]:
            series.append(i)
        
        result = series * 2
        initial = list(result)
        assert initial == [2.0, 4.0, 6.0]
        
        # Append to source
        series.append(4.0)
        updated = list(result)
        assert updated == [2.0, 4.0, 6.0, 8.0]
    
    def test_chained_arithmetic_operations(self):
        """Test chaining multiple arithmetic operations."""
        series = Series("float")
        for i in [1.0, 2.0, 3.0]:
            series.append(i)
        
        # (series + 10) * 2 - 5
        result = ((series + 10) * 2) - 5
        assert isinstance(result, DerivedSeries)
        assert list(result) == pytest.approx([17.0, 19.0, 21.0])
    
    def test_arithmetic_with_derived_series(self):
        """Test arithmetic operations with DerivedSeries."""
        series = Series("float")
        for i in [1.0, 2.0, 3.0, 4.0, 5.0]:
            series.append(i)
        
        # Create derived series
        avg_series = series.avg(window=3)
        
        # Perform arithmetic on derived series
        result = avg_series + 10
        assert isinstance(result, DerivedSeries)
        # avg[2] = (1+2+3)/3 = 2, so result[2] = 12
        assert result[2] == pytest.approx(12.0)
    
    def test_mixed_types_arithmetic(self):
        """Test arithmetic with int and float."""
        series = Series("int")
        for i in [2, 4, 6]:
            series.append(i)
        
        # Integer series with float scalar
        result = series * 2.5
        values = list(result)
        assert values == pytest.approx([5.0, 10.0, 15.0])
    
    def test_complex_expression(self):
        """Test complex arithmetic expression."""
        series = Series("float")
        for i in [2.0, 4.0, 6.0]:
            series.append(i)
        
        # (series * 3 + 5) / 2
        result = (series * 3 + 5) / 2
        assert list(result) == pytest.approx([5.5, 8.5, 11.5])


# New tests for shift operation
class TestShiftOperation:
    """Test suite for Series shift operation."""

    def test_shift_forward_basic(self):
        """Test basic forward shift (positive periods)."""
        series = Series("float")
        for i in [1.0, 2.0, 3.0, 4.0, 5.0]:
            series.append(i)
        
        shifted = series.shift(1)
        result = list(shifted)
        
        # First value should be NaN, rest shifted
        assert len(result) == 5
        assert pytest.approx(result[1:]) == [1.0, 2.0, 3.0, 4.0]
        # Check NaN separately
        import math
        assert math.isnan(result[0])
    
    def test_shift_backward_basic(self):
        """Test basic backward shift (negative periods)."""
        series = Series("float")
        for i in [1.0, 2.0, 3.0, 4.0, 5.0]:
            series.append(i)
        
        shifted = series.shift(-1)
        result = list(shifted)
        
        # Last value should be NaN, rest shifted
        assert len(result) == 5
        assert pytest.approx(result[:-1]) == [2.0, 3.0, 4.0, 5.0]
        import math
        assert math.isnan(result[-1])
    
    def test_shift_forward_multiple(self):
        """Test forward shift by multiple periods."""
        series = Series("float")
        for i in [1.0, 2.0, 3.0, 4.0, 5.0]:
            series.append(i)
        
        shifted = series.shift(2)
        result = list(shifted)
        
        assert len(result) == 5
        assert pytest.approx(result[2:]) == [1.0, 2.0, 3.0]
        import math
        assert math.isnan(result[0])
        assert math.isnan(result[1])
    
    def test_shift_backward_multiple(self):
        """Test backward shift by multiple periods."""
        series = Series("float")
        for i in [1.0, 2.0, 3.0, 4.0, 5.0]:
            series.append(i)
        
        shifted = series.shift(-2)
        result = list(shifted)
        
        assert len(result) == 5
        assert pytest.approx(result[:3]) == [3.0, 4.0, 5.0]
        import math
        assert math.isnan(result[3])
        assert math.isnan(result[4])
    
    def test_shift_zero(self):
        """Test shift by zero (no shift)."""
        series = Series("float")
        for i in [1.0, 2.0, 3.0]:
            series.append(i)
        
        shifted = series.shift(0)
        assert list(shifted) == pytest.approx([1.0, 2.0, 3.0])
    
    def test_shift_int_series_default_fill(self):
        """Test shift on int series with default fill value."""
        series = Series("int")
        for i in [1, 2, 3, 4, 5]:
            series.append(i)
        
        shifted = series.shift(1)
        result = list(shifted)
        
        # Default fill for int is 0
        assert result == [0, 1, 2, 3, 4]
    
    def test_shift_custom_fill_value(self):
        """Test shift with custom fill value."""
        series = Series("float")
        for i in [1.0, 2.0, 3.0, 4.0, 5.0]:
            series.append(i)
        
        # Forward shift with custom fill
        shifted = series.shift(2, fill_value=-999.0)
        result = list(shifted)
        assert result == pytest.approx([-999.0, -999.0, 1.0, 2.0, 3.0])
        
        # Backward shift with custom fill
        shifted_back = series.shift(-1, fill_value=0.0)
        result_back = list(shifted_back)
        assert result_back == pytest.approx([2.0, 3.0, 4.0, 5.0, 0.0])
    
    def test_shift_empty_series(self):
        """Test shift on empty series."""
        series = Series("float")
        shifted = series.shift(1)
        assert list(shifted) == []
    
    def test_shift_single_element(self):
        """Test shift on series with single element."""
        series = Series("float")
        series.append(42.0)
        
        # Shift forward
        shifted_fwd = series.shift(1)
        result_fwd = list(shifted_fwd)
        import math
        assert len(result_fwd) == 1
        assert math.isnan(result_fwd[0])
        
        # Shift backward
        shifted_back = series.shift(-1)
        result_back = list(shifted_back)
        assert len(result_back) == 1
        assert math.isnan(result_back[0])
    
    def test_shift_large_periods(self):
        """Test shift by periods larger than series length."""
        series = Series("float")
        for i in [1.0, 2.0, 3.0]:
            series.append(i)
        
        # Shift by more than length
        shifted = series.shift(5)
        result = list(shifted)
        
        assert len(result) == 3
        import math
        assert all(math.isnan(val) for val in result)
    
    def test_shift_auto_update(self):
        """Test that shifted series auto-updates when source changes."""
        series = Series("float")
        for i in [1.0, 2.0, 3.0]:
            series.append(i)
        
        shifted = series.shift(1)
        initial = list(shifted)
        assert len(initial) == 3
        
        # Append to source
        series.append(4.0)
        updated = list(shifted)
        assert len(updated) == 4
        assert pytest.approx(updated[1:]) == [1.0, 2.0, 3.0]
    
    def test_shift_with_derived_series(self):
        """Test shift on DerivedSeries."""
        series = Series("float")
        for i in [1.0, 2.0, 3.0, 4.0, 5.0]:
            series.append(i)
        
        # Create derived series first
        avg_series = series.avg(window=3)
        
        # Shift the derived series
        shifted_avg = avg_series.shift(1)
        
        # avg[0] = 1.0, avg[1] = 1.5, avg[2] = 2.0, avg[3] = 3.0, avg[4] = 4.0
        # After shift(1): [NaN, 1.0, 1.5, 2.0, 3.0]
        result = list(shifted_avg)
        assert len(result) == 5
        import math
        assert math.isnan(result[0])
        assert pytest.approx(result[1:]) == [1.0, 1.5, 2.0, 3.0]
    
    def test_shift_chained_operations(self):
        """Test shift combined with arithmetic operations."""
        series = Series("float")
        for i in [2.0, 4.0, 6.0, 8.0, 10.0]:
            series.append(i)
        
        # Compute difference: series - series.shift(1)
        # This is like pandas df.diff()
        shifted = series.shift(1)
        diff = series - shifted
        
        result = list(diff)
        # result[0] = 2.0 - NaN = NaN
        # result[1] = 4.0 - 2.0 = 2.0
        # result[2] = 6.0 - 4.0 = 2.0
        # result[3] = 8.0 - 6.0 = 2.0
        # result[4] = 10.0 - 8.0 = 2.0
        import math
        assert math.isnan(result[0])
        assert result[1:] == pytest.approx([2.0, 2.0, 2.0, 2.0])
    
    def test_shift_timestamp_series(self):
        """Test shift on timestamp series."""
        series = Series("timestamp")
        import time
        base_time = time.time()
        for i in range(5):
            series.append(base_time + i)
        
        shifted = series.shift(1)
        result = list(shifted)
        
        assert len(result) == 5
        import math
        assert math.isnan(result[0])
        assert pytest.approx(result[1:]) == [base_time, base_time + 1, base_time + 2, base_time + 3]
    
    def test_shift_preserves_length(self):
        """Test that shift always preserves series length."""
        series = Series("float")
        for i in range(10):
            series.append(float(i))
        
        # Various shift amounts
        for periods in [-5, -2, -1, 0, 1, 2, 5]:
            shifted = series.shift(periods)
            assert len(shifted) == len(series), f"Length mismatch for shift({periods})"
    
    def test_shift_type_consistency(self):
        """Test that shift returns DerivedSeries."""
        series = Series("float")
        for i in [1.0, 2.0, 3.0]:
            series.append(i)
        
        shifted = series.shift(1)
        assert isinstance(shifted, DerivedSeries)
        assert shifted.dtype == "float"


# Tests for slicing with complex operations
class TestSlicingWithOperations:
    """Test suite for slicing combined with various operations."""

    def test_slicing_with_cumsum(self):
        """Test slicing followed by cumsum."""
        series = Series("int")
        for i in range(10):
            series.append(i)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        # Slice with step 2: [0, 2, 4, 6, 8]
        sliced = series[::2]
        assert isinstance(sliced, Series)
        assert list(sliced) == [0, 2, 4, 6, 8]
        
        # Cumsum of sliced: [0, 2, 6, 12, 20]
        cumsum_result = sliced.cumsum()
        assert isinstance(cumsum_result, DerivedSeries)
        assert list(cumsum_result) == [0, 2, 6, 12, 20]
    
    def test_slicing_cumsum_shift_difference(self):
        """Test the complex operation: s[::2].cumsum() - s[::2].cumsum().shift(1)."""
        series = Series("float")
        for i in range(10):
            series.append(float(i))  # [0.0, 1.0, 2.0, ..., 9.0]
        
        # Slice with step 2: [0, 2, 4, 6, 8]
        sliced = series[::2]
        assert list(sliced) == pytest.approx([0.0, 2.0, 4.0, 6.0, 8.0])
        
        # Cumsum: [0, 2, 6, 12, 20]
        cumsum_result = sliced.cumsum()
        cumsum_list = list(cumsum_result)
        assert cumsum_list == pytest.approx([0.0, 2.0, 6.0, 12.0, 20.0])
        
        # Shift cumsum by 1: [NaN, 0, 2, 6, 12]
        shifted = cumsum_result.shift(1)
        shifted_list = list(shifted)
        assert len(shifted_list) == 5
        import math
        assert math.isnan(shifted_list[0])
        assert shifted_list[1:] == pytest.approx([0.0, 2.0, 6.0, 12.0])
        
        # Difference: cumsum - shifted = [0-NaN, 2-0, 6-2, 12-6, 20-12]
        # Expected: [NaN, 2, 4, 6, 8]
        diff = cumsum_result - shifted
        diff_list = list(diff)
        assert len(diff_list) == 5
        assert math.isnan(diff_list[0])
        assert diff_list[1:] == pytest.approx([2.0, 4.0, 6.0, 8.0])
        
        # This should give us back the original sliced values (except first is NaN)
        # Because cumsum[i] - cumsum[i-1] = original[i]
        original_sliced = list(sliced)
        assert diff_list[1:] == pytest.approx(original_sliced[1:])
    
    def test_slicing_with_arithmetic(self):
        """Test slicing followed by arithmetic operations."""
        series = Series("float")
        for i in range(10):
            series.append(float(i))
        
        # Slice: [0, 2, 4, 6, 8]
        sliced = series[::2]
        
        # Arithmetic on sliced
        doubled = sliced * 2
        assert list(doubled) == pytest.approx([0.0, 4.0, 8.0, 12.0, 16.0])
        
        added = sliced + 10
        assert list(added) == pytest.approx([10.0, 12.0, 14.0, 16.0, 18.0])
    
    def test_slicing_with_avg(self):
        """Test slicing followed by avg."""
        series = Series("float")
        for i in range(10):
            series.append(float(i))
        
        # Slice: [0, 2, 4, 6, 8]
        sliced = series[::2]
        
        # Avg with window
        avg_result = sliced.avg(window=3)
        avg_list = list(avg_result)
        
        # avg[0] = 0/1 = 0
        # avg[1] = (0+2)/2 = 1
        # avg[2] = (0+2+4)/3 = 2
        # avg[3] = (2+4+6)/3 = 4
        # avg[4] = (4+6+8)/3 = 6
        assert avg_list == pytest.approx([0.0, 1.0, 2.0, 4.0, 6.0])
    
    def test_slicing_negative_indices(self):
        """Test slicing with negative indices and operations."""
        series = Series("int")
        for i in range(10):
            series.append(i)
        
        # Last 5 elements: [5, 6, 7, 8, 9]
        sliced = series[-5:]
        assert list(sliced) == [5, 6, 7, 8, 9]
        
        # Sum of last 5
        sum_result = sliced.sum()
        # Expanding sum: [5, 11, 18, 26, 35]
        assert list(sum_result) == [5, 11, 18, 26, 35]
    
    def test_slicing_range_with_operations(self):
        """Test range slicing with operations."""
        series = Series("float")
        for i in range(10):
            series.append(float(i))
        
        # Middle range: [2, 3, 4, 5, 6]
        sliced = series[2:7]
        assert list(sliced) == pytest.approx([2.0, 3.0, 4.0, 5.0, 6.0])
        
        # Shift and difference
        shifted = sliced.shift(1)
        diff = sliced - shifted
        
        diff_list = list(diff)
        import math
        assert math.isnan(diff_list[0])
        # All differences should be 1.0
        assert diff_list[1:] == pytest.approx([1.0, 1.0, 1.0, 1.0])
    
    def test_double_slicing(self):
        """Test slicing of already sliced series."""
        series = Series("int")
        for i in range(20):
            series.append(i)
        
        # First slice: every 2nd element [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        sliced1 = series[::2]
        assert len(sliced1) == 10
        
        # Second slice: first 5 elements [0, 2, 4, 6, 8]
        sliced2 = sliced1[:5]
        assert list(sliced2) == [0, 2, 4, 6, 8]
        
        # Operations on double-sliced
        cumsum = sliced2.cumsum()
        assert list(cumsum) == [0, 2, 6, 12, 20]
    
    def test_slicing_preserves_dtype(self):
        """Test that slicing preserves dtype."""
        series_int = Series("int")
        series_float = Series("float")
        series_timestamp = Series("timestamp")
        
        for i in range(10):
            series_int.append(i)
            series_float.append(float(i))
            series_timestamp.append(float(i))
        
        assert series_int[::2].dtype == "int"
        assert series_float[::2].dtype == "float"
        assert series_timestamp[::2].dtype == "timestamp"
    
    def test_complex_chained_operations_on_slice(self):
        """Test complex chain: slice -> cumsum -> shift -> arithmetic -> sum."""
        series = Series("float")
        for i in range(1, 11):
            series.append(float(i))  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        # Slice: [1, 3, 5, 7, 9]
        sliced = series[::2]
        
        # Cumsum: [1, 4, 9, 16, 25]
        cumsum = sliced.cumsum()
        
        # Shift: [NaN, 1, 4, 9, 16]
        shifted = cumsum.shift(1)
        
        # Difference: [NaN, 3, 5, 7, 9] (this is sliced values except first)
        diff = cumsum - shifted
        
        # Add 1: [NaN, 4, 6, 8, 10]
        added = diff + 1
        
        # Sum with window 2: [NaN, NaN+4, 4+6, 6+8, 8+10]
        # Note: NaN in arithmetic produces NaN
        sum_result = added.sum(window=2)
        result_list = list(sum_result)
        
        import math
        # First element involves NaN
        assert math.isnan(result_list[0])
        # Second element: NaN + 4 = NaN
        assert math.isnan(result_list[1])
        # Rest should be valid
        assert result_list[2:] == pytest.approx([10.0, 14.0, 18.0])
    
    def test_slicing_auto_updates_with_source(self):
        """Test that operations on sliced series update when original updates."""
        series = Series("float")
        for i in range(5):
            series.append(float(i))  # [0, 1, 2, 3, 4]
        
        # Slice: [0, 2, 4]
        sliced = series[::2]
        
        # Note: Slicing creates a new Series, not a DerivedSeries
        # So it won't auto-update. Let's verify this behavior
        initial = list(sliced)
        assert initial == pytest.approx([0.0, 2.0, 4.0])
        
        # Add to original
        series.append(5.0)
        series.append(6.0)
        
        # Sliced is a snapshot, won't change
        assert list(sliced) == pytest.approx([0.0, 2.0, 4.0])
        
        # Need to re-slice to get updated values
        new_sliced = series[::2]
        assert list(new_sliced) == pytest.approx([0.0, 2.0, 4.0, 6.0])
    
    def test_derived_operations_on_slice_auto_update(self):
        """Test that DerivedSeries operations on slice track the slice correctly."""
        series = Series("float")
        for i in range(5):
            series.append(float(i))
        
        sliced = series[::2]  # [0, 2, 4]
        
        # Create derived series from slice
        cumsum = sliced.cumsum()
        assert list(cumsum) == pytest.approx([0.0, 2.0, 6.0])
        
        # Append to sliced series (not original)
        sliced.append(6.0)
        
        # Cumsum should update
        assert list(cumsum) == pytest.approx([0.0, 2.0, 6.0, 12.0])
    
    def test_slice_step_negative(self):
        """Test slicing with negative step (reverse)."""
        series = Series("int")
        for i in range(5):
            series.append(i)  # [0, 1, 2, 3, 4]
        
        # Reverse: [4, 3, 2, 1, 0]
        reversed_slice = series[::-1]
        assert list(reversed_slice) == [4, 3, 2, 1, 0]
        
        # Operations on reversed
        cumsum = reversed_slice.cumsum()
        assert list(cumsum) == [4, 7, 9, 10, 10]


# Tests for diff operation
class TestDiffOperation:
    """Test suite for Series diff operation."""

    def test_diff_basic(self):
        """Test basic diff operation."""
        series = Series("float")
        for i in [2.0, 5.0, 9.0, 14.0, 20.0]:
            series.append(i)
        
        diff = series.diff()
        result = list(diff)
        
        import math
        assert math.isnan(result[0])
        assert result[1:] == pytest.approx([3.0, 4.0, 5.0, 6.0])
    
    def test_diff_periods_2(self):
        """Test diff with periods=2."""
        series = Series("float")
        for i in [1.0, 3.0, 6.0, 10.0, 15.0]:
            series.append(i)
        
        diff = series.diff(periods=2)
        result = list(diff)
        
        import math
        assert math.isnan(result[0])
        assert math.isnan(result[1])
        # 6-1=5, 10-3=7, 15-6=9
        assert result[2:] == pytest.approx([5.0, 7.0, 9.0])
    
    def test_diff_equivalent_to_shift_subtract(self):
        """Test that diff(n) equals self - self.shift(n)."""
        series = Series("float")
        for i in range(10):
            series.append(float(i))
        
        # Using diff
        diff_result = series.diff(1)
        
        # Using shift
        manual_result = series - series.shift(1)
        
        # Should be equivalent
        assert list(diff_result) == pytest.approx(list(manual_result))
    
    def test_diff_on_derived_series(self):
        """Test diff on DerivedSeries."""
        series = Series("float")
        for i in range(10):
            series.append(float(i))
        
        # Create derived series
        avg = series.avg(window=3)
        
        # Diff of derived
        diff = avg.diff()
        
        # Should work correctly
        result = list(diff)
        assert len(result) == 10
    
    def test_diff_use_case_rate_of_change(self):
        """Test diff for calculating rate of change."""
        series = Series("float")
        values = [100.0, 103.0, 107.0, 112.0, 118.0]
        for v in values:
            series.append(v)
        
        # Calculate change per step
        change = series.diff()
        result = list(change)
        
        import math
        assert math.isnan(result[0])
        assert result[1:] == pytest.approx([3.0, 4.0, 5.0, 6.0])
    
    def test_diff_auto_update(self):
        """Test that diff auto-updates when source changes."""
        series = Series("float")
        for i in [1.0, 2.0, 3.0]:
            series.append(i)
        
        diff = series.diff()
        initial = list(diff)
        
        # Append to source
        series.append(5.0)
        
        # Should update
        updated = list(diff)
        assert len(updated) == 4
        import math
        assert math.isnan(updated[0])
        # 2-1=1, 3-2=1, 5-3=2
        assert updated[1:] == pytest.approx([1.0, 1.0, 2.0])
    
    def test_diff_int_series(self):
        """Test diff on int series."""
        series = Series("int")
        for i in [10, 15, 22, 31, 42]:
            series.append(i)
        
        diff = series.diff()
        result = list(diff)
        
        # First element: 10 - 0 (default fill for int) = 10
        assert result[0] == 10
        # Rest: differences
        assert result[1:] == [5, 7, 9, 11]
    
    def test_diff_returns_derived_series(self):
        """Test that diff returns DerivedSeries."""
        series = Series("float")
        for i in [1.0, 2.0, 3.0]:
            series.append(i)
        
        diff = series.diff()
        assert isinstance(diff, DerivedSeries)
    
    def test_diff_chained_with_operations(self):
        """Test chaining diff with other operations."""
        series = Series("float")
        for i in range(10):
            series.append(float(i * i))  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
        
        # Diff gives us the differences
        diff = series.diff()  # [NaN, 1, 3, 5, 7, 9, 11, 13, 15, 17]
        
        # Average of differences
        avg_diff = diff.avg(window=3)
        
        # Should work
        result = list(avg_diff)
        assert len(result) == 10
