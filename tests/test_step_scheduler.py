"""Tests for StepScheduler class."""
import pytest
from argparse import Namespace


# Mock args object for testing
def create_mock_args(gradient_accumulation_steps=4, 
                     logging_per_step=100, 
                     save_checkpoint_per_step=1000):
    """Create a mock args object for testing."""
    return Namespace(
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_per_step=logging_per_step,
        save_checkpoint_per_step=save_checkpoint_per_step
    )


class TestStepScheduler:
    """Test suite for StepScheduler class."""

    def test_initialization(self):
        """Test StepScheduler initialization."""
        from muse.training.common import StepScheduler
        
        args = create_mock_args()
        scheduler = StepScheduler(args)
        
        assert scheduler.gradient_accumulation_steps == 4
        assert scheduler.logging_per_step == 100
        assert scheduler.save_checkpoint_per_step == 1000
        assert scheduler.micro_step == 0
        assert scheduler.global_step == 0

    def test_step_increment(self):
        """Test that step() increments micro_step."""
        from muse.training.common import StepScheduler
        
        args = create_mock_args(gradient_accumulation_steps=1)
        scheduler = StepScheduler(args)
        
        assert scheduler.micro_step == 0
        scheduler.step()
        assert scheduler.micro_step == 1
        scheduler.step()
        assert scheduler.micro_step == 2

    def test_global_step_increment_at_boundary(self):
        """Test that global_step increments at gradient accumulation boundaries."""
        from muse.training.common import StepScheduler
        
        args = create_mock_args(gradient_accumulation_steps=4)
        scheduler = StepScheduler(args)
        
        assert scheduler.global_step == 0
        
        # First 3 steps: no global step increment
        for i in range(3):
            scheduler.step()
            assert scheduler.global_step == 0
        
        # 4th step: should increment global_step
        scheduler.step()
        assert scheduler.micro_step == 4
        assert scheduler.global_step == 1
        
        # Next 4 steps
        for i in range(4):
            scheduler.step()
        assert scheduler.micro_step == 8
        assert scheduler.global_step == 2

    def test_is_gradient_accumulation_boundary(self):
        """Test gradient accumulation boundary detection."""
        from muse.training.common import StepScheduler
        
        args = create_mock_args(gradient_accumulation_steps=4)
        scheduler = StepScheduler(args)
        
        # Not at boundary initially
        assert not scheduler.is_gradient_accumulation_boundary()
        
        # Steps 1-3: not at boundary
        for i in range(3):
            scheduler.step()
            assert not scheduler.is_gradient_accumulation_boundary()
        
        # Step 4: at boundary
        scheduler.step()
        assert scheduler.is_gradient_accumulation_boundary()
        
        # Steps 5-7: not at boundary
        for i in range(3):
            scheduler.step()
            assert not scheduler.is_gradient_accumulation_boundary()
        
        # Step 8: at boundary
        scheduler.step()
        assert scheduler.is_gradient_accumulation_boundary()

    def test_should_logging_basic(self):
        """Test should_logging() basic behavior."""
        from muse.training.common import StepScheduler
        
        args = create_mock_args(
            gradient_accumulation_steps=4, 
            logging_per_step=2
        )
        scheduler = StepScheduler(args)
        
        # Should not log before reaching boundary
        for i in range(3):
            scheduler.step()
            assert not scheduler.should_logging()
        
        # At step 4 (global_step 1): not divisible by logging_per_step (2)
        scheduler.step()
        assert not scheduler.should_logging()
        
        # At step 8 (global_step 2): should log (2 % 2 == 0)
        for i in range(4):
            scheduler.step()
        assert scheduler.should_logging()

    def test_should_logging_frequency(self):
        """Test should_logging() with specific frequency."""
        from muse.training.common import StepScheduler
        
        args = create_mock_args(
            gradient_accumulation_steps=1,
            logging_per_step=100
        )
        scheduler = StepScheduler(args)
        
        log_steps = []
        for i in range(301):
            scheduler.step()
            if scheduler.should_logging():
                log_steps.append(scheduler.global_step)
        
        # Should log at steps 100, 200, 300
        assert log_steps == [100, 200, 300]

    def test_should_save_checkpoint_basic(self):
        """Test should_save_checkpoint() basic behavior."""
        from muse.training.common import StepScheduler
        
        args = create_mock_args(
            gradient_accumulation_steps=4,
            save_checkpoint_per_step=2
        )
        scheduler = StepScheduler(args)
        
        # Should not save before reaching boundary
        for i in range(3):
            scheduler.step()
            assert not scheduler.should_save_checkpoint()
        
        # At step 4 (global_step 1): not divisible by save_checkpoint_per_step (2)
        scheduler.step()
        assert not scheduler.should_save_checkpoint()
        
        # At step 8 (global_step 2): should save (2 % 2 == 0)
        for i in range(4):
            scheduler.step()
        assert scheduler.should_save_checkpoint()

    def test_should_save_checkpoint_frequency(self):
        """Test should_save_checkpoint() with specific frequency."""
        from muse.training.common import StepScheduler
        
        args = create_mock_args(
            gradient_accumulation_steps=1,
            save_checkpoint_per_step=1000
        )
        scheduler = StepScheduler(args)
        
        checkpoint_steps = []
        for i in range(3001):
            scheduler.step()
            if scheduler.should_save_checkpoint():
                checkpoint_steps.append(scheduler.global_step)
        
        # Should save at steps 1000, 2000, 3000
        assert checkpoint_steps == [1000, 2000, 3000]

    def test_combined_workflow(self):
        """Test combined workflow simulating real training loop."""
        from muse.training.common import StepScheduler
        
        args = create_mock_args(
            gradient_accumulation_steps=4,
            logging_per_step=5,
            save_checkpoint_per_step=10
        )
        scheduler = StepScheduler(args)
        
        optimizer_updates = 0
        logs_written = 0
        checkpoints_saved = 0
        
        # Simulate 50 micro-steps
        for i in range(50):
            scheduler.step()
            
            # Update optimizer at boundaries
            if scheduler.is_gradient_accumulation_boundary():
                optimizer_updates += 1
            
            # Log at specified intervals
            if scheduler.should_logging():
                logs_written += 1
            
            # Save checkpoint at specified intervals
            if scheduler.should_save_checkpoint():
                checkpoints_saved += 1
        
        # 50 micro-steps / 4 acc_steps = 12 optimizer updates (global_steps)
        assert optimizer_updates == 12
        assert scheduler.global_step == 12
        
        # Logging every 5 global steps: 5, 10
        assert logs_written == 2
        
        # Checkpointing every 10 global steps: 10
        assert checkpoints_saved == 1

    def test_zero_at_start(self):
        """Test that should_logging and should_save_checkpoint are False at step 0."""
        from muse.training.common import StepScheduler
        
        args = create_mock_args(
            gradient_accumulation_steps=1,
            logging_per_step=1,
            save_checkpoint_per_step=1
        )
        scheduler = StepScheduler(args)
        
        # At initialization (before any step)
        assert not scheduler.should_logging()
        assert not scheduler.should_save_checkpoint()

    def test_properties_are_readonly(self):
        """Test that micro_step and global_step properties are read-only."""
        from muse.training.common import StepScheduler
        
        args = create_mock_args()
        scheduler = StepScheduler(args)
        
        # These should raise AttributeError
        with pytest.raises(AttributeError):
            scheduler.micro_step = 100
        
        with pytest.raises(AttributeError):
            scheduler.global_step = 50

    def test_different_accumulation_steps(self):
        """Test with different gradient accumulation steps."""
        from muse.training.common import StepScheduler
        
        for acc_steps in [1, 2, 4, 8, 16]:
            args = create_mock_args(gradient_accumulation_steps=acc_steps)
            scheduler = StepScheduler(args)
            
            # Run for multiple cycles
            for i in range(acc_steps * 3):
                scheduler.step()
            
            # Should have 3 global steps
            assert scheduler.global_step == 3
            assert scheduler.micro_step == acc_steps * 3
