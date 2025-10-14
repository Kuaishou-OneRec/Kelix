from typing import Union, List, Optional
from .base import DistributedDataset
from .recoverable import RecoverableDistributedDataset


def create_dataset(sources: Union[str, List[str]],
                  rank: int = 0,
                  world_size: int = 1,
                  num_workers: int = 8,
                  seed: int = 1024,
                  num_epochs: int = 1,
                  shard_by: str = "auto",
                  shuffle_buffer_size: int = 0,
                  enable_checkpointing: bool = False,
                  checkpoint_dir: str = None,
                  checkpoint_interval: int = 1000,
                  **kwargs) -> Union[DistributedDataset, RecoverableDistributedDataset]:
    """
    Factory function to create the appropriate dataset based on requirements.
    
    Args:
        sources: Data source paths or patterns
        rank: Rank of current process  
        world_size: Total number of processes
        num_workers: Number of data loader workers
        seed: Random seed
        num_epochs: Number of epochs
        shard_by: Sharding strategy ("auto", "files", "samples")
        shuffle_buffer_size: Size of shuffle buffer (0 to disable)
        enable_checkpointing: Whether to enable checkpoint recovery
        checkpoint_dir: Directory for checkpoint files
        checkpoint_interval: Interval for saving checkpoints
        **kwargs: Additional arguments passed to DistributedDataset
        
    Returns:
        DistributedDataset if no advanced features needed,
        RecoverableDistributedDataset if buffering or checkpointing enabled
    """
    
    # Create base dataset
    base_dataset = DistributedDataset(
        sources=sources,
        rank=rank,
        world_size=world_size,
        num_workers=num_workers,
        seed=seed,
        num_epochs=num_epochs,
        shard_by=shard_by,
        **kwargs
    )
    
    # Wrap with RecoverableDistributedDataset if advanced features needed
    if shuffle_buffer_size > 0 or enable_checkpointing:
        return RecoverableDistributedDataset(
            dataset=base_dataset,
            shuffle_buffer_size=shuffle_buffer_size,
            enable_checkpointing=enable_checkpointing,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval
        )
    else:
        # Return base dataset for maximum performance when no advanced features needed
        return base_dataset


# Backward compatibility alias
def get_recoverable_dataset(*args, **kwargs):
    """Alias for create_dataset for backward compatibility"""
    return create_dataset(*args, **kwargs)
