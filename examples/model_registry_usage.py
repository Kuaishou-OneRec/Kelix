"""
Example demonstrating how to use the model registry to load models by name.

This shows how training scripts can use get_model_class() instead of 
hardcoding model imports.
"""

from muse.models import get_model_class, list_models, MODEL_REGISTRY
from muse.config import Qwen3Config


def example_basic_usage():
    """Basic usage: Load a model by name."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # List all available models
    print("\nAvailable models:", list_models())
    
    # Get model class by name
    model_class = get_model_class("Qwen3Model")
    print(f"\nLoaded model class: {model_class}")
    print(f"Model class name: {model_class.__name__}")
    
    # Create model instance
    config = Qwen3Config(
        vocab_size=1000,
        embed_dim=512,
        num_layers=4,
        num_heads=8,
        max_seq_len=512
    )
    model = model_class(config)
    print(f"\nCreated model instance: {type(model).__name__}")
    print(f"Model config: {model.config}")


def example_training_script_pattern():
    """Example pattern for training scripts with argparse."""
    print("\n" + "=" * 60)
    print("Example 2: Training Script Pattern")
    print("=" * 60)
    
    # Simulating argparse args
    class Args:
        model_class = "Qwen3Model"
        # ... other training args
    
    args = Args()
    
    # In your training script, you would do:
    print(f"\nLoading model: {args.model_class}")
    model_cls = get_model_class(args.model_class)
    
    # Create config and model
    config = Qwen3Config(
        vocab_size=1000,
        embed_dim=512,
        num_layers=4,
        num_heads=8,
        max_seq_len=512
    )
    model = model_cls(config)
    print(f"Successfully created {type(model).__name__}")
    

def example_error_handling():
    """Example of error handling for non-existent models."""
    print("\n" + "=" * 60)
    print("Example 3: Error Handling")
    print("=" * 60)
    
    try:
        model_cls = get_model_class("NonExistentModel")
    except KeyError as e:
        print(f"\nExpected error caught: {e}")


def example_direct_registry_access():
    """Example of directly accessing the registry."""
    print("\n" + "=" * 60)
    print("Example 4: Direct Registry Access")
    print("=" * 60)
    
    print("\nDirect registry access:")
    for name, cls in MODEL_REGISTRY.items():
        print(f"  - {name}: {cls}")


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# Model Registry Usage Examples")
    print("#" * 60 + "\n")
    
    example_basic_usage()
    example_training_script_pattern()
    example_error_handling()
    example_direct_registry_access()
    
    print("\n" + "#" * 60)
    print("# All examples completed successfully!")
    print("#" * 60 + "\n")

