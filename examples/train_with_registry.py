"""
Example training script demonstrating model registry usage.

This shows the recommended pattern for using get_model_class() in training scripts
instead of hardcoding model imports.

Usage:
    python train_with_registry.py --model_class Qwen3Model
"""

import argparse
from muse.models import get_model_class, list_models
from muse.config import Qwen3Config


def get_argument_parser():
    """Create argument parser with model_class argument."""
    parser = argparse.ArgumentParser(
        description="Training script with model registry support"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_class", 
        type=str, 
        default="Qwen3Model",
        help=f"Model class name to use for training. "
             f"Available models: {', '.join(list_models())}"
    )
    
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Directory containing pretrained model weights"
    )
    
    # Config arguments (simplified for example)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--embed_dim", type=int, default=2048)
    parser.add_argument("--num_layers", type=int, default=24)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    
    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    
    return parser


def create_model_config(args):
    """Create model config based on model class and args."""
    # For this example, we use Qwen3Config
    # In a real scenario, you might have different configs for different models
    config = Qwen3Config(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
    )
    return config


def train(args):
    """Main training function."""
    print("=" * 70)
    print(f"Training with Model Registry")
    print("=" * 70)
    
    # 1. List available models
    print(f"\nAvailable models: {list_models()}")
    
    # 2. Load model class dynamically using registry
    print(f"\nLoading model class: {args.model_class}")
    try:
        model_cls = get_model_class(args.model_class)
        print(f"✓ Successfully loaded: {model_cls.__name__}")
    except KeyError as e:
        print(f"✗ Error: {e}")
        print(f"  Please choose from: {list_models()}")
        return
    
    # 3. Create model config
    print(f"\nCreating model configuration...")
    config = create_model_config(args)
    print(f"✓ Config created: {config}")
    
    # 4. Instantiate model
    print(f"\nInstantiating model...")
    model = model_cls(config)
    print(f"✓ Model instantiated: {type(model).__name__}")
    
    # 5. Load pretrained weights (if provided)
    if args.model_dir:
        print(f"\nLoading pretrained weights from: {args.model_dir}")
        # model.from_pretrained(args.model_dir)
        print("✓ Weights loaded (commented out for demo)")
    
    # 6. Setup training (simplified)
    print(f"\n{'='*70}")
    print("Training Configuration:")
    print(f"  Model: {args.model_class}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"{'='*70}")
    
    # In a real training script, you would:
    # - Setup optimizer: optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # - Setup dataloader: dataloader = get_dataloader(...)
    # - Training loop: for epoch in range(args.num_epochs): ...
    
    print("\n✓ Training script setup complete!")
    print("  (Actual training loop would go here)")


def main():
    """Entry point."""
    parser = get_argument_parser()
    args = parser.parse_args()
    
    train(args)


if __name__ == "__main__":
    main()

