#!/usr/bin/env python
"""Example script demonstrating the usage of the Muse configuration module.

This script shows various ways to create, use, and manage configurations.
"""

import argparse
import json
from pathlib import Path

from muse.config import (
    Qwen3Config,
    DatasetConfig,
    TrainingConfig,
    OptimizerConfig,
    SchedulerConfig,
    CheckpointConfig,
    LoggingConfig,
    create_config_from_args_and_file,
    save_all_configs_to_checkpoint,
    load_configs_from_checkpoint,
    load_config_from_file,
)


def example_1_create_from_code():
    """Example 1: Create configurations directly in code."""
    print("=" * 60)
    print("Example 1: Create configurations from code")
    print("=" * 60)
    
    # Create model config
    model_config = Qwen3Config(
        model_class="Qwen3Model",
        model_dir="/path/to/qwen3",
        vocab_size=151936,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        use_flash_attention_2=True,
        attention_function="flash",
    )
    
    # Create dataset config
    dataset_config = DatasetConfig(
        dataset="my_dataset",
        data_format="chatml",
        max_length=2048,
        batch_size=4,
    )
    
    # Create training config
    training_config = TrainingConfig(
        optimizer=OptimizerConfig(
            learning_rate=2e-4,
            weight_decay=0.1,
        ),
        scheduler=SchedulerConfig(
            lr_scheduler_type="cosine_with_min_lr",
            num_training_steps=10000,
            num_warmup_steps=100,
        ),
        checkpoint=CheckpointConfig(
            output_dir="/path/to/output",
            save_checkpoint_per_step=1000,
        ),
        logging=LoggingConfig(
            comment="Example training run",
        ),
        seed=42,
    )
    
    print("Model config created:")
    print(f"  - Model class: {model_config.model_class}")
    print(f"  - Hidden size: {model_config.hidden_size}")
    print(f"  - Attention function: {model_config.attention_function}")
    
    print("\nDataset config created:")
    print(f"  - Data format: {dataset_config.data_format}")
    print(f"  - Max length: {dataset_config.max_length}")
    print(f"  - Batch size: {dataset_config.batch_size}")
    
    print("\nTraining config created:")
    print(f"  - Learning rate: {training_config.optimizer.learning_rate}")
    print(f"  - Training steps: {training_config.scheduler.num_training_steps}")
    print(f"  - Seed: {training_config.seed}")
    print()


def example_2_load_from_file():
    """Example 2: Load configurations from a JSON file."""
    print("=" * 60)
    print("Example 2: Load configurations from file")
    print("=" * 60)
    
    # Check if example config exists
    config_path = Path(__file__).parent / "config_example.json"
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        print("Skipping this example.\n")
        return
    
    # Load config dict from file
    config_dict = load_config_from_file(str(config_path))
    
    # Create config objects
    dataset_config = DatasetConfig(**config_dict['dataset'])
    
    print(f"Loaded config from: {config_path}")
    print(f"Dataset config:")
    print(f"  - Data format: {dataset_config.data_format}")
    print(f"  - Max visual tokens: {dataset_config.max_visual_tokens}")
    print(f"  - Batch size: {dataset_config.batch_size}")
    print()


def example_3_serialization():
    """Example 3: Serialize and save configurations."""
    print("=" * 60)
    print("Example 3: Serialize and save configurations")
    print("=" * 60)
    
    # Create a simple config
    model_config = Qwen3Config(
        model_class="Qwen3Model",
        model_dir="/path/to/model",
        hidden_size=2048,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=128,
    )
    
    # Convert to dict
    config_dict = model_config.to_dict()
    print("Config as dictionary:")
    print(json.dumps(config_dict, indent=2)[:200] + "...")
    
    # Convert to JSON string
    json_str = model_config.to_json()
    print("\nConfig as JSON string:")
    print(json_str[:200] + "...")
    
    # Save to file
    output_path = Path("/tmp/example_model_config.json")
    model_config.save(str(output_path))
    print(f"\nConfig saved to: {output_path}")
    
    # Load back
    loaded_config = Qwen3Config.load(str(output_path))
    print(f"Config loaded back: {loaded_config.model_class}, hidden_size={loaded_config.hidden_size}")
    print()


def example_4_validation():
    """Example 4: Configuration validation."""
    print("=" * 60)
    print("Example 4: Configuration validation")
    print("=" * 60)
    
    # Valid config
    try:
        config = Qwen3Config(
            model_class="Qwen3Model",
            model_dir="/path/to/model",
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=32,
            head_dim=128,  # 32 * 128 = 4096 ✓
        )
        print("✓ Valid config created successfully")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Invalid config - head_dim mismatch
    try:
        config = Qwen3Config(
            model_class="Qwen3Model",
            model_dir="/path/to/model",
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=32,
            head_dim=64,  # 32 * 64 = 2048 ≠ 4096 ✗
        )
        print("✓ Config created (unexpected)")
    except Exception as e:
        print(f"✓ Validation caught error: {str(e)[:80]}...")
    
    # Invalid config - num_heads not divisible
    try:
        config = Qwen3Config(
            model_class="Qwen3Model",
            model_dir="/path/to/model",
            hidden_size=4096,
            num_attention_heads=33,  # Not divisible by 32 ✗
            num_key_value_heads=32,
            head_dim=128,
        )
        print("✓ Config created (unexpected)")
    except Exception as e:
        print(f"✓ Validation caught error: {str(e)[:80]}...")
    
    print()


def example_5_merging():
    """Example 5: Merge configurations."""
    print("=" * 60)
    print("Example 5: Merge configurations")
    print("=" * 60)
    
    # Base config
    base_config = OptimizerConfig(
        learning_rate=1e-4,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
    )
    print(f"Base config: LR={base_config.learning_rate}, WD={base_config.weight_decay}")
    
    # Merge with override
    override = {"learning_rate": 2e-4, "beta2": 0.999}
    merged_config = base_config.merge(override)
    print(f"Merged config: LR={merged_config.learning_rate}, WD={merged_config.weight_decay}, beta2={merged_config.beta2}")
    print()


def example_6_argparse_integration():
    """Example 6: Integration with argparse."""
    print("=" * 60)
    print("Example 6: Argparse integration (simulated)")
    print("=" * 60)
    
    # Simulate argparse Namespace
    class Args:
        config_file = None
        model_class = "Qwen3Model"
        model_dir = "/path/to/model"
        learning_rate = 1e-4
        output_dir = "/path/to/output"
        num_training_steps = 5000
        batch_size = 8
        seed = 123
        
        # Add other required fields with defaults
        use_flash_attention_2 = True
        enable_gradient_checkpointing = False
        compile = False
        fp32_weight = False
        fp32_reduce = False
        reshard_after_forward = False
        prefetch_parameters = False
        model_processor = "Qwen2_5_VLProcessor_moonvit"
        allow_random_init_params = ""
        
        dataset_config = None
        dataset = "my_dataset"
        data_format = "chatml"
        min_visual_tokens = 16
        max_visual_tokens = 512
        max_length = 2048
        
        vision_learning_rate = -1.0
        vision_lr_layer_decay = 1.0
        weight_decay = 0.1
        beta1 = 0.9
        beta2 = 0.95
        
        lr_scheduler_type = "cosine_with_min_lr"
        num_warmup_steps = 100
        num_decay_steps = 5000
        num_epochs = 1
        min_lr = 1e-6
        
        save_checkpoint_per_step = 1000
        save_checkpoint_every_epoch = False
        resume_from = None
        resume_from_tag = None
        resume_dataloader = False
        load_weights_only = False
        auto_resume_local_latest = False
        merge_checkpoint = False
        merge_checkpoint_dtype = "fp16"
        merge_checkpoint_output_file = "pytorch_model.bin"
        
        logging_per_step = 100
        monitor_datasource_loss = False
        monitor_datasource_cnt = False
        monitor_image_tokens = False
        comment = "Argparse example"
        commit_id = "abc123"
        heartbeat_monitor = False
        enable_profile = False
        
        gradient_accumulation_steps = 4
        clip_range = 1.0
        freeze_llm = False
        freeze_visual = False
        freeze_projector = False
        update_vision_tower = False
        sequence_parallel_size = 1
        kml_id = None
        kml_task_id = None
    
    args = Args()
    
    # Create configs from args
    model_config, dataset_config, training_config = create_config_from_args_and_file(
        args=args,
        config_file=args.config_file,
        model_config_class=Qwen3Config,
    )
    
    print(f"Created configs from argparse:")
    print(f"  - Model: {model_config.model_class}")
    print(f"  - Learning rate: {training_config.optimizer.learning_rate}")
    print(f"  - Output dir: {training_config.checkpoint.output_dir}")
    print(f"  - Batch size: {dataset_config.batch_size}")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Muse Configuration Module - Usage Examples")
    print("=" * 60 + "\n")
    
    example_1_create_from_code()
    example_2_load_from_file()
    example_3_serialization()
    example_4_validation()
    example_5_merging()
    example_6_argparse_integration()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

