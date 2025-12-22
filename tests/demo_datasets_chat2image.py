"""
Demo script for Chat2ImageDataset - demonstrates dataset outputs with and without collator.
"""
import json
import torch
import tempfile
import os
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import Dict, Any, List
import torch.distributed as dist

from muse.data.datasets.image import Chat2ImageDataset, Token2ImageDataset

Chat2ImageDataset = Token2ImageDataset
# Real processor path for testing
PROCESSOR_PATH = "/llm_reco_ssd/zhouyang12/models/muse/KeyeTokenizer/"
PROCESSOR_AVAILABLE = os.path.exists(PROCESSOR_PATH)

if not PROCESSOR_AVAILABLE:
    print(f"⚠️  Processor not found at {PROCESSOR_PATH}")
    print("Please update PROCESSOR_PATH to a valid processor location")
    exit(1)


def init_distributed_for_test():
    """Initialize distributed environment for testing."""
    if not dist.is_initialized():
        # Set environment variables for single-process distributed mode
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'
        
        # Initialize process group
        dist.init_process_group(backend='gloo', rank=0, world_size=1)
        print("✅ Distributed environment initialized for testing")


def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("✅ Distributed environment cleaned up")


def create_test_image(width=100, height=100, color='red', mode='RGB'):
    """Create a test PIL Image."""
    return Image.new(mode, (width, height), color=color)


def create_test_parquet(tmp_path):
    """Create a test parquet file with chat-style data."""
    # Create test images
    img1 = create_test_image(256, 256, color='red')
    img2 = create_test_image(300, 200, color='blue')
    
    # Save images to temp files
    img1_path = tmp_path / "test_img1.png"
    img2_path = tmp_path / "test_img2.png"
    img1.save(img1_path)
    img2.save(img2_path)
    
    # Create chat-style messages
    message1 = [
        {"role": "user", "content": "Generate an image of a beautiful sunset over the ocean"},
        {"role": "assistant", "content": [{"type": "image", "image": "img1"}]}
    ]
    
    message2 = [
        {"role": "user", "content": "Create an image of a cute cat playing with a ball"},
        {"role": "assistant", "content": [{"type": "image", "image": "img2"}]}
    ]
    
    data = {
        'uuid': ['1', '2'],
        'source': ['test', 'test'],
        'image': ['img1', 'img2'],
        'message': [json.dumps(message1), json.dumps(message2)],
        'images': [
            json.dumps({"img1": str(img1_path)}),
            json.dumps({"img2": str(img2_path)})
        ],
        'metadata': [
            json.dumps({
                "images_info": {
                    "img1": {"height": 256, "width": 256},
                    "img2": {"height": 300, "width": 200}
                }
            }),
            json.dumps({
                "images_info": {
                    "img1": {"height": 256, "width": 256},
                    "img2": {"height": 300, "width": 200}
                }
            })
        ]
    }
    
    df = pd.DataFrame(data)
    parquet_path = tmp_path / "test_chat.parquet"
    df.to_parquet(parquet_path)
    return str(parquet_path)


def print_sample_info(sample, title):
    """Print detailed information about a sample."""
    print(f"\n{'='*60}")
    print(f"📊 {title}")
    print(f"{'='*60}")
    
    if sample is None:
        print("❌ Sample is None")
        return
    
    print(f"📋 Sample keys: {list(sample.keys())}")
    print(f"📏 Sample type: {type(sample)}")
    
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"🔹 {key}: Tensor with shape {value.shape}, dtype {value.dtype}")
            if value.numel() <= 10:  # Only print small tensors
                print(f"   Values: {value}")
        elif isinstance(value, (list, dict)):
            print(f"🔹 {key}: {type(value).__name__} with {len(value)} elements")
        else:
            print(f"🔹 {key}: {type(value).__name__} = {value}")


def demo_single_samples(dataset):
    """Demo: Iterate through individual samples without collator."""
    print("\n" + "🎯" * 30)
    print("🎯 DEMO 1: Single Samples (Without Collator)")
    print("🎯" * 30)
    
    print("🔍 Iterating through dataset samples using iterator:")
    
    # Use iterator to get samples
    sample_count = 0
    for i, sample in enumerate(dataset):
        if sample_count >= 2:  # Show only first 2 samples
            break
            
        print(f"\n📦 Sample {i}:")
        print_sample_info(sample, f"Sample {i} - Raw Output")
        
        if sample and "pixel_values" in sample:
            print(f"   🖼️  Pixel values range: [{sample['pixel_values'].min():.3f}, {sample['pixel_values'].max():.3f}]")
        
        if sample and "image_grid_thw" in sample:
            print(f"   📐 Image grid shape: {sample['image_grid_thw']}")
        
        sample_count += 1


def demo_with_collator(dataset):
    """Demo: Use collator to batch samples."""
    print("\n" + "🚀" * 30)
    print("🚀 DEMO 2: Batch Processing (With Collator)")
    print("🚀" * 30)
    
    print("🔍 Creating a batch of samples and applying collate_fn:")
    
    # Create a batch of samples using iterator
    batch_samples = []
    sample_count = 0
    for sample in dataset:
        if sample_count >= 2:  # Use first 2 samples
            break
        if sample is not None:
            batch_samples.append(sample)
            print(f"📥 Added sample {sample_count} to batch")
            sample_count += 1
    
    if not batch_samples:
        print("❌ No valid samples found for batching")
        return
    
    print(f"\n📦 Batch contains {len(batch_samples)} samples")
    
    # Apply collator
    print("\n🔄 Applying collate_fn...")
    collated_batch = dataset.collate_fn(batch_samples)
    
    print_sample_info(collated_batch, "Collated Batch Output")
    
    # Show detailed batch information
    print("\n📊 Detailed Batch Analysis:")
    for key, value in collated_batch.items():
        if isinstance(value, torch.Tensor):
            print(f"   🔹 {key}: shape={value.shape}, dtype={value.dtype}")
            if "pixel_values" in key:
                print(f"      📊 Stats: min={value.min():.3f}, max={value.max():.3f}, mean={value.mean():.3f}")
            elif "ids" in key or "mask" in key:
                print(f"      🔢 Unique values: {torch.unique(value)}")


def demo_processor_outputs(dataset):
    """Demo: Show processor output fields."""
    print("\n" + "🔧" * 30)
    print("🔧 DEMO 3: Processor Output Analysis")
    print("🔧" * 30)
    
    # Get first sample using iterator
    sample = None
    for s in dataset:
        sample = s
        break
    
    if sample is None:
        print("❌ No sample available for analysis")
        return
    
    print("🔍 Analyzing processor output fields:")
    
    # Categorize fields
    image_fields = [k for k in sample.keys() if 'image' in k.lower() or 'pixel' in k.lower()]
    text_fields = [k for k in sample.keys() if 'id' in k.lower() or 'mask' in k.lower() or 'position' in k.lower()]
    other_fields = [k for k in sample.keys() if k not in image_fields + text_fields]
    
    print(f"🖼️  Image-related fields: {image_fields}")
    print(f"📝 Text-related fields: {text_fields}")
    print(f"📦 Other fields: {other_fields}")
    
    print("\n📋 Field details:")
    for field in image_fields + text_fields + other_fields:
        if field in sample:
            value = sample[field]
            if isinstance(value, torch.Tensor):
                print(f"   🔹 {field}: {value.shape} {value.dtype}")
            else:
                print(f"   🔹 {field}: {type(value)}")


def main():
    """Main demo function."""
    print("🚀 Starting Chat2ImageDataset Demo")
    print("=" * 60)
    
    # Initialize distributed environment for testing
    init_distributed_for_test()
    
    try:
        # Create temporary directory for test data
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            print("📁 Creating test data...")
            # parquet_path = create_test_parquet(tmp_path)
            parquet_path = "/llm_reco/vlm/datahub/datasets/Sana_pretrain/0.0.0/index/parquet.json"
            
            print("🔧 Initializing Chat2ImageDataset...")
            dataset = Chat2ImageDataset(
                sources=parquet_path,
                image_size=256,
                processor_path=PROCESSOR_PATH,
                num_workers=1
            )
            print(dataset.collate_fn); exit()
            print("📊 Dataset type: IterableDataset (use iterator, not indexing)")
            
            # Run demos
            demo_single_samples(dataset)
            demo_with_collator(dataset)
            demo_processor_outputs(dataset)
            
            print("\n" + "✅" * 30)
            print("✅ Demo Completed Successfully!")
            print("✅" * 30)
            print("\n📝 Summary:")
            print("   • IterableDataset: use iterator (for sample in dataset)")
            print("   • Single samples show individual processor outputs")
            print("   • Collator concatenates sequence-based fields along dim=0")
            print("   • Image tensors are stacked along batch dimension")
            print("   • All processor output fields are preserved")
    
    finally:
        # Clean up distributed environment
        cleanup_distributed()


if __name__ == "__main__":
    main()