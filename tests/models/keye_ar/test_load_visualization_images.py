import os
import sys
import tempfile
import pandas as pd
import json
import torch
from PIL import Image
from pathlib import Path
from muse.training.common import set_default_dtype

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
import traceback
# 从train_sana_ar_ae.py导入load_visualization_images函数
# 从muse.data.datasets.chat2image导入Chat2ImageDataset
from muse.data.datasets.chat2image import Chat2ImageDataset
# 从muse.config.dataset_config导入DatasetConfig
from muse.config.dataset_config import DatasetConfig
# 从muse.config导入load_config函数，与训练代码保持一致
from muse.config import load_config
from muse.training.common import (
    set_default_dtype,
    get_torch_dtype,
    clip_grad_by_value, 
    compute_fsdp_zero2_grad_norm
)
from keye_vl_utils import process_vision_info
from typing import Optional, List, Union, Any



def load_visualization_images(
    parquet_path: str,  # 改为接收parquet_path参数
    dataset,  # 保留dataset参数用于处理方法
    processor,
    image_size: int,
    max_condition_length: int,
    device: torch.device,
    dtype: torch.dtype,
    num_images: Optional[int] = None
) -> tuple:
    """Load and preprocess images from parquet file for visualization.
    
    Args:
        parquet_path: Path to parquet file containing samples
        dataset: Chat2ImageDataset instance for processing methods
        processor: AutoProcessor instance for image preprocessing
        image_size: Target image size
        max_condition_length: Maximum condition sequence length for image tokenizer
        device: Target device
        dtype: Target dtype
        num_images: Maximum number of images to load (None for all)
    
    Returns:
        Tuple of (original_images, pixel_values, image_grid_thw, vae_input_images)
        - original_images: List of PIL images (for visualization)
        - pixel_values: Tensor for image tokenizer
        - image_grid_thw: Grid info tensor for image tokenizer
        - vae_input_images: Tensor for VAE encoding [B, C, H, W] in [-1, 1]
    """
    from PIL import Image
    from torchvision import transforms
    import pandas as pd
    import json
    print(f"load_visualization_images={load_visualization_images}")
    # Read parquet file
    try:
        df = pd.read_parquet(parquet_path)
        if num_images is not None:
            df = df.head(num_images)
        
        if df.empty:
            print(f"Warning: No samples found in {parquet_path}")
            return None, None, None, None
        
        print(f"Loading {len(df)} samples from {parquet_path}")
        
        # Process samples using dataset's process method
        processed_samples = []
        texts = []
        for _, row in df.iterrows():
            # Convert parquet row to sample format expected by dataset
            sample = {
                "__key__": row.get("__key__", ""),
                "messages": json.loads(row["messages"]) if isinstance(row["messages"], str) else row["messages"],
                "images": json.loads(row["images"]) if isinstance(row["images"], str) else row["images"],
                "source": row.get("source", "")
            }
            # messages=[{'role': 'user', 'content': [{'type': 'text', 'text': '这是第0张图像的描述'}]}, {'role': 'assistant', 'content': [{'type': 'image', 'image': '/tmp/tmpmah5htt0/images/image_0.jpg'}]}]
            print(f"messages={sample['messages']}")
            print(f"sample={sample}")
            # Use dataset's process method
            processed_sample = dataset.process(sample)
            processed_samples.append(processed_sample)
            text = sample["messages"][0]["content"][0]["text"]
            texts.append(text)
        
        print(f"processed_samples={processed_samples}")

        # Use dataset's collate_fn to batch the samples
        batch = dataset.collate_fn(processed_samples)
        
        # Extract original images from the batch
        original_images = []
        for i in range(len(df)):
            # Get the image from the batch (assuming it's in 'image' field)
            if 'image' in batch:
                img_tensor = batch['image'][i]
                # Convert tensor back to PIL image for visualization
                img = transforms.ToPILImage()(img_tensor.cpu())
                original_images.append(img)
            else:
                # Fallback: try to get from pixel_values if available
                if 'pixel_values' in batch:
                    # This is more complex as pixel_values are processed, so we'll use the baseline approach
                    # For now, we'll use the baseline approach
                    break
        
        # If we couldn't extract original images from batch, use baseline approach
        if not original_images:
            # Use baseline approach: create fake messages and process images
            # This maintains the original vae_input_images generation logic
            fake_original_images = []
            for i in range(len(df)):
                # Create a dummy image of the correct size
                img = Image.new('RGB', (image_size, image_size), color='white')
                fake_original_images.append(img)
            original_images = fake_original_images
    
    except Exception as e:
        print(f"Error loading parquet file {parquet_path}: {e}")
        traceback.print_exc()
        return None, None, None, None, None
    
    # Prepare messages format for processor (keye_vl_utils format) - BASELINE LOGIC
    fake_messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": img,
                "min_pixels": 4 * 28 * 28,
                "max_pixels": max_condition_length * 28 * 28
            } for img in original_images],
    }]
    text = processor.apply_chat_template(
        fake_messages,
        tokenize=False
    )
    # Process using keye_vl_utils
    image_inputs, _, _ = process_vision_info(fake_messages)
    
    # Use processor to get pixel_values and image_grid_thw - BASELINE LOGIC
    inputs = processor(
        text=text,
        images=image_inputs,
        padding=False,
        truncation=False,
        return_tensors="pt",
    )
    
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
    image_grid_thw = inputs["image_grid_thw"].to(device=device)
    
    # Prepare images for VAE (normalize to [-1, 1]) - BASELINE LOGIC
    vae_transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 1]
        transforms.Normalize([0.5], [0.5]),  # [-1, 1]
    ])
    vae_input_images = torch.stack([vae_transform(img) for img in original_images])
    vae_input_images = vae_input_images.to(device=device, dtype=dtype)
    print(f"return_text={text}")
    return text, original_images, pixel_values, image_grid_thw, vae_input_images





def create_test_image(width=512, height=512, color='red', mode='RGB'):
    """创建一个测试PIL Image."""
    return Image.new(mode, (width, height), color=color)

def create_test_parquet(tmp_path, image_size=512):
    """创建一个测试parquet文件，符合实际训练数据格式."""
    parquet_path = tmp_path / "test_visualization.parquet"
    
    # 创建测试图像并保存
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    
    # 创建模拟数据
    data = []
    for i in range(2):  # 创建2个样本
        # 保存测试图像
        img = create_test_image(width=image_size, height=image_size, color='blue')
        img_path = images_dir / f"image_{i}.jpg"
        img.save(img_path)
        
        # 创建符合Chat2ImageDataset预期的样本结构
        sample = {
            "__key__": f"sample_{i}",
            # messages字段需要与实际训练数据格式一致
            "messages": json.dumps([
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"这是第{i}张图像的描述"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "image", "image": str(img_path)}
                    ]
                }
            ]),
            # images字段需要与实际训练数据格式一致
            "images": json.dumps({
                str(img_path): str(img_path)
            }),
            "source": "test_source"
        }
        data.append(sample)
    
    df = pd.DataFrame(data)
    df.to_parquet(parquet_path, index=False)
    
    return parquet_path, tmp_path

def test_load_visualization_images_with_real_env():
    """使用实际训练环境的模型和处理器测试load_visualization_images函数."""
    # 创建测试目录和parquet文件
    with tempfile.TemporaryDirectory() as tmp_path:
        tmp_path = Path(tmp_path)
        # parquet_path, tmp_root = create_test_parquet(tmp_path, image_size=512)
        
        try:
            # 配置参数 - 从run_ar_ae_lzx_4096.sh读取
            KEYE_AR_DIR = "/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.7/run_8b_vis_stage3.29_1e-4/step18000/global_step18000/muse_converted"
            MODEL_DIR = "/llm_reco_ssd/zhouyang12/models/muse/Sana_1600M_1024px/"
            MODEL_CONFIG = "/llm_reco_ssd/zhouyang12/models/muse/Sana_1600M_1024px/config.json"
            VISUALIZE_PARQUET_PATH = "/llm_reco/lingzhixin/recovlm_data/datasets/Gen_qwen_image_position/0.0.0/part/rank0-0.parquet"# str(parquet_path)
            IMAGE_SIZE = 512
            MAX_CONDITION_LENGTH = 324
            NUM_VIS_IMAGES = 2
            
            # 配置设备和数据类型
            device = torch.device("cpu")  # 或者使用"cuda"如果可用
            dtype = torch.float32
            
            # 1. 加载model_config获取model_class_name，与训练代码保持一致
            model_config = load_config(MODEL_CONFIG)
            model_class_name = model_config.model_class
            print(f"从model_config获取的model_class_name: {model_class_name}")
            
            # 2. 加载实际的数据集配置
            # 从examples/sana/ar-ae-mix.json加载配置
            dataset_config_path = Path("examples/sana/ar-ae-mix.json")
            with open(dataset_config_path, "r") as f:
                dataset_config_dict = json.load(f)
            
            # 更新数据集配置，与训练代码保持一致
            dataset_config_dict.update({
                "sources": [str(parquet_path)],
                "image_size": IMAGE_SIZE,
                "max_condition_length": MAX_CONDITION_LENGTH,
                "processor_path": KEYE_AR_DIR,  # 改为使用KEYE_AR_DIR，与训练代码保持一致
                "model_class": model_class_name,  # 添加model_class字段，与训练代码保持一致
                "num_workers": 1,
                "center_crop": True,
                "packing": False
            })
            
            print(f"使用数据集配置: {dataset_config_dict}")
            
            # 3. 初始化实际的数据集
            dataset = Chat2ImageDataset(**dataset_config_dict)
            
            # 4. 加载实际的处理器，从KEYE_AR_DIR加载，与训练代码保持一致
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(KEYE_AR_DIR, trust_remote_code=True)
            
            print(f"加载的处理器: {processor}")
            
            # 4. 调用load_visualization_images函数
            print(f"开始调用load_visualization_images函数...")
            result = load_visualization_images(
                parquet_path=VISUALIZE_PARQUET_PATH,
                dataset=dataset,
                processor=processor,
                image_size=IMAGE_SIZE,
                max_condition_length=MAX_CONDITION_LENGTH,
                device=device,
                dtype=dtype,
                num_images=NUM_VIS_IMAGES
            )
            
            # 5. 验证输出
            text, original_images, pixel_values, image_grid_thw, vae_input_images = result
            
            print(f"函数调用成功，返回结果:")
            print(f"  text: {text}")
            print(f"  original_images数量: {len(original_images) if original_images else 0}")
            print(f"  pixel_values形状: {pixel_values.shape if pixel_values is not None else None}")
            print(f"  image_grid_thw形状: {image_grid_thw.shape if image_grid_thw is not None else None}")
            print(f"  vae_input_images形状: {vae_input_images.shape if vae_input_images is not None else None}")
            
            # 验证返回值是否符合预期
            assert text is not None, "text should not be None"
            assert original_images is not None and len(original_images) == NUM_VIS_IMAGES, f"Should have {NUM_VIS_IMAGES} original_images"
            assert pixel_values is not None, "pixel_values should not be None"
            assert image_grid_thw is not None, "image_grid_thw should not be None"
            assert vae_input_images is not None, "vae_input_images should not be None"
            
            # 验证数据类型和形状
            assert isinstance(pixel_values, torch.Tensor), "pixel_values should be a torch.Tensor"
            assert isinstance(image_grid_thw, torch.Tensor), "image_grid_thw should be a torch.Tensor"
            assert isinstance(vae_input_images, torch.Tensor), "vae_input_images should be a torch.Tensor"
            
            print("\n✅ 所有验证通过！测试成功！")
            
        except Exception as e:
            print(f"\n❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == "__main__":
    print("开始使用实际训练环境测试load_visualization_images函数...")
    print("=" * 80)
    
    test_load_visualization_images_with_real_env()
    
    print("\n🎉 测试完成！")