# the code find all resolutions for vae, which are divisible by vae_patch_size, and consumes less than vit_max_tokens tokens
from functools import lru_cache
from PIL import Image
import math

class ResolutionFinder:
    # The ResolutionFinder class is used to find optimal resolutions for a given maximum resolution and aspect ratio.
    # It also gives the system prompt for LLM.
    def __init__(self,
                 vit_max_tokens=324,
                 vit_patch_size=28,
                 vae_patch_size=32,
                 max_aspect_ratio=1.5,
                 max_resolution=1024,
                 system_prompt_template=None,
                 predifined_resolutions=None,
                 only_general_template=False
                 ):
        if system_prompt_template is None:
            system_prompt_template = {
                "generation": [
                    # 生成理解通用模板
                    """You are a helpful assistant.""",
                    """You are a helpful assistant adept at chatting, image editing, understanding and generation.""",
                    """You are a capable assistant skilled in dialogue, image understanding and creative generation.""",
                    """You are a helpful assistant proficient at conversation, image analysis and visual content generation.""",
                     
                    # 生成通用模板
                    """You are a high-performance model for image generation, specialized in turning creative ideas into high-quality visuals.""",
                    """You are a user-friendly image creation model that excels at translating natural language into high-quality images.""",
                    """You are an expert in image generation, capable of producing professional-grade visuals from simple textual prompts.""",

                    ],
                "edit": [
                    # 生成理解通用模板
                    """You are a helpful assistant.""",
                    """You are a helpful assistant adept at chatting, image editing, understanding and generation.""",
                    """You are a capable assistant skilled in dialogue, image understanding and creative generation.""",
                    """You are a helpful assistant proficient at conversation, image analysis and visual content generation.""",

                    # 编辑通用模板
                    """You are a high-performance model for image editing, specialized in enhancing and refining visuals to meet user requirements.""",
                    """You are a user-friendly image editing model that excels at adjusting and optimizing images based on natural language instructions.""",
                    """You are an expert in image editing, capable of performing professional-grade retouching, enhancement and modification on existing images.""",
                    """An adept image editing model focused on refining visual details, adjusting styles and improving image quality from textual prompts.""",
                    """You are a high-performance image editing model specialized in semantic-aware modification, style transfer and quality enhancement for various visual content.""",
                    """You are a user-friendly image editing system that excels at interpreting complex textual instructions to perform precise retouching, cropping and color grading.""",

                    ],
                "understanding": [
                    """You are a helpful assistant.""",
                    """You are a helpful assistant adept at chatting, image editing, understanding and generation.""",
                    """You are a capable assistant skilled in dialogue, image understanding and creative generation.""",
                    """You are a helpful assistant proficient at conversation, image analysis and visual content generation.""",
                    
                    """You are a helpful multimodal large language model. You can engage in natural language conversations and directly generate high-quality images based on user requests.""",
                    """You are a helpful assistant in the field of image understanding.""",
                    """You are a helpful assistant specialized in visual content understanding and natural language dialogue, adept at answering questions about images and engaging in context-aware conversations.""",
                    """You are a user-friendly assistant specialized in visual comprehension and dialogue, capable of interpreting complex image details and engaging in natural, context-aware conversations.""",
                    """You are an expert assistant in image understanding and conversational AI, proficient at recognizing image content, answering related questions and maintaining smooth dialogue flow.""",

                    ]
                }
            if only_general_template:
                print(f"only_general_template=True")
                system_prompt_template = {
                    "generation": [
                        # 生成理解通用模板
                        """You are a helpful assistant.""",
                        """You are a helpful assistant adept at chatting, image editing, understanding and generation.""",
                        """You are a capable assistant skilled in dialogue, image understanding and creative generation.""",
                        """You are a helpful assistant proficient at conversation, image analysis and visual content generation.""",
                        
                        # 生成通用模板
                        """You are a high-performance model for image generation, specialized in turning creative ideas into high-quality visuals.""",
                        """You are a user-friendly image creation model that excels at translating natural language into high-quality images.""",
                        """You are an expert in image generation, capable of producing professional-grade visuals from simple textual prompts.""",

                        ],
                    "edit": [
                        # 生成理解通用模板
                        """You are a helpful assistant.""",
                        """You are a helpful assistant adept at chatting, image editing, understanding and generation.""",
                        """You are a capable assistant skilled in dialogue, image understanding and creative generation.""",
                        """You are a helpful assistant proficient at conversation, image analysis and visual content generation.""",

                        # 编辑通用模板
                        """You are a high-performance model for image editing, specialized in enhancing and refining visuals to meet user requirements.""",
                        """You are a user-friendly image editing model that excels at adjusting and optimizing images based on natural language instructions.""",
                        """You are an expert in image editing, capable of performing professional-grade retouching, enhancement and modification on existing images.""",
                        """An adept image editing model focused on refining visual details, adjusting styles and improving image quality from textual prompts.""",
                        """You are a high-performance image editing model specialized in semantic-aware modification, style transfer and quality enhancement for various visual content.""",
                        """You are a user-friendly image editing system that excels at interpreting complex textual instructions to perform precise retouching, cropping and color grading.""",
                        ],
                    "understanding": [
                        """You are a helpful assistant.""",
                        """You are a helpful assistant adept at chatting, image editing, understanding and generation.""",
                        """You are a capable assistant skilled in dialogue, image understanding and creative generation.""",
                        """You are a helpful assistant proficient at conversation, image analysis and visual content generation.""",
                        
                        """You are a helpful multimodal large language model. You can engage in natural language conversations and directly generate high-quality images based on user requests.""",
                        """You are a helpful assistant in the field of image understanding.""",
                        """You are a helpful assistant specialized in visual content understanding and natural language dialogue, adept at answering questions about images and engaging in context-aware conversations.""",
                        """You are a user-friendly assistant specialized in visual comprehension and dialogue, capable of interpreting complex image details and engaging in natural, context-aware conversations.""",
                        """You are an expert assistant in image understanding and conversational AI, proficient at recognizing image content, answering related questions and maintaining smooth dialogue flow.""",

                        ]
                    }
        self.max_resolution = max_resolution
        self.system_prompt_template = system_prompt_template
        self.max_aspect_ratio = max_aspect_ratio
        self.vit_max_tokens = vit_max_tokens
        self.vit_patch_size = vit_patch_size
        self.vae_patch_size = vae_patch_size
        self.predifined_resolutions = predifined_resolutions
    
    @lru_cache(maxsize=None)
    def find_optimal_resolutions(self):
        """
        Find resolutions that:
        1. Are divisible by vae_patch_size (32)
        2. Consume close to vit_max_tokens (324), meaning that increasing by vae_patch_size in either dimension 
        would exceed the token limit
        
        Args:
            max_resolution: Maximum resolution to consider (default 1024)
        
        Returns:
            List of (width, height) tuples that are optimal for token usage
        """
        if self.predifined_resolutions:
            return self.predifined_resolutions
        max_resolution = self.max_resolution
        vae_patch_size = self.vae_patch_size
        vit_patch_size = self.vit_patch_size
        vit_max_tokens = self.vit_max_tokens
        max_aspect_ratio = self.max_aspect_ratio
        optimal_resolutions = []
        for width in range(vae_patch_size, max_resolution + 1, vae_patch_size):
            for height in range(vae_patch_size, max_resolution + 1, vae_patch_size):
                # Calculate current token count
                tokens_current = (width // vit_patch_size) * (height // vit_patch_size)
                
                # Check if current resolution is within limit
                if tokens_current <= vit_max_tokens:
                    # Check if increasing by vae_patch_size in either dimension would exceed limit
                    tokens_width_inc = ((width + vae_patch_size) // vit_patch_size) * (height // vit_patch_size)
                    tokens_height_inc = (width // vit_patch_size) * ((height + vae_patch_size) // vit_patch_size)
                    # Check aspect ratio constraint
                    if width / height <= max_aspect_ratio and height / width <= max_aspect_ratio:
                        # Add to optimal if increasing either dimension would exceed the limit
                        if tokens_width_inc > vit_max_tokens or tokens_height_inc > vit_max_tokens:
                            optimal_resolutions.append((width, height))
        
        return optimal_resolutions

    def get_system_prompt(self, mode="generation"):
        all_resolutions = self.find_optimal_resolutions()
        resolutions = [f"{w}x{h}" for w, h in all_resolutions]
        resolutions_str = ', '.join(resolutions[:-1]) + " and " + resolutions[-1]
        system_prompt = self.system_prompt_template[mode]
        for i in range(len(system_prompt)):
            if "{num}" in system_prompt[i]:
                system_prompt[i] = system_prompt[i].format(num=len(resolutions), resolutions=resolutions_str)
        return system_prompt
    
    def get_all_resolutions(self):
        return self.find_optimal_resolutions()
    
    def get_resolution_from_cropped_image_size(self, cropped_width: int, cropped_height: int):
        '''
        根据裁剪后的图片尺寸推断出使用的原始 optimal resolution
        
        Args:
            cropped_width: 裁剪后的图片宽度
            cropped_height: 裁剪后的图片高度
            
        Returns:
            最可能使用的原始 optimal resolution (width, height)
        '''
        optimal_resolutions = self.find_optimal_resolutions()
        if not optimal_resolutions:
            return None
            
        # 计算裁剪图片的宽高比
        cropped_aspect = cropped_width / cropped_height
        
        # 找到与裁剪尺寸宽高比最接近的optimal resolution
        closest_resolution = None
        min_aspect_diff = float('inf')
        
        for res in optimal_resolutions:
            res_aspect = res[0] / res[1]
            aspect_diff = abs(res_aspect - cropped_aspect)
            
            if aspect_diff < min_aspect_diff:
                min_aspect_diff = aspect_diff
                closest_resolution = res
            elif aspect_diff == min_aspect_diff:
                # 如果宽高比相同，选择面积更大的resolution
                if res[0] * res[1] > closest_resolution[0] * closest_resolution[1]:
                    closest_resolution = res
        
        return closest_resolution
        
    def crop_and_resize_image(self, image: Image.Image):
        '''
        对于输入的图片，这个函数会选择与原始图片长宽比最接近的optimal分辨率，
        然后按比例缩放到刚好覆盖该分辨率，最后裁剪到长宽都是patch_size的倍数
        
        Args:
            image: PIL Image object
            
        Returns:
            PIL Image object after cropping and resizing
        '''
        from PIL import Image
        import math
        
        # 获取原始图片尺寸
        orig_width, orig_height = image.size
        orig_aspect_ratio = orig_width / orig_height
        
        # 获取所有最优分辨率
        optimal_resolutions = self.find_optimal_resolutions()
        
        if not optimal_resolutions:
            # 如果没有最优分辨率，直接按patch_size裁剪
            final_width = (orig_width // self.vit_patch_size) * self.vit_patch_size
            final_height = (orig_height // self.vit_patch_size) * self.vit_patch_size
            final_width = max(self.vit_patch_size, final_width)
            final_height = max(self.vit_patch_size, final_height)
            
            # 中心裁剪
            left = (orig_width - final_width) // 2
            top = (orig_height - final_height) // 2
            right = left + final_width
            bottom = top + final_height
            
            return image.crop((left, top, right, bottom))
        
        # 1. 找到与原始图片长宽比最接近的optimal分辨率
        closest_resolution = None
        min_aspect_diff = float('inf')
        
        for res in optimal_resolutions:
            res_aspect_ratio = res[0] / res[1]
            aspect_diff = abs(res_aspect_ratio - orig_aspect_ratio)
            
            # 优先选择长宽比最接近的分辨率
            if aspect_diff < min_aspect_diff:
                min_aspect_diff = aspect_diff
                closest_resolution = res
            elif aspect_diff == min_aspect_diff:
                # 如果长宽比相同，选择面积更大的分辨率
                if res[0] * res[1] > closest_resolution[0] * closest_resolution[1]:
                    closest_resolution = res
        
        target_width, target_height = closest_resolution
        
        # 2. 按比例缩放到刚好覆盖目标分辨率的尺寸
        width_ratio = target_width / orig_width
        height_ratio = target_height / orig_height
        
        # 选择较大的缩放比例以确保图片完全覆盖目标尺寸
        scale_factor = max(width_ratio, height_ratio)
        
        # 计算缩放后的尺寸
        scaled_width = int(orig_width * scale_factor)
        scaled_height = int(orig_height * scale_factor)
        
        # 缩放图片
        scaled_image = image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)

        # 3. 从缩放后的图片中裁剪到长宽都是28的倍数
        # 计算最大的28的倍数尺寸，不超过缩放后的尺寸
        croped = [
            (
                math.ceil(scaled_width / self.vit_patch_size) * self.vit_patch_size,
                math.ceil(scaled_height / self.vit_patch_size) * self.vit_patch_size,
            ),
            (
                math.floor(scaled_width / self.vit_patch_size) * self.vit_patch_size,
                math.ceil(scaled_height / self.vit_patch_size) * self.vit_patch_size,
            ),
            (
                math.ceil(scaled_width / self.vit_patch_size) * self.vit_patch_size,
                math.floor(scaled_height / self.vit_patch_size) * self.vit_patch_size,
            ),
            (
                math.floor(scaled_width / self.vit_patch_size) * self.vit_patch_size,
                math.floor(scaled_height / self.vit_patch_size) * self.vit_patch_size,
            ),
        ]
        best_crop = croped[-1]
        for crop in croped:
            crop_width, crop_height = crop
            if crop_width * crop_height / self.vit_patch_size / self.vit_patch_size > self.vit_max_tokens:
                continue
            if crop_width * crop_height > best_crop[0] * best_crop[1]:
                best_crop = crop
                break
    
        crop_width, crop_height = best_crop

        # 确保裁剪尺寸至少为28x28
        crop_width = max(self.vit_patch_size, crop_width)
        crop_height = max(self.vit_patch_size, crop_height)
        
        # 计算中心裁剪位置
        left = (scaled_width - crop_width) // 2
        top = (scaled_height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height
        
        # 进行中心裁剪
        cropped_image = scaled_image.crop((left, top, right, bottom))
        
        return cropped_image


def test_crop_and_resize_image():
    """
    测试crop_and_resize_image函数，生成各种尺寸的测试图片并验证处理结果
    """
    import os
    from PIL import Image, ImageDraw
    import tempfile
    
    # 创建测试目录
    test_dir = "test_images"
    os.makedirs(test_dir, exist_ok=True)
    
    # 创建ResolutionFinder实例
    reso_finder = ResolutionFinder()
    
    # 定义各种测试尺寸 (小、中、大、超大)
    test_sizes = [
        (64, 64),      # 很小
        (128, 128),    # 小
        (3000, 2000),  # 超大
        (1024, 767)
    ]
    
    results = []
    
    for i, (width, height) in enumerate(test_sizes):
        # 创建测试图片
        img = Image.new('RGB', (width, height), color=(i*30, i*20, i*10))
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"Original: {width}x{height}", fill=(255, 255, 255))
        
        # 保存原始图片
        orig_path = os.path.join(test_dir, f"test_{i}_original_{width}x{height}.png")
        img.save(orig_path)
        
        # 调用crop_and_resize_image
        processed_img = reso_finder.crop_and_resize_image(img)
        processed_width, processed_height = processed_img.size
        
        # 检查处理后的尺寸是否符合要求
        # 1. 应该是vit_patch_size(28)的倍数
        assert processed_width % reso_finder.vit_patch_size == 0, \
            f"Width {processed_width} not multiple of {reso_finder.vit_patch_size}"
        assert processed_height % reso_finder.vit_patch_size == 0, \
            f"Height {processed_height} not multiple of {reso_finder.vit_patch_size}"
        
        # 2. token数应该在限制范围内
        tokens = (processed_width // reso_finder.vit_patch_size) * \
                 (processed_height // reso_finder.vit_patch_size)
        assert tokens <= reso_finder.vit_max_tokens, \
            f"Token count {tokens} exceeds max {reso_finder.vit_max_tokens}"
        
        # 保存处理后的图片
        processed_path = os.path.join(test_dir, f"test_{i}_processed_{processed_width}x{processed_height}.png")
        processed_img.save(processed_path)
        
        results.append({
            "original": (width, height),
            "processed": (processed_width, processed_height),
            "tokens": tokens,
            "orig_path": orig_path,
            "proc_path": processed_path
        })
    
    # 打印测试结果
    print("=" * 80)
    print("Crop and Resize Test Results:")
    print("=" * 80)
    for i, result in enumerate(results):
        print(f"Test {i}:")
        print(f"  Original: {result['original'][0]}x{result['original'][1]}")
        print(f"  Processed: {result['processed'][0]}x{result['processed'][1]}")
        print(f"  Tokens: {result['tokens']}/{reso_finder.vit_max_tokens}")
        print(f"  Original saved: {result['orig_path']}")
        print(f"  Processed saved: {result['proc_path']}")
        print("-" * 40)
    
    return results

def test_get_resolution_from_cropped_image_size():
    """
    测试get_resolution_from_cropped_image_size函数
    """
    reso_finder = ResolutionFinder()
    optimal_resolutions = reso_finder.get_all_resolutions()
    
    # 测试用例：使用optimal resolutions作为输入
    print("\n" + "="*80)
    print("Testing get_resolution_from_cropped_image_size with optimal resolutions:")
    print("="*80)
    for res in optimal_resolutions:
        # 模拟裁剪后的尺寸（可能比optimal resolution小）
        cropped_width = (res[0] // 28) * 28
        cropped_height = (res[1] // 28) * 28
        
        inferred_res = reso_finder.get_resolution_from_cropped_image_size(
            cropped_width, cropped_height
        )
        print(f"Input: {cropped_width}x{cropped_height} -> Inferred: {inferred_res[0]}x{inferred_res[1]}")
        assert inferred_res == res, "Inferred resolution should match original optimal resolution"

if __name__ == "__main__":
    # 测试分辨率查找功能
    reso_finder = ResolutionFinder()
    print(reso_finder.get_system_prompt("generation"))
    print(reso_finder.get_system_prompt("edit"))
    print("Optimal resolutions:", reso_finder.get_all_resolutions())
    
    print("\n" + "="*80)
    print("Running crop_and_resize_image tests...")
    print("="*80)
    
    # 运行测试
    test_results = test_crop_and_resize_image()
    
    # 测试get_resolution_from_cropped_image_size
    test_get_resolution_from_cropped_image_size()
    
    print("\nAll tests completed successfully!")
    print(f"Generated {len(test_results)} test cases in 'test_images/' directory")


