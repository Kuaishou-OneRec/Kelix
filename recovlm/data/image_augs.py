from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import os

class AutoAugmentWrapper:
    """
    PyTorch AutoAugment数据增强包装类
    
    参数:
        policy: 预训练策略，可选 'imagenet', 'cifar10', 'svhn'
        interpolation: 插值方法，默认为双线性插值
        fill: 填充值，用于边界外填充
    """
    def __init__(self, 
                 policy: str = 'grounding_ocr_imagenet', 
                 interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BILINEAR,
                 fill: int = None):
        # 选择对应数据集的预训练策略
        if policy.lower() == 'grounding_ocr_imagenet':
            grounding_ocr_imagenet_policy = [
                # 亮度/对比度调整组合
                (("AutoContrast", 0.6, None), ("Brightness", 0.4, 8)),
                (("Contrast", 0.8, 7), ("Equalize", 0.6, None)),
                (("AutoContrast", 0.8, None), ("Sharpness", 0.4, 9)),
                (("Brightness", 0.6, 6), ("Contrast", 0.6, 6)),
                
                # 颜色空间变换（保留文字形状）
                (("Solarize", 0.3, 5), ("AutoContrast", 0.7, None)),  # 降低Solarize概率
                (("Posterize", 0.3, 6), ("Equalize", 0.7, None)),    # 降低Posterize强度
                (("Color", 0.5, 4), ("Contrast", 0.8, 8)),           # 调整色彩饱和度
                
                # 轻微几何变换（小角度旋转，无翻转）
                (("Rotate", 0.3, 3), ("AutoContrast", 0.7, None)),   # 仅小角度旋转
                (("ShearX", 0.2, 2), ("Equalize", 0.8, None)),       # 极轻微水平剪切
                
                # 噪声和模糊（模拟扫描缺陷）
                (("GaussianBlur", 0.2, 1), ("AutoContrast", 0.8, None)),  # 新增模糊
                (("Noise", 0.3, 10), ("Equalize", 0.7, None)),           # 新增噪声（假设自定义实现）
                
                # 直方图均衡化组合
                (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
                (("Equalize", 0.6, None), ("AutoContrast", 0.6, None)),
                
                # 保留但调整强度的组合
                (("Color", 0.4, 3), ("Contrast", 0.9, 7)),
                (("Solarize", 0.2, 6), ("Sharpness", 0.6, 8)),
                (("Posterize", 0.2, 7), ("Brightness", 0.6, 7)),
            ]
            _policy = grounding_ocr_imagenet_policy
        else:
            raise ValueError(f"Unsupported policy: {policy}->{_policy}. Available: 'grounding_ocr_imagenet'")
        
        # 创建AutoAugment增强器
        self.augmenter = transforms.AutoAugment(
            policy=_policy,
            interpolation=interpolation,
            fill=fill
        )
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        对输入的PIL图像应用AutoAugment增强
        
        参数:
            img: 输入的PIL图像
        
        返回:
            增强后的PIL图像
        """
        return self.augmenter(img)


def create_test_images(size=(200, 200)):
    """创建测试图像：圆形、正方形和单词"""
    # 创建圆形图像
    circle_img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(circle_img)
    draw.ellipse((50, 50, 150, 150), fill='blue')
    
    # 创建正方形图像
    square_img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(square_img)
    draw.rectangle((50, 50, 150, 150), fill='red')
    
    # 创建单词图像
    word_img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(word_img)
    
    # 尝试加载系统字体，否则使用默认字体
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        font = ImageFont.load_default()
    
    draw.text((70, 80), "AI", fill='green', font=font)
    
    return circle_img, square_img, word_img


from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import os
import random



def create_random_color_words(num=60, size=(200, 200), text="AI"):
    """生成指定数量的随机颜色单词图像"""
    images = []
    for _ in range(num):
        # 生成随机颜色（非白色）
        r, g, b = random.randint(0, 200), random.randint(0, 200), random.randint(0, 200)
        while r + g + b > 550:  # 确保非亮色
            r, g, b = random.randint(0, 200), random.randint(0, 200), random.randint(0, 200)
        
        img = Image.new('RGB', size, color='white')
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except IOError:
            font = ImageFont.load_default()
        
        # 计算文本位置
        text_width, text_height = draw.textsize(text, font=font)
        x = (size[0] - text_width) // 2
        y = (size[1] - text_height) // 2
        draw.text((x, y), text, fill=(r, g, b), font=font)
        images.append(img)
    return images


def visualize_word_augmentation(augmenter, words, save_prefix='word_augmentation'):
    """可视化单词增强效果，生成原始和增强图像网格"""
    grid_rows = 10
    grid_cols = 6
    img_size = words[0].size
    spacing = 10
    
    # 计算总尺寸
    total_width = grid_cols * img_size[0] + (grid_cols - 1) * spacing
    total_height = grid_rows * img_size[1] + (grid_rows - 1) * spacing
    
    # 创建原始图像网格
    original_grid = Image.new('RGB', (total_width, total_height), color='white')
    augmented_grid = Image.new('RGB', (total_width, total_height), color='white')
    
    for i in range(grid_rows * grid_cols):
        img = words[i]
        x = (img_size[0] + spacing) * (i % grid_cols)
        y = (img_size[1] + spacing) * (i // grid_cols)
        
        # 原始图像
        original_grid.paste(img, (x, y))
        
        # 增强图像
        augmented_img = augmenter(img.copy())
        augmented_grid.paste(augmented_img, (x, y))
    
    # 保存图像
    original_grid.save(f"{save_prefix}_original.jpg")
    augmented_grid.save(f"{save_prefix}_augmented.jpg")
    print(f"图像已保存：{save_prefix}_original.jpg 和 {save_prefix}_augmented.jpg")


if __name__ == "__main__":
    # 创建增强器
    augmenter = AutoAugmentWrapper(policy='grounding_ocr_imagenet', fill=255)
    
    # 生成60个随机颜色单词
    test_words = create_random_color_words(num=60, text="hello")
    
    # 可视化增强效果
    visualize_word_augmentation(augmenter, test_words, save_prefix='ocr_word_augment')