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
                 policy: str = 'imagenet', 
                 interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BILINEAR,
                 fill: int = None):
        # 选择对应数据集的预训练策略
        if policy.lower() == 'imagenet':
            # policy = transforms.AutoAugmentPolicy.IMAGENET
            policy = [
                (("Posterize", 0.4, 8), ("Rotate", 0.6, 9)),
                (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
                (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
                (("Posterize", 0.6, 7), ("Posterize", 0.6, 6)),
                (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
                (("Equalize", 0.4, None), ("Rotate", 0.8, 8)),
                (("Solarize", 0.6, 3), ("Equalize", 0.6, None)),
                (("Posterize", 0.8, 5), ("Equalize", 1.0, None)),
                (("Rotate", 0.2, 3), ("Solarize", 0.6, 8)),
                (("Equalize", 0.6, None), ("Posterize", 0.4, 6)),
                (("Rotate", 0.8, 8), ("Color", 0.4, 0)),
                (("Rotate", 0.4, 9), ("Equalize", 0.6, None)),
                (("Equalize", 0.0, None), ("Equalize", 0.8, None)),
                (("Invert", 0.6, None), ("Equalize", 1.0, None)),
                (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
                (("Rotate", 0.8, 8), ("Color", 1.0, 2)),
                (("Color", 0.8, 8), ("Solarize", 0.8, 7)),
                (("Sharpness", 0.4, 7), ("Invert", 0.6, None)),
                (("ShearX", 0.6, 5), ("Equalize", 1.0, None)),
                (("Color", 0.4, 0), ("Equalize", 0.6, None)),
                (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
                (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
                (("Invert", 0.6, None), ("Equalize", 1.0, None)),
                (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
                (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
            ]
            # elif policy.lower() == 'cifar10':
            #     policy = transforms.AutoAugmentPolicy.CIFAR10
            # elif policy.lower() == 'svhn':
            #     policy = transforms.AutoAugmentPolicy.SVHN
        else:
            raise ValueError(f"Unsupported policy: {policy}. Available: 'imagenet', 'cifar10', 'svhn'")
        
        # 创建AutoAugment增强器
        self.augmenter = transforms.AutoAugment(
            policy=policy,
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


def visualize_augmentation(augmenter, save_path='augmentation_demo.jpg'):
    """可视化增强效果：创建原始和增强图像并拼接展示"""
    # 创建测试图像
    circle, square, word = create_test_images()
    
    # 应用增强
    augmented_circle = augmenter(circle)
    augmented_square = augmenter(square)
    augmented_word = augmenter(word)
    
    # 创建拼接图像
    total_width = 620  # 3张图，每张200宽，间隔10
    total_height = 420  # 2行，每行200高，间隔20
    
    combined_img = Image.new('RGB', (total_width, total_height), color='white')
    
    # 第一行：原始图像
    combined_img.paste(circle, (10, 10))
    combined_img.paste(square, (220, 10))
    combined_img.paste(word, (430, 10))
    
    # 第二行：增强图像
    combined_img.paste(augmented_circle, (10, 230))  # 修正位置
    combined_img.paste(augmented_square, (220, 230))
    combined_img.paste(augmented_word, (430, 230))
    
    # 添加标题
    draw = ImageDraw.Draw(combined_img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    draw.text((90, 220), "Original", fill='black', font=font)
    draw.text((90, 440), "Augmented", fill='black', font=font)  # 修正位置
    
    # 保存图像
    combined_img.save(save_path)
    print(f"增强效果对比图已保存至: {save_path}")


if __name__ == "__main__":
    # 创建CIFAR10预训练策略的增强器
    augmenter = AutoAugmentWrapper(policy='cifar10', fill=128)
    
    # 可视化增强效果
    visualize_augmentation(augmenter)