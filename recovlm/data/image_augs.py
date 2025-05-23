from torchvision import transforms
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageEnhance, ImageOps  # 确保导入ImageOps

from PIL import Image
import torch
import random
import math

class CustomNoise:
    """自定义高斯噪声 (PIL图像版)"""
    def __init__(self, intensity=10):
        self.intensity = intensity  # 0-255范围

    def __call__(self, img):
        if isinstance(img, Image.Image):
            tensor = transforms.functional.to_tensor(img)
        else:
            tensor = img
        
        noise = torch.randn_like(tensor) * (self.intensity / 255.0)
        noisy_tensor = torch.clamp(tensor + noise, 0, 1)
        
        return transforms.functional.to_pil_image(noisy_tensor)

class AutoAugmentWrapper:
    """
    严格适配OCR策略的AutoAugment实现
    
    参数:
        policy: 策略名称 ('grounding_ocr_imagenet' 或 None)
        interpolation: 插值方法 (默认 BILINEAR)
        fill: 填充颜色
    """
    def __init__(self, 
                 policy: str = None,
                 interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BILINEAR,
                 fill: int = None):
        print("Using custom auto augment wrapper with policy {}".format(policy))
        if policy is None: 
            self.policy = None
            return
        self.policy_name = policy.lower()
        self.interpolation = interpolation
        self.fill = fill
        self.policy = self._parse_policy()

    def _parse_policy(self):
        # 策略原始定义
        if self.policy_name == 'grounding_ocr_imagenet':
            # 降低颜色相关操作的强度
            policy = [
                # 亮度/对比度调整（降低强度）
                (("AutoContrast", 0., None), ("Brightness", 0.8, 3),),  # 原8→3
                (("Contrast", 1, 3), ("Equalize", 0., None)),        # 原7→3
                (("AutoContrast", 0.0, None),  ("Sharpness", 0.4, 9),),   # 锐度不影响颜色
                (("Brightness", 0.6, 3), ("Contrast", 0.6, 3)),         # 原6→3
                
                # 颜色空间变换（降低强度和概率）
                # (("Solarize", 0.05, 1), ("AutoContrast", 0.7, None),),    # 原0.3,5→0.1,2
                (("Posterize", 0.3, 7), ("Equalize", 0., None)),       # 原0.3,6→0.1,7（保留更多位）
                (("Color", 0.2, 2), ("Contrast", 0.8, 3)),              # 原0.5,4→0.2,2
                
                # 几何变换（不影响颜色）
                (("Rotate", 0.6, 5), ("AutoContrast", 0.0, None)),
                (("ShearX", 0.8, 4), ("Equalize", 0., None)),
                (("ShearY", 0.8, 4), ("Equalize", 0., None)),

                # 噪声模糊（不影响颜色）
                (("GaussianBlur", 0.4, 1), ("AutoContrast", 0.0, None)
                 ),
                (("Noise", 0.8, 10), ("Equalize", 0., None)),
                
                # 直方图均衡（保留但降低概率）
                #(("Equalize", 0., None), ("Equalize", 0., None)),     # 原0.8→0.5
                # (("Equalize", 0., None), ("AutoContrast", 0.0, None)
                #  ), # 原0.6→0.4
                
                # 其他组合（降低颜色强度）
                (("Color", 0.2, 2), ("Contrast", 0.9, 3)),              # 原0.4,3→0.2,2
                (("Solarize", 0.0, 2), ("Sharpness", 0.9, 8),),          # 原0.2,6→0.1,2
                (("Posterize", 0.1, 7), ("Brightness", 0.6, 3)),        # 原0.2,7→0.1,7
            ]
        elif self.policy_name == 'grounding_ocr_imagenet2':
            # 降低颜色相关操作的强度
            policy = [
                # 亮度/对比度调整（降低强度）
                (("AutoContrast", 0., None), ("Brightness", 0.8, 5),),  # 原8→3
                (("Contrast", 1, 5), ("Equalize", 0., None)),        # 原7→3
                (("AutoContrast", 0.0, None),  ("Sharpness", 0.9, 9),),   # 锐度不影响颜色
                (("Brightness", 0.8, 5), ("Contrast", 0.8, 5)),         # 原6→3
                
                # 颜色空间变换（降低强度和概率）
                # (("Solarize", 0.05, 1), ("AutoContrast", 0.7, None),),    # 原0.3,5→0.1,2
                (("Posterize", 0.5, 7), ("Equalize", 0., None)),       # 原0.3,6→0.1,7（保留更多位）
                (("Color", 0.5, 4), ("Contrast", 0.8, 5)),              # 原0.5,4→0.2,2
                
                # 几何变换（不影响颜色）
                (("Rotate", 0.9, 8), ("AutoContrast", 0.0, None)),
                (("ShearX", 0.9, 6), ("Equalize", 0., None)),
                (("ShearY", 0.9, 6), ("Equalize", 0., None)),

                # 噪声模糊（不影响颜色）
                # (("GaussianBlur", 0.6, 1), ("AutoContrast", 0.0, None)
                #  ),
                # (("Noise", 0.8, 10), ("Equalize", 0., None)),
                
                # 直方图均衡（保留但降低概率）
                #(("Equalize", 0., None), ("Equalize", 0., None)),     # 原0.8→0.5
                # (("Equalize", 0., None), ("AutoContrast", 0.0, None)
                #  ), # 原0.6→0.4
                
                # 其他组合（降低颜色强度）
                (("Color", 0.5, 4), ("Contrast", 0.9, 4)),              # 原0.4,3→0.2,2
                # (("Solarize", 0.0, 2), ("Sharpness", 0.9, 8),),          # 原0.2,6→0.1,2
                (("Posterize", 0.2, 7), ("Brightness", 0.8, 5)),        # 原0.2,7→0.1,7
            ]
        else:
            raise ValueError(f"Unsupported policy: {self.policy_name}")

        # 转换为PyTorch transforms
        aug_policy = []
        for sub_policy in policy:
            transforms_list = []
            for op in sub_policy:
                print(op)
                t = self._create_operation(
                    name=op[0],
                    p=op[1],
                    magnitude=op[2]
                )
                if t is not None:
                    transforms_list.append(t)
            if transforms_list:
                aug_policy.append(transforms.RandomOrder(transforms_list))
        
        return transforms.RandomChoice(aug_policy)

    def _create_operation(self, name, p, magnitude):
        """将策略参数映射到具体操作"""
        if p == 0:  # 概率为0时跳过
            return None

        # 公共参数
        kw = {"interpolation": self.interpolation}
        if self.fill is not None:
            kw["fill"] = self.fill
        else:
            import numpy as np
            kw["fill"] = np.random.randint(0, 255)

        # 操作映射字典
        op_map = {
            "AutoContrast": lambda: transforms.RandomAutocontrast(p=1),
            "Equalize": lambda: transforms.RandomEqualize(p=1),
            "Brightness": lambda: transforms.RandomApply(
                [transforms.ColorJitter(brightness=(1-(magnitude/10), 1+(magnitude/10)))], p=p),
            "Contrast": lambda: transforms.RandomApply(
                [transforms.ColorJitter(contrast=(1-(magnitude/10), 1+(magnitude/10)))], p=p),
            "Sharpness": lambda: transforms.RandomAdjustSharpness(
                sharpness_factor=1 + magnitude/10, p=p),
            "Solarize": lambda: transforms.RandomSolarize(
                threshold=int(255*(1 - magnitude/10)), p=p),
            "Posterize": lambda: transforms.RandomPosterize(
                bits=8 - magnitude, p=p),
            "Color": lambda: transforms.RandomApply(
                [transforms.ColorJitter(saturation=(1-(magnitude/10), 1+(magnitude/10)))], p=p),
            "Rotate": lambda: transforms.RandomApply(
                [transforms.RandomRotation(degrees=(-magnitude, magnitude), **kw)], p=p),
            "ShearX": lambda: transforms.RandomApply(
                [transforms.RandomAffine(degrees=0, shear=(-magnitude, magnitude, 0, 0), **kw)], p=p),
            "ShearY": lambda: transforms.RandomApply(
                [transforms.RandomAffine(degrees=0, shear=(0,0,-magnitude, magnitude), **kw)], p=p),
            "GaussianBlur": lambda: transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=max(3, magnitude*2+1))], p=p),
            "Noise": lambda: transforms.RandomApply(
                [CustomNoise(intensity=magnitude)], p=p)
        }

        if name not in op_map:
            raise ValueError(f"Unsupported operation: {name}")

        return op_map[name]()

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.policy is None:
            return img
        try:
            return self.policy(img)
        except Exception as e:
            import numpy as np
            if np.random.rand() < 0.01:
                print(f"Report_ratio=0.01. Failed to apply augmentation for {img}: {e}")
            return img
            # raise ValueError(f"Failed to apply augmentation for {img}: {e}")




def create_test_images(size=(200, 200), font_size=80):
    """创建测试图像，增加font_size参数控制文字大小"""
    circle_img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(circle_img)
    draw.ellipse((50, 50, 150, 150), fill='blue')
    
    square_img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(square_img)
    draw.rectangle((50, 50, 150, 150), fill='red')
    
    word_img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(word_img)
    try:
        font = ImageFont.truetype("Arial.ttf", font_size)  # 增加字体大小参数
    except IOError:
        font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), "AI", font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    draw.text((x, y), "AI", fill='green', font=font)
    
    return circle_img, square_img, word_img

def create_random_color_words(num=60, size=(200, 200), text="AI", font_size=80):
    """生成随机颜色单词，增加font_size参数"""
    images = []
    for _ in range(num):
        r, g, b = random.randint(0, 200), random.randint(0, 200), random.randint(0, 200)
        while r + g + b > 550:
            r, g, b = random.randint(0, 200), random.randint(0, 200), random.randint(0, 200)
        
        img = Image.new('RGB', size, color='white')
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)  # 使用更大的字体
        except IOError:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (size[0] - text_width) // 2
        y = (size[1] - text_height) // 2
        draw.text((x, y), text, fill=(r, g, b), font=font)
        images.append(img)
    return images

def visualize_word_augmentation(augmenter, words, save_prefix='word_augmentation'):
    """可视化函数保持不变"""
    grid_rows = 6
    grid_cols = 10
    img_size = words[0].size
    spacing = 10
    
    total_width = grid_cols * img_size[0] + (grid_cols - 1) * spacing
    total_height = grid_rows * img_size[1] + (grid_rows - 1) * spacing
    
    original_grid = Image.new('RGB', (total_width, total_height), color='white')
    augmented_grid = Image.new('RGB', (total_width, total_height), color='white')
    
    for i in range(grid_rows * grid_cols):
        img = words[i]
        x = (img_size[0] + spacing) * (i % grid_cols)
        y = (img_size[1] + spacing) * (i // grid_cols)
        
        original_grid.paste(img, (x, y))
        img = img.convert("RGB")
        augmented_img = augmenter(img.copy())
        augmented_grid.paste(augmented_img, (x, y))
    
    original_grid.save(f"{save_prefix}_original.jpg")
    augmented_grid.save(f"{save_prefix}_augmented.jpg")
    print(f"图像已保存：{save_prefix}_original.jpg 和 {save_prefix}_augmented.jpg")



from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

def visualize_single_ops(img: Image.Image, save_path: str, spacing: int = 10):
    """
    对每个数据增强操作单独应用并可视化结果
    
    参数:
        img: 输入的PIL图像（需包含"hello"文字）
        save_path: 结果保存路径
        spacing: 图片间距（像素）
    """
    # 定义操作信息（名称，对应强度值）
    op_info = [
        ("AutoContrast", None),
        ("Equalize", None),
        ("Brightness", 3),
        ("Contrast", 2),
        ("Sharpness", 9),
        # ("Solarize", 1),
        ("Posterize", 3),
        ("Color", 2),
        ("Rotate", 5),
        ("ShearX", 5),
        ("ShearY", 5),
        ("GaussianBlur", 1),
        ("Noise", 10),
    ]
    
    # 公共变换参数
    transform_kwargs = {
        "interpolation": InterpolationMode.BILINEAR,
        "fill": 255  # 填充颜色为白色
    }
    
    # 创建变换列表
    transforms_list = []
    for name, mag in op_info:
        if name == "AutoContrast":
            t = transforms.RandomAutocontrast(p=1)
        elif name == "Equalize":
            t = transforms.RandomEqualize(p=1)
        elif name == "Brightness":
            t = transforms.ColorJitter(brightness=(1 - mag/10, 1 + mag/10))
        elif name == "Contrast":
            t = transforms.ColorJitter(contrast=(1 - mag/10, 1 + mag/10))
        elif name == "Sharpness":
            t = transforms.RandomAdjustSharpness(sharpness_factor=1 + mag/10, p=1)
        elif name == "Solarize":
            t = transforms.RandomSolarize(threshold=int(255*(1 - mag/10)), p=1)
        elif name == "Posterize":
            t = transforms.RandomPosterize(bits=8 - mag, p=1)
        elif name == "Color":
            t = transforms.ColorJitter(saturation=(1 - mag/10, 1 + mag/10))
        elif name == "Rotate":
            t = transforms.RandomRotation(degrees=(-mag, mag), **transform_kwargs)
        elif name == "ShearX":
            t = transforms.RandomAffine(degrees=0, shear=(-mag, mag, 0, 0), **transform_kwargs)
        elif name == "ShearY":
            t = transforms.RandomAffine(degrees=0, shear=(0, 0, -mag, mag), **transform_kwargs)
        elif name == "GaussianBlur":
            t = transforms.GaussianBlur(kernel_size=max(3, mag*2+1))
        elif name == "Noise":
            t = CustomNoise(intensity=mag)
        else:
            raise ValueError(f"不支持的操作: {name}")
        
        transforms_list.append((name, t))
    
    # 图像基础信息
    img_size = img.size
    total_ops = len(transforms_list)
    rows = 6  # 每个操作生成6张图片
    font = ImageFont.truetype("arial.ttf", 80) 

    # 计算网格尺寸
    grid_width = total_ops * (img_size[0] + spacing) - spacing
    grid_height = rows * (img_size[1] + spacing) - spacing + 20  # 预留标签空间
    grid = Image.new('RGB', (grid_width, grid_height), 'white')
    grid_draw = ImageDraw.Draw(grid)  # 用于在网格上绘制的draw对象
    
    # 绘制每个操作的变换结果
    for col, (op_name, transform) in enumerate(transforms_list):
        x = col * (img_size[0] + spacing)
        
        # 测量文本尺寸（使用临时draw对象）
        bbox = grid_draw.textbbox((0, 0), op_name, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        # 在网格顶部绘制操作名称
        grid_draw.text(
            (x + (img_size[0] - text_w) // 2, 5),  # 居中显示
            op_name,
            font=ImageFont.truetype("arial.ttf", 20) ,
            fill='black'
        )
        
        # 生成并粘贴变换结果
        for row in range(rows):
            y = 20 + row * (img_size[1] + spacing)  # 跳过标签区域
            transformed_img = transform(img.copy())
            grid.paste(transformed_img, (x, y))
    
    # 保存结果
    grid.save(save_path)
    print(f"可视化结果已保存至: {save_path}")


def vis_single():
    # 生成测试图像（需包含"hello"文字）
    test_img = Image.new('RGB', (200, 200), 'white')
    draw = ImageDraw.Draw(test_img)
    font = ImageFont.truetype("arial.ttf", 30)

    # 绘制黑色"hello"
    text = "hello"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # 黑色文字位置（左侧居中）
    x_black = (50 - text_width) // 2
    y_black = (200 - text_height) // 2
    draw.text((x_black, y_black), text, fill=(0, 0, 0), font=font)  # 黑色

    # 红色文字位置（右侧居中，间隔20像素）
    x_red = x_black + text_width + 5
    draw.text((x_red, y_black), text, fill=(255, 0, 0), font=font)  # 红色
    
    # 可视化单个操作效果
    visualize_single_ops(test_img, "single_op_visualization.jpg")



def debug_aug():
    # vis_single(); exit()
    augmenter = AutoAugmentWrapper(policy='grounding_ocr_imagenet', fill=255)
    
    # 生成大字体的测试单词（字体大小设为80）
    test_words = create_random_color_words(num=60, text="hello", font_size=80)
    
    visualize_word_augmentation(augmenter, test_words, save_prefix='ocr_word_augment')


def debug_shape():
    # 生成测试图像（需包含"hello"文字）
    augmenter = AutoAugmentWrapper(policy='grounding_ocr_imagenet', fill=255)
    for h in range(1, 100):
        for w in range(1, 100):
            try:
                test_img = Image.new('RGB', (h, w), 'white')
                augmenter(test_img)
            except Exception as e:
                print(test_img)
                print(e); continue


if __name__ == "__main__":
    debug_shape()
