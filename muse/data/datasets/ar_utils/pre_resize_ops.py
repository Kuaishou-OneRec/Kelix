from PIL import Image, ImageDraw, ImageFont
import os

# 全局变量用于统计图片比例异常情况
total_images_processed = 0
exception_images_count = 0


class BadAspectRatioException(Exception):
    """自定义异常类，用于处理图片长宽比超出阈值的情况"""
    pass


def resize_and_center_crop(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """
    对PIL图像进行等比例缩放后居中裁剪至目标尺寸
    
    参数:
        image: 输入的PIL Image对象
        target_width: 目标宽度（像素）
        target_height: 目标高度（像素）
    
    返回:
        处理后的PIL Image对象
    """
    # 获取原始图像尺寸
    original_width, original_height = image.size
    
    # 计算缩放比例（保证图像能覆盖目标尺寸，取较大的缩放比例）
    scale_width = target_width / original_width
    scale_height = target_height / original_height
    scale = max(scale_width, scale_height)
    
    # 计算等比例缩放后的尺寸
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # 等比例缩放图像（兼容PIL不同版本）
    try:
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    except AttributeError:
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # 计算居中裁剪的坐标（四舍五入避免浮点误差）
    left = round((new_width - target_width) / 2)
    top = round((new_height - target_height) / 2)
    right = left + target_width
    bottom = top + target_height
    
    # 执行居中裁剪
    cropped_image = resized_image.crop((left, top, right, bottom))
    
    return cropped_image


def resize_with_aspect_ratio_check(image: Image.Image, target_width: int, target_height: int, aspect_ratio_threshold: float = 0.75, 
                                   ) -> Image.Image:
    """
    检查图片的长宽比是否在指定阈值内，如果在范围内则调整到目标尺寸，否则抛出异常
    
    参数:
        image: 输入的PIL Image对象
        aspect_ratio_threshold: 长宽比阈值（float），图片的长宽比与目标长宽比的比值必须在此阈值内
        target_width: 目标宽度（像素）
        target_height: 目标高度（像素）
    
    返回:
        处理后的PIL Image对象
    
    异常:
        ValueError: 当图片长宽比不在指定阈值范围内时抛出
    """
    global total_images_processed, exception_images_count
    
    # 更新处理图片总数
    total_images_processed += 1

    if total_images_processed % 1000 == 0:
        print(f"resize_with_aspect_ratio_check: Processed image count: {total_images_processed}, Exception images count: {exception_images_count}, target_width= {target_width}, target_height= {target_height}")

    assert target_width == target_height  # 确保目标尺寸是正方形
    assert aspect_ratio_threshold <= 1.0

    original_width, original_height = image.size
    if not (aspect_ratio_threshold < original_width / original_height <= 1.0) \
        and not (aspect_ratio_threshold < original_height / original_width <= 1.0):
        # 异常图片计数
        exception_images_count += 1
        raise BadAspectRatioException(f"图片长宽比超出阈值，请检查图片：{original_width}x{original_height}, exception_images_count/total_images_count={exception_images_count}/{total_images_processed}")
    
    image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return image


def get_exception_ratio() -> float:
    """
    获取图片比例异常的占比
    
    返回:
        异常图片比例（float）
    """
    global total_images_processed, exception_images_count
    
    if total_images_processed == 0:
        return 0.0
    
    return exception_images_count / total_images_processed


def generate_test_image(width: int, height: int, text: str = "Test Image") -> Image.Image:
    """
    生成带文字和网格的测试图像（便于观察裁剪效果）
    
    参数:
        width: 测试图宽度
        height: 测试图高度
        text: 测试图上的文字
    
    返回:
        生成的PIL Image对象
    """
    # 创建白色背景图像
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    # 绘制网格（便于观察比例和裁剪区域）
    grid_step = min(width, height) // 10  # 网格步长
    for x in range(0, width, grid_step):
        draw.line([(x, 0), (x, height)], fill=(220, 220, 220), width=1)
    for y in range(0, height, grid_step):
        draw.line([(0, y), (width, y)], fill=(220, 220, 220), width=1)
    
    # 绘制中心十字线（突出中心位置）
    draw.line([(width//2, 0), (width//2, height)], fill=(255, 0, 0), width=2)
    draw.line([(0, height//2), (width, height//2)], fill=(255, 0, 0), width=2)
    
    # 绘制文字（显示尺寸信息）
    try:
        # 尝试加载系统字体（Windows/Mac/Linux通用）
        if os.name == 'nt':  # Windows
            font = ImageFont.truetype("arial.ttf", 24)
        else:  # Mac/Linux
            font = ImageFont.truetype("/Library/Fonts/Arial.ttf" if os.uname().sysname == 'Darwin' else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        # 备用默认字体
        font = ImageFont.load_default(size=24)
    
    # 文字内容：尺寸信息
    text_content = f"{text}\n{width}x{height}"
    # 计算文字位置（居中）
    text_bbox = draw.textbbox((0, 0), text_content, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = (width - text_width) // 2
    text_y = (height - text_height) // 2
    
    # 绘制文字
    draw.text((text_x, text_y), text_content, fill=(0, 0, 255), font=font, align="center")
    
    return image

# ------------------- 主测试流程 -------------------
if __name__ == "__main__":
    # 1. 配置参数
    TEST_IMAGE_SIZE = (800, 500)  # 原始测试图尺寸（宽x高）
    TARGET_SIZE = (400, 400)      # 目标裁剪尺寸（宽x高）
    ASPECT_RATIO_THRESHOLD = 1.5   # 长宽比阈值
    OUTPUT_DIR = "test/resize_and_crop"  # 结果保存目录
    
    # 2. 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 3. 生成测试图像
    original_image = generate_test_image(
        width=TEST_IMAGE_SIZE[0],
        height=TEST_IMAGE_SIZE[1],
        text="Original"
    )
    
    # 4. 保存原始图像
    original_save_path = os.path.join(OUTPUT_DIR, "01_original_image.jpg")
    original_image.save(original_save_path, quality=95)
    print(f"原始图像已保存：{original_save_path}")
    
    # 5. 测试正常情况（长宽比在阈值内）
    try:
        processed_image = resize_with_aspect_ratio_check(
            image=original_image,
            aspect_ratio_threshold=ASPECT_RATIO_THRESHOLD,
            target_width=TARGET_SIZE[0],
            target_height=TARGET_SIZE[1]
        )
        
        # 保存处理后的图像
        processed_save_path = os.path.join(OUTPUT_DIR, f"02_processed_{TARGET_SIZE[0]}x{TARGET_SIZE[1]}.jpg")
        processed_image.save(processed_save_path, quality=95)
        print(f"处理后图像已保存：{processed_save_path}")
        print(f"当前异常图片比例：{get_exception_ratio():.2%}")
    except ValueError as e:
        print(f"处理失败：{e}")
        print(f"当前异常图片比例：{get_exception_ratio():.2%}")
    
    # 6. 测试异常情况（创建一个长宽比差异较大的图片）
    extreme_ratio_image = generate_test_image(
        width=1000,
        height=100,
        text="Extreme Ratio"
    )
    
    extreme_save_path = os.path.join(OUTPUT_DIR, "03_extreme_ratio_image.jpg")
    extreme_ratio_image.save(extreme_save_path, quality=95)
    print(f"极端比例图像已保存：{extreme_save_path}")
    
    try:
        processed_extreme = resize_with_aspect_ratio_check(
            image=extreme_ratio_image,
            aspect_ratio_threshold=ASPECT_RATIO_THRESHOLD,
            target_width=TARGET_SIZE[0],
            target_height=TARGET_SIZE[1]
        )
        processed_extreme_save_path = os.path.join(OUTPUT_DIR, f"04_processed_extreme_{TARGET_SIZE[0]}x{TARGET_SIZE[1]}.jpg")
        processed_extreme.save(processed_extreme_save_path, quality=95)
        print(f"极端比例处理后图像已保存：{processed_extreme_save_path}")
    except ValueError as e:
        print(f"极端比例处理失败（预期行为）：{e}")
    
    # 7. 打印最终的异常比例
    print(f"最终异常图片比例：{get_exception_ratio():.2%}")
    print(f"总共处理图片数：{total_images_processed}")
    print(f"异常图片数：{exception_images_count}")

    # 7. 额外：生成缩放后未裁剪的中间图像（便于理解过程）
    # 计算缩放后的尺寸
    scale_w = TARGET_SIZE[0]/TEST_IMAGE_SIZE[0]
    scale_h = TARGET_SIZE[1]/TEST_IMAGE_SIZE[1]
    scale = max(scale_w, scale_h)
    scaled_size = (int(TEST_IMAGE_SIZE[0]*scale), int(TEST_IMAGE_SIZE[1]*scale))
    # 生成缩放后的中间图
    scaled_image = generate_test_image(
        width=scaled_size[0],
        height=scaled_size[1],
        text="Scaled (before crop)"
    )
    # 绘制裁剪框（红色虚线）
    draw = ImageDraw.Draw(scaled_image)
    crop_left = round((scaled_size[0] - TARGET_SIZE[0])/2)
    crop_top = round((scaled_size[1] - TARGET_SIZE[1])/2)
    crop_right = crop_left + TARGET_SIZE[0]
    crop_bottom = crop_top + TARGET_SIZE[1]
    # 绘制裁剪区域框
    draw.rectangle(
        [crop_left, crop_top, crop_right, crop_bottom],
        outline=(255, 0, 0),
        width=3,
        # dash=(10, 5)  # 虚线样式
    )
    # 保存中间图像
    scaled_save_path = os.path.join(OUTPUT_DIR, "03_scaled_before_crop.jpg")
    scaled_image.save(scaled_save_path, quality=95)
    print(f"缩放后（裁剪前）图像已保存：{scaled_save_path}")
    
    print("\n===== 测试完成 =====")
    print(f"原始尺寸：{TEST_IMAGE_SIZE}")
    print(f"目标尺寸：{TARGET_SIZE}")
    print(f"结果目录：{os.path.abspath(OUTPUT_DIR)}")