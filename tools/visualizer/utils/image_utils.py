from PIL import Image, ImageDraw
import io
import base64
import numpy as np

def resize_base64_image(base64_str, max_pixels):
    """调整base64编码图片的大小"""
    try:
        # 解码base64字符串
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))
        
        # 获取原始尺寸
        width, height = img.size
        
        # 如果图片尺寸小于最大像素，直接返回原图
        if width * height <= max_pixels:
            return base64_str
            
        # 计算调整比例
        ratio = np.sqrt(max_pixels / (width * height))
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        # 调整图片大小
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 转回base64
        buffer = io.BytesIO()
        resized_img.save(buffer, format=img.format)
        return base64.b64encode(buffer.getvalue()).decode()
    except:
        return None

def draw_boxes_on_image(base64_img: str, boxes: list, polygons: list = None) -> str:
    """Draw boxes and polygons on a base64 encoded image"""
    try:
        img_data = base64.b64decode(base64_img)
        img = Image.open(io.BytesIO(img_data))
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        for box in boxes:
            x1, y1 = box[0][0] * width / 1000, box[0][1] * height / 1000
            x2, y2 = box[1][0] * width / 1000, box[1][1] * height / 1000
            draw.rectangle([(x1, y1), (x2, y2)], outline='red', width=4)
            
        if polygons:
            for points in polygons:
                scaled_points = [(p[0] * width / 1000, p[1] * height / 1000) for p in points]
                draw.polygon(scaled_points, outline='blue', width=4)
        
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        print(f"Error drawing shapes: {str(e)}")
        return base64_img 