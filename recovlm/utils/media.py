import os
import cv2
import base64

# TODO: 640，480
def resize(image):
    """
    调整图片大小以适应指定的尺寸。
    参数:
        image (numpy.ndarray): 输入的图片，格式为numpy数组。
    返回:
        numpy.ndarray: 调整大小后的图片。
    """
    # 获取图片的原始高度和宽度
    height, width = image.shape[:2]
    # 根据图片的宽高比确定目标尺寸
    if height < width:
        target_height, target_width = 480, 640
    else:
        target_height, target_width = 640, 480
    # 如果图片尺寸已经小于或等于目标尺寸，则直接返回原图片
    if height <= target_height and width <= target_width:
        return image
    # 计算新的高度和宽度，保持图片的宽高比
    if height / target_height < width / target_width:
        new_width = target_width
        new_height = int(height * (new_width / width))
    else:
        new_height = target_height
        new_width = int(width * (new_height / height))
    # 调整图片大小
    return cv2.resize(image, (new_width, new_height))
  
# 定义方法将指定路径图片转为Base64编码
def encode_image(image_path):
  """
  将指定路径的图片进行编码
  参数:
      image_path (str): 图片文件的路径
  返回:
      str: 编码后的图片字符串
  """
  # 读取图片
  image = cv2.imread(image_path)
  # 调整图片大小
  image_resized = resize(image)
  # 将图片编码为JPEG格式
  _, encoded_image = cv2.imencode(".jpg", image_resized)
  # 将编码后的图片转换为Base64字符串
  return base64.b64encode(encoded_image).decode("utf-8")

def get_pid_folder(pid, output_dir):
  """
  根据PID获取视频路径
  参数:
      pid (str): 视频的PID
  返回:
      str: 视频路径
  """
  folder = str(int(pid[-4:]))
  os.makedirs(output_dir / folder, exist_ok=True)
  return output_dir / folder