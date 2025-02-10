import os
from PIL import Image
import io
import time

def resize_image(img, max_size):
    # 获取图片的宽度和高度
    width, height = img.size
    
    # 计算缩放比例
    if width > height:
        ratio = max_size / width
    else:
        ratio = max_size / height
    
    # 计算新的宽度和高度
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    
    # 缩放图片
    img = img.resize((new_width, new_height))
    return img

def read_image(img_path, max_size=0):
    try:
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            if max_size > 0 and max(img.size) > max_size:
                img = resize_image(img, max_size)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()
            return img_bytes
    except Exception as e:
        print(f"Skipped {filename}, err_msg={e}")
        return None

if __name__ == "__main__":
    from client import GPT4oClient
    directory = '/llm_reco_ssd/zhangzixing/XFUND_img'
    max_size = 1080
    gpt4o_client = GPT4oClient()
    prompt = "请深呼吸并一步步仔细观察这张图片，针对这张图片生成html代码，要求尽可能还原图片中的格式，忽略图中的条形码。注意，只输出html代码，不要包含其他文字。"

    ans = 0
    ts = time.time()

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            img_bytes = read_image(file_path, max_size)
            if img_bytes is not None:
                ans += 1
                answer = gpt4o_client.chat(prompt, [img_bytes])
                print(answer)
                if ans > 10:
                    break

    print(time.time() - ts)