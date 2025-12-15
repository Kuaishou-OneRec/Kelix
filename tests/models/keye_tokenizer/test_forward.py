import torch

from muse.models.keye_tokenizer import KeyeImageTokenizer
from muse.training.common import set_default_dtype
from PIL import Image, ImageDraw
from transformers import AutoProcessor
from keye_vl_utils import process_vision_info

MODEL_PATH = "/llm_reco_ssd/zhouyang12/models/muse/KeyeTokenizer/"

def generate_circle_image(
        size=(100, 100),
        fill_color=(0, 0, 0),
        outline_color=(255, 255, 255),
        outline_width=5):
    """
    Generate a PIL Image object containing a circle for testing.
    
    :param size: Size of the image, defaults to (100, 100)
    :param fill_color: Fill color of the circle, defaults to black (0, 0, 0)
    :param outline_color: Outline color of the circle, defaults to white (255, 255, 255)
    :param outline_width: Outline width of the circle, defaults to 5
    :return: Generated PIL Image object
    """
    # 创建一个新的图像对象
    image = Image.new('RGB', size, color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    # 计算圆的坐标（图像中心为圆心）
    x_center, y_center = size[0] // 2, size[1] // 2
    radius = min(size[0], size[1]) // 2
    # 绘制圆
    draw.ellipse([x_center - radius, y_center - radius, x_center + radius, y_center + radius],
                 fill=fill_color,
                 outline=outline_color,
                 width=outline_width)
    return image

def test_forward():

    with set_default_dtype("bfloat16"), torch.device("cuda"):
        tokenizer = KeyeImageTokenizer.from_pretrained(MODEL_PATH)

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": generate_circle_image()},
        ],
    }]

    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True  # 开启生成提示
    )

    image_inputs, _, _ = process_vision_info(messages)

    # 构建原始输入（纯有效Token，无任何Pad）
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=False,  # 强制关闭Pad，确保原始输入无多余Token
        truncation=False,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        vq_out = tokenizer(
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"]
        )

    indices = torch.stack([x_i for x_i in vq_out['indices']], 0).T 
    aligned_indices = 151936 + indices + torch.arange(8).\
        to("cuda")[None] * tokenizer.config.codebook_size // 8

    answer = torch.LongTensor(
        [[157696, 161428, 172176, 176543, 191772, 198720, 201995, 209754],
        [155872, 161428, 172475, 178883, 189182, 195380, 203676, 215606],
        [152707, 162232, 170817, 177567, 190989, 194128, 203905, 211502],
        [155872, 161428, 169345, 181246, 186400, 193588, 203676, 210683],
        [156170, 164215, 175621, 180185, 192150, 197081, 208022, 216938],
        [153068, 160320, 172737, 178954, 185988, 198887, 203676, 212428],
        [152707, 167798, 174112, 182049, 187728, 193269, 206012, 214532],
        [156614, 166166, 172568, 184363, 189079, 199763, 207621, 212810],
        [152707, 160713, 172989, 181146, 192749, 195380, 207734, 215305],
        [159674, 166192, 171391, 181686, 190231, 195874, 203905, 215606],
        [152707, 161925, 174112, 182049, 190989, 195761, 201154, 214532],
        [156614, 167079, 172568, 182049, 185202, 195874, 206012, 214334],
        [156614, 160645, 175321, 181067, 189079, 199763, 202266, 212062],
        [153068, 160645, 172568, 178561, 187447, 199763, 207621, 214334],
        [152707, 167053, 174112, 182049, 187728, 195874, 201204, 214532],
        [153068, 162232, 171391, 178561, 192513, 195874, 204058, 210162]]
    ).to("cuda")
    torch.testing.assert_close(aligned_indices, answer)
