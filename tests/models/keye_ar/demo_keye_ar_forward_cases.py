"""
多模态模型测试框架
支持添加多个测试用例并批量运行
"""

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["nosp"] = 'true'
from tqdm import tqdm
import json
import csv
import IPython
import sys
from pathlib import Path
import torch
import re
import time
from PIL import Image, ImageDraw
from transformers import AutoProcessor
import argparse  # 添加argparse模块导入

# 设置环境和路径
current_script = Path(__file__).resolve()

# 导入模型相关模块
from muse.models.keye_ar import KeyeARModel
from keye_vl_utils import process_vision_info


class VLMTestFramework:
    """
    多模态语言模型测试框架
    负责模型和处理器的初始化，以及管理和运行测试用例
    """
    def __init__(self, device=1, args=None):
        """
        初始化测试框架
        
        Args:
            device: 使用的设备ID
        """
        self.device = device
        self.model = None
        self.processor = None
        self.test_cases = []
        self.args = args
        # self._setup_torch_precision()
    
    def _setup_torch_precision(self):
        """设置PyTorch打印选项以提高可读性"""
        torch.set_printoptions(
            threshold=float('inf'),
            edgeitems=1000,
            linewidth=200,
            sci_mode=False,
            precision=3)
    
    def initialize_model(self, base_model_dir, model_dir, step, auto_prepare=True):
        """
        初始化模型和处理器
        
        Args:
            base_model_dir: 基础模型目录
            model_dir: 模型目录
            step: 模型步数
            auto_prepare: 是否自动准备模型检查点
        """
        output_model_dir = f"{model_dir}/step{step}/global_step{step}/converted"
        
        # 检查并准备模型检查点
        if auto_prepare and not os.path.exists(output_model_dir + "/model.safetensors.index.json"):
            print(f"Preparing model checkpoint: {output_model_dir}")
            os.system(
                f"cd /llm_reco/lingzhixin/recovlm_vlmevalkit/vlmevalkit; "
                f"PYTHONPATH=. python3 dcp2torch_save.py "
                f"--dcp_path {model_dir} --step step{step} --base_model_path {base_model_dir}"
            )
        
        # 加载模型
        print(f"Loading model from: {output_model_dir}")
        self.model = KeyeForConditionalGeneration.from_pretrained(
            output_model_dir, 
            _attn_implementation="flash_attention_2", 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True
        )
        self.model.config.output_one_token = self.model.output_one_token = False
        self.model.token_head.use_flash_attn = True
        self.model = self.model.to(self.device).bfloat16()
        
        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(
            output_model_dir, 
            trust_remote_code=True
        )
        
        print("Model and processor initialized successfully!")
    
    def process_message(self, messages, add_generation_prompt=True, padding=False):
        """
        处理消息并返回模型输入
        
        Args:
            messages: 消息列表
            add_generation_prompt: 是否添加生成提示
            padding: 是否进行填充
            
        Returns:
            处理后的模型输入
        """
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=add_generation_prompt
        )
        
        # print(f"text={text}")
        
        image_inputs, video_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=padding,
            truncation=False,
            return_tensors="pt",
        ).to(self.device)
        return inputs
    
    def add_test_case(self, name, test_func):
        """
        添加测试用例
        
        Args:
            name: 测试用例名称
            test_func: 测试函数，接收框架实例作为参数
        """
        self.test_cases.append((name, test_func))
        print(f"Added test case: {name}")
    
    def run_all_tests(self):
        """
        运行所有测试用例
        """
        if not self.model or not self.processor:
            raise ValueError("Model and processor not initialized. Call initialize_model first.")
        
        print(f"\n{'=' * 60}")
        print(f"Running {len(self.test_cases)} test cases...")
        print(f"{'=' * 60}\n")
        
        for i, (name, test_func) in tqdm(enumerate(self.test_cases, 1), total=len(self.test_cases)):
            print(f"\n{'=' * 60}")
            print(f"Test Case {i}/{len(self.test_cases)}: {name}")
            print(f"{'=' * 60}")
            try:
                test_func(self)
                print(f"Test '{name}' completed successfully!")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error in test '{name}': {str(e)}")
        
        print(f"\n{'=' * 60}")
        print("All tests completed!")
        print(f"{'=' * 60}")


def generate_circle_image(size=(100, 100), fill_color=(0, 0, 0), outline_color=(255, 255, 255), outline_width=5):
    """
    生成一个包含一个圆的 PIL Image 对象，用于测试。
    
    Args:
        size: 图像的大小，默认为 (100, 100)
        fill_color: 圆的填充颜色，默认为黑色 (0, 0, 0)
        outline_color: 圆的轮廓颜色，默认为白色 (255, 255, 255)
        outline_width: 圆的轮廓宽度，默认为 5
        
    Returns:
        生成的 PIL Image 对象
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


# 测试用例函数
def test_forward_image_tokens(framework):
    """测试前向图像tokens"""
    print("Testing forward_image_tokens...")
    inputs = framework.process_message([{
            "role": "user",
            "content": [
                {"type": "image", "image": "/mmu_mllm_hdd_2/zhouyang12/media/images/09/76/92/96/e7/09769296e728cb3fbc3c0ecf4f336586.jpg"},
            ],
        }])
    print(inputs)
    labels = framework.model.forward_image_tokens(
        **inputs
    )
    # 保存labels供后续测试使用
    framework.labels = labels
    print(f"labels=\n{labels}")
    print("forward_image_tokens test completed.")

def test_image_qa(framework):
    """测试图像QA"""
    print("Testing image QA...")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": "/mmu_mllm_hdd_2/penghao03/Bench/images/OCRBench/186.jpg"},
            {"type": "text", "text": "What's in the image?"}
        ],
    }]
    
    inputs = framework.process_message(messages)
    output_ids = framework.model.generate_multimodal(**inputs, top_k=1, max_new_tokens=400)

    print(f"messages={messages}")
    print("inputs=", inputs["input_ids"])
    
    content = framework.processor.decode(output_ids[0,inputs["input_ids"].shape[1]:,0].long().tolist())
    print(f"content={content}")

def test_text_qa(framework):
    """测试文本QA"""
    print("Testing text QA...")
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "How are you?"}
        ],
    }]
    
    inputs = framework.process_message(messages)
    output_ids = framework.model.generate_multimodal(**inputs, top_k=1, max_new_tokens=400)
    
    print(f"messages={messages}")
    print("inputs=", inputs["input_ids"])
    
    content = framework.processor.decode(output_ids[0,inputs["input_ids"].shape[1]:,0].long().tolist())
    print(f"content={content}")

def test_custom_input_ids(framework):
    """测试自定义input_ids"""
    print("Testing custom input_ids...")
    custom_input_ids = [
        [151644, 151681, 151681, 151681, 151681, 151681, 151681, 151681, 151681],
        [  8948, 151681, 151681, 151681, 151681, 151681, 151681, 151681, 151681],
        [   198, 151681, 151681, 151681, 151681, 151681, 151681, 151681, 151681],
        [  2610, 151681, 151681, 151681, 151681, 151681, 151681, 151681, 151681],
        [   525, 151681, 151681, 151681, 151681, 151681, 151681, 151681, 151681],
        [   264, 151681, 151681, 151681, 151681, 151681, 151681, 151681, 151681],
        [ 10950, 151681, 151681, 151681, 151681, 151681, 151681, 151681, 151681],
        [ 17847, 151681, 151681, 151681, 151681, 151681, 151681, 151681, 151681],
        [    13, 151681, 151681, 151681, 151681, 151681, 151681, 151681, 151681],
        [151645, 151681, 151681, 151681, 151681, 151681, 151681, 151681, 151681],
        [   198, 151681, 151681, 151681, 151681, 151681, 151681, 151681, 151681],
        [151652, 151681, 151681, 151681, 151681, 151681, 151681, 151681, 151681]
    ]
    
    # 处理labels并连接
    if hasattr(framework, 'labels'):
        labels = torch.nn.functional.pad(framework.labels, (0, 1, 0, 0), value=151681)
        input_ids = torch.cat([torch.tensor(custom_input_ids).to(labels), labels[:3]], dim=0)
        print(f"input_ids={input_ids}")
        
        inputs = {"input_ids": input_ids[None].long().to(framework.device),}
        output_ids = framework.model.generate_multimodal(**inputs, top_k=1, max_new_tokens=600)
        
        print(f"output_ids={output_ids.shape}")
        print(f"新生成的Token数: {output_ids.shape[1]}") 
        print(f"output_ids=\n{output_ids}")
        
        output_ids = output_ids[0,inputs["input_ids"].shape[1]:]
        content = framework.processor.decode(output_ids[:,0].long().tolist())
        print(f"生成内容: {content}")
    else:
        print("Warning: labels not available. Skipping input_ids concatenation.")

def test_nemotron(framework):
    """测试Nemotron_CC_v2_part0"""
    print("Testing Nemotron_CC_v2_part0...")
    nemotron_input_ids = [13218, 26423, 271, 13218, 26423, 271, 13218, 26423, 702, 264, 17011, 369, 1181, 11050, 323, 73901, 5426, 3922, 13, 6771, 752, 1896, 498, 389, 264, 11618, 1526, 279, 29691, 92900, 553, 13569, 382, 13218, 1038, 1163, 1735, 323, 2463, 55889, 22356, 1447, 50, 10098, 320, 30124, 896, 3135, 10559, 46307, 33, 29560, 1648, 29121, 748, 5313, 39006, 2813, 11, 328, 10098]
    input_ids_tensor = torch.tensor(nemotron_input_ids).to(framework.device)
    inputs = {"input_ids": input_ids_tensor[None].long().to(framework.device),}
    output_ids = framework.model.generate_multimodal(**inputs, top_k=1, max_new_tokens=400)
    print(f"input=\n{framework.processor.decode(input_ids_tensor.long().tolist())}")
    output_ids = output_ids[0,inputs["input_ids"].shape[1]:]
    print(f"output_ids=\n{output_ids}")
    content = framework.processor.decode(output_ids[:,0].long().tolist())
    print(f"生成内容: {content}")

def test_omnicorpus(framework):
    """测试OmniCorpus"""
    print("Testing OmniCorpus...")
    image0 = "/mmu_mllm_hdd_2/zhouyang12/media/images/70/c2/9d/ca/19/70c29dca197cf75da9d4f83f3e0a0d82.jpg"
    
    omni_messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image0, "min_pixels": 48*28**2, "max_pixels": 64*28**2},
        ],
    }]
    
    # 对于特殊格式的输入，使用自定义处理
    text = "<|vision_start|><|image_pad|><|vision_end|>"
    image_inputs, video_inputs = process_vision_info(omni_messages)
    
    inputs = framework.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        truncation=False,
        return_tensors="pt",
    ).to(framework.device)
    
    output_ids = framework.model.generate_multimodal(**inputs, top_k=1, max_new_tokens=400)
    print(f"input=\n{text}")
    output_ids = output_ids[0,inputs["input_ids"].shape[1]:]
    print(f"output_ids=\n{output_ids}")
    content = framework.processor.decode(output_ids[:,0].long().tolist())
    print(f"生成内容: {content}")


def test_image_generation(framework):
    """测试OmniCorpus"""
    print("Testing Image Generation...")
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Generate an image of a cat"},
        ],
    }]
    inputs = framework.process_message(messages)
    output_ids = framework.model.generate_multimodal(**inputs, top_k=1, max_new_tokens=450)
    print(f"input=\n{messages}")
    output_ids = output_ids[0,inputs["input_ids"].shape[1]:]
    print(f"output_ids=\n{output_ids}")
    content = framework.processor.decode(output_ids[:,0].long().tolist())
    print(f"生成内容: {content}")


def _prepare_image_edit_test_data():
    """
    准备图像编辑测试数据
    
    Returns:
        dict: 包含图像路径和测试消息的字典
    """
    image = "/mmu_mllm_hdd_2/zhouyang12/media/images/b4/56/a0/33/9e/b456a0339e1dbc326020ab4372d40927.jpg"
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image, "min_pixels": 48*28**2, "max_pixels": 64*28**2},
                {"type": "text", "text": "Add a woman wearing a yellow top and black shorts."}
            ],
        }
    ]
    
    return {"image_path": image, "messages": messages}

def _process_image_edit_input(framework, messages):
    """
    处理图像编辑测试的输入消息
    
    Args:
        framework: VLM测试框架实例
        messages: 包含图像和文本消息的列表
        
    Returns:
        dict: 处理后的模型输入
    """
    return framework.process_message(messages)

def _generate_image_edit_output(framework, inputs):
    """
    生成图像编辑的输出结果
    
    Args:
        framework: VLM测试框架实例
        inputs: 模型输入数据
        
    Returns:
        tuple: 包含原始输出ID和处理后的输出ID
    """
    output_ids = framework.model.generate_multimodal(**inputs, top_k=1, max_new_tokens=400)
    processed_output_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    return output_ids, processed_output_ids

def _decode_and_print_result(framework, output_ids, processed_output_ids, messages):
    """
    解码并打印图像编辑结果
    
    Args:
        framework: VLM测试框架实例
        output_ids: 原始输出ID
        processed_output_ids: 处理后的输出ID
        messages: 原始输入消息
    """
    print(f"input=\n{messages}")
    print(f"output_ids=\n{output_ids}")
    content = framework.processor.decode(processed_output_ids[:,0].long().tolist())
    print(f"生成内容: {content}")

def test_image_edit(framework):
    """测试OmniCorpus的图像编辑功能"""
    print("Testing Image Generation...")
    
    # 准备测试数据
    test_data = _prepare_image_edit_test_data()
    messages = test_data["messages"]
    
    # 处理输入
    inputs = _process_image_edit_input(framework, messages)
    
    # 生成输出
    output_ids, processed_output_ids = _generate_image_edit_output(framework, inputs)
    
    # 解码并打印结果
    _decode_and_print_result(framework, output_ids, processed_output_ids, messages)



def test_generate_and_understanding(framework):
    print("Testing generate and understanding...")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Generate an image of a cat."}
            ],
        }
    ]
    inputs = framework.process_message(messages)
    output_ids = framework.model.generate_multimodal(**inputs.to(framework.model.device), top_k=1, max_new_tokens=450)
    output_ids = output_ids[0,inputs["input_ids"].shape[1]:]
    content = framework.processor.decode(output_ids[:,0].long().tolist())
    print(f"输入:\n{messages}")
    print(f"生成内容: {content}\n")
    print(f"output_ids=\n{output_ids}")
    print()
    input_image_ids = framework.model.extract_image_tokens(output_ids)
    print(f"input_image_ids({[x.shape for x in input_image_ids]})=\n{input_image_ids}")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<|vision_start|><|vision_end|>what's in the image? I need a detailed description."}
            ],
        }
    ]
    inputs = framework.process_message(messages)
    inputs["input_ids"] = framework.model.fill_image_tokens(inputs["input_ids"], input_image_ids)
    inputs["input_image_ids"] = torch.cat(input_image_ids, 0)
    output_ids = framework.model.generate_multimodal(**inputs.to(framework.model.device), top_k=1, max_new_tokens=450)
    output_ids = output_ids[0,inputs["input_ids"].shape[1]:]
    content = framework.processor.decode(output_ids[:,0].long().tolist())
    print(f"输入:\n{messages}")
    print(f"生成内容: {content}\n")
    print(f"output_ids=\n{output_ids}")




def test_geneval_generate_and_understanding(framework):
    print("Testing GenEval generate and understanding...")

    def get_data_by_interval(file_path="/llm_reco/lingzhixin/recovlm_data/generation_data/GenEval.tsv", interval=50):
        """
        每隔n行（interval）返回一行的question、include_class、include_count
        :param file_path: TSV文件路径
        :param interval: 间隔行数（默认每1行取1行）
        :yield: (question, include_class, include_count)
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')  # 用DictReader读取TSV
            for idx, row in enumerate(reader):
                # 跳过表头，且满足间隔条件（从第0行开始计数）
                if (idx + 1) % interval == 0:
                    yield (
                        row['question'],
                        row['tag'],
                        row['include_class']
                    )

    interval = int(os.environ.get("INTERVAL", 50))
    for question, tag, include_class in get_data_by_interval(interval=interval):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Generate an image based on the description: {question}"}
                ],
            }
        ]
        inputs = framework.process_message(messages)
        output_ids = framework.model.generate_multimodal(**inputs.to(framework.model.device), top_k=1, max_new_tokens=400)
        output_ids = output_ids[0,inputs["input_ids"].shape[1]:]
        content = framework.processor.decode(output_ids[:,0].long().tolist())
        print(f"tag={tag}, include_class={include_class}")
        print(f"输入:\n{messages}")
        print(f"生成内容: {content}\n")
        print(f"output_ids=\n{output_ids.shape}")
        print()
        input_image_ids = framework.model.extract_image_tokens(output_ids)
        print(f"input_image_ids({[x.shape for x in input_image_ids]})\n")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<|vision_start|><|vision_end|>what's in the image? I need a detailed description."}
                ],
            }
        ]
        inputs = framework.process_message(messages)
        inputs["input_ids"] = framework.model.fill_image_tokens(inputs["input_ids"], input_image_ids)
        inputs["input_image_ids"] = torch.cat(input_image_ids, 0)
        output_ids = framework.model.generate_multimodal(**inputs.to(framework.model.device), top_k=1, max_new_tokens=400)
        output_ids = output_ids[0,inputs["input_ids"].shape[1]:]
        content = framework.processor.decode(output_ids[:,0].long().tolist())
        print(f"输入:\n{messages}")
        print(f"生成内容: {content}\n\n")



def test_geneval_generate_and_understanding_v2(framework):
    print("Testing GenEval generate and understanding...")

    def get_data_by_interval(file_path="/llm_reco/lingzhixin/recovlm_data/generation_data/GenEval.tsv", interval=50):
        """
        每隔n行（interval）返回一行的question、include_class、include_count
        :param file_path: TSV文件路径
        :param interval: 间隔行数（默认每1行取1行）
        :yield: (question, include_class, include_count)
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')  # 用DictReader读取TSV
            for idx, row in enumerate(reader):
                # 跳过表头，且满足间隔条件（从第0行开始计数）
                if (idx + 1) % interval == 0:
                    yield (
                        row['question'],
                        row['tag'],
                        row['include_class']
                    )

    interval = int(os.environ.get("INTERVAL", 50))
    # Strictly follow all requirements, generate an image based on the description: {IMAGE_DESCRIPTION}. Ensure the main subject, color(s), and quantity specified in the description are 100% consistent and not altered in any way.
    # template = "Strictly follow all requirements, generate an image based on the description: {IMAGE_DESCRIPTION}. Ensure the main subject, color(s), and quantity specified in the description are 100% consistent and not altered in any way."
    template = getattr(framework.args, "gen_im_template", "{IMAGE_DESCRIPTION}")
    output_dir = getattr(framework.args, "output_dir", "")
    for index, (question, tag, include_class) in enumerate(get_data_by_interval(interval=interval)):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": template.format(IMAGE_DESCRIPTION=question).replace("..", "."),
            }
        ]
        inputs = framework.process_message(messages)
        output_ids = framework.model.generate_multimodal(**inputs.to(framework.model.device), top_k=1, max_new_tokens=400)
        output_ids = output_ids[0,inputs["input_ids"].shape[1]:]
        content = framework.processor.decode(output_ids[:,0].long().tolist())
        print(f"index={index}, tag={tag}, include_class={include_class}")
        print(f"输入:\n{messages}")
        print(f"生成内容: {content}\n")
        print(f"output_ids=\n{output_ids.shape}")
        print()
        input_image_ids = framework.model.extract_image_tokens(output_ids)
        print(f"input_image_ids({[x.shape for x in input_image_ids]})\n")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<|vision_start|><|vision_end|>what's in the image? I need a detailed description, include the main subject, color(s), number(s), background, and quantity specified in the description."}
                ],
            }
        ]
        inputs = framework.process_message(messages)
        inputs["input_ids"] = framework.model.fill_image_tokens(inputs["input_ids"], input_image_ids)
        inputs["input_image_ids"] = torch.cat(input_image_ids, 0)
        output_ids = framework.model.generate_multimodal(**inputs.to(framework.model.device), top_k=1, max_new_tokens=400)
        output_ids = output_ids[0,inputs["input_ids"].shape[1]:]
        content = framework.processor.decode(output_ids[:,0].long().tolist())
        print(f"输入:\n{messages}")
        print(f"生成内容: {content}\n\n")
        
        if output_dir != "":
            output_dir = getattr(framework.args, "output_dir")
            os.makedirs(output_dir, exist_ok=True)

            # jsonl dumping
            with open(os.path.join(output_dir, f"GenEval_v2_{index}.jsonl"), "w") as f:
                f.write(json.dumps({
                    "index": index,
                    "tag": tag,
                    "include_class": include_class,
                    "question": question,
                    "input_image_ids": input_image_ids[0].cpu().tolist(),
                    "content": content,
                }, ensure_ascii=False) + "\n")
        else:
            print(f"output_dir={output_dir}, skip")
    


def test_ketu_generate_and_understanding(framework):
    print("Testing ketu (in domain) generate and understanding...")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": """Construct a graphic representation for this content:
The image depicts a waterfall cascading down a rocky cliff face. The waterfall is the main element, with water flowing vigorously over the rugged, textured rocks. The background consists of the cliff face, which is composed of various shades of gray and brown rocks, with some patches of green vegetation. The scene is set in a natural environment, likely in a mountainous area."""}
            ],
        }
    ]
    inputs = framework.process_message(messages)
    output_ids = framework.model.generate_multimodal(**inputs.to(framework.model.device), top_k=1, max_new_tokens=450)
    output_ids = output_ids[0,inputs["input_ids"].shape[1]:]
    content = framework.processor.decode(output_ids[:,0].long().tolist())
    print(f"输入:\n{messages}")
    print(f"生成内容: {content}\n")
    print(f"output_ids=\n{output_ids}")
    print()
    input_image_ids = framework.model.extract_image_tokens(output_ids)
    print(f"input_image_ids({[x.shape for x in input_image_ids]})=\n{input_image_ids}")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<|vision_start|><|vision_end|>what's in the image? I need a detailed description."}
            ],
        }
    ]
    inputs = framework.process_message(messages)
    inputs["input_ids"] = framework.model.fill_image_tokens(inputs["input_ids"], input_image_ids)
    inputs["input_image_ids"] = torch.cat(input_image_ids, 0)
    output_ids = framework.model.generate_multimodal(**inputs.to(framework.model.device), top_k=1, max_new_tokens=400)
    output_ids = output_ids[0,inputs["input_ids"].shape[1]:]
    content = framework.processor.decode(output_ids[:,0].long().tolist())
    print(f"输入:\n{messages}")
    print(f"生成内容: {content}\n")
    print(f"output_ids=\n{output_ids}")


def test_edit_and_understanding(framework):
    """测试编辑和理解"""
    print("Testing edit and understanding...")
    for size in [100, 600]:
        print(f"Generating circle image of size {size}x{size}...")
        # 生成圆形测试图像
        image = generate_circle_image((size,size))

        # 构建第一次查询：图像内容描述请求
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "What's in the image? I need detailed description."}
                ],
            }
        ]
        # 处理消息并生成多模态输出
        inputs = framework.process_message(messages)
        output_ids = framework.model.generate_multimodal(**inputs, top_k=1, max_new_tokens=400)

        # 解码输出token为文本内容
        content = framework.processor.decode(output_ids[0,inputs["input_ids"].shape[1]:,0].long().tolist())
        print(f"输入:\n{messages}")
        # print(f"生成内容: {content}\n")
        print()

        # 构建第二次查询：图像编辑指令（将图像颜色变为红色）
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Turn the image color into red, but keep the background the same."}
                ],
            }
        ]
        # 处理编辑指令并生成输出
        inputs = framework.process_message(messages)
        output_ids = framework.model.generate_multimodal(**inputs, top_k=1, max_new_tokens=450)
        output_ids = output_ids[0,inputs["input_ids"].shape[1]:]
        content = framework.processor.decode(output_ids[:,0].long().tolist())
        print(f"输入:\n{messages}")
        print(f"生成内容: {content}\n")
        print(f"output_ids=\n{output_ids}")
        print()

        # 从输出中提取图像token用于后续分析
        input_image_ids = framework.model.extract_image_tokens(output_ids)
        print(f"input_image_ids({[x.shape for x in input_image_ids]})=\n{input_image_ids}")
        # 构建第三次查询：纯文本查询（使用特殊token标记图像位置）
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<|vision_start|><|vision_end|>what's in the image? I need a detailed description."}
                ],
            }
        ]
        # 处理纯文本查询并生成输出
        inputs = framework.process_message(messages)
        inputs["input_ids"] = framework.model.fill_image_tokens(inputs["input_ids"], input_image_ids)
        inputs["input_image_ids"] = torch.cat(input_image_ids, 0)
        output_ids = framework.model.generate_multimodal(**inputs.to(framework.model.device), top_k=1, max_new_tokens=400)
        output_ids = output_ids[0,inputs["input_ids"].shape[1]:]
        content = framework.processor.decode(output_ids[:,0].long().tolist())
        print(f"setting: size={size}x{size}")
        print(f"输入:\n{messages}")
        print(f"生成内容: {content}\n")
        # print(f"output_ids=\n{output_ids}")


def test_complex_edit_and_understanding(framework):
    """
    测试OmniCorpus
    {'uuid': '15ba24e1-25c1-4b91-bb25-b7364149e66c', 
    'metadata': '{"images_info": {"image1": {"width": 1024, "height": 1024, "format": "JPEG"}, 
    "image2": {"width": 1024, "height": 1024, "format": "JPEG"}}}', 
    'images': '{"image1": "/mmu_mllm_hdd_2/zhouyang12/media/images/c5/43/9f/53/8d/c5439f538d8e7d6ba6e75705aa36df63.jpg", 
    "image2": "/mmu_mllm_hdd_2/zhouyang12/media/images/d7/28/0c/85/fe/d7280c85feab8a674e73a1ed350d85dc.jpg"}', 
    'videos': '{}', 'source': 'Gen_ShareGPT-4o-Image_edit', 'messages': 
    '[{"role": "user", "content": 
    [{"type": "text", "text": "Substitute a coffee mug for one of the wine bottles on the table."}, 
    {"type": "image", "image": "image1"}]}, 
    {"role": "assistant", "content": [{"type": "image_gen", "image": "image2"}]}]', 
    'segments': None, 'image': None, 'video': None, 'text': None, 'label': None}
    """

    print("Testing OmniCorpus Image Edit and Understanding...")
    # image = "/mmu_mllm_hdd_2/zhouyang12/media/images/b4/56/a0/33/9e/b456a0339e1dbc326020ab4372d40927.jpg"
    image = "/mmu_mllm_hdd_2/zhouyang12/media/images/c5/43/9f/53/8d/c5439f538d8e7d6ba6e75705aa36df63.jpg"
    # 构建第一次查询：图像内容描述请求
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What's in the image? I need detailed description."},
            ],
        }
    ]
    # 处理消息并生成多模态输出
    inputs = framework.process_message(messages)
    output_ids = framework.model.generate_multimodal(**inputs, top_k=1, max_new_tokens=400)
    # 解码输出token为文本内容
    output_ids = output_ids[0,inputs["input_ids"].shape[1]:]
    content = framework.processor.decode(output_ids[:,0].long().tolist())
    print(f"输入:\n{messages}")
    print(f"生成内容: {content}\n")
    print(f"output_ids=\n{output_ids}")
    print()

    # 构建第二次查询：图像编辑指令（将图像颜色变为红色）
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Substitute a coffee mug for one of the wine bottles on the table."},
                {"type": "image", "image": image},
            ],
        }
    ]
    # 处理编辑指令并生成输出
    inputs = framework.process_message(messages)
    output_ids = framework.model.generate_multimodal(**inputs.to(framework.model.device), top_k=1, max_new_tokens=450)
    output_ids = output_ids[0,inputs["input_ids"].shape[1]:]
    content = framework.processor.decode(output_ids[:,0].long().tolist())
    print(f"输入:\n{messages}")
    print(f"生成内容: {content}\n")
    print(f"output_ids=\n{output_ids}")
    print()

    # 从输出中提取图像token用于后续分析
    input_image_ids = framework.model.extract_image_tokens(output_ids)
    print(f"input_image_ids({[x.shape for x in input_image_ids]})=\n{input_image_ids}")
    # 构建第三次查询：纯文本查询（使用特殊token标记图像位置）
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<|vision_start|><|vision_end|>what's in the image? I need a detailed description."}
            ],
        }
    ]
    # 处理纯文本查询并生成输出
    inputs = framework.process_message(messages)
    inputs["input_ids"] = framework.model.fill_image_tokens(inputs["input_ids"], input_image_ids)
    inputs["input_image_ids"] = torch.cat(input_image_ids, 0)
    output_ids = framework.model.generate_multimodal(**inputs.to(framework.model.device), top_k=1, max_new_tokens=400)
    output_ids = output_ids[0,inputs["input_ids"].shape[1]:]
    content = framework.processor.decode(output_ids[:,0].long().tolist())
    print(f"输入:\n{messages}")
    print(f"生成内容: {content}\n")
    print(f"output_ids=\n{output_ids}")



def test_text_generate_and_understanding(framework):
    print("Testing Text generate and understanding...")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Generate an image of a text 'Hello World'."}
            ],
        }
    ]
    inputs = framework.process_message(messages)
    output_ids = framework.model.generate_multimodal(**inputs.to(framework.model.device), top_k=1, max_new_tokens=450)
    output_ids = output_ids[0,inputs["input_ids"].shape[1]:]
    content = framework.processor.decode(output_ids[:,0].long().tolist())
    print(f"输入:\n{messages}")
    print(f"生成内容: {content}\n")
    print(f"output_ids=\n{output_ids}")
    print()
    input_image_ids = framework.model.extract_image_tokens(output_ids)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<|vision_start|><|vision_end|>what's in the image? I need a detailed description."}
            ],
        }
    ]
    inputs = framework.process_message(messages)
    inputs["input_ids"] = framework.model.fill_image_tokens(inputs["input_ids"], input_image_ids)
    inputs["input_image_ids"] = torch.cat(input_image_ids, 0)
    output_ids = framework.model.generate_multimodal(**inputs.to(framework.model.device), top_k=1, max_new_tokens=450)
    output_ids = output_ids[0,inputs["input_ids"].shape[1]:]
    content = framework.processor.decode(output_ids[:,0].long().tolist())
    print(f"输入:\n{messages}")
    print(f"生成内容: {content}\n")
    print(f"output_ids=\n{output_ids}")



def test_simple_obj_generate_and_understanding(framework):
    print("Testing simple object generate and understanding...")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Generate an image of a circle."}
            ],
        }
    ]
    inputs = framework.process_message(messages)
    output_ids = framework.model.generate_multimodal(**inputs.to(framework.model.device), top_k=1, max_new_tokens=450)
    output_ids = output_ids[0,inputs["input_ids"].shape[1]:]
    content = framework.processor.decode(output_ids[:,0].long().tolist())
    print(f"输入:\n{messages}")
    print(f"生成内容: {content}\n")
    print(f"output_ids=\n{output_ids}")
    print()
    input_image_ids = framework.model.extract_image_tokens(output_ids)
    print(f"input_image_ids({[x.shape for x in input_image_ids]})=\n{input_image_ids}")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<|vision_start|><|vision_end|>what's in the image? I need a detailed description."}
            ],
        }
    ]
    inputs = framework.process_message(messages)
    inputs["input_ids"] = framework.model.fill_image_tokens(inputs["input_ids"], input_image_ids)
    inputs["input_image_ids"] = torch.cat(input_image_ids, 0)
    output_ids = framework.model.generate_multimodal(**inputs.to(framework.model.device), top_k=1, max_new_tokens=450)
    output_ids = output_ids[0,inputs["input_ids"].shape[1]:]
    content = framework.processor.decode(output_ids[:,0].long().tolist())
    print(f"输入:\n{messages}")
    print(f"生成内容: {content}\n")
    print(f"output_ids=\n{output_ids}")



def monitor_and_test_new_checkpoint(base_model_dir, model_dir, device, tests, args):
    """
    实时监控目录中是否出现新的checkpoint，并在发现新checkpoint时立即运行测试
    每次处理step最大且未评估过的checkpoint
    
    Args:
        base_model_dir: 基础模型目录
        model_dir: 模型目录
        device: 使用的设备ID
    """
    print(f"开始实时监控新的checkpoint，监控目录: {model_dir}")
    
    # 正则表达式用于匹配step目录
    step_pattern = re.compile(r'step(\d+)$')
    
    # 记录已经处理过的step，避免重复处理
    processed_steps = set()
    
    try:
        while True:
            # 收集所有符合条件的step目录
            unprocessed_steps = []
            
            try:
                for item in list(os.listdir(model_dir)):
                    item_path = os.path.join(model_dir, item)

                    # 检查是否是目录且符合step格式
                    if os.path.isdir(item_path) and step_pattern.match(item):
                        # 提取step数字
                        step_match = step_pattern.match(item)
                        if step_match:
                            step = int(step_match.group(1))
                            
                            # 检查该step是否未处理过且.metadata文件存在
                            if step not in processed_steps:
                                metadata_path = os.path.join(item_path, f"global_step{step}", ".metadata")
                                if os.path.exists(metadata_path):
                                    unprocessed_steps.append(int(step))
                
                # 如果有未处理的step，选择最大的进行处理
                if unprocessed_steps:
                    max_step = max(unprocessed_steps)
                    max_step_dir = os.path.join(model_dir, f"step{max_step}")
                    
                    print(f"发现新的checkpoint: step={max_step} (最大未处理step)")
                    processed_steps.add(max_step)
                    
                    # 为最大的checkpoint创建测试框架实例
                    print(f"\n{'=' * 80}")
                    print(f"开始使用最大的新checkpoint运行测试: step={max_step}")
                    print(f"{'=' * 80}\n")
                    
                    # 初始化测试框架并运行测试
                    run_test_with_step(base_model_dir, model_dir, max_step, device, tests, args)
                    
                    print(f"\n{'=' * 80}")
                    print(f"测试完成: step={max_step}")
                    print(f"{'=' * 80}\n")
                
                else:
                    print("未发现新的checkpoint，5秒后再次检查...")
                    time.sleep(5)
                
            except Exception as e:
                print(f"监控过程中发生错误: {str(e)}")
                time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n监控已被用户中断")


def run_test_with_step(base_model_dir, model_dir, step, device, tests=None, args=None):
    """
    使用指定的step运行测试
    
    Args:
        base_model_dir: 基础模型目录
        model_dir: 模型目录
        step: 模型步数
        device: 使用的设备ID
        tests: 要运行的测试函数名列表，如果为None则运行所有测试
    """
    # 初始化测试框架
    framework = VLMTestFramework(device=device, args=args)
    
    try:
        # 初始化模型
        framework.initialize_model(base_model_dir, model_dir, step)
        
        # 创建测试函数映射表
        test_functions = {
            "forward_image_tokens": test_forward_image_tokens,
            "image_qa": test_image_qa,
            "custom_input_ids": test_custom_input_ids,
            "ketu_generate_and_understanding": test_ketu_generate_and_understanding,
            "geneval_generate_and_understanding": test_geneval_generate_and_understanding,
            "geneval_generate_and_understanding_v2": test_geneval_generate_and_understanding_v2,
            "generate_and_understanding": test_generate_and_understanding,
            "edit_and_understanding": test_edit_and_understanding,
            "complex_edit_and_understanding": test_complex_edit_and_understanding,
            "text_generate_and_understanding": test_text_generate_and_understanding,
            "simple_obj_generate_and_understanding": test_simple_obj_generate_and_understanding,
            "interactive_generate_and_understanding": test_interactive_generate_and_understanding
        }
        
        # 添加测试用例
        if tests is None:
            # 运行所有测试（除了交互式测试）
            framework.add_test_case("Forward Image Tokens", test_forward_image_tokens)
            framework.add_test_case("Image QA", test_image_qa)
            framework.add_test_case("Custom Image Input IDs", test_custom_input_ids)
            framework.add_test_case("Image Ketu Generation and Understanding", test_ketu_generate_and_understanding)
            framework.add_test_case("Generate and Understanding", test_generate_and_understanding)
            framework.add_test_case("Edit and Understanding", test_edit_and_understanding)
            framework.add_test_case("Complex Image Edit and Understanding", test_complex_edit_and_understanding)
            framework.add_test_case("Text Generation", test_text_generate_and_understanding)
            framework.add_test_case("Simple Object Generation and Understanding", test_simple_obj_generate_and_understanding)
            framework.add_test_case("GenEval Generation and Understanding", test_geneval_generate_and_understanding)
        else:
            # 只运行指定的测试
            for test_name in tests:
                if test_name in test_functions:
                    # 使用函数名作为测试用例名的一部分
                    framework.add_test_case(f"{test_name.replace('_', ' ').title()}", test_functions[test_name])
                else:
                    print(f"警告: 未知的测试函数名 '{test_name}'")

        # 运行所有添加的测试
        framework.run_all_tests()
        
    except Exception as e:
        print(f"执行测试过程中发生错误: {str(e)}")
        # 出错时继续监控，不中断整个程序



def test_interactive_generate_and_understanding(framework):
    print("Testing interactive generatation and understanding...")
    while True:
        generation_prompt = input("Generation prompt: ")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": generation_prompt}
                ],
            }
        ]
        inputs = framework.process_message(messages)
        output_ids = framework.model.generate_multimodal(**inputs.to(framework.model.device), top_k=1, max_new_tokens=450)
        output_ids = output_ids[0,inputs["input_ids"].shape[1]:]
        content = framework.processor.decode(output_ids[:,0].long().tolist())
        print(f"输入:\n{messages}")
        print(f"生成内容: {content}\n")
        print(f"output_ids=\n{output_ids}")
        print()
        input_image_ids = framework.model.extract_image_tokens(output_ids)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<|vision_start|><|vision_end|>what's in the image? I need a detailed description."}
                ],
            }
        ]
        inputs = framework.process_message(messages)
        inputs["input_ids"] = framework.model.fill_image_tokens(inputs["input_ids"], input_image_ids)
        inputs["input_image_ids"] = torch.cat(input_image_ids, 0)
        output_ids = framework.model.generate_multimodal(**inputs.to(framework.model.device), top_k=1, max_new_tokens=450)
        output_ids = output_ids[0,inputs["input_ids"].shape[1]:]
        content = framework.processor.decode(output_ids[:,0].long().tolist())
        print(f"generation_prompt=\n{generation_prompt}")
        print(f"输入:\n{messages}")
        print(f"生成内容: {content}\n")
        print(f"output_ids=\n{output_ids}")


def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='VLM 测试框架')
    
    # 添加命令行参数，使用原来的硬编码值作为默认值
    parser.add_argument('--base_model_dir', 
                        default="/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.7/run_8b_vis_stage1_transformer_tokenhead512_allavg_train_textemb_v4/step3500/global_step3500/converted/_stage2/",
                        help='基础模型目录')
    parser.add_argument('--model_dir', 
                        default="/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.7/run_8b_vis_stage3_transformer_tokenhead_allavg_v5.3_debug_from11k_stage2/",
                        help='模型目录')
    parser.add_argument('--step', 
                        default=7000,
                        type=int,
                        help='模型步数，设置为-1时实时监控新的checkpoint')
    parser.add_argument('--device', 
                        default=1,
                        type=int,
                        help='使用的设备ID')
    parser.add_argument('--tests', 
                        nargs='+',
                        help='要运行的测试函数名列表，例如: --tests interactive_generate_and_understanding image_qa')
    parser.add_argument('--gen_im_template', 
                        default="{IMAGE_DESCRIPTION}",
                        help='生成图像的模板')
    parser.add_argument('--output_dir', 
                        default="",
                        help='输出目录')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    print(f"base_model_dir={args.base_model_dir}")
    print(f"model_dir={args.model_dir}")
    print(f"step={args.step}")
    print(f"device={args.device}")
    if args.tests:
        print(f"指定的测试函数: {args.tests}")

    # 处理step参数
    if args.step == -1:
        # 实时监控模式：持续监控并为每个新checkpoint运行测试
        # 注意：在监控模式下，如果指定了tests参数，将在每次发现新checkpoint时使用这些测试函数
        print("警告: 在实时监控模式下使用--tests参数，将对每个新checkpoint运行相同的测试函数")
        
        # 修改monitor_and_test_new_checkpoint函数来传递tests参数
        # def monitored_run_test_with_step(base_model_dir, model_dir, step, device, _=None):
        #     run_test_with_step(base_model_dir, model_dir, step, device, args.tests)
        
        # 保存原始函数的引用
        # original_run_test_with_step = globals().get('run_test_with_step')
        
        # 临时替换run_test_with_step函数
        # globals()['run_test_with_step'] = monitored_run_test_with_step
        
        # try:
        monitor_and_test_new_checkpoint(args.base_model_dir, args.model_dir, args.device, args.tests, args)
        # run_test_with_step(args.base_model_dir, args.model_dir, args.device)
        # finally:
        #     # 恢复原始函数
        #     if original_run_test_with_step:
        #         globals()['run_test_with_step'] = original_run_test_with_step
    else:
        # 单次运行模式：使用指定的step运行一次测试
        print(f"使用指定的step运行测试: {args.step}")
        run_test_with_step(args.base_model_dir, args.model_dir, args.step, args.device, args.tests, args)


if __name__ == "__main__":
    main()