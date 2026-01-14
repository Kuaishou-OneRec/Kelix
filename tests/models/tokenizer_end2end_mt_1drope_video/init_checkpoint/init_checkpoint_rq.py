import os
os.system("pip3 install transformers==4.53; pip3 install torchao==0.10; pip3 install easydict") # 请你在import之前安装

import sys
from pathlib import Path
import os
import shutil
os.environ["nosp"] = 'true'
current_script = Path(__file__).resolve()
# print(f"current_script.parent.parent={current_script.parent}")
# sys.path.append(str(current_script.parent))
# pip3 install transformers==4.53; pip3 install torchao==0.10 
from recovlm.models.tokenizer_end2end_mt_1drope_v2.configuration_keye import KeyeConfig
from recovlm.models.tokenizer_end2end_mt_1drope_v2.modeling_keye import KeyeForConditionalGeneration,KeyeImageTokenizer
from recovlm.models.tokenizer_end2end_mt_1drope_v2.keye_vl_utils import process_vision_info
from PIL import Image, ImageDraw
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
import random
import torch
import json
from torch import nn


# /llm_reco_ssd/chuchenglong/data/kmeans/models/RQ/8192_3/model_20251103.pt


path = '/llm_reco_ssd/zhouyang12/models/KeyeImageTokenizer_end2end_init_Keye1_5_init'
tokenizer_path = '/llm_reco_ssd/zhouyang12/models/KeyeImageTokenizer_65536_codebooksize_128hid_kmeans_init'
Keye_path = '/llm_reco_ssd/zhouyang12/models/Keye-VL-1_5-8B-Base/'

qwen3_path = '/llm_reco_ssd/zhouyang12/models/Qwen3-0.6B/'

def generate_circle_image(size=(64, 64), fill_color=(0, 0, 0), outline_color=(255, 255, 255), outline_width=5):
    """
    生成一个包含一个圆的 PIL Image 对象。

    :param size: 图像的大小，默认为 (200, 200)
    :param fill_color: 圆的填充颜色，默认为黑色 (0, 0, 0)
    :param outline_color: 圆的轮廓颜色，默认为白色 (255, 255, 255)
    :param outline_width: 圆的轮廓宽度，默认为 5
    :return: 生成的 PIL Image 对象
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


def make_inputs(processor, with_im=True):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": generate_circle_image() } if with_im else {"type": "text", "text": ""},
                {"type": "text", "text": "what's in the image"},
            ],
        }
    ]


    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs

# n_q_tokens=1,
# split_dim=False,

def load_config(path):
    with open(path + '/config.json', 'r') as f:
        config = json.load(f)
    import easydict
    
    res = easydict.EasyDict(config)
    print("\n\n")
    print("=" * 30)
    print(path)
    print(res)
    return res


from safetensors.torch import load_file
def load_codebook_weights(path):
    # with open(path + '/model.safetensors.index.json', 'r') as f:
    #     model_json = json.load(f)
    # safetensor_name = model_json['weight_map']['visual_tokenizer.quantizer.embedding.weight']
    # base_model_sd = load_file(path + '/' + safetensor_name)
    base_model_sd = {}
    import tqdm
    for sub_path in tqdm.tqdm(os.listdir(path)):
        if sub_path.endswith('.safetensors'):
            base_model_sd.update(load_file(path + '/' + sub_path))
    return base_model_sd

base_model_sd = load_codebook_weights(path)
# print(base_model_sd.keys()); exit()



'''
self.split_dim = config.split_dim # False
self.split_voc = config.split_voc # 1
self.add_voc_reducer = config.add_voc_reducer # False
'''
import easydict
import copy


def generate_settings():
    default_config = {
        "vision_config.split_voc": 1,
        "vision_config.add_voc_reducer": False,
        "vision_config.split_dim": False,
        "vision_config.n_q_tokens": 1,
        "vision_config.embedding_dim": 128,
        "vision_config.pre_llm_align": False,
        "pool": "avg"
    }
    configs = []
    def add_configs(kargs):
        new_config = copy.deepcopy(default_config)
        new_config.update(**kargs)
        new_config = easydict.EasyDict(new_config)
        configs.append(new_config)

    # add_configs({"vision_config.n_q_tokens":1, "vision_config.split_dim":0, "meta":'1.6.1'})
    # add_configs({"vision_config.n_q_tokens":8, "vision_config.split_dim":1, "meta":'1.6.2'})
    # add_configs({"vision_config.n_q_tokens":8, "vision_config.split_dim":False, "vision_config.add_voc_reducer":True, "vision_config.split_voc":8, "meta":'1.6.3'})
    # add_configs({"vision_config.n_q_tokens":8, "vision_config.split_voc":8, "meta":'1.6.4'})
    # add_configs({"vision_config.n_q_tokens":4, "vision_config.split_voc":4, "meta":'1.6.5'})
    # add_configs({"vision_config.n_q_tokens":2, "vision_config.split_voc":2, "meta":'1.6.6'})
    # add_configs({"vision_config.n_q_tokens":16, "vision_config.split_voc":16, "meta":'1.6.7'})
    # add_configs({"vision_config.n_q_tokens":8, "vision_config.split_voc":8, "pool": "sum", "meta":'1.6.8'})
    add_configs({"vision_config.n_q_tokens":8, "vision_config.split_voc":8, "meta":'1.6.9', "vision_config.pre_llm_align": True})
    for config in configs:
        print(f"runnig config: {config}")
        yield config
        

'''
self.n_q_tokens = config.n_q_tokens
self.split_dim = config.split_dim # False
self.split_voc = config.split_voc # 1
self.add_voc_reducer = config.add_voc_reducer # False
'''


# for n_tokens in [1,2,4,8]:
#     for embedding_dim in [16,32,64,128]:
#         for split in [True, False]:

with torch.no_grad():
    for gen_config in generate_settings():
        print("\n\n\n\n\n")
        print("=" * 40)
        output_path = f"/mmu_mllm_hdd_2/lingzhixin/models/tokenizers_1103/KeyeImageTokenizer_Qwen3-0.6B_init_mt_{gen_config.meta}/"
        if os.path.exists(output_path):
            print(f"{output_path} exists, remove it")
            shutil.rmtree(output_path)
        
        config = KeyeConfig()
        config_base = load_config(path)
        config_keye = load_config(Keye_path)
        qwen3_config = load_config(qwen3_path)
        for k, v in list(config_keye.items())  + list(config_base.items()) + list(qwen3_config.items()):
            if not hasattr(config, k) and k not in ["head_dim"] and 'token_id' not in k:
                print(f"skip {k}:{v}")
                continue
            print(f"{k}:{v} copy from qwen3_config to config")
            if k == 'rope_scaling' and v is None or 'vision_config' in k:
                continue
            print(f"set {k}:{v}")
            setattr(config, k, v)
        
        config.vision_config = AutoConfig.from_pretrained(path, trust_remote_code=True).vision_config
        
        for k,v in gen_config.items():
            if k.startswith('vision_config.'):
                k = k[len('vision_config.'):]
                print(f"{k}:{v} copy from gen_config to vision config")
                setattr(config.vision_config, k, v)
            else:
                print(f"{k}:{v} copy from gen_config to config")
                setattr(config, k, v)

        print("\n\n")
        print(config)

        model = KeyeForConditionalGeneration(config=config)
        model_state_dict = model.state_dict()

        keye =  AutoModelForCausalLM.from_pretrained(
                Keye_path,        
                trust_remote_code=True)
        keye_state_dict = keye.state_dict()

        qwen3 =  AutoModelForCausalLM.from_pretrained(
                qwen3_path,        
                trust_remote_code=True)
        qwen3_state_dict = qwen3.state_dict()

        for k, v in keye_state_dict.items():
            if k in model_state_dict:
                if k in model_state_dict:
                    model_state_dict[k] = v 
                    print(f"{k}:{v.shape} copied from keye_state_dict")

            visual_k = 'visual_tokenizer.' + k
            if visual_k in model_state_dict:
                model_state_dict[visual_k] = v 
                print(f"{visual_k}:{v.shape} copied from keye_state_dict")


        for k, v in qwen3_state_dict.items():
            if k in model_state_dict:
                model_state_dict[k] = v
                print(f"{k}:{v.shape} copied from qwen3_state_dict")

        for name, param in model.named_parameters():
            param.detach_()
            if not name.startswith('visual_tokenizer'):
                continue
            if 'weight' in name and param.ndim==2:
                print(name, 'kaiming inited')
                nn.init.kaiming_normal_(param, a=0, mode='fan_in', nonlinearity='relu')
            
            if not name.startswith('visual_tokenizer.quantizer'):
                continue

            if name.startswith('visual_tokenizer.quantizer') and name.endswith('embedding.weight'):
                print(f"{name}:{v.shape} copied from base_model_sd")
                param.copy_(base_model_sd["visual_tokenizer.quantizer.embedding.weight"].detach())
                continue

            old_name = name.replace('visual_tokenizer.quantizer.0.', 'visual_tokenizer.quantizer.')
            if old_name in base_model_sd and base_model_sd[old_name].shape == param.shape:
                print(f'copied from base_model_sd, {old_name} -> {name}')
                param.copy_(base_model_sd[old_name].detach())

        model.load_state_dict(model_state_dict)
        model = model.half()
        model.save_pretrained(output_path)

        with open(output_path + 'config.json', 'r') as f:
            print(output_path)
            info = json.loads(f.read())
            info["meta"] = f"base_config={path},tokenizer_path={tokenizer_path},Keye_path={Keye_path},qwen3_path={qwen3_path}"

        with open(output_path + 'config.json', 'w') as f:
            json.dump(info, f, indent=2)


        template_dir = "/llm_reco_ssd/zhouyang12/models/KeyeImageTokenizer_exp_121_old"
        for py_name in ["image_processing_keye.py", "processing_keye.py"]:
            import shutil
            print(f"copy {py_name} to {output_path}")
            shutil.copyfile(f"{template_dir}/{py_name}", f"{output_path}/{py_name}")

        for json_name in os.listdir(template_dir):
            if json_name.endswith('.json'):
                if os.path.exists(f"{output_path}/{json_name}"):
                    print(f"skip {json_name}")
                    continue
                print(f"copy {template_dir}/{json_name} to {output_path}")
                shutil.copyfile(f"{template_dir}/{json_name}", f"{output_path}/{json_name}")

        for py_name in os.listdir(current_script.parent.parent):
            target_path = f"{output_path}/{py_name}"
            if os.path.exists(target_path):
                print(f"skip {py_name}")
                continue

            import shutil
            source_name = f"{current_script.parent.parent}/{py_name}"
            if Path(source_name).is_dir():
                print(f"skip {py_name}")
                continue
            print(f"copy {source_name} to {output_path}")
            shutil.copyfile(source_name, target_path)

        print("test init...", output_path)
        model = KeyeForConditionalGeneration.from_pretrained(output_path, _attn_implementation='flash_attention_2').to(0).half()
        processor = AutoProcessor.from_pretrained(output_path, trust_remote_code=True)

        for with_im in [True, False]:
            inputs = make_inputs(processor, with_im).to(0)
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(0)
                    if v.dtype in [torch.float32, torch.float64]:
                        inputs[k] = v.half()
            print("testing...")
            print("inputs:", inputs, "inputs.input_ids:", inputs.input_ids)
            generated = model.generate(**inputs, max_new_tokens=4)
            output_ids = generated[0][len(inputs.input_ids[0]):].tolist() 
            content = processor.decode(output_ids[0:], skip_special_tokens=True).strip("\n")
            print(f"with_im={with_im}, content={content}")