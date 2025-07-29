from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import json
import uuid
import pandas as pd

# MODEL_PATH = "/llm_reco/chuchenglong/R3/converted_models/sft_401"
MODEL_PATH = "/llm_reco_ssd/zangdunju/models/Comment"
MODEL_PATH = "/llm_reco_ssd/luoxinchen/output3/RecoVLM-Base/0.3.0/cmt/global_step4001/merged_4001"
MODEL_PATH = "/mmu_mllm_hdd_2/wenbin/release/Keye-8B-0-9-1-Mixed-Mode-RL-0618"
CONFIG_PATH = "/llm_reco/chuchenglong/R3/deom_visual/result_0317/config2.json"
CONFIG_PATH = "/llm_reco/chuchenglong/R3/deom_visual/result_0317/config2.json"

output_file = "/llm_reco/chuchenglong/R3/deom_visual/comment_only_msy.json"


llm = LLM(
    model=MODEL_PATH,
    limit_mm_per_prompt={"image": 10, "video": 10},
)

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    repetition_penalty=1.05,
    max_tokens=256,
    stop_token_ids=[],
)

def load_config(config_path):
    """从JSON文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        
    # 验证配置格式
    if "videos" not in config:
        raise ValueError("配置文件缺少'videos'字段")
    if "prompts" not in config:
        raise ValueError("配置文件缺少'prompts'字段")
    if not isinstance(config["videos"], list) or not config["videos"]:
        raise ValueError("'videos'必须是非空列表")
    if not isinstance(config["prompts"], list) or not config["prompts"]:
        raise ValueError("'prompts'必须是非空列表")
        
    return config

def generate_video_messages(video_url, prompt_text):
    """根据视频URL和提示文本生成messages"""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "video", 
                    "video": video_url,
                    "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 28
                }
            ]
        },
    ]

config = load_config(CONFIG_PATH)  # 假设配置文件名为config.json

# 从配置中获取视频URL列表和提示列表
videos = config["videos"]
prompts = config["prompts"]

# 为每个视频和提示组合创建messages列表
all_messages = []
video_prompt_pairs = []  # 记录每个消息对应的视频和提示
for video_url in videos:
    for prompt_text in prompts:
        for i in range(10):
            all_messages.append(generate_video_messages(video_url, prompt_text))
            video_prompt_pairs.append({"video": video_url, "prompt": prompt_text})

# 存储所有结果的字典

# 检查是否存在现有结果文件，如果有则加载，否则创建新字典
try:
    with open(output_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    print(f"已加载现有结果文件: {output_file}")
except (FileNotFoundError, json.JSONDecodeError):
    results = {}
    print(f"创建新的结果文件: {output_file}")

# 将处理器初始化移到循环外
processor = AutoProcessor.from_pretrained(MODEL_PATH)

total_tasks = len(all_messages)
print(f"开始处理 {len(videos)} 个视频与 {len(prompts)} 个提示的组合，共 {total_tasks} 个任务")

# 处理每个视频和提示组合
for i, messages in enumerate(all_messages):
    try:
        video_url = video_prompt_pairs[i]["video"]
        prompt_text = video_prompt_pairs[i]["prompt"]
        
        print(f"处理视频 {video_url} 和提示 '{prompt_text}'")
        
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }

        outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        # 使用新的数据结构保存结果
        if video_url not in results:
            results[video_url] = []
        
        # 创建新的采样记录
        sample = {
            "sample_id": str(uuid.uuid4()),  # 生成唯一ID
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": {}
        }
        
        # 添加当前提示的结果
        sample["results"][prompt_text] = generated_text
        
        # 添加到视频对应的采样列表中
        results[video_url].append(sample)
        
        # 每处理完一个组合就保存一次结果
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        print(f"已完成: {i+1}/{total_tasks} ({(i+1)/total_tasks*100:.1f}%)，结果已保存")
    except Exception as e:
        print(f"处理失败: {e}")
        # 记录失败信息
        if video_url not in results:
            results[video_url] = []
        results[video_url].append({
            "sample_id": str(uuid.uuid4()),  # 生成唯一ID
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": {prompt_text: f"处理失败: {str(e)}"}
        })
        
        # 保存失败信息
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

print(f"所有结果已保存到 {output_file}")
