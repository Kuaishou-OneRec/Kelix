import unittest
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, 
import torch.distributed as dist


import os
from recovlm.models.keye_vitrope_slowfast_v2.modeling_keye import KeyeForConditionalGeneration


# --- 1. 配置模型路径 ---
QWEN3_PATH = '/llm_reco_ssd/zhouyang12/models/Qwen3-0.6B'
KEYE_PATH = '/llm_reco_ssd/zhouyang12/models/KeyeImageTokenizer_exp_121'

class TestTextGenerationConsistency(unittest.TestCase):
    """
    一个简洁的单元测试，用于验证 KeyeForConditionalGeneration 在纯文本任务上
    是否与原始的 Qwen3 模型表现完全一致。
    """
    
    original_model = None
    keye_model = None
    tokenizer = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*60)
    print("正在初始化单进程分布式环境...")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # 使用一个随机的空闲端口
    # 如果是CPU环境，使用 'gloo' 后端
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend=backend, rank=0, world_size=1)
    print("分布式环境初始化完成。")




    @classmethod
    def setUpClass(cls):
        """在所有测试开始前，加载一次模型和分词器。"""
        print("="*60)
        print("正在设置测试环境 (加载模型中，请稍候)...")
        print(f"使用的设备: {cls.device}")

        # --- 2. 加载通用的文本分词器 ---
        # 对于纯文本任务，两个模型使用相同的原始分词器。
        print(f"-> 正在从 '{QWEN3_PATH}' 加载 Tokenizer...")
        cls.tokenizer = AutoTokenizer.from_pretrained(QWEN3_PATH, trust_remote_code=True)

        # --- 3. 加载模型 ---
        # 使用 bfloat16 以获得更好的性能，并确保数据类型一致
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        print(f"使用的数据类型: {dtype}")

        # 加载原始 Qwen3 模型 (作为比较基准)
        print("-> 正在加载原始 Qwen3 模型 (AutoModelForCausalLM)...")
        cls.original_model = AutoModelForCausalLM.from_pretrained(
            QWEN3_PATH,
            torch_dtype=dtype,
            device_map=cls.device,
            trust_remote_code=True
        ).eval() # 设置为评估模式

        # 加载你的自定义 Keye 模型
        # 直接从 Keye 的权重加载，这是最简洁的方式
        print("-> 正在加载自定义 Keye 模型 (KeyeForConditionalGeneration)...")
        cls.keye_model = KeyeForConditionalGeneration.from_pretrained(
            KEYE_PATH,
            torch_dtype=dtype,
            device_map=cls.device,
            trust_remote_code=True,
            ignore_mismatched_sizes=True
        ).eval() # 同样设置为评估模式

        print("模型加载完成！")
        print("="*60)

    def test_text_output_is_identical(self):
        """
        核心测试：给定相同的纯文本输入，断言两个模型的输出完全相同。
        """
        # --- 准备输入 ---
        prompt_text = "你好，请你用中文介绍一下北京这座城市。"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text}
        ]
        
        # 使用聊天模板格式化输入，这是推荐的做法
        text_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text_input], return_tensors="pt").to(self.device)

        # --- 设定完全相同的生成参数 ---
        # `do_sample=False` 使用贪心搜索，确保生成过程是确定性的
        generation_params = {
            'max_new_tokens': 50,
            'do_sample': False,
            'pad_token_id': self.tokenizer.eos_token_id,
        }
        
        print("\n--- 开始生成内容对比测试 ---")
        print(f"输入 Prompt: {prompt_text}")
        print(f"生成参数: {generation_params}")

        # 使用 torch.no_grad() 进行推理，以节省显存和计算资源
        with torch.no_grad():
            # --- [A] 使用原始模型生成 ---
            print("\n[1] 原始模型正在生成...")
            original_output_ids = self.original_model.generate(**model_inputs, **generation_params)
            
            # --- [B] 使用 Keye 模型生成 ---
            print("[2] Keye 模型正在生成...")
            keye_output_ids = self.keye_model.generate(**model_inputs, **generation_params)

        # --- 解码并比较结果 ---
        # 计算输入部分的长度，以便我们只解码新生成的内容
        input_length = model_inputs.input_ids.shape[1]
        
        original_decoded_text = self.tokenizer.decode(original_output_ids[0, input_length:], skip_special_tokens=True)
        keye_decoded_text = self.tokenizer.decode(keye_output_ids[0, input_length:], skip_special_tokens=True)
        
        print("\n--- 生成结果 ---")
        print(f"原始模型输出:\n{original_decoded_text}")
        print("-" * 20)
        print(f"Keye 模型输出:\n{keye_decoded_text}")
        print("--- 结果对比 ---")

        # --- 断言 ---
        # 使用 self.assertEqual 进行严格比较
        self.assertEqual(
            original_decoded_text.strip(), 
            keye_decoded_text.strip(),
            "失败：合并前后的模型对于纯文本输入的生成结果不一致！"
        )
        
        print("✅ 测试通过！两个模型的纯文本生成结果完全相同。")


# --- 运行测试 ---
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)