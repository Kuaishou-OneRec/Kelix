import sys
import os
import torch
import pytest

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 导入模型类
from muse.models.keye_ar.token_decoder_ori import PureDecoderTransformer
from muse.models.keye_ar.unified_token_decoder import TokenDecoder

@pytest.fixture
def base_config():
    """基础配置参数"""
    return {
        "vocab_size": 1000,
        "max_length": 30,
        "d_model": 128,
        "eos_token": 999,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 512,
        "use_gradient_checkpointing": False,
        "reduce": False,
        "attention_function": "eager"
    }

@pytest.fixture
def token_embedding(base_config):
    """共享的token embedding"""
    return torch.nn.Embedding(base_config["vocab_size"], base_config["d_model"])

@pytest.fixture
def test_input_ids(base_config):
    """测试输入IDs"""
    batch_size = 2
    seq_len = 5
    return torch.randint(0, base_config["vocab_size"], (batch_size, seq_len))

@pytest.fixture
def test_input_embeddings(base_config):
    """测试输入embeddings"""
    batch_size = 2
    seq_len = 5
    return torch.randn(batch_size, seq_len, base_config["d_model"])