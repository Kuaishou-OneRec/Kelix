import torch

from muse.models.keye_tokenizer import KeyeImageTokenizer
from muse.training.common import set_default_dtype

TOKENIZER_PATH = ""

def test_forward():
    with set_default_dtype("bfloat16"), torch.device("cuda"):
        tokenizer = KeyeImageTokenizer.from_pretrained(
            "/llm_reco_ssd/zhouyang12/models/muse/KeyeTokenizer/"
        )
    
