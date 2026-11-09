"""Check standalone and embedded tokenizer loading without the AR model."""

import json

import pytest
import torch
from safetensors.torch import save_file

from demo.infer_aedit import load_image_tokenizer
from muse.config.model_config import KeyeTokenizerConfig, KeyeVisionConfig
from muse.models.keye_tokenizer import KeyeImageTokenizer


@pytest.fixture
def tokenizer():
    return KeyeImageTokenizer(KeyeTokenizerConfig(
        vision_config=KeyeVisionConfig(
            image_size=28, hidden_size=16, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=32, attention_function="eager",
        ),
        codebook_size=16, n_q_tokens=2, split_voc=2, embedding_dim=4,
        init_embedding_dim=8, llm_hidden_size=8, output_dim=6,
    )).eval()


def save_embedded_tokenizer(folder, tokenizer, omit=None):
    config = {"tokenizer_config": tokenizer.config.model_dump()}
    (folder / "config.json").write_text(json.dumps(config))
    state = {f"visual_tokenizer.{k}": v for k, v in tokenizer.state_dict().items()
             if k != omit}
    keys = list(state)
    index = {"model.language_weight": "ar-not-downloaded.safetensors"}
    for number, shard_keys in enumerate((keys[::2], keys[1::2])):
        name = f"tokenizer-{number}.safetensors"
        save_file({k: state[k] for k in shard_keys}, str(folder / name))
        index.update({k: name for k in shard_keys})
    (folder / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": index})
    )


@pytest.mark.parametrize("embedded", [False, True])
def test_load_image_tokenizer(tmp_path, tokenizer, embedded):
    if embedded:
        save_embedded_tokenizer(tmp_path, tokenizer)
    else:
        tokenizer.save_pretrained(str(tmp_path))
    loaded, processor_dir = load_image_tokenizer(str(tmp_path), "cpu", torch.float32)
    assert processor_dir == tmp_path
    assert loaded.config.fusion_type == "mean"
    assert not loaded.training
    expected = tokenizer.state_dict()
    actual = loaded.state_dict()
    assert expected.keys() == actual.keys()
    for name in expected:
        torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)
    expected_buffers = dict(tokenizer.named_buffers())
    for name, buffer in loaded.named_buffers():
        torch.testing.assert_close(buffer, expected_buffers[name], rtol=0, atol=0)


def test_embedded_tokenizer_rejects_missing_weights(tmp_path, tokenizer):
    save_embedded_tokenizer(tmp_path, tokenizer, omit="up_projectors.0.weight")
    with pytest.raises(RuntimeError, match="up_projectors.0.weight"):
        load_image_tokenizer(str(tmp_path), "cpu", torch.float32)
