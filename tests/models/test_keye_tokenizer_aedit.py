"""Regression coverage for the image and discrete-token conditioning paths."""

from unittest.mock import patch

import pytest
import torch

from muse.config.model_config import KeyeTokenizerConfig, KeyeVisionConfig
from muse.models.keye_tokenizer import KeyeImageTokenizer


@pytest.mark.parametrize("fusion_type", ["mean", "sum"])
@pytest.mark.parametrize("max_length", [2, 4])
def test_image_and_index_conditions_match(fusion_type, max_length):
    torch.manual_seed(42)
    config = KeyeTokenizerConfig(
        vision_config=KeyeVisionConfig(
            image_size=28,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            attention_function="eager",
        ),
        codebook_size=16,
        n_q_tokens=2,
        split_voc=2,
        embedding_dim=4,
        init_embedding_dim=8,
        llm_hidden_size=8,
        output_dim=6,
        fusion_type=fusion_type,
    )
    tokenizer = KeyeImageTokenizer(config).eval()
    # Supply encoder features on CPU; quantization, fusion, and padding are real.
    features = torch.randn(5, config.llm_hidden_size)
    grid = torch.tensor([[1, 2, 4], [1, 2, 6]])
    pixels = torch.empty(0)
    with torch.inference_mode(), patch.object(
        tokenizer, "get_image_embeds", return_value=features
    ):
        outputs = tokenizer(pixels, grid)
        indices = torch.stack(outputs["indices"], dim=-1)
        torch.testing.assert_close(
            outputs["token_embeds"], tokenizer.tokenize_indices(indices)
        )
        image_cond, image_mask, image_length = tokenizer.tokenize(
            pixels, grid, max_pad_to=max_length
        )
        token_cond, token_mask, token_length = tokenizer.tokenize_indices(
            indices, lengths=[2, 3], max_pad_to=max_length
        )
        vocab_ids = tokenizer.forward_image_tokens(pixels, grid, vocab_size=100)

    expected_length = min(3, max_length)
    assert image_length == token_length == expected_length
    assert image_cond.shape == (2, expected_length, config.output_dim)
    torch.testing.assert_close(image_cond, token_cond)
    torch.testing.assert_close(image_mask, token_mask)
    expected_mask = torch.zeros(2, 1, 1, expected_length)
    expected_mask[0, ..., :min(2, expected_length)] = 1
    expected_mask[1, ...] = 1
    torch.testing.assert_close(image_mask, expected_mask)
    if expected_length > 2:
        assert torch.count_nonzero(image_cond[0, 2:]) == 0
    offsets = torch.arange(config.n_q_tokens) * (
        config.codebook_size // config.n_q_tokens
    )
    torch.testing.assert_close(vocab_ids, 100 + indices + offsets)
