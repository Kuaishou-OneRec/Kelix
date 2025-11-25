"""
Unit tests for muse.config.model_config.
"""

import json

import pytest

from muse.config import Qwen3Config


def test_qwen3_config_defaults():
    """Qwen3Config should expose sensible defaults."""
    cfg = Qwen3Config(model_class="Qwen3Model")

    assert cfg.model_class == "Qwen3Model"
    assert cfg.vocab_size == 151936
    assert cfg.embed_dim == 4096
    assert cfg.num_heads == 32
    assert cfg.num_kv_heads == 32
    assert cfg.head_dim == 128
    assert cfg.embed_dim == cfg.num_heads * cfg.head_dim
    assert cfg.num_heads % cfg.num_kv_heads == 0


def test_qwen3_config_dict_roundtrip():
    """to_dict/from_dict should round-trip the same values."""
    cfg = Qwen3Config(model_class="Qwen3Model")
    cfg_dict = cfg.to_dict()

    restored = Qwen3Config.from_dict(cfg_dict)

    assert restored == cfg
    assert restored.to_dict() == cfg_dict


def test_qwen3_config_json_roundtrip():
    """to_json/from_json should faithfully serialize the config."""
    cfg = Qwen3Config(model_class="Qwen3Model")
    json_str = cfg.to_json()

    # Ensure JSON is valid and contains key fields
    parsed = json.loads(json_str)
    assert parsed["model_class"] == "Qwen3Model"

    restored = Qwen3Config.from_json(json_str)
    assert restored == cfg


def test_qwen3_config_save_and_load(tmp_path):
    """save/load helpers should persist configs to disk."""
    cfg = Qwen3Config(model_class="Qwen3Model")
    cfg_path = tmp_path / "qwen3.json"

    cfg.save(cfg_path.as_posix())

    loaded = Qwen3Config.load(cfg_path.as_posix())
    assert loaded == cfg


def test_qwen3_config_merge_creates_new_instance():
    """merge should override fields and keep original intact."""
    cfg = Qwen3Config(model_class="Qwen3Model")

    overrides = {
        "head_dim": 64,
        "num_heads": 64,
        "num_kv_heads": 8,
        "attn_dropout": 0.1,
        "embed_dim": 4096,  # 64 heads * 64 dim still equals 4096
    }
    merged = cfg.merge(overrides)

    # Original remains unchanged
    assert cfg.head_dim == 128
    assert cfg.num_heads == 32

    # New config reflects overrides
    assert merged.head_dim == 64
    assert merged.num_heads == 64
    assert merged.num_kv_heads == 8
    assert merged.attn_dropout == pytest.approx(0.1)
    assert merged.embed_dim == merged.num_heads * merged.head_dim


def test_qwen3_config_validators_trigger_on_invalid_values():
    """Validators should reject incompatible head settings."""
    with pytest.raises(ValueError, match="num_heads"):
        Qwen3Config(
            model_class="Qwen3Model",
            num_heads=40,
            num_kv_heads=6,
            embed_dim=40 * 128,
        )

    with pytest.raises(ValueError, match="embed_dim"):
        Qwen3Config(
            model_class="Qwen3Model",
            num_heads=32,
            num_kv_heads=32,
            head_dim=256,
            embed_dim=4096,  # should be 32 * 256
        )

