"""
Keye tokenizer video model implementation (using transformers-style modeling from end2end repo).

This module uses the modeling code from end2end/muse/recovlm/models/tokenizer_end2end_mt_1drope_video/
to test cross-repository compatibility and identify training differences.
"""

from muse.models.keye_tokenizer_end2end_video_transformers.modeling import (
    KeyeForConditionalGeneration,
)

# Create an alias for consistency with muse naming conventions
KeyeTokenizerEnd2EndVideoTransformers = KeyeForConditionalGeneration

# Register the model (import here to avoid circular imports)
# The registration decorator is applied when this module is imported by muse.models
from muse.models.base import Model

# Apply decorator manually to avoid circular import issues
def _register_keye_tokenizer_end2end_video_transformers():
    """Register KeyeTokenizerEnd2EndVideoTransformers in the model registry."""
    try:
        from muse.models import register_model
        # Register the model with both names for flexibility
        register_model("KeyeTokenizerEnd2EndVideoTransformers")(KeyeForConditionalGeneration)
        register_model("KeyeForConditionalGeneration")(KeyeForConditionalGeneration)
    except ImportError:
        # Registry not yet available during initial import
        pass

_register_keye_tokenizer_end2end_video_transformers()

__all__ = ["KeyeTokenizerEnd2EndVideoTransformers", "KeyeForConditionalGeneration"]

