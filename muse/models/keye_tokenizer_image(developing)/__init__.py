"""
Keye tokenizer video model implementation.
"""

from muse.models.keye_tokenizer_image.modeling import (
    KeyeImageTokenizer,
    KeyeForConditionalGeneration,
)

# Register the model (import here to avoid circular imports)
# The registration decorator is applied when this module is imported by muse.models
from muse.models.base import Model

# Apply decorator manually to avoid circular import issues
def _register_keye_tokenizer_video():
    """Register KeyeImageTokenizer and KeyeForConditionalGeneration in the model registry."""
    try:
        from muse.models import register_model
        # Register the model
        # register_model("KeyeImageTokenizer")(KeyeImageTokenizer)
        register_model("KeyeForConditionalGeneration")(KeyeForConditionalGeneration)
    except ImportError:
        # Registry not yet available during initial import
        pass

_register_keye_tokenizer_video()

__all__ = [ "KeyeForConditionalGeneration"]

