"""
Keye tokenizer video model implementation.
"""

from muse.models.keye_tokenizer_end2end_image.modeling import (
    KeyeTokenizerEnd2EndVideo,
)

# Register the model (import here to avoid circular imports)
# The registration decorator is applied when this module is imported by muse.models
from muse.models.base import Model

# Apply decorator manually to avoid circular import issues
def _register_keye_tokenizer_end2end_image():
    """Register KeyeTokenizerEnd2EndVideo in the model registry."""
    try:
        from muse.models import register_model
        # Register the model    
        register_model("KeyeTokenizerEnd2EndVideo")(KeyeTokenizerEnd2EndVideo)
    except ImportError:
        # Registry not yet available during initial import
        pass

_register_keye_tokenizer_end2end_image()

__all__ = [ "KeyeTokenizerEnd2EndVideo"]

