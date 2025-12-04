"""
Qwen3 model implementation.
"""

from muse.models.keye_vit.modeling import KeyeVisionTransformer

# Register the model (import here to avoid circular imports)
# The registration decorator is applied when this module is imported by muse.models
from muse.models.base import Model

# Apply decorator manually to avoid circular import issues
def _register_keye_vision():
    """Register KeyeVisionTransformer in the model registry."""
    try:
        from muse.models import register_model
        # Register the model
        register_model("KeyeVisionTransformer")(KeyeVisionTransformer)
    except ImportError:
        # Registry not yet available during initial import
        pass

_register_keye_vision()

__all__ = ["KeyeVisionTransformer"]

