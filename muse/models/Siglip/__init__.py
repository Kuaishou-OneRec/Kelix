"""
Qwen3 model implementation.
"""

from muse.models.Siglip.modeling import SiglipVisionModel

# Register the model (import here to avoid circular imports)
# The registration decorator is applied when this module is imported by muse.models
from muse.models.base import Model

# Apply decorator manually to avoid circular import issues
def _register_siglip_vision():
    """Register SiglipVisionModel in the model registry."""
    try:
        from muse.models import register_model
        # Register the model
        register_model("SiglipVisionModel")(SiglipVisionModel)
    except ImportError:
        # Registry not yet available during initial import
        pass

_register_siglip_vision()

__all__ = ["SiglipVisionModel"]

