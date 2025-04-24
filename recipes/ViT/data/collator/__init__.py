from recipes.ViT.helpers.common import filter_function_arguments
from .single_image_text import SingleImageTextPairCollator
from .identity import IdentityCollator
from .vision_packing import VisionPackingCollator


def build_collator(**kwargs):
    name = kwargs["collate_fn"]
    collator_class = eval(name)
    kwargs = filter_function_arguments(collator_class.__init__, kwargs, new_obj=True, exclude_keys=["collate_fn"])
    return collator_class(**kwargs)
