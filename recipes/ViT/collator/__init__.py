from recipes.ViT.common import filter_function_arguments
from .single_image_text import SingleImageTextPairCollator


def build_collator(**kwargs):
    name = kwargs["collate_fn"]
    collator_class = eval(name)
    kwargs = filter_function_arguments(collator_class.__init__, kwargs, new_obj=False, exclude_keys=["collate_fn"])
    return collator_class(**kwargs)
