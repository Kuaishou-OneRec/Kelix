from typing import Dict, List, Optional, Union


class ConverterBase(object):

    def __call__(self, src: Dict[str, any]) -> Optional[Union[Dict[str, any], List[Dict[str, any]]]]:
        raise NotImplementedError

class EmptyConverter(ConverterBase):

    def __call__(self, src: Dict[str, any]) -> Optional[Dict[str, any]]:
        print('src', src.keys())
        return None

    
def render_image_text(images):
    text = []
    for key in images.keys():
        text.append({
            "type": "image",
            "image": f"{key}"
        })
    return text