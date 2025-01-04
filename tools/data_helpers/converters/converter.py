from typing import Dict, List


class Converter(object):

    def __call__(self, src: Dict[str, any]) -> Dict[str, any]:
        raise NotImplementedError
    
def render_image_text(images):
    text = []
    for key in images.keys():
        text.append({
            "type": "image",
            "image": f"{key}"
        })
    return text