import json
from typing import Dict, List, Optional, Union

class ConverterBase(object):

    def __call__(self, src: Dict[str, any]) -> Optional[Union[Dict[str, any], List[Dict[str, any]]]]:
        raise NotImplementedError
    
class EmptyConverter(ConverterBase):

    def __call__(self, src: Dict[str, any]) -> Optional[Dict[str, any]]:
        print('src', src.keys())
        return None

class IdentityConverter(ConverterBase):

    def __call__(self, src: Dict[str, any]) -> Optional[Dict[str, any]]:
        return src

    
def render_image_text(images):
    text = []
    for key in images.keys():
        text.append({
            "type": "image",
            "image": f"{key}"
        })
    return text

def iter_messages(messages, text_fn):
    if messages is None:
        return None
    messages = json.loads(messages)
    if messages is None or messages == 'null':
        return None
    for msg in messages:
        content = msg['content']
        if isinstance(content, str):
            msg['content'] = text_fn(content)
        elif isinstance(content, List):
            for i in range(len(content)):
                c = content[i]
                if isinstance(c, Dict) and c['type'] == 'text':
                    c['text'] = text_fn(c['text'])
        else:
            raise ValueError(f"invalid msg {msg}")
    return json.dumps(messages)

def iter_segments(segments, text_fn, start_offset: int = 0):
    if segments is None:
        return None
    if segments is None or segments == 'null':
        return None
    segments = json.loads(segments)
    for i in range(start_offset, len(segments)):
        seg = segments[i]
        if isinstance(seg, str):
            segments[i] = text_fn(seg)
        elif isinstance(seg, Dict) and seg['type'] == 'text':
            seg['text'] = text_fn(seg['text'])
    return json.dumps(segments)


