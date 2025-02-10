import json
from typing import Dict, Optional, List


class FilterBase(object):

    def __call__(self, src: Dict[str, any]) -> bool:
        raise NotImplementedError
    
    def extract_all_text(self, src):
        text = "" 
        if src['messages'] is not None:
            messages = json.loads(src['messages'])
            if messages is not None and messages != 'null':
                for msg in messages:
                    content = msg['content']
                    if isinstance(content, str):
                        text += content
                    elif isinstance(content, List):
                        for sub in content:
                            if sub['type'] == 'text':
                                text += sub['text']
                    else:
                        raise ValueError(f"msg {msg} is not valid")

        if src['segments'] is not None:
            segments = json.loads(src['segments'])
            if segments is not None and segments != 'null':
                for seg in segments:
                    if isinstance(seg, str):
                        text += seg
                    elif isinstance(seg, Dict):
                        if seg['type'] == 'text':
                            text += seg['text']
                    else:
                        raise ValueError(f"seg {seg} is not valid")
        return text
    def get_metadata(self, src):
        if "metadata" in src and isinstance(src['metadata'], str):
            return json.loads(src['metadata'])
        else:
            return dict()
