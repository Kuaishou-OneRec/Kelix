import json
import uuid
from typing import Dict, List, Optional

from .converter import (
    ConverterBase,
)

ROLE_MAP = {"human": "user", "gpt": "assistant"}

class InfinityInstructConverter(ConverterBase):

    def __call__(self, src: Dict[str, any]) -> List[Dict[str, any]]:

        def cvt_msg(m):
            msg = dict()
            msg['role'] = ROLE_MAP[m['from']]
            msg['content'] = m['value']
            return msg

        conversations = src['conversations']
        messages = [cvt_msg(x) for x in conversations]
        images = dict()
        videos = dict()
        source = src['source']
        
        result = {
            "images": json.dumps(images),
            "videos": json.dumps(videos),
            "source": source,
            "messages": json.dumps(messages),
            "segments": None,
            "metadata": None,
            "uuid": str(uuid.uuid1()),
        }
        return result