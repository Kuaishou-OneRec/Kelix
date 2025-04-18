import json
import hashlib
import uuid
import base64
from typing import Dict, Optional
from .converter import ConverterBase
import base64
import requests
import os

def image_key_to_base64(temp_path):
    temp_path = temp_path[4:]#remove the first 4 char
    image_path = f"/llm_reco/luoxinchen/dataset/coco2017/coco2017/pic/{temp_path}"#use key's pre 5 char to find the image
    if not os.path.exists(image_path):
        return None
    try:
        with open(image_path, 'rb') as image_file:
            image_bytes = image_file.read()
            base64_data = base64.b64encode(image_bytes).decode("ascii") # 转换为 Base64 字符串
            return base64_data
    except Exception as e:
        print(f"Error: {e}")
        return None


def convert_to_messages(conversation_list):
    """
    Convert a conversation list to messages format.
    
    Args:
        conversation_list: List of conversation dictionaries with 'from' and 'value' keys
        
    Returns:
        List of messages in the desired format
    """
    messages = []
 
    for i, item in enumerate(conversation_list):
        content = item['value']
        tempitem = []
        if '<image>' in content:
            content = content.replace('<image>', '')
            tempitem.append({
                "type": "image",
                "image": "0.jpg"
            })
        tempitem.append({
            "type": "text",
            "text": content
        })
 
        if item['from'] == 'human':
            # Replace '<image>' tag with an actual image placeholder if needed
            messages.append({
                "role": "user",
                "content": tempitem
            })
        elif item['from'] == 'gpt':
            messages.append({
                "role": "assistant",
                "content": tempitem
            })
 
    return messages


class ConversationCaptionConverter(ConverterBase):

    def __init__(
        self,
        source: str,
    ):
        self.source = source

    def __call__(self, src: Dict[str, any]) -> Optional[Dict[str, any]]:
        image_path = src['image']
        conversations = src['conversations']
        image_bytes = image_key_to_base64(image_path)
        if image_bytes is None:
            return None
        messages = convert_to_messages(conversations)
        print(messages)

        images = {"0.jpg": image_bytes}
        
        segments = None



        metadata = None
        result = {
            "images": json.dumps(images),
            "videos": json.dumps(None),
            "source": self.source,
            "messages": json.dumps(messages),
            "segments": None,
            "metadata": json.dumps(metadata),
            "uuid": str(uuid.uuid1()),
        }
        return result