import os
import json
import uuid
import numpy as np
from typing import Dict, Optional
from .converter import (
    ConverterBase,
    render_image_text
)

class PubTabNetConverter(ConverterBase):
    def __init__(self, source: str):
        self.source = source
        # 内置多样化的中英文提示语
        self.prompts = [
            # 英文提示
            "Generate the HTML code for this table.",
            "Please convert this table image to HTML format.",
            "What's the HTML representation of this table?",
            "Can you provide the HTML code that would recreate this table?",
            "Transform this table into HTML code.",
            "Write the HTML markup for this table structure.",
            
            # 中文提示
            "请你生成这个表格对应的HTML代码。",
            "将这个表格转换成HTML格式。",
            "请提供这个表格的HTML代码实现。",
            "把这个表格图片转成HTML代码。",
            "请用HTML代码重现这个表格的结构。",
            "生成一段HTML代码来表示这个表格。",
        ]

    def __call__(self, src: Dict[str, any]) -> Optional[Dict[str, any]]:
        # 确保输入数据包含必要字段
        if 'image' not in src or 'html' not in src:
            return None

        # 准备图像数据
        images = {"0.jpg": src['image']}

        # 构建对话消息
        messages = [
            {
                "role": "user",
                "content": render_image_text(images) + [
                    {"type": "text", "text": np.random.choice(self.prompts)}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": src['html']}
                ]
            }
        ]

        # 构建返回结果，对齐 VlmSftTextConverter 的格式
        result = {
            "images": json.dumps(images),
            "videos": json.dumps([]),
            "source": self.source,
            "messages": json.dumps(messages),
            "segments": None,
            "metadata": None,
            "uuid": str(uuid.uuid1()),
        }
        return result
