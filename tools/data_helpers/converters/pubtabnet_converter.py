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
            # English prompts
            "Can you convert this table image into an HTML document?",
            "Can you provide the HTML code that would recreate this table?",
            "Convert this table to its HTML equivalent.",
            "Could you show me how this table translates to HTML?",
            "Create HTML code based on this table image.",
            "Generate an HTML version of this table.",
            "Generate the HTML code for this table.",
            "Give me the HTML implementation of this table.",
            "Help me convert this table into HTML format.",
            "How can I represent this table in HTML?",
            "How would you represent this table in HTML?",
            "I need the HTML version of this table.",
            "Parse this table and output its HTML code.",
            "Please convert this table image to HTML format.",
            "Please provide the HTML structure for this table image.",
            "Show me the HTML structure for this table.",
            "Transform this table into HTML code.",
            "Translate this table into HTML markup.",
            "Turn this table image into HTML code, please.",
            "What HTML code would reproduce this table?",
            "What HTML tags would recreate this table?",
            "What is the HTML markup for this table?",
            "What would the HTML code for this table look like?",
            "What's the HTML representation of this table?",
            "Write the HTML markup for this table structure.",
            
            
            # Chinese prompts
            "把这个表格的内容用HTML代码表达出来。",
            "把这个表格图片转成HTML代码。",
            "把这个表格转换成对应的HTML标记语言。",
            "帮我把这个表格转换成HTML的形式。",
            "帮我把这个表格转换成HTML格式。",
            "给出这个表格的HTML代码表示方式。",
            "将这个表格结构转换为HTML格式。",
            "将这个表格转换成HTML格式。",
            "你能把这个表格图片转换成HTML文档吗？",
            "你能告诉我这个表格如何转换成HTML吗？",
            "请把这个表格图片转换成HTML代码。",
            "请帮我将这个表格转成HTML代码格式。",
            "请解析这个表格并生成HTML代码。",
            "请你生成这个表格对应的HTML代码。",
            "请提供这个表格的HTML代码实现。",
            "请提供这个表格图片的HTML结构。",
            "请用HTML代码重现这个表格的结构。",
            "如何用HTML表示这个表格？",
            "如何用HTML代码来构建这个表格？",
            "生成一段HTML代码来表示这个表格。",
            "我需要这个表格的HTML版本。",
            "用哪些HTML标签可以重现这个表格？",
            "用HTML代码描述这个表格的结构。",
            "这个表格的HTML标记是什么？",
            "这个表格的HTML代码是什么样的？",
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
