import os
import json
import uuid
import numpy as np
from typing import Dict, Optional
from .converter import (
    ConverterBase,
    render_image_text
)

class FinTabNetConverter(ConverterBase):
    def __init__(self, source: str):
        self.source = source
        self.prompts = [
            # English prompts
            "Analyze this financial table and convert it to HTML format.",
            "Can you convert this marked table into HTML code?",
            "Convert this highlighted table to HTML structure.",
            "Create HTML markup for this bordered table.",
            "Detect the table structure and generate corresponding HTML.",
            "Extract the table structure and provide HTML code.",
            "Generate HTML code for this marked financial table.",
            "Given the highlighted table regions, create HTML markup.",
            "How would you represent these table boundaries in HTML?",
            "I need HTML code for this table with marked boundaries.",
            "Identify the table structure and convert to HTML.",
            "Parse this bordered table and generate HTML code.",
            "Please convert this marked table region to HTML.",
            "Provide HTML code based on the table boundaries shown.",
            "Recognize this table structure and output HTML.",
            "Transform these table markings into HTML code.",
            "Translate these table boundaries into HTML format.",
            "Turn these table annotations into valid HTML.",
            "What HTML structure matches these table markings?",
            "Write HTML code that matches this table layout.",

            # Chinese prompts
            "分析这个带标记的表格并转换成HTML代码。",
            "把这个已标注边界的表格转成HTML格式。",
            "根据表格的边界标记生成HTML代码。",
            "将这个带框的表格区域转换为HTML结构。",
            "能否根据标记的边界生成对应的HTML代码？",
            "请解析这个已标注的表格并输出HTML。",
            "请根据表格的框线生成HTML代码。",
            "识别这个带标记的表格并转换成HTML。",
            "提取表格的结构信息并生成HTML代码。",
            "为这个已标注边界的表格创建HTML代码。",
            "依据表格的标记边界生成HTML结构。",
            "用HTML代码表示这个已标记的表格结构。",
            "这个带框的表格应该如何用HTML表示？",
            "这些表格边界标记对应的HTML是什么？",
            "这个表格的边界标记如何转换成HTML？",
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

        # 构建返回结果
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
