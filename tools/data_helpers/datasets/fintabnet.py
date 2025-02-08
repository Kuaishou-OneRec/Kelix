import os
import json
import base64
import numpy as np
from PIL import Image, ImageDraw
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup as bs
from html import escape
from tools.data_helpers.datasets.dataset import DistDataset
from io import BytesIO
import random  # 添加 random 导入

class FinTabNetDataset(DistDataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.jsonl_path = os.path.join(path, "FinTabNet_1.0.0_table_train.jsonl")
        
        # 移除固定颜色定义，因为我们将使用随机颜色
        self.categories = ["table", "cell"]
        
        # Read and shard the jsonl file
        with open(self.jsonl_path, "r") as f:
            lines = f.readlines()
        shard_size = len(lines) // self.world_size
        self.lines = lines[self.rank * shard_size: (self.rank + 1) * shard_size]
    
    def __len__(self):
        return len(self.lines)

    def markup_annotations(self, image, annotations, pdf_height):
        """只绘制 table 的边框，使用随机颜色"""
        draw = ImageDraw.Draw(image, 'RGBA')
        for annotation in annotations:
            # 只处理 table (category_id == 1)
            if annotation['category_id'] != 1:
                continue
                
            # 生成随机颜色
            random_color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            
            # Draw bbox with coordinate conversion
            orig_annotation = annotation['bbox'].copy()
            annotation['bbox'][3] = pdf_height - orig_annotation[1]
            annotation['bbox'][1] = pdf_height - orig_annotation[3]
            
            # Draw rectangle with random color
            draw.rectangle(
                (annotation['bbox'][0],
                 annotation['bbox'][1],
                 annotation['bbox'][2],
                 annotation['bbox'][3]),
                outline=random_color + (255,),
                width=2
            )
            
        return np.array(image)

    def format_html(self, sample):
        """Format HTML table from the sample data"""
        html = ""
        annotations = []
        
        # Process cells
        for token in sample["html"]["cells"]:
            if "bbox" in token:
                annotations.append({"category_id": 2, "bbox": token["bbox"]})
        
        # Build HTML table
        cnt = 0
        for token in sample["html"]["structure"]["tokens"]:
            html += token
            if token == "<td>":
                html += "".join(sample["html"]["cells"][cnt]["tokens"])
                cnt += 1
                
        # Add table annotation
        annotations.append({"category_id": 1, "bbox": sample["bbox"]})
        
        # Wrap with full HTML structure
        html = f'''<html>
                   <head>
                   <meta charset="UTF-8">
                   <style>
                   table, th, td {{
                     border: 1px solid black;
                     font-size: 10px;
                   }}
                   </style>
                   </head>
                   <body>
                   <table frame="hsides" rules="groups" width="100%">
                     {html}
                   </table>
                   </body>
                   </html>'''
                   
        # Prettify HTML
        soup = bs(html)
        return soup.prettify(), annotations

    def __iter__(self):
        for line in self.lines:
            if line.strip() != '':
                sample = json.loads(line)
                filename = sample['filename']
                
                # Process PDF file
                pdf_path = os.path.join(self.path, "pdf", filename)
                if os.path.exists(pdf_path):
                    # Get PDF dimensions
                    pdf_page = PdfReader(open(pdf_path, 'rb')).pages[0]
                    pdf_shape = pdf_page.mediabox
                    pdf_height = pdf_shape[3] - pdf_shape[1]
                    pdf_width = pdf_shape[2] - pdf_shape[0]
                    
                    # Convert PDF to image
                    converted_images = convert_from_path(pdf_path, size=(pdf_width, pdf_height))
                    img = converted_images[0]
                    
                    # Format HTML and get annotations
                    html, annotations = self.format_html(sample)
                    
                    # Apply annotations markup
                    marked_image = self.markup_annotations(img, annotations, pdf_height)
                    
                    # Convert image to base64
                    img_byte_arr = Image.fromarray(marked_image)
                    img_byte_arr = img_byte_arr.convert('RGB')
                    img_buffer = BytesIO()
                    img_byte_arr.save(img_buffer, format='JPEG')
                    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('ascii')
                    
                    # Prepare output
                    data = {
                        'filename': filename,
                        'image': img_base64,
                        'html': html,
                        'annotations': annotations
                    }
                    yield data
