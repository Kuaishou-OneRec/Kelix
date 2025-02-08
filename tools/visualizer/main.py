from flask import Flask, request, render_template, redirect, url_for, jsonify
import pandas as pd
import numpy as np
import json
import os
from PIL import Image
import io
import base64
import argparse
import pyarrow.parquet as pq
import glob
from urllib.parse import urlparse
import random
import re
from PIL import ImageDraw

app = Flask(__name__)

# 配置参数
DATA_PATH = ""  # 将被用户输入覆盖
ERROR_TYPES = [
    "未标注",
    "拼写错误",
    "难以辨认",
    "表格理解",
    "图表理解",
    "图片理解",
    "超高分辨率",
    "文档理解",
    "手写公式"
]  # 默认错误类型列表
MAX_PIXELS = 512 * 28 * 28  # 设置最大像素值

def load_data():
    """加载数据并确保保存目录存在"""
    # os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
    data = json.load(open(DATA_PATH))
    return data

def extract_messages(data):
    """提取messages字段"""
    content = data['content']
    rst = ""
    for s in content:
        if s['type'] == 'text':
            rst += s['text']
    return rst

def extract_image(data):
    """提取image字段"""
    content = data['content']
    rst = []
    for s in content:
        if s['type'] == 'image':
            rst.append(s['image'])
    return rst

def resize_base64_image(base64_str, max_pixels):
    """调整base64编码图片的大小"""
    try:
        # 解码base64字符串
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))
        
        # 获取原始尺寸
        width, height = img.size
        
        # 如果图片尺寸小于最大像素，直接返回原图
        if width * height <= max_pixels:
            return base64_str
            
        # 计算调整比例
        ratio = np.sqrt(max_pixels / (width * height))
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        # 调整图片大小
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 转回base64
        buffer = io.BytesIO()
        resized_img.save(buffer, format=img.format)
        return base64.b64encode(buffer.getvalue()).decode()
    except:
        return None  # 如果处理失败，返回原图

@app.route('/')
def index():
    """索引页面，提供导航到不同功能"""
    return render_template('index.html')

@app.route('/visualize_eval_result', methods=['GET', 'POST'])
def visualize_eval_result():
    """原来的 index 功能，用于可视化评估结果"""
    global DATA_PATH, ERROR_TYPES
    
    if request.method == 'POST':
        # 处理错误类型更新
        if 'update_error_types' in request.form:
            error_types = request.form.get('error_types', '')
            if error_types:
                ERROR_TYPES = [t.strip() for t in error_types.split(',')]
            return redirect(url_for('visualize_eval_result'))
            
        # 获取数据文件路径
        data_path = request.form.get('data_path')
        if data_path:  # 如果提供了新的数据路径
            DATA_PATH = data_path
            return redirect(url_for('visualize_eval_result'))
            
        # 获取保存路径
        save_path = request.form.get('save_path', 'labeled_data.json')
        
        # 如果没有设置数据路径，返回错误信息
        if not DATA_PATH:
            return render_template('visualize_eval_result.html', 
                                error="请先设置数据文件路径",
                                label_options=ERROR_TYPES)

        # 获取标注结果
        labeled_data = []
        data = load_data()
        label_stats = {}  # 用于统计各标签数量
        
        for i in range(len(data)):
            label = request.form.get(f'label_{i}')
            item = data[i].copy()
            item['label'] = label
            labeled_data.append(item)
            
            # 统计标签数量
            label_stats[label] = label_stats.get(label, 0) + 1
            
        # 保存标注结果
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(labeled_data, f, ensure_ascii=False, indent=2)
            
        # 准备统计信息
        stats_message = "标注统计结果：\n"
        total = len(labeled_data)
        for label, count in label_stats.items():
            percentage = (count / total) * 100
            stats_message += f"{label}: {count}个 ({percentage:.1f}%)\n"
            
        return render_template('visualize_eval_result.html',
                            #  data=labeled_data,
                             label_options=ERROR_TYPES,
                             current_error_types=','.join(ERROR_TYPES),
                             current_data_path=DATA_PATH,
                             stats_message=stats_message)
        
        # TODO: 处理标注数据的保存逻辑
        
    try:
        data = load_data() if DATA_PATH else []
        df = pd.DataFrame(data)

        if 'image' not in df.columns:
            df['image'] = df['messages'].apply(lambda x: extract_image(x))
        
        if not df.empty:
            # 添加resize_image列
            df['resized_image'] = df['image'].apply(lambda x: resize_base64_image(x, MAX_PIXELS))
            df['messages'] = df['messages'].apply(lambda x: extract_messages(x))
    except Exception as e:
        return render_template('visualize_eval_result.html', 
                             error=f"加载数据失败: {str(e)}",
                             label_options=ERROR_TYPES)
    
    return render_template('visualize_eval_result.html', 
                         data=df.to_dict('records') if not df.empty else [],
                         label_options=ERROR_TYPES,
                         current_error_types=','.join(ERROR_TYPES),
                         current_data_path=DATA_PATH)

def read_parquet_with_nrows(data_path, nrows=None, shuffle=False):
    """读取parquet文件，支持行数限制和HDFS路径，并随机打乱数据"""
    # 判断是否为HDFS路径
    is_hdfs = data_path.startswith('viewfs://')
    
    if is_hdfs:
        import pyarrow.hdfs as hdfs
        # 使用默认配置连接HDFS
        fs = hdfs.connect()
        
        # 检查路径是否为文件
        if data_path.endswith('.parquet'):
            files = [data_path]  # 直接使用单个parquet文件
        else:
            # 获取目录下所有parquet文件
            files = fs.ls(data_path)
            files = [f for f in files if f.endswith('.parquet')]
    else:
        # 本地文件系统
        if os.path.isfile(data_path) and data_path.endswith('.parquet'):
            files = [data_path]  # 直接使用单个parquet文件
        elif os.path.isdir(data_path):
            files = glob.glob(os.path.join(data_path, '*.parquet'))
        else:
            files = [data_path]  # 如果是其他类型的路径，假设是文件

    # 随机打乱文件顺序
    if shuffle:
        random.shuffle(files)
    
    # 读取数据
    dfs = []
    rows_read = 0
    
    for file in files:
        if nrows is not None and rows_read >= nrows:
            break
            
        # 读取单个文件
        table = pq.read_table(file)
        df = table.to_pandas()
        
        # 随机采样数据
        if len(df) > 0 and shuffle:
            df = df.sample(frac=1.0, random_state=None)
        
        if nrows is not None:
            remaining_rows = nrows - rows_read
            if len(df) > remaining_rows:
                df = df.iloc[:remaining_rows]
            
        dfs.append(df)
        rows_read += len(df)
    
    if not dfs:
        return pd.DataFrame()
        
    return pd.concat(dfs, ignore_index=True)

def draw_boxes_on_image(base64_img: str, boxes: list, polygons: list = None) -> str:
    """Draw boxes and polygons on a base64 encoded image and return the modified base64 string"""
    try:
        # Decode base64 image
        img_data = base64.b64decode(base64_img)
        img = Image.open(io.BytesIO(img_data))
        
        # Create draw object
        draw = ImageDraw.Draw(img)
        
        # Get image dimensions
        width, height = img.size
        
        # Draw each box
        for box in boxes:
            # Convert normalized coordinates to actual image coordinates
            x1, y1 = box[0][0] * width / 1000, box[0][1] * height / 1000
            x2, y2 = box[1][0] * width / 1000, box[1][1] * height / 1000
            
            # Draw rectangle with thicker outline
            draw.rectangle([(x1, y1), (x2, y2)], outline='red', width=4)
            
        # Draw each polygon
        for points in polygons:
            # Convert normalized coordinates to actual image coordinates
            scaled_points = [
                (p[0] * width / 1000, p[1] * height / 1000)
                for p in points
            ]
            # Draw polygon with thicker outline
            draw.polygon(scaled_points, outline='blue', width=4)
        
        # Convert back to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        print(f"Error drawing shapes: {str(e)}")
        return base64_img

def extract_boxes_from_text(text: str) -> tuple:
    """Extract box and polygon coordinates from text containing box annotations"""
    boxes = []
    polygons = []
    
    # Extract regular boxes
    box_pattern = r'<\|box_start\|>\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)<\|box_end\|>'
    box_matches = re.findall(box_pattern, text)
    
    for match in box_matches:
        x1, y1, x2, y2 = map(int, match)
        boxes.append(((x1, y1), (x2, y2)))
    
    # Extract polygons (quads or any number of points)
    # First find all content between quad_start and quad_end tags
    quad_pattern = r'<\|quad_start\|>(.*?)<\|quad_end\|>'
    quad_matches = re.findall(quad_pattern, text)
    
    for match in quad_matches:
        # Extract all coordinate pairs from the matched content
        point_pattern = r'\((\d+),\s*(\d+)\)'
        points = re.findall(point_pattern, match)
        
        if points:
            # Convert all points to integer coordinates
            polygon_points = [(int(x), int(y)) for x, y in points]
            polygons.append(polygon_points)
    
    return boxes, polygons

@app.route('/visualize_data', methods=['GET', 'POST'])
def visualize_data():
    """数据可视化页面"""
    if request.method == 'POST':
        data_path = request.form.get('data_path')
        nrows = request.form.get('nrows', type=int)
        shuffle = request.form.get('shuffle') == 'true'
        
        if data_path:
            try:
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    df = read_parquet_with_nrows(data_path, nrows, shuffle)
                    
                    samples = []
                    for _, row in df.iterrows():
                        # Parse the basic sample data
                        sample = {
                            'source': row['source'],
                            'images': json.loads(row['images']) if row['images'] else None,
                            'messages': json.loads(row['messages']) if row['messages'] else None,
                            'segments': json.loads(row['segments']) if row['segments'] else None
                        }
                        
                        # Process messages and segments to find boxes
                        if sample['images'] and (sample['messages'] or sample['segments']):
                            boxes = []
                            polygons = []
                            
                            # Extract boxes from messages
                            if sample['messages']:
                                for msg in sample['messages']:
                                    if isinstance(msg.get('content'), list):
                                        for content in msg['content']:
                                            if content.get('type') == 'text':
                                                b, p = extract_boxes_from_text(content['text'])
                                                boxes.extend(b)
                                                polygons.extend(p)
                            
                            # Extract boxes from segments
                            if sample['segments']:
                                for segment in sample['segments']:
                                    if segment.get('type') == 'text':
                                        b, p = extract_boxes_from_text(segment['text'])
                                        boxes.extend(b)
                                        polygons.extend(p)
                            
                            # Draw boxes and polygons on images if any were found
                            if boxes or polygons:
                                for img_key in sample['images']:
                                    sample['images'][img_key] = draw_boxes_on_image(
                                        sample['images'][img_key], 
                                        boxes,
                                        polygons
                                    )
                        
                        samples.append(sample)
                    
                    return jsonify({'success': True, 'samples': samples})
                return render_template('visualize_data.html')
            except Exception as e:
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({'success': False, 'error': str(e)})
                return render_template('visualize_data.html', error=f"加载数据失败: {str(e)}")
        else:
            return render_template('visualize_data.html', error="请提供文件路径")
    
    return render_template('visualize_data.html')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Flask application')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the server on')
    parser.add_argument('--debug', type=bool, default=False, help='Debug mode')
    args = parser.parse_args()
    app.run(debug=args.debug, host=args.host, port=args.port)