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

def read_parquet_with_nrows(data_path, nrows=None):
    """读取parquet文件，支持行数限制和HDFS路径"""
    # 判断是否为HDFS路径
    is_hdfs = data_path.startswith('viewfs://')
    
    if is_hdfs:
        import pyarrow.hdfs as hdfs
        # 使用默认配置连接HDFS
        fs = hdfs.connect()
        # 获取目录下所有parquet文件
        files = fs.ls(data_path) if fs.isdir(data_path) else [data_path]
        files = [f for f in files if f.endswith('.parquet')]
    else:
        # 本地文件系统
        if os.path.isdir(data_path):
            files = glob.glob(os.path.join(data_path, '*.parquet'))
        else:
            files = [data_path]
    
    # 读取数据
    dfs = []
    rows_read = 0
    
    for file in files:
        if nrows is not None and rows_read >= nrows:
            break
            
        # 读取单个文件
        table = pq.read_table(file)
        df = table.to_pandas()
        
        if nrows is not None:
            remaining_rows = nrows - rows_read
            if len(df) > remaining_rows:
                df = df.iloc[:remaining_rows]
            
        dfs.append(df)
        rows_read += len(df)
    
    if not dfs:
        return pd.DataFrame()
        
    return pd.concat(dfs, ignore_index=True)

@app.route('/visualize_data', methods=['GET', 'POST'])
def visualize_data():
    """数据可视化页面"""
    if request.method == 'POST':
        data_path = request.form.get('data_path')
        nrows = request.form.get('nrows', type=int)  # 获取nrows参数
        
        if data_path:
            try:
                # 添加加载状态返回
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    # 使用新的读取函数
                    df = read_parquet_with_nrows(data_path, nrows)
                    
                    # 处理数据
                    samples = []
                    for _, row in df.iterrows():
                        sample = {
                            'source': row['source'],
                            'images': json.loads(row['images']) if row['images'] else None,
                            'messages': json.loads(row['messages']) if row['messages'] else None,
                            'segments': json.loads(row['segments']) if row['segments'] else None
                        }
                        samples.append(sample)
                    return jsonify({'success': True, 'samples': samples})
                return render_template('visualize_data.html')
            except Exception as e:
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({'success': False, 'error': str(e)})
                return render_template('visualize_data.html', error=f"加载数据失败: {str(e)}")
        else:
            return render_template('visualize_data.html', 
                                error="请提供文件路径")
    
    return render_template('visualize_data.html')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Flask application')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the server on')
    
    args = parser.parse_args()
    app.run(debug=True, host=args.host, port=args.port)