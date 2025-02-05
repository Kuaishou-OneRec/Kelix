from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import numpy as np
import json
import os
from PIL import Image
import io
import base64

app = Flask(__name__)

# 配置参数
DATA_PATH = ""  # 将被用户输入覆盖
ERROR_TYPES = [
    "未标注",
    "拼写错误",
    "难以辨认",
    "表格理解",
    "图表理解",
    "超高分辨率",
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

@app.route('/', methods=['GET', 'POST'])
def index():
    global DATA_PATH, ERROR_TYPES
    
    if request.method == 'POST':
        # 处理错误类型更新
        if 'update_error_types' in request.form:
            error_types = request.form.get('error_types', '')
            if error_types:
                ERROR_TYPES = [t.strip() for t in error_types.split(',')]
            return redirect(url_for('index'))
            
        # 获取数据文件路径
        data_path = request.form.get('data_path')
        if data_path:  # 如果提供了新的数据路径
            DATA_PATH = data_path
            return redirect(url_for('index'))
            
        # 获取保存路径
        save_path = request.form.get('save_path', 'labeled_data.json')
        
        # 如果没有设置数据路径，返回错误信息
        if not DATA_PATH:
            return render_template('index.html', 
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
            
        return render_template('index.html',
                            #  data=labeled_data,
                             label_options=ERROR_TYPES,
                             current_error_types=','.join(ERROR_TYPES),
                             current_data_path=DATA_PATH,
                             stats_message=stats_message)
        
        # TODO: 处理标注数据的保存逻辑
        
    try:
        data = load_data() if DATA_PATH else []
        df = pd.DataFrame(data)
        
        if not df.empty:
            # 添加resize_image列
            df['resized_image'] = df['image'].apply(lambda x: resize_base64_image(x, MAX_PIXELS))
            df['messages'] = df['messages'].apply(lambda x: extract_messages(x))
    except Exception as e:
        return render_template('index.html', 
                             error=f"加载数据失败: {str(e)}",
                             label_options=ERROR_TYPES)
    
    return render_template('index.html', 
                         data=df.to_dict('records') if not df.empty else [],
                         label_options=ERROR_TYPES,
                         current_error_types=','.join(ERROR_TYPES),
                         current_data_path=DATA_PATH)

if __name__ == '__main__':
    app.run(debug=True)