from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import json
import os
import subprocess
import tempfile
from PIL import Image
import base64
from io import BytesIO
import time
import random  # 添加在文件顶部的导入部分

app = Flask(__name__)
app.secret_key = 'latex_viewer_secret_key'

def load_jsonl(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    except Exception as e:
        return None, str(e)

def render_latex(latex_code):
    try:
        # 创建临时目录
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 保存 LaTeX 文档到临时文件
            tex_path = os.path.join(tmp_dir, "temp.tex")
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(latex_code)
            
            # 切换到临时目录执行命令
            current_dir = os.getcwd()
            os.chdir(tmp_dir)
            
            # 编译 LaTeX (pdflatex)
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "temp.tex"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode != 0:
                os.chdir(current_dir)
                return None, f"LaTeX 编译错误: {result.stderr}"
            
            # 将 PDF 转换为图片
            try:
                # 尝试使用 pdf2image
                from pdf2image import convert_from_path
                images = convert_from_path("temp.pdf")
                image = images[0]
            except ImportError:
                # 如果 pdf2image 不可用，尝试使用 pdftoppm
                img_path = os.path.join(tmp_dir, "output.png")
                subprocess.run(
                    ["pdftoppm", "-png", "-singlefile", "temp.pdf", "output"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                image = Image.open("output.png")
            
            # 返回工作目录
            os.chdir(current_dir)
            
            # 将图片转换为 base64 以便在网页中显示
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return img_str, None
            
    except Exception as e:
        return None, f"渲染失败: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 获取文件路径
        file_path = request.form.get('file_path', '').strip()
        
        if not file_path:
            flash('请输入文件路径')
            return redirect(request.url)
        
        if not os.path.exists(file_path):
            flash(f'文件不存在: {file_path}')
            return redirect(request.url)
        
        # 加载数据
        data = load_jsonl(file_path)
        
        if isinstance(data, tuple) and data[0] is None:
            flash(f'加载文件失败: {data[1]}')
            return redirect(request.url)
        
        # 将数据存储在会话中
        session['file_path'] = file_path
        # 不直接存储数据到会话中，而是保存文件路径
        # 这样可以避免会话大小限制问题
        return redirect(url_for('view_data', index=0))
    
    return render_template('index.html')

@app.route('/view/<int:index>')
def view_data(index):
    file_path = session.get('file_path')
    if not file_path:
        flash('请先指定JSONL文件路径')
        return redirect(url_for('index'))
    
    # 每次从文件重新加载数据，避免会话大小问题
    data = load_jsonl(file_path)
    if isinstance(data, tuple):
        flash(f'加载文件失败: {data[1]}')
        return redirect(url_for('index'))
    
    if index < 0 or index >= len(data):
        flash('无效的索引')
        return redirect(url_for('index'))
    
    sample = data[index]
    
    # 获取原始图片
    image_path = sample['source']
    original_image = None
    if os.path.exists(image_path):
        with open(image_path, 'rb') as img_file:
            original_image = base64.b64encode(img_file.read()).decode()
    
    # 获取LaTeX代码
    latex_code = sample['responses'][0]  # 假设第一个响应是 LaTeX 代码
    
    # 渲染LaTeX
    latex_content = latex_code.replace("```latex\n", "").replace("\n```", "")
    rendered_image, error = render_latex(latex_content)
    
    # 提取额外字段
    source = sample.get('source', 'N/A')
    url = sample.get('__url__', 'N/A')
    key = sample.get('__key__', 'N/A')
    
    return render_template('view.html', 
                          index=index, 
                          total=len(data), 
                          original_image=original_image,
                          latex_code=latex_code,
                          rendered_image=rendered_image,
                          error=error,
                          source=source,
                          url=url,
                          key=key)

@app.route('/random')
def random_sample():
    file_path = session.get('file_path')
    if not file_path:
        flash('请先指定JSONL文件路径')
        return redirect(url_for('index'))
    
    # 加载数据
    data = load_jsonl(file_path)
    if isinstance(data, tuple):
        flash(f'加载文件失败: {data[1]}')
        return redirect(url_for('index'))
    
    if not data:
        flash('数据为空')
        return redirect(url_for('index'))
    
    # 随机选择一个索引
    random_index = random.randint(0, len(data) - 1)
    
    # 重定向到随机选择的样例
    return redirect(url_for('view_data', index=random_index))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True) 