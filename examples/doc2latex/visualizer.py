# streamlit>=1.28.0
# Pillow>=10.0.0
# pdf2image>=1.16.0 
import streamlit as st
import json
import os
import subprocess
import tempfile
from PIL import Image

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

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
                st.error(f"LaTeX 编译错误: {result.stderr}")
                os.chdir(current_dir)
                return None
            
            # 将 PDF 转换为图片 (使用 pdftoppm 或 pdf2image)
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
            return image
            
    except Exception as e:
        st.error(f"渲染失败: {str(e)}")
        return None

def main():
    st.title("LaTeX 可视化工具")
    
    # 文件上传
    uploaded_file = st.file_uploader("选择 JSONL 文件", type=['jsonl'])
    
    if uploaded_file:
        # 保存上传的文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jsonl') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        # 加载数据
        data = load_jsonl(temp_path)
        os.unlink(temp_path)  # 删除临时文件
        
        # 创建选择器
        if data:
            selected_index = st.selectbox("选择样本", range(len(data)))
            sample = data[selected_index]
            
            # 显示原始图片
            st.subheader("原始图片")
            image_path = sample['source']
            if os.path.exists(image_path):
                image = Image.open(image_path)
                st.image(image)
            else:
                st.error("图片文件不存在")
            
            # 显示 LaTeX 代码
            st.subheader("LaTeX 代码")
            latex_code = sample['responses'][0]  # 假设第一个响应是 LaTeX 代码
            st.code(latex_code, language='latex')
            
            # 显示渲染结果
            st.subheader("渲染结果")
            # 提取实际的 LaTeX 代码（去除 Markdown 代码块标记）
            latex_content = latex_code.replace("```latex\n", "").replace("\n```", "")
            rendered_image = render_latex(latex_content)
            if rendered_image:
                st.image(rendered_image)
            else:
                st.error("LaTeX 渲染失败")

if __name__ == "__main__":
    main() 