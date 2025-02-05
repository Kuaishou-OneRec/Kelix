from flask import Flask, request, render_template
import pandas as pd
import json
import os

app = Flask(__name__)

# 配置参数
CSV_PATH = "data.csv"          # 原始数据文件路径
JSON_PATH = "labeled_data.json"  # 标注结果保存路径
LABEL_OPTIONS = ["类型A", "类型B", "类型C"]  # 可选的标注类型

def load_data():
    """加载数据并确保保存目录存在"""
    os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
    return pd.read_csv(CSV_PATH)

@app.route('/', methods=['GET', 'POST'])
def index():
    df = load_data()
    
    if request.method == 'POST':
        # 处理标注结果
        labels = {k.split("_")[1]: v for k, v in request.form.items() if k.startswith("label_")}
        
        # 添加标注列并保存
        df["label"] = df.index.astype(str).map(labels)
        df.to_json(JSON_PATH, orient="records", force_ascii=False)
        
    # 转换为字典列表便于模板渲染
    data = df.to_dict(orient='records')
    return render_template('index.html', 
                         data=data,
                         label_options=LABEL_OPTIONS)

if __name__ == '__main__':
    app.run(debug=True)