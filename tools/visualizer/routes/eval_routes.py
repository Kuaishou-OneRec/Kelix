from flask import render_template, request, redirect, url_for
import pandas as pd
from tools.visualizer.utils.data_loader import load_data
from tools.visualizer.utils.text_utils import extract_messages, extract_image
from tools.visualizer.utils.image_utils import resize_base64_image
from tools.visualizer.config import DATA_PATH, ERROR_TYPES, MAX_PIXELS
import json

def register_eval_routes(app):
    @app.route('/visualize_eval_result', methods=['GET', 'POST'])
    def visualize_eval_result():
        """评估结果可视化"""
        global DATA_PATH, ERROR_TYPES
        
        if request.method == 'POST':
            if 'update_error_types' in request.form:
                error_types = request.form.get('error_types', '')
                if error_types:
                    ERROR_TYPES = [t.strip() for t in error_types.split(',')]
                return redirect(url_for('visualize_eval_result'))
                
            data_path = request.form.get('data_path')
            if data_path:
                DATA_PATH = data_path
                return redirect(url_for('visualize_eval_result'))
                
            save_path = request.form.get('save_path', 'labeled_data.json')
            
            if not DATA_PATH:
                return render_template('visualize_eval_result.html', 
                                    error="请先设置数据文件路径",
                                    label_options=ERROR_TYPES)

            labeled_data = []
            data = load_data(DATA_PATH)
            label_stats = {}
            
            for i in range(len(data)):
                label = request.form.get(f'label_{i}')
                item = data[i].copy()
                item['label'] = label
                labeled_data.append(item)
                label_stats[label] = label_stats.get(label, 0) + 1
                
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(labeled_data, f, ensure_ascii=False, indent=2)
                
            stats_message = "标注统计结果：\n"
            total = len(labeled_data)
            for label, count in label_stats.items():
                percentage = (count / total) * 100
                stats_message += f"{label}: {count}个 ({percentage:.1f}%)\n"
                
            return render_template('visualize_eval_result.html',
                                 label_options=ERROR_TYPES,
                                 current_error_types=','.join(ERROR_TYPES),
                                 current_data_path=DATA_PATH,
                                 stats_message=stats_message)
            
        try:
            data = load_data(DATA_PATH) if DATA_PATH else []
            df = pd.DataFrame(data)

            if 'image' not in df.columns:
                df['image'] = df['messages'].apply(lambda x: extract_image(x))
            
            if not df.empty:
                df['resized_image'] = df['image'].apply(lambda x: resize_base64_image(x, MAX_PIXELS))
                df['messages'] = df['messages'].apply(lambda x: extract_messages(x))
                
                # 按source分组计算统计信息
                dataset_stats = {}
                if 'dataset_name' in df.columns:
                    # 如果数据还未标注，只显示各source的总数
                    for dataset_name, group in df.groupby('dataset_name'):
                        dataset_stats[dataset_name] = {
                            "count": len(group),
                            "ratio": len(group) / len(df)
                        }
                
        except Exception as e:
            return render_template('visualize_eval_result.html', 
                                 error=f"加载数据失败: {str(e)}",
                                 label_options=ERROR_TYPES)
        
        return render_template('visualize_eval_result.html', 
                             data=df.to_dict('records') if not df.empty else [],
                             label_options=ERROR_TYPES,
                             current_error_types=','.join(ERROR_TYPES),
                             current_data_path=DATA_PATH,
                             dataset_stats=dataset_stats) 