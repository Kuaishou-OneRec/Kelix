from flask import render_template, request, jsonify
from tools.visualizer.utils.data_loader import read_parquet_with_nrows
from tools.visualizer.utils.text_utils import extract_boxes_from_text
from tools.visualizer.utils.image_utils import draw_boxes_on_image
import json

def register_data_routes(app):
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
                            sample = {
                                'source': row['source'],
                                'images': json.loads(row['images']) if row['images'] else None,
                                'messages': json.loads(row['messages']) if row['messages'] else None,
                                'segments': json.loads(row['segments']) if row['segments'] else None
                            }
                            
                            if sample['images'] and (sample['messages'] or sample['segments']):
                                boxes = []
                                polygons = []
                                
                                if sample['messages']:
                                    for msg in sample['messages']:
                                        if isinstance(msg.get('content'), list):
                                            for content in msg['content']:
                                                if content.get('type') == 'text':
                                                    b, p = extract_boxes_from_text(content['text'])
                                                    boxes.extend(b)
                                                    polygons.extend(p)
                                
                                if sample['segments']:
                                    for segment in sample['segments']:
                                        if segment.get('type') == 'text':
                                            b, p = extract_boxes_from_text(segment['text'])
                                            boxes.extend(b)
                                            polygons.extend(p)
                                
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