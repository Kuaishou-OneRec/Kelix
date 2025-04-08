from flask import Flask, render_template, request, jsonify, send_file, url_for
import json
import os
from pathlib import Path
from recovlm.utils.media import get_pid_folder

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
CACHE_FOLDER = '/llm_reco/zhouyang12/.cache/Photo'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_file', methods=['POST'])
def load_file():
    data = request.get_json()
    file_path = data.get('file_path')
    start_index = data.get('start_index', 0)
    count = data.get('count', 10)
    
    if not file_path:
        return jsonify({'error': 'No file path provided'}), 400
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        items = []
        with open(file_path, 'r', encoding='utf-8') as f:
            # Skip to start_index
            for _ in range(start_index):
                try:
                    next(f)
                except StopIteration:
                    break
            
            # Read count items
            for _ in range(count):
                try:
                    line = next(f)
                    item = json.loads(line.strip())
                    items.append(item)
                except (StopIteration, json.JSONDecodeError):
                    break
        
        return jsonify({
            'items': items,
            'has_more': len(items) == count  # Indicates if there might be more items
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_media_info/<key>')
def get_media_info(key):
    json_path = get_pid_folder(
        key, Path(CACHE_FOLDER)) / f"{key}.json"
    
    if not os.path.exists(json_path):
        return jsonify({'error': 'Media info not found'}), 404
    
    with open(json_path, 'r', encoding='utf-8') as f:
        media_info = json.load(f)
    
    # Convert file paths to URLs
    if media_info['media_type'] == 'video':
        media_info['media_path'] = url_for('serve_media', key=key, filename=os.path.basename(media_info['media_path']))
    else:
        media_info['media_path'] = [url_for('serve_media', key=key, filename=os.path.basename(path)) 
                                  for path in media_info['media_path']]
    
    return jsonify(media_info)

@app.route('/media/<key>/<path:filename>')
def serve_media(key, filename):
    media_folder = get_pid_folder(key, Path(CACHE_FOLDER))
    file_path = media_folder / filename
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'Media file not found'}), 404
    
    return send_file(file_path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8888)