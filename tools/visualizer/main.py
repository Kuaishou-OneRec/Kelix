import sys
import os
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from flask import Flask, render_template
from tools.visualizer.routes.eval_routes import register_eval_routes
from tools.visualizer.routes.data_routes import register_data_routes
import argparse

def create_app():
    app = Flask(__name__)
    
    # 注册路由
    register_eval_routes(app)
    register_data_routes(app)
    
    @app.route('/')
    def index():
        """索引页面，提供导航到不同功能"""
        return render_template('index.html')
    
    return app

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Flask application')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the server on')
    parser.add_argument('--debug', type=bool, default=False, help='Debug mode')
    args = parser.parse_args()
    
    app = create_app()
    app.run(debug=args.debug, host=args.host, port=args.port)