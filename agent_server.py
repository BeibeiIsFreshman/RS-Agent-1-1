# agent_server.py
import os
import uuid
import json
import time
from flask import Flask, request, jsonify, render_template_string, send_file, current_app
from flask_cors import CORS
from detection_agent import run_agent
from shared_state import get_uploaded_images, get_temp_dir
from pathlib import Path
import config

app = Flask(__name__)
CORS(app, resources=r'/*')

# 初始化全局状态
app.config['UPLOADED_IMAGES'] = {}
TEMP_IMAGE_DIR = get_temp_dir()

TEMPLATE_PATH = Path(__file__).parent / 'agent_web_template.html'
with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
    WEB_TEMPLATE = f.read()

@app.route('/')
def index():
    return render_template_string(WEB_TEMPLATE)

@app.route('/ai/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "no file"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "empty file"}), 400

    image_id = str(uuid.uuid4())
    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else 'jpg'
    filepath = TEMP_IMAGE_DIR / f"{image_id}.{ext}"
    file.save(str(filepath))

    get_uploaded_images()[image_id] = str(filepath)
    return jsonify({"image_id": image_id})

@app.route('/ai/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        data = {}

    user_query = data.get('query', '').strip()
    image_id = data.get('image_id', '').strip() if data.get('image_id') is not None else ''

    full_query = f"{user_query}\n[IMAGE_ID]{image_id}[/IMAGE_ID]" if image_id else user_query
    result = run_agent(full_query)
    return jsonify(result)

# 新增：下载结果图
@app.route('/download/<path:filename>')
def download_result(filename):
    file_path = Path(config.RESULT_IMG_DIR) / filename
    if file_path.exists() and file_path.is_file():
        return send_file(str(file_path), as_attachment=True, download_name=filename)
    return jsonify({"error": "文件不存在"}), 404

# 清理过期图片
def cleanup_old_images():
    import time
    now = time.time()
    images = get_uploaded_images()
    for image_id, path in list(images.items()):
        try:
            if now - os.path.getctime(path) > 3600:
                Path(path).unlink(missing_ok=True)
                del images[image_id]
        except:
            pass

if __name__ == '__main__':
    from my_inference import model
    import threading
    threading.Thread(target=lambda: [time.sleep(300), cleanup_old_images()], daemon=True).start()
    print("\n" + "="*60)
    print("DINOv3 LLM Agent 服务启动")
    print("="*60)
    print(f"  访问: http://{config.SERVER_HOST}:{config.SERVER_PORT}")
    print("="*60 + "\n")
    app.run(host=config.SERVER_HOST, port=config.SERVER_PORT, threaded=True)