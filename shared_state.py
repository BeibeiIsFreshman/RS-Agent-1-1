# shared_state.py
from flask import current_app

def get_uploaded_images():
    """安全获取 Flask app 的全局图片字典"""
    if 'UPLOADED_IMAGES' not in current_app.config:
        current_app.config['UPLOADED_IMAGES'] = {}
    return current_app.config['UPLOADED_IMAGES']

def get_temp_dir():
    from config import TEMP_DIR
    temp_dir = TEMP_DIR / "uploaded_images"
    temp_dir.mkdir(exist_ok=True)
    return temp_dir