"""
配置文件 - 存储所有路径和参数
"""
from pathlib import Path

# ============ 路径配置 ============
# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

# 模型相关路径
REPO_DIR = '/media/tbb/shuju/RS-Agent/dinov3-main'
BACKBONE_WEIGHTS = '/media/tbb/shuju/Vehicle/weight/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'
HEAD_WEIGHTS = '/media/tbb/shuju/Vehicle/Rsult/weight/best.pth'

# 类别文件路径
CLASS_FILE = '/media/tbb/shuju/RS-Agent/AgentV-1-1/class_with_id.txt'

# 结果保存路径
RESULT_DIR = PROJECT_ROOT / 'result'
RESULT_IMG_DIR = PROJECT_ROOT / 'result_img'
TEMP_DIR = Path('/tmp')

# ============ 模型参数 ============
NUM_CLASSES = 3
IMG_SIZE = 640
STRIDE = 16
FREEZE_BACKBONE = True

# ============ 检测参数 ============
DEFAULT_CONFIDENCE = 0.1
NMS_THRESHOLD = 0.5

# 图像尺寸限制
MIN_IMAGE_SIZE = 100
MAX_IMAGE_SIZE = 10000

# ============ 服务器配置 ============
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 5051
MAX_TASK_NUM = 2

# ============ 设备配置 ============
DEVICE = 'cuda'  # 'cuda' or 'cpu'

# ============ 其他配置 ============
# 创建必要的目录
RESULT_DIR.mkdir(exist_ok=True)
RESULT_IMG_DIR.mkdir(exist_ok=True)


def load_label_map(class_file=CLASS_FILE):
    """从文件加载类别映射"""
    label_map = {}
    try:
        with open(class_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        idx, name = parts
                        label_map[int(idx)] = name
        print(f"✓ 加载类别映射: {label_map}")
    except Exception as e:
        print(f"⚠️  加载类别文件失败: {e}")
        # 默认映射
        label_map = {0: "Missle", 1: "car", 2: "tank"}
    
    return label_map


# 全局类别映射
LABEL_MAP = load_label_map()


def get_supported_classes():
    return list(LABEL_MAP.values())