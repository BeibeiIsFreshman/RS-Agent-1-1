from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, ToolException
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, ChatMessagePromptTemplate
from pydantic import SecretStr
import config
import uuid
import base64
import json
from pathlib import Path

# ---------- 加载大模型，需要用户自己设计对应的API接口，可以利用阿里巴巴的百炼平台获取 ----------
llm = ChatOpenAI(
    model="",
    api_key=SecretStr(""),
    base_url="",
    streaming=True
)

# ---------- 系统提示 ----------
system_message_template = ChatMessagePromptTemplate.from_template(
    template="你是一名{role}专家，擅长{domain}。请根据用户查询智能调用工具。如果工具返回错误，请直接回复用户。",
    role="system",
)

human_message_template = ChatMessagePromptTemplate.from_template(
    template="用户问题：{question}",
    role="CHA",
)

chat_prompt_template = ChatPromptTemplate.from_messages([
    system_message_template,
    human_message_template,
])

# ---------- 工具参数 ----------
class DetectInputArgs(BaseModel):
    image_id: str = Field(description="上传图片返回的 image_id")
    confidence: float = Field(default=0.1, description="置信度阈值")
    target_class: str = Field(default="", description="要检测的目标类别")

# ---------- 检测工具 ----------
# common.py → detect_image 函数
@tool(
    description="使用DINOv3模型检测图像中的指定目标。只支持 config.LABEL_MAP 中的类别。",
    args_schema=DetectInputArgs,
    return_direct=False,
)
def detect_image(image_id: str = "", confidence: float = 0.1, target_class: str = "") -> dict:
    from my_inference import mmyolo_test
    from shared_state import get_uploaded_images
    import uuid, json
    import shutil  # ← 新增
    from pathlib import Path
    import config

    uploaded_images = get_uploaded_images()

    if not image_id or image_id not in uploaded_images:
        return {"error": "图像未找到或已过期，请重新上传。"}

    img_path = uploaded_images[image_id]
    task_id = str(uuid.uuid4())

    detect_path = config.TEMP_DIR / f"{task_id}.jpg"
    shutil.copy(img_path, detect_path)  # ← 正确复制

    task = {
        "taskId": task_id,
        "filePath": str(detect_path),
        "confidence": confidence,
    }

    result = mmyolo_test(task)

    json_path = config.RESULT_DIR / f"{task_id}.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    detect_path.unlink(missing_ok=True)

    return result

def create_detection_tools():
    return [detect_image]