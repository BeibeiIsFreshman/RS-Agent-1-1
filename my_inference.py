# my_inference.py
"""
推理模块 - 处理检测任务
"""
import base64
import cv2
import numpy as np
from pathlib import Path
import config
from detect_tool import detect_large_image
from SimpleDetector import get_model

# 全局模型初始化
print("="*60)
print("初始化检测系统...")
print("="*60)
model = get_model()
print("系统初始化完成\n")


def mmyolo_test(task):
    """
    执行检测任务
    
    Args:
        task: 任务字典，包含 filePath, confidence, taskId 等
        
    Returns:
        检测结果字典（不含大图 base64）
    """
    img_path = task.get("filePath", "")
    confidence_threshold = task.get("confidence", config.DEFAULT_CONFIDENCE)
    task_id = task.get("taskId", "unknown")
    
    print(f"\n{'='*60}")
    print(f"开始检测任务: {task_id}")
    print(f"   图片路径: {img_path}")
    print(f"   置信度阈值: {confidence_threshold}")
    print(f"{'='*60}\n")
    
    # 检查文件是否存在
    if not Path(img_path).exists():
        print(f"文件不存在: {img_path}")
        return {
            "taskId": task_id,
            "targets": [],
            "targetsNum": 0,
            "error": "文件不存在"
        }
    
    try:
        # 执行检测
        detections, result_img = detect_large_image(
            model, 
            img_path, 
            confidence=confidence_threshold
        )
        
        # 获取图像尺寸
        h, w = result_img.shape[:2]
        img_size = f"{w}×{h}"
        
        # 保存结果图像
        save_path = config.RESULT_IMG_DIR / f"{Path(img_path).stem}_result.jpg"
        cv2.imwrite(str(save_path), result_img)
        print(f"结果图已保存: {save_path}")
        
        # 构建目标列表
        targets = []
        for box, score, label in zip(
            detections['boxes'], 
            detections['scores'], 
            detections['labels']
        ):
            targets.append({
                "bbox": [float(x) for x in box],
                "score": float(score),
                "label": int(label),
                "label_name": config.LABEL_MAP.get(int(label), f'class{label}')
            })
        
        print(f"\n{'='*60}")
        print(f"检测完成")
        print(f"   检测到目标数: {len(targets)}")
        print(f"   图像尺寸: {img_size}")
        print(f"{'='*60}\n")
        
        return {
            "taskId": task_id,
            "targets": targets,
            "targetsNum": len(targets),
            "imageSize": img_size,
            "result_image_path": str(save_path)  # ← 关键：返回路径
        }
        
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"检测失败:\n{error_msg}")
        
        return {
            "taskId": task_id,
            "targets": [],
            "targetsNum": 0,
            "error": str(e)
        }