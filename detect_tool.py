"""
检测工具函数 - 图像切片、拼接、坐标映射等
"""
import cv2
import numpy as np
import torch
from typing import List, Tuple, Dict
from pathlib import Path
import config


def validate_image_size(img: np.ndarray) -> Tuple[bool, str]:
    """验证图像尺寸是否符合要求"""
    h, w = img.shape[:2]
    
    if h < config.MIN_IMAGE_SIZE or w < config.MIN_IMAGE_SIZE:
        return False, f"图像尺寸过小 ({w}x{h}), 最小要求 {config.MIN_IMAGE_SIZE}x{config.MIN_IMAGE_SIZE}"
    
    if h > config.MAX_IMAGE_SIZE or w > config.MAX_IMAGE_SIZE:
        return False, f"图像尺寸过大 ({w}x{h}), 最大限制 {config.MAX_IMAGE_SIZE}x{config.MAX_IMAGE_SIZE}"
    
    return True, "OK"


def slice_image(img: np.ndarray, tile_size: int = 640, overlap: int = 0) -> List[Dict]:
    """
    将大图切片成多个小块
    
    Args:
        img: 原始图像 (H, W, C)
        tile_size: 切片大小
        overlap: 重叠像素数
        
    Returns:
        List of dict with 'image', 'x_offset', 'y_offset'
    """
    h, w = img.shape[:2]
    tiles = []
    
    stride = tile_size - overlap
    
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # 计算切片区域
            x_end = min(x + tile_size, w)
            y_end = min(y + tile_size, h)
            
            # 如果切片太小，调整起始位置
            if x_end - x < tile_size and x > 0:
                x = max(0, x_end - tile_size)
            if y_end - y < tile_size and y > 0:
                y = max(0, y_end - tile_size)
            
            # 提取切片
            tile = img[y:y_end, x:x_end].copy()
            
            # 如果切片小于目标尺寸，进行padding
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=img.dtype)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded
            
            tiles.append({
                'image': tile,
                'x_offset': x,
                'y_offset': y,
                'tile_h': y_end - y,
                'tile_w': x_end - x
            })
    
    return tiles


def preprocess_image(img: np.ndarray, target_size: int = 640) -> torch.Tensor:
    """
    预处理图像用于模型推理
    
    Args:
        img: 输入图像 (H, W, C) BGR格式
        target_size: 目标尺寸
        
    Returns:
        处理后的tensor (1, 3, H, W)
    """
    # 确保图像是640x640
    if img.shape[0] != target_size or img.shape[1] != target_size:
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    # BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 归一化到 [0, 1]
    img_float = img_rgb.astype(np.float32) / 255.0
    
    # 转换为tensor: (H, W, C) -> (1, C, H, W)
    img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor


def merge_detections(all_detections: List[Dict], img_shape: Tuple[int, int], 
                     nms_threshold: float = 0.5) -> Dict:
    """
    合并所有切片的检测结果，并进行NMS
    
    Args:
        all_detections: 所有切片的检测结果
        img_shape: 原图尺寸 (H, W)
        nms_threshold: NMS阈值
        
    Returns:
        合并后的检测结果 {'boxes': [], 'scores': [], 'labels': []}
    """
    if not all_detections:
        return {'boxes': [], 'scores': [], 'labels': []}
    
    all_boxes = []
    all_scores = []
    all_labels = []
    
    # 收集所有检测框
    for det in all_detections:
        if len(det['boxes']) > 0:
            all_boxes.extend(det['boxes'])
            all_scores.extend(det['scores'])
            all_labels.extend(det['labels'])
    
    if not all_boxes:
        return {'boxes': [], 'scores': [], 'labels': []}
    
    # 转换为numpy数组
    boxes = np.array(all_boxes)
    scores = np.array(all_scores)
    labels = np.array(all_labels)
    
    # 裁剪到图像边界
    boxes[:, 0] = np.clip(boxes[:, 0], 0, img_shape[1])
    boxes[:, 1] = np.clip(boxes[:, 1], 0, img_shape[0])
    boxes[:, 2] = np.clip(boxes[:, 2], 0, img_shape[1])
    boxes[:, 3] = np.clip(boxes[:, 3], 0, img_shape[0])
    
    # 分类别进行NMS
    final_boxes = []
    final_scores = []
    final_labels = []
    
    for cls_id in range(config.NUM_CLASSES):
        cls_mask = labels == cls_id
        if not cls_mask.any():
            continue
        
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        
        # 使用torchvision的NMS
        boxes_tensor = torch.from_numpy(cls_boxes).float()
        scores_tensor = torch.from_numpy(cls_scores).float()
        
        from torchvision.ops import nms
        keep_idx = nms(boxes_tensor, scores_tensor, nms_threshold)
        keep_idx = keep_idx.numpy()
        
        final_boxes.extend(cls_boxes[keep_idx].tolist())
        final_scores.extend(cls_scores[keep_idx].tolist())
        final_labels.extend([cls_id] * len(keep_idx))
    
    return {
        'boxes': final_boxes,
        'scores': final_scores,
        'labels': final_labels
    }


def draw_detections(img: np.ndarray, detections: Dict, label_map: Dict = None) -> np.ndarray:
    """
    在图像上绘制检测框
    
    Args:
        img: 原始图像
        detections: 检测结果
        label_map: 类别映射
        
    Returns:
        绘制后的图像
    """
    if label_map is None:
        label_map = config.LABEL_MAP
    
    img_result = img.copy()
    boxes = detections['boxes']
    scores = detections['scores']
    labels = detections['labels']
    
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
    
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box)
        
        # 绘制框
        color = colors[label % len(colors)]
        cv2.rectangle(img_result, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签
        label_text = label_map.get(label, f'class{label}')
        label_str = f"{label_text}:{score:.2f}"
        
        (text_w, text_h), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_result, (x1, y1 - text_h - 4), 
                     (x1 + text_w, y1), color, -1)
        cv2.putText(img_result, label_str, (x1, y1 - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img_result


def detect_large_image(model, img_path: str, confidence: float = 0.1) -> Tuple[Dict, np.ndarray]:
    """
    检测大图 - 自动切片、检测、合并
    
    Args:
        model: 检测模型
        img_path: 图像路径
        confidence: 置信度阈值
        
    Returns:
        (检测结果, 结果图像)
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")
    
    # 验证尺寸
    valid, msg = validate_image_size(img)
    if not valid:
        raise ValueError(msg)
    
    h, w = img.shape[:2]
    print(f"📐 原图尺寸: {w}x{h}")
    
    # 判断是否需要切片
    if w <= config.IMG_SIZE and h <= config.IMG_SIZE:
        print("📦 图像尺寸小于640x640，直接检测")
        tiles = [{'image': img, 'x_offset': 0, 'y_offset': 0, 
                 'tile_h': h, 'tile_w': w}]
    else:
        print(f"✂️  图像较大，切片处理...")
        tiles = slice_image(img, tile_size=config.IMG_SIZE, overlap=50)
        print(f"   共切分为 {len(tiles)} 个切片")
    
    # 对每个切片进行检测
    all_detections = []
    device = torch.device(config.DEVICE)
    
    for idx, tile_info in enumerate(tiles):
        tile_img = tile_info['image']
        x_offset = tile_info['x_offset']
        y_offset = tile_info['y_offset']
        
        # 预处理
        img_tensor = preprocess_image(tile_img, config.IMG_SIZE).to(device)
        
        # 推理
        with torch.no_grad():
            cls_pred, reg_pred = model(img_tensor)
            preds = model.decode_predictions(
                cls_pred, reg_pred,
                conf_threshold=confidence,
                nms_threshold=config.NMS_THRESHOLD
            )
        
        # 提取检测结果
        detected_boxes = preds[0]['boxes'].cpu().numpy()
        detected_scores = preds[0]['scores'].cpu().numpy()
        detected_labels = preds[0]['labels'].cpu().numpy()
        
        if len(detected_boxes) > 0:
            print(f"   切片 {idx+1}/{len(tiles)}: 检测到 {len(detected_boxes)} 个目标")
            
            # 将坐标映射回原图
            mapped_boxes = []
            for box in detected_boxes:
                x1, y1, x2, y2 = box
                mapped_boxes.append([
                    x1 + x_offset,
                    y1 + y_offset,
                    x2 + x_offset,
                    y2 + y_offset
                ])
            
            all_detections.append({
                'boxes': mapped_boxes,
                'scores': detected_scores.tolist(),
                'labels': detected_labels.tolist()
            })
    
    # 合并所有检测结果
    print("🔗 合并检测结果...")
    final_detections = merge_detections(all_detections, (h, w), config.NMS_THRESHOLD)
    
    print(f"✅ 最终检测到 {len(final_detections['boxes'])} 个目标")
    
    # 绘制结果
    result_img = draw_detections(img, final_detections)
    
    return final_detections, result_img