"""
极简小目标检测模型 - 基于DINOv3
去除所有复杂机制，只保留核心功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms
import numpy as np
import config


class SimpleDetectionHead(nn.Module):
    """极简检测头 - 无anchor，直接预测"""

    def __init__(self, in_channels=256, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

        # 共享特征提取
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # 分类分支：每个位置预测num_classes个分数
        self.cls_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1)
        )

        # 回归分支：每个位置预测4个值 [x_offset, y_offset, w, h]
        self.reg_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 4, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """简单的初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # 分类头的bias初始化为负值，让初始预测倾向于背景
        bias_value = -np.log(9)
        nn.init.constant_(self.cls_head[-1].bias, bias_value)

    def forward(self, x):
        """前向传播"""
        feat = self.shared(x)
        cls_pred = self.cls_head(feat)
        reg_pred = self.reg_head(feat)
        return cls_pred, reg_pred


class SimpleDetector(nn.Module):
    """完整的简单检测器"""

    def __init__(self, dinov3_backbone, num_classes=2, freeze_backbone=True,
                 img_size=640, stride=16):
        super().__init__()
        self.backbone = dinov3_backbone
        self.num_classes = num_classes
        self.img_size = img_size
        self.stride = stride

        # 冻结backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # DINOv3输出是1024维
        backbone_dim = 1024

        # 特征投影：1024 -> 256
        self.proj = nn.Sequential(
            nn.Conv2d(backbone_dim, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # 检测头
        self.head = SimpleDetectionHead(in_channels=256, num_classes=num_classes)

    def forward(self, x):
        """前向传播"""
        B = x.size(0)

        # 提取DINOv3特征
        if not self.training:
            with torch.no_grad():
                features_dict = self.backbone.forward_features(x)
        else:
            features_dict = self.backbone.forward_features(x)

        # 获取patch tokens并重塑
        patch_tokens = features_dict['x_norm_patchtokens']
        B, num_patches, C = patch_tokens.shape
        H = W = int(num_patches ** 0.5)

        features = patch_tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # 特征投影
        features = self.proj(features)

        # 检测头
        cls_pred, reg_pred = self.head(features)

        return cls_pred, reg_pred

    def decode_predictions(self, cls_pred, reg_pred, conf_threshold=0.05, nms_threshold=0.5):
        """解码预测结果"""
        B, C, H, W = cls_pred.shape
        device = cls_pred.device

        results = []

        for b in range(B):
            cls_scores = torch.sigmoid(cls_pred[b])
            reg_b = reg_pred[b]

            # 生成坐标网格
            y_indices, x_indices = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij'
            )

            x_centers = (x_indices.float() + 0.5) * self.stride
            y_centers = (y_indices.float() + 0.5) * self.stride

            # Flatten
            cls_flat = cls_scores.reshape(C, -1).permute(1, 0)
            reg_flat = reg_b.reshape(4, -1).permute(1, 0)
            x_centers_flat = x_centers.reshape(-1)
            y_centers_flat = y_centers.reshape(-1)

            max_scores, labels = cls_flat.max(dim=1)

            # 置信度过滤
            keep = max_scores > conf_threshold
            if not keep.any():
                results.append({
                    'boxes': torch.zeros((0, 4), device=device),
                    'scores': torch.zeros((0,), device=device),
                    'labels': torch.zeros((0,), dtype=torch.long, device=device)
                })
                continue

            scores = max_scores[keep]
            labels = labels[keep]
            reg_keep = reg_flat[keep]
            x_centers_keep = x_centers_flat[keep]
            y_centers_keep = y_centers_flat[keep]

            # 解码bbox
            dx = reg_keep[:, 0]
            dy = reg_keep[:, 1]
            w = torch.exp(reg_keep[:, 2].clamp(min=-10, max=10))
            h = torch.exp(reg_keep[:, 3].clamp(min=-10, max=10))

            cx = x_centers_keep + dx * self.stride
            cy = y_centers_keep + dy * self.stride

            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            boxes = torch.stack([x1, y1, x2, y2], dim=1)

            boxes[:, 0::2].clamp_(min=0, max=self.img_size)
            boxes[:, 1::2].clamp_(min=0, max=self.img_size)

            # 过滤无效框
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            valid = (widths > 1) & (heights > 1)

            boxes = boxes[valid]
            scores = scores[valid]
            labels = labels[valid]

            # NMS
            keep_nms = []
            for cls_id in range(self.num_classes):
                cls_mask = labels == cls_id
                if not cls_mask.any():
                    continue

                cls_boxes = boxes[cls_mask]
                cls_scores = scores[cls_mask]

                keep_cls = nms(cls_boxes, cls_scores, nms_threshold)
                keep_nms.append(torch.where(cls_mask)[0][keep_cls])

            if keep_nms:
                keep_nms = torch.cat(keep_nms)
                boxes = boxes[keep_nms]
                scores = scores[keep_nms]
                labels = labels[keep_nms]
            else:
                boxes = torch.zeros((0, 4), device=device)
                scores = torch.zeros((0,), device=device)
                labels = torch.zeros((0,), dtype=torch.long, device=device)

            results.append({
                'boxes': boxes,
                'scores': scores,
                'labels': labels
            })

        return results


def get_model():
    """加载模型 - 包含backbone和权重"""
    print("🚀 初始化模型...")
    
    # 加载backbone
    dinov3_backbone = torch.hub.load(
        config.REPO_DIR, 'dinov3_vitl16',
        source='local',
        weights=config.BACKBONE_WEIGHTS
    )
    
    # 构建检测器
    model = SimpleDetector(
        dinov3_backbone=dinov3_backbone,
        num_classes=config.NUM_CLASSES,
        freeze_backbone=config.FREEZE_BACKBONE,
        img_size=config.IMG_SIZE,
        stride=config.STRIDE
    )
    
    # 加载权重
    device = torch.device(config.DEVICE)
    print(f"  ├─ 加载权重: {config.HEAD_WEIGHTS}")
    
    try:
        checkpoint = torch.load(config.HEAD_WEIGHTS, map_location=device, weights_only=False)
    except RuntimeError:
        checkpoint = torch.load(config.HEAD_WEIGHTS, map_location=device)
    
    # 获取state_dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 去除module.前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace('module.', '')] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    print("  └─ ✓ 模型加载完成\n")
    
    return model