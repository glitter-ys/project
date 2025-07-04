#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于深度学习的漂浮窗无障碍合规性检测器

使用YOLO等目标检测模型来检测漂浮窗、关闭按钮和文字按钮
"""

import cv2
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Any
import logging
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import yaml

class DeepLearningFloatingWindowDetector:
    """
    基于深度学习的漂浮窗检测器
    """
    
    def __init__(self, model_path=None, device='auto'):
        """
        初始化深度学习检测器
        
        Args:
            model_path: 训练好的模型路径，如果为None则使用预训练模型
            device: 计算设备 ('auto', 'cpu', 'cuda')
        """
        self.logger = logging.getLogger(__name__)
        
        # 设置设备
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.logger.info(f"使用设备: {self.device}")
        
        # 类别定义
        self.class_names = ['floating_window', 'close_button', 'text_button']
        self.class_to_id = {name: i for i, name in enumerate(self.class_names)}
        
        # 加载模型
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            self.logger.info(f"已加载训练好的模型: {model_path}")
        else:
            # 使用预训练的YOLOv8模型
            self.model = YOLO('yolov8n.pt')  # 或 yolov8s.pt, yolov8m.pt
            self.logger.info("使用预训练YOLOv8模型")
    
    def create_training_dataset(self, annotation_file: str, images_dir: str, output_dir: str):
        """
        创建YOLO格式的训练数据集
        
        Args:
            annotation_file: 标注文件路径 (支持COCO JSON或自定义格式)
            images_dir: 图片文件夹路径
            output_dir: 输出数据集路径
        """
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)
        
        # 读取标注文件
        if annotation_file.endswith('.json'):
            annotations = self._parse_coco_annotations(annotation_file)
        else:
            annotations = self._parse_custom_annotations(annotation_file)
        
        # 划分训练集和验证集
        train_annotations, val_annotations = train_test_split(
            annotations, test_size=0.2, random_state=42
        )
        
        # 转换格式并保存
        self._convert_and_save_dataset(train_annotations, images_dir, 
                                     os.path.join(output_dir, 'images', 'train'),
                                     os.path.join(output_dir, 'labels', 'train'))
        
        self._convert_and_save_dataset(val_annotations, images_dir,
                                     os.path.join(output_dir, 'images', 'val'),
                                     os.path.join(output_dir, 'labels', 'val'))
        
        # 创建数据集配置文件
        self._create_dataset_yaml(output_dir)
        
        self.logger.info(f"数据集创建完成: {output_dir}")
        
    def _parse_coco_annotations(self, annotation_file: str) -> List[Dict]:
        """解析COCO格式的标注文件"""
        with open(annotation_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # 构建图片ID到文件名的映射
        image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
        
        # 按图片分组标注
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = {
                    'filename': image_id_to_filename[image_id],
                    'annotations': []
                }
            annotations_by_image[image_id]['annotations'].append(ann)
        
        return list(annotations_by_image.values())
    
    def _parse_custom_annotations(self, annotation_file: str) -> List[Dict]:
        """解析自定义格式的标注文件"""
        # 这里可以根据实际的标注格式进行解析
        # 示例格式：每行包含 filename,class,x,y,w,h
        annotations_by_image = {}
        
        with open(annotation_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split(',')
                filename = parts[0]
                class_name = parts[1]
                x, y, w, h = map(float, parts[2:6])
                
                if filename not in annotations_by_image:
                    annotations_by_image[filename] = {
                        'filename': filename,
                        'annotations': []
                    }
                
                annotations_by_image[filename]['annotations'].append({
                    'category_id': self.class_to_id.get(class_name, 0),
                    'bbox': [x, y, w, h]
                })
        
        return list(annotations_by_image.values())
    
    def _convert_and_save_dataset(self, annotations: List[Dict], images_dir: str, 
                                output_images_dir: str, output_labels_dir: str):
        """转换并保存数据集"""
        for ann_data in annotations:
            filename = ann_data['filename']
            image_path = os.path.join(images_dir, filename)
            
            if not os.path.exists(image_path):
                self.logger.warning(f"图片不存在: {image_path}")
                continue
            
            # 复制图片
            import shutil
            shutil.copy2(image_path, os.path.join(output_images_dir, filename))
            
            # 读取图片尺寸
            image = cv2.imread(image_path)
            if image is None:
                continue
            height, width = image.shape[:2]
            
            # 转换标注为YOLO格式
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(output_labels_dir, label_filename)
            
            with open(label_path, 'w') as f:
                for ann in ann_data['annotations']:
                    class_id = ann['category_id']
                    bbox = ann['bbox']
                    
                    # 转换为YOLO格式 (center_x, center_y, width, height)，归一化到[0,1]
                    x, y, w, h = bbox
                    center_x = (x + w/2) / width
                    center_y = (y + h/2) / height
                    norm_w = w / width
                    norm_h = h / height
                    
                    f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")
    
    def _create_dataset_yaml(self, output_dir: str):
        """创建数据集配置文件"""
        yaml_content = {
            'path': output_dir,
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        yaml_path = os.path.join(output_dir, 'dataset.yaml')
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
    
    def train_model(self, dataset_yaml: str, epochs: int = 100, imgsz: int = 640, 
                   batch_size: int = 16, output_dir: str = 'runs/train'):
        """
        训练模型
        
        Args:
            dataset_yaml: 数据集配置文件路径
            epochs: 训练轮数
            imgsz: 输入图片尺寸
            batch_size: 批处理大小
            output_dir: 输出目录
        """
        self.logger.info("开始训练模型...")
        
        # 训练模型
        results = self.model.train(
            data=dataset_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=self.device,
            project=output_dir,
            save_period=10,  # 每10轮保存一次
            patience=20,     # 早停耐心值
            plots=True,      # 保存训练图表
        )
        
        self.logger.info("模型训练完成!")
        return results
    
    def detect_objects(self, image: np.ndarray, conf_threshold: float = 0.3) -> List[Dict]:
        """
        使用训练好的模型检测对象
        
        Args:
            image: 输入图像
            conf_threshold: 置信度阈值
            
        Returns:
            检测结果列表
        """
        # 使用模型进行推理
        results = self.model(image, conf=conf_threshold, device=self.device)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    if class_id < len(self.class_names):
                        class_name = self.class_names[class_id]
                        
                        detections.append({
                            'bbox': (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                            'confidence': float(confidence),
                            'class': class_name,
                            'class_id': class_id
                        })
        
        return detections
    
    def check_accessibility_compliance_dl(self, image_path: str, conf_threshold: float = 0.3) -> Dict[str, Any]:
        """
        使用深度学习模型检查无障碍合规性
        
        Args:
            image_path: 图片路径
            conf_threshold: 置信度阈值
            
        Returns:
            检测结果字典
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            return {
                'image_path': image_path,
                'is_pass': 0,
                'error': 'Cannot load image',
                'detections': []
            }
        
        # 检测对象
        detections = self.detect_objects(image, conf_threshold)
        
        # 分类检测结果
        floating_windows = [d for d in detections if d['class'] == 'floating_window']
        close_buttons = [d for d in detections if d['class'] == 'close_button']
        text_buttons = [d for d in detections if d['class'] == 'text_button']
        
        # 判断是否通过测试
        if not floating_windows:
            # 没有检测到漂浮窗，通过
            is_pass = 1
            reason = "No floating windows detected"
        else:
            # 检测到漂浮窗，需要有关闭按钮或文字按钮
            has_close_button = len(close_buttons) > 0
            has_text_button = len(text_buttons) > 0
            
            is_pass = 1 if (has_close_button or has_text_button) else 0
            
            if is_pass:
                if has_close_button and has_text_button:
                    reason = "Has both close button and text button"
                elif has_close_button:
                    reason = "Has close button"
                elif has_text_button:
                    reason = "Has text button"
            else:
                reason = "No close button or text button found"
        
        return {
            'image_path': image_path,
            'is_pass': is_pass,
            'reason': reason,
            'detections': detections,
            'floating_windows': floating_windows,
            'close_buttons': close_buttons,
            'text_buttons': text_buttons
        }
    
    def batch_check_dl(self, image_folder: str, output_file: str = "pred_dl.json", 
                      conf_threshold: float = 0.3):
        """
        使用深度学习模型批量检测
        
        Args:
            image_folder: 图片文件夹路径
            output_file: 输出结果文件路径
            conf_threshold: 置信度阈值
        """
        results = []
        
        # 支持的图片格式
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # 遍历文件夹中的所有图片
        for filename in os.listdir(image_folder):
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in supported_formats:
                image_path = os.path.join(image_folder, filename)
                
                try:
                    result = self.check_accessibility_compliance_dl(image_path, conf_threshold)
                    
                    # 格式化输出结果
                    pred_result = {
                        "imgname": filename,
                        "Ispass": result['is_pass']
                    }
                    results.append(pred_result)
                    
                    self.logger.info(f"{filename}: {'PASS' if result['is_pass'] else 'FAIL'} - {result.get('reason', '')}")
                    
                except Exception as e:
                    self.logger.error(f"处理图片 {filename} 时出错: {str(e)}")
                    results.append({
                        "imgname": filename,
                        "Ispass": 0
                    })
        
        # 保存结果到JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"深度学习检测完成，结果已保存到: {output_file}")
        
        # 统计结果
        pass_count = sum(1 for r in results if r['Ispass'] == 1)
        fail_count = len(results) - pass_count
        self.logger.info(f"通过: {pass_count}, 不通过: {fail_count}")
        
        return results
    
    def create_sample_training_data(self, output_dir: str = "sample_training_data"):
        """
        创建示例训练数据
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建示例标注文件
        sample_annotations = [
            "sample1.jpg,floating_window,100,150,300,400",
            "sample1.jpg,close_button,350,160,30,30",
            "sample1.jpg,text_button,120,480,80,40",
            "sample2.jpg,floating_window,50,100,400,500",
            "sample2.jpg,text_button,200,520,120,50",
            "sample3.jpg,floating_window,80,200,350,300",
            "sample3.jpg,close_button,400,210,25,25",
        ]
        
        annotation_file = os.path.join(output_dir, "annotations.txt")
        with open(annotation_file, 'w', encoding='utf-8') as f:
            for ann in sample_annotations:
                f.write(ann + '\n')
        
        # 创建说明文件
        readme_content = """
# 漂浮窗检测训练数据说明

## 数据格式
标注文件格式：filename,class,x,y,w,h
- filename: 图片文件名
- class: 类别 (floating_window, close_button, text_button)
- x,y: 边界框左上角坐标
- w,h: 边界框宽度和高度

## 类别说明
1. floating_window: 漂浮窗区域
2. close_button: 关闭按钮 (包括X按钮、关闭图标等)
3. text_button: 文字按钮 (包含可点击文字的按钮区域)

## 使用方法
1. 准备图片数据，放在images文件夹中
2. 按照上述格式创建标注文件
3. 使用create_training_dataset()方法转换为YOLO格式
4. 使用train_model()方法训练模型
        """
        
        readme_file = os.path.join(output_dir, "README.md")
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        self.logger.info(f"示例训练数据已创建: {output_dir}")
    
    def visualize_detections(self, image_path: str, detections: List[Dict], 
                           output_path: str = None):
        """
        可视化检测结果
        
        Args:
            image_path: 原始图片路径
            detections: 检测结果
            output_path: 输出图片路径
        """
        image = cv2.imread(image_path)
        if image is None:
            return
        
        # 定义颜色
        colors = {
            'floating_window': (255, 0, 0),    # 蓝色
            'close_button': (0, 255, 0),       # 绿色
            'text_button': (0, 0, 255)         # 红色
        }
        
        # 绘制检测框
        for detection in detections:
            x, y, w, h = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            color = colors.get(class_name, (128, 128, 128))
            
            # 绘制边界框
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            
            # 绘制标签
            label = f"{class_name}: {confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(image, (x, y-text_height-10), (x+text_width, y), color, -1)
            cv2.putText(image, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 保存或显示结果
        if output_path:
            cv2.imwrite(output_path, image)
            self.logger.info(f"可视化结果已保存: {output_path}")
        else:
            cv2.imshow('Detection Results', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()