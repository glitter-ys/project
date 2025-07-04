#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度学习模型训练脚本

演示如何训练自定义的漂浮窗检测模型
"""

import os
import sys
import argparse
import logging
from deep_learning_detector import DeepLearningFloatingWindowDetector

def create_sample_training_dataset():
    """创建示例训练数据集"""
    print("正在创建示例训练数据集...")
    
    # 创建训练图片文件夹
    os.makedirs("training_data/images", exist_ok=True)
    
    # 创建示例图片（实际使用时应该用真实的标注数据）
    import cv2
    import numpy as np
    
    # 创建示例图片1: 包含漂浮窗、关闭按钮和文字按钮
    img1 = np.ones((640, 480, 3), dtype=np.uint8) * 240
    cv2.rectangle(img1, (100, 100), (380, 400), (100, 100, 255), -1)  # 漂浮窗
    cv2.rectangle(img1, (100, 100), (380, 400), (0, 0, 0), 2)
    cv2.circle(img1, (360, 120), 12, (200, 200, 200), -1)  # 关闭按钮
    cv2.line(img1, (352, 112), (368, 128), (0, 0, 0), 2)
    cv2.line(img1, (352, 128), (368, 112), (0, 0, 0), 2)
    cv2.rectangle(img1, (150, 320), (230, 360), (100, 255, 100), -1)  # 文字按钮
    cv2.putText(img1, "OK", (175, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.imwrite("training_data/images/train_001.jpg", img1)
    
    # 创建示例图片2: 只包含漂浮窗
    img2 = np.ones((640, 480, 3), dtype=np.uint8) * 240
    cv2.rectangle(img2, (80, 150), (400, 450), (255, 255, 100), -1)  # 漂浮窗
    cv2.rectangle(img2, (80, 150), (400, 450), (0, 0, 0), 2)
    cv2.putText(img2, "Advertisement", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imwrite("training_data/images/train_002.jpg", img2)
    
    # 创建示例图片3: 包含漂浮窗和文字按钮
    img3 = np.ones((640, 480, 3), dtype=np.uint8) * 240
    cv2.rectangle(img3, (120, 80), (360, 380), (255, 150, 150), -1)  # 漂浮窗
    cv2.rectangle(img3, (120, 80), (360, 380), (0, 0, 0), 2)
    cv2.rectangle(img3, (160, 320), (280, 360), (150, 150, 255), -1)  # 文字按钮
    cv2.putText(img3, "Continue", (175, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.imwrite("training_data/images/train_003.jpg", img3)
    
    # 创建标注文件
    annotations = [
        # 图片1的标注
        "train_001.jpg,floating_window,100,100,280,300",
        "train_001.jpg,close_button,348,108,24,24",
        "train_001.jpg,text_button,150,320,80,40",
        
        # 图片2的标注
        "train_002.jpg,floating_window,80,150,320,300",
        
        # 图片3的标注
        "train_003.jpg,floating_window,120,80,240,300",
        "train_003.jpg,text_button,160,320,120,40",
    ]
    
    with open("training_data/annotations.txt", 'w', encoding='utf-8') as f:
        for ann in annotations:
            f.write(ann + '\n')
    
    print("示例训练数据集创建完成:")
    print("- training_data/images/: 训练图片")
    print("- training_data/annotations.txt: 标注文件")

def prepare_yolo_dataset():
    """准备YOLO格式的数据集"""
    print("正在转换为YOLO格式...")
    
    detector = DeepLearningFloatingWindowDetector()
    
    # 转换数据集格式
    detector.create_training_dataset(
        annotation_file="training_data/annotations.txt",
        images_dir="training_data/images",
        output_dir="yolo_dataset"
    )
    
    print("YOLO数据集准备完成: yolo_dataset/")

def train_model(epochs=50, batch_size=8, image_size=640):
    """训练模型"""
    print(f"开始训练模型... (epochs={epochs}, batch_size={batch_size}, image_size={image_size})")
    
    try:
        detector = DeepLearningFloatingWindowDetector()
        
        # 检查数据集配置文件
        dataset_yaml = "yolo_dataset/dataset.yaml"
        if not os.path.exists(dataset_yaml):
            print("错误: 数据集配置文件不存在，请先运行数据准备步骤")
            return None
        
        # 训练模型
        results = detector.train_model(
            dataset_yaml=dataset_yaml,
            epochs=epochs,
            imgsz=image_size,
            batch_size=batch_size,
            output_dir="training_runs"
        )
        
        print("模型训练完成!")
        print("训练结果保存在: training_runs/")
        
        return results
        
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_model(model_path=None):
    """评估训练好的模型"""
    print("正在评估模型...")
    
    try:
        # 查找最新的训练结果
        if model_path is None:
            # 尝试找到最新的训练模型
            if os.path.exists("training_runs"):
                for root, dirs, files in os.walk("training_runs"):
                    for file in files:
                        if file == "best.pt":
                            model_path = os.path.join(root, file)
                            break
                    if model_path:
                        break
        
        if model_path and os.path.exists(model_path):
            print(f"使用模型: {model_path}")
            
            # 初始化检测器
            detector = DeepLearningFloatingWindowDetector(model_path=model_path)
            
            # 在测试图片上评估
            test_images = ["training_data/images/train_001.jpg", 
                          "training_data/images/train_002.jpg",
                          "training_data/images/train_003.jpg"]
            
            for img_path in test_images:
                if os.path.exists(img_path):
                    result = detector.check_accessibility_compliance_dl(img_path)
                    print(f"图片: {img_path}")
                    print(f"  结果: {'通过' if result['is_pass'] else '不通过'}")
                    print(f"  原因: {result['reason']}")
                    print(f"  检测到的对象: {len(result['detections'])}")
                    
                    # 可视化检测结果
                    if result['detections']:
                        output_path = f"eval_{os.path.basename(img_path)}"
                        detector.visualize_detections(img_path, result['detections'], output_path)
                        print(f"  可视化结果: {output_path}")
                    print()
        else:
            print("未找到训练好的模型，请先运行训练步骤")
            
    except Exception as e:
        print(f"评估过程中出现错误: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='漂浮窗检测模型训练脚本')
    parser.add_argument('--action', 
                       choices=['prepare', 'train', 'evaluate', 'all'],
                       default='all',
                       help='执行的操作')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=8, help='批处理大小')
    parser.add_argument('--image-size', type=int, default=640, help='输入图片尺寸')
    parser.add_argument('--model-path', help='评估时使用的模型路径')
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    print("漂浮窗检测模型训练工具")
    print("=" * 50)
    
    try:
        if args.action == 'prepare' or args.action == 'all':
            create_sample_training_dataset()
            prepare_yolo_dataset()
        
        if args.action == 'train' or args.action == 'all':
            if not os.path.exists("yolo_dataset/dataset.yaml"):
                print("数据集未准备，先执行数据准备步骤...")
                create_sample_training_dataset()
                prepare_yolo_dataset()
            
            train_model(args.epochs, args.batch_size, args.image_size)
        
        if args.action == 'evaluate' or args.action == 'all':
            evaluate_model(args.model_path)
            
        print("=" * 50)
        print("训练流程完成!")
        print("\n注意事项:")
        print("1. 这里使用的是示例数据，实际使用时需要准备真实的标注数据")
        print("2. 训练数据量较小，实际效果可能有限")
        print("3. 建议使用更多真实的移动应用截图数据进行训练")
        print("4. 可以调整训练参数以获得更好的效果")
        
    except KeyboardInterrupt:
        print("\n训练已中断")
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()