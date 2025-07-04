#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
漂浮窗检测器测试脚本

演示如何使用传统CV、深度学习和集成检测器
"""

import os
import sys
import json
import cv2
import numpy as np
import logging
from floating_window_detector import FloatingWindowDetector
from deep_learning_detector import DeepLearningFloatingWindowDetector  
from integrated_detector import IntegratedFloatingWindowDetector

def create_sample_images():
    """创建示例测试图像"""
    os.makedirs("test_images", exist_ok=True)
    
    # 创建示例图像1: 有漂浮窗，有关闭按钮
    img1 = np.ones((800, 600, 3), dtype=np.uint8) * 240  # 浅灰色背景
    
    # 绘制漂浮窗
    cv2.rectangle(img1, (150, 200), (450, 550), (255, 100, 100), -1)  # 蓝色漂浮窗
    cv2.rectangle(img1, (150, 200), (450, 550), (0, 0, 0), 2)  # 黑色边框
    
    # 绘制关闭按钮 (X)
    cv2.circle(img1, (420, 230), 15, (200, 200, 200), -1)  # 圆形背景
    cv2.line(img1, (410, 220), (430, 240), (0, 0, 0), 3)  # X的一笔
    cv2.line(img1, (410, 240), (430, 220), (0, 0, 0), 3)  # X的另一笔
    
    # 添加文字按钮
    cv2.rectangle(img1, (180, 480), (280, 520), (100, 255, 100), -1)  # 绿色按钮
    cv2.putText(img1, "OK", (220, 505), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    cv2.imwrite("test_images/sample_pass.jpg", img1)
    
    # 创建示例图像2: 有漂浮窗，无关闭按钮和文字按钮
    img2 = np.ones((800, 600, 3), dtype=np.uint8) * 240
    
    # 绘制漂浮窗
    cv2.rectangle(img2, (100, 150), (500, 600), (255, 255, 100), -1)  # 黄色漂浮窗
    cv2.rectangle(img2, (100, 150), (500, 600), (0, 0, 0), 2)
    
    # 只添加一些装饰，不是按钮
    cv2.putText(img2, "Advertisement", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    cv2.putText(img2, "Content here...", (150, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    
    cv2.imwrite("test_images/sample_fail.jpg", img2)
    
    # 创建示例图像3: 无漂浮窗
    img3 = np.ones((800, 600, 3), dtype=np.uint8) * 240
    
    # 绘制正常界面元素
    cv2.rectangle(img3, (50, 50), (550, 100), (200, 200, 200), -1)  # 标题栏
    cv2.putText(img3, "Normal App Interface", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # 绘制一些按钮
    cv2.rectangle(img3, (100, 200), (200, 250), (150, 150, 255), -1)
    cv2.putText(img3, "Button 1", (120, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    cv2.rectangle(img3, (250, 200), (350, 250), (150, 255, 150), -1)
    cv2.putText(img3, "Button 2", (270, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    cv2.imwrite("test_images/sample_no_floating.jpg", img3)
    
    print("示例测试图像已创建在 test_images/ 文件夹中")

def test_cv_detector():
    """测试传统CV检测器"""
    print("=" * 50)
    print("测试传统CV检测器")
    print("=" * 50)
    
    detector = FloatingWindowDetector(debug_mode=True)
    
    # 测试单张图片
    test_image = "test_images/sample_pass.jpg"
    if os.path.exists(test_image):
        result = detector.check_accessibility_compliance(test_image)
        print(f"图片: {test_image}")
        print(f"结果: {'通过' if result['is_pass'] else '不通过'}")
        print(f"原因: {result['reason']}")
        print(f"检测到漂浮窗: {len(result['floating_windows'])}")
        print(f"检测到关闭按钮: {len(result['close_buttons'])}")
        print(f"检测到文字按钮: {len(result['text_buttons'])}")
    
    # 批量测试
    if os.path.exists("test_images"):
        print("\n批量测试...")
        results = detector.batch_check("test_images", "pred_cv_test.json")
        print(f"批量测试完成，结果保存在 pred_cv_test.json")

def test_dl_detector():
    """测试深度学习检测器"""
    print("=" * 50)
    print("测试深度学习检测器")
    print("=" * 50)
    
    try:
        detector = DeepLearningFloatingWindowDetector()
        
        # 创建示例训练数据
        detector.create_sample_training_data()
        
        # 测试单张图片
        test_image = "test_images/sample_pass.jpg"
        if os.path.exists(test_image):
            result = detector.check_accessibility_compliance_dl(test_image)
            print(f"图片: {test_image}")
            print(f"结果: {'通过' if result['is_pass'] else '不通过'}")
            print(f"原因: {result['reason']}")
            print(f"检测到的对象数: {len(result['detections'])}")
        
        # 批量测试
        if os.path.exists("test_images"):
            print("\n批量测试...")
            results = detector.batch_check_dl("test_images", "pred_dl_test.json")
            print(f"批量测试完成，结果保存在 pred_dl_test.json")
            
    except Exception as e:
        print(f"深度学习检测器测试失败: {str(e)}")
        print("注意: 深度学习检测器需要PyTorch和YOLO模型")

def test_integrated_detector():
    """测试集成检测器"""
    print("=" * 50)
    print("测试集成检测器")
    print("=" * 50)
    
    try:
        detector = IntegratedFloatingWindowDetector(debug_mode=True)
        
        # 测试单张图片
        test_image = "test_images/sample_pass.jpg"
        if os.path.exists(test_image):
            result = detector.check_accessibility_compliance_integrated(test_image)
            print(f"图片: {test_image}")
            print(f"结果: {'通过' if result['is_pass'] else '不通过'}")
            print(f"原因: {result['reason']}")
            print(f"检测方法: {result['detection_method']}")
            print(f"检测到漂浮窗: {len(result['floating_windows'])}")
            print(f"检测到关闭按钮: {len(result['close_buttons'])}")
            print(f"检测到文字按钮: {len(result['text_buttons'])}")
        
        # 批量测试
        if os.path.exists("test_images"):
            print("\n批量测试...")
            results = detector.batch_check_integrated("test_images", "pred_integrated_test.json")
            print(f"批量测试完成，结果保存在 pred_integrated_test.json")
            
        # 方法比较
        if os.path.exists("test_images"):
            print("\n进行方法比较...")
            comparison_results = detector.compare_methods("test_images", "test_comparison")
            print("方法比较完成，结果保存在 test_comparison/ 文件夹中")
            
    except Exception as e:
        print(f"集成检测器测试失败: {str(e)}")

def evaluate_results():
    """评估检测结果"""
    print("=" * 50)
    print("评估检测结果")
    print("=" * 50)
    
    # 预期结果 (ground truth)
    ground_truth = {
        "sample_pass.jpg": 1,      # 应该通过
        "sample_fail.jpg": 0,      # 应该不通过
        "sample_no_floating.jpg": 1 # 没有漂浮窗，应该通过
    }
    
    # 评估不同方法的结果
    results_files = [
        ("pred_cv_test.json", "传统CV方法"),
        ("pred_dl_test.json", "深度学习方法"),
        ("pred_integrated_test.json", "集成方法")
    ]
    
    for result_file, method_name in results_files:
        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                predictions = json.load(f)
            
            # 计算准确率
            correct = 0
            total = 0
            
            pred_dict = {p['imgname']: p['Ispass'] for p in predictions}
            
            for img_name, gt_value in ground_truth.items():
                if img_name in pred_dict:
                    pred_value = pred_dict[img_name]
                    if pred_value == gt_value:
                        correct += 1
                    total += 1
                    print(f"{method_name} - {img_name}: 预测={pred_value}, 真实={gt_value}, {'✓' if pred_value==gt_value else '✗'}")
            
            if total > 0:
                accuracy = correct / total * 100
                print(f"{method_name} 准确率: {accuracy:.1f}% ({correct}/{total})")
            print()

def create_prediction_json():
    """创建标准格式的预测结果文件"""
    if os.path.exists("pred_integrated_test.json"):
        # 复制集成方法的结果作为最终预测结果
        import shutil
        shutil.copy2("pred_integrated_test.json", "pred.json")
        print("已创建最终预测结果文件: pred.json")
    else:
        # 创建一个示例预测结果
        sample_predictions = [
            {"imgname": "sample_pass.jpg", "Ispass": 1},
            {"imgname": "sample_fail.jpg", "Ispass": 0},
            {"imgname": "sample_no_floating.jpg", "Ispass": 1}
        ]
        
        with open("pred.json", 'w', encoding='utf-8') as f:
            json.dump(sample_predictions, f, ensure_ascii=False, indent=2)
        print("已创建示例预测结果文件: pred.json")

def main():
    """主测试函数"""
    print("漂浮窗无障碍合规性检测器测试")
    print("=" * 60)
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        # 创建示例图像
        create_sample_images()
        
        # 测试传统CV检测器
        test_cv_detector()
        
        # 测试深度学习检测器
        test_dl_detector()
        
        # 测试集成检测器
        test_integrated_detector()
        
        # 评估结果
        evaluate_results()
        
        # 创建最终预测结果文件
        create_prediction_json()
        
        print("=" * 60)
        print("所有测试完成!")
        print("生成的文件:")
        print("- test_images/: 示例测试图像")
        print("- pred_cv_test.json: 传统CV方法结果")
        print("- pred_dl_test.json: 深度学习方法结果")
        print("- pred_integrated_test.json: 集成方法结果")
        print("- pred.json: 最终预测结果")
        print("- test_comparison/: 方法比较结果")
        print("- debug_*.jpg: 调试可视化图像")
        
    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()