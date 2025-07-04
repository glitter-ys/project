#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成漂浮窗无障碍合规性检测器

结合传统计算机视觉方法和深度学习方法，提供更准确和鲁棒的检测结果
"""

import cv2
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Any
import logging
from floating_window_detector import FloatingWindowDetector
from deep_learning_detector import DeepLearningFloatingWindowDetector

class IntegratedFloatingWindowDetector:
    """
    集成漂浮窗检测器
    
    结合传统CV方法和深度学习方法的优势
    """
    
    def __init__(self, dl_model_path=None, use_paddle_ocr=True, debug_mode=False):
        """
        初始化集成检测器
        
        Args:
            dl_model_path: 深度学习模型路径
            use_paddle_ocr: 是否使用PaddleOCR
            debug_mode: 是否开启调试模式
        """
        self.debug_mode = debug_mode
        self.logger = logging.getLogger(__name__)
        
        # 初始化传统CV检测器
        self.cv_detector = FloatingWindowDetector(
            use_paddle_ocr=use_paddle_ocr,
            debug_mode=debug_mode
        )
        
        # 初始化深度学习检测器
        self.dl_detector = DeepLearningFloatingWindowDetector(
            model_path=dl_model_path
        )
        
        self.logger.info("集成检测器初始化完成")
    
    def detect_floating_windows_integrated(self, image: np.ndarray) -> List[Dict]:
        """
        使用集成方法检测漂浮窗
        
        Args:
            image: 输入图像
            
        Returns:
            检测到的漂浮窗列表
        """
        # 传统CV方法检测
        cv_windows = self.cv_detector.detect_floating_windows(image)
        
        # 深度学习方法检测
        dl_detections = self.dl_detector.detect_objects(image, conf_threshold=0.3)
        dl_windows = [d for d in dl_detections if d['class'] == 'floating_window']
        
        # 融合结果
        integrated_windows = self._merge_detections(cv_windows, dl_windows, image.shape[:2])
        
        return integrated_windows
    
    def detect_close_buttons_integrated(self, image: np.ndarray, 
                                      window_bboxes: List[Tuple]) -> List[Dict]:
        """
        使用集成方法检测关闭按钮
        
        Args:
            image: 输入图像
            window_bboxes: 漂浮窗边界框列表
            
        Returns:
            检测到的关闭按钮列表
        """
        all_close_buttons = []
        
        # 深度学习方法检测
        dl_detections = self.dl_detector.detect_objects(image, conf_threshold=0.3)
        dl_close_buttons = [d for d in dl_detections if d['class'] == 'close_button']
        
        # 传统CV方法检测
        for window_bbox in window_bboxes:
            cv_close_buttons = self.cv_detector.detect_close_buttons(image, window_bbox)
            all_close_buttons.extend(cv_close_buttons)
        
        # 添加深度学习检测的结果
        all_close_buttons.extend(dl_close_buttons)
        
        # 去重和合并
        merged_buttons = self._merge_button_detections(all_close_buttons)
        
        return merged_buttons
    
    def detect_text_buttons_integrated(self, image: np.ndarray, 
                                     window_bboxes: List[Tuple]) -> List[Dict]:
        """
        使用集成方法检测文字按钮
        
        Args:
            image: 输入图像
            window_bboxes: 漂浮窗边界框列表
            
        Returns:
            检测到的文字按钮列表
        """
        all_text_buttons = []
        
        # 深度学习方法检测
        dl_detections = self.dl_detector.detect_objects(image, conf_threshold=0.3)
        dl_text_buttons = [d for d in dl_detections if d['class'] == 'text_button']
        
        # 传统CV+OCR方法检测
        for window_bbox in window_bboxes:
            cv_text_buttons = self.cv_detector.detect_text_buttons(image, window_bbox)
            all_text_buttons.extend(cv_text_buttons)
        
        # 添加深度学习检测的结果
        all_text_buttons.extend(dl_text_buttons)
        
        # 去重和合并
        merged_buttons = self._merge_button_detections(all_text_buttons)
        
        return merged_buttons
    
    def _merge_detections(self, cv_windows: List[Dict], dl_windows: List[Dict], 
                         image_shape: Tuple[int, int]) -> List[Dict]:
        """
        合并传统CV和深度学习的漂浮窗检测结果
        
        Args:
            cv_windows: 传统CV检测结果
            dl_windows: 深度学习检测结果
            image_shape: 图像尺寸 (height, width)
            
        Returns:
            合并后的检测结果
        """
        all_windows = []
        
        # 添加传统CV结果
        for window in cv_windows:
            all_windows.append({
                'bbox': window['bbox'],
                'confidence': window['confidence'],
                'method': 'cv',
                'source': window.get('method', 'unknown')
            })
        
        # 添加深度学习结果
        for window in dl_windows:
            all_windows.append({
                'bbox': window['bbox'],
                'confidence': window['confidence'],
                'method': 'dl',
                'source': 'yolo'
            })
        
        if not all_windows:
            return []
        
        # 使用非极大值抑制去重
        boxes = np.array([w['bbox'] for w in all_windows])
        scores = np.array([w['confidence'] for w in all_windows])
        
        # 转换为 [x1, y1, x2, y2] 格式
        boxes_nms = np.column_stack([
            boxes[:, 0],  # x1
            boxes[:, 1],  # y1
            boxes[:, 0] + boxes[:, 2],  # x2
            boxes[:, 1] + boxes[:, 3]   # y2
        ])
        
        indices = cv2.dnn.NMSBoxes(
            boxes_nms.tolist(), scores.tolist(), 0.3, 0.5
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            merged_windows = [all_windows[i] for i in indices]
        else:
            merged_windows = all_windows
        
        return merged_windows
    
    def _merge_button_detections(self, buttons: List[Dict]) -> List[Dict]:
        """
        合并按钮检测结果，去除重复
        
        Args:
            buttons: 按钮检测结果列表
            
        Returns:
            合并后的按钮列表
        """
        if not buttons:
            return []
        
        # 使用非极大值抑制去重
        boxes = np.array([b['bbox'] for b in buttons])
        scores = np.array([b.get('confidence', 0.5) for b in buttons])
        
        # 转换为 [x1, y1, x2, y2] 格式
        boxes_nms = np.column_stack([
            boxes[:, 0],  # x1
            boxes[:, 1],  # y1
            boxes[:, 0] + boxes[:, 2],  # x2
            boxes[:, 1] + boxes[:, 3]   # y2
        ])
        
        indices = cv2.dnn.NMSBoxes(
            boxes_nms.tolist(), scores.tolist(), 0.2, 0.4
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            merged_buttons = [buttons[i] for i in indices]
        else:
            merged_buttons = buttons
        
        return merged_buttons
    
    def check_accessibility_compliance_integrated(self, image_path: str) -> Dict[str, Any]:
        """
        使用集成方法检查无障碍合规性
        
        Args:
            image_path: 图片路径
            
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
                'floating_windows': [],
                'close_buttons': [],
                'text_buttons': []
            }
        
        self.logger.info(f"正在使用集成方法检测图片: {image_path}")
        
        # 检测漂浮窗
        floating_windows = self.detect_floating_windows_integrated(image)
        
        if not floating_windows:
            # 没有检测到漂浮窗，认为通过
            return {
                'image_path': image_path,
                'is_pass': 1,
                'reason': 'No floating windows detected',
                'floating_windows': [],
                'close_buttons': [],
                'text_buttons': [],
                'detection_method': 'integrated'
            }
        
        # 提取窗口边界框
        window_bboxes = [w['bbox'] for w in floating_windows]
        
        # 检测关闭按钮
        close_buttons = self.detect_close_buttons_integrated(image, window_bboxes)
        
        # 检测文字按钮
        text_buttons = self.detect_text_buttons_integrated(image, window_bboxes)
        
        # 判断是否通过测试
        has_close_button = len(close_buttons) > 0
        has_text_button = len(text_buttons) > 0
        
        is_pass = 1 if (has_close_button or has_text_button) else 0
        
        reason = ""
        if is_pass:
            if has_close_button and has_text_button:
                reason = "Has both close button and text button"
            elif has_close_button:
                reason = "Has close button"
            elif has_text_button:
                reason = "Has text button"
        else:
            reason = "No close button or text button found"
        
        result = {
            'image_path': image_path,
            'is_pass': is_pass,
            'reason': reason,
            'floating_windows': floating_windows,
            'close_buttons': close_buttons,
            'text_buttons': text_buttons,
            'detection_method': 'integrated'
        }
        
        # 保存调试图片
        if self.debug_mode:
            self._save_debug_image(image, result, image_path)
        
        return result
    
    def _save_debug_image(self, image: np.ndarray, result: Dict, image_path: str):
        """保存带有检测结果标注的调试图片"""
        debug_image = image.copy()
        
        # 绘制漂浮窗
        for window in result['floating_windows']:
            x, y, w, h = window['bbox']
            method = window.get('method', 'unknown')
            color = (255, 0, 0) if method == 'cv' else (0, 255, 255)  # 蓝色/黄色
            cv2.rectangle(debug_image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(debug_image, f"Window-{method}({window['confidence']:.2f})", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 绘制关闭按钮
        for button in result['close_buttons']:
            x, y, w, h = button['bbox']
            cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(debug_image, "Close", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 绘制文字按钮
        for button in result['text_buttons']:
            x, y, w, h = button['bbox']
            cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
            text_label = button.get('text', 'Text')[:10]
            cv2.putText(debug_image, f"Text: {text_label}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 保存调试图片
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        debug_path = f"debug_{base_name}_integrated_result.jpg"
        cv2.imwrite(debug_path, debug_image)
        self.logger.info(f"集成方法调试图片已保存: {debug_path}")
    
    def batch_check_integrated(self, image_folder: str, output_file: str = "pred_integrated.json"):
        """
        使用集成方法批量检测
        
        Args:
            image_folder: 图片文件夹路径
            output_file: 输出结果文件路径
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
                    result = self.check_accessibility_compliance_integrated(image_path)
                    
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
        
        self.logger.info(f"集成方法检测完成，结果已保存到: {output_file}")
        
        # 统计结果
        pass_count = sum(1 for r in results if r['Ispass'] == 1)
        fail_count = len(results) - pass_count
        self.logger.info(f"通过: {pass_count}, 不通过: {fail_count}")
        self.logger.info(f"通过率: {pass_count/len(results)*100:.1f}%")
        
        return results
    
    def compare_methods(self, image_folder: str, output_dir: str = "method_comparison"):
        """
        比较不同方法的检测结果
        
        Args:
            image_folder: 图片文件夹路径
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 使用三种方法分别检测
        cv_results = self.cv_detector.batch_check(
            image_folder, os.path.join(output_dir, "pred_cv.json")
        )
        
        dl_results = self.dl_detector.batch_check_dl(
            image_folder, os.path.join(output_dir, "pred_dl.json")
        )
        
        integrated_results = self.batch_check_integrated(
            image_folder, os.path.join(output_dir, "pred_integrated.json")
        )
        
        # 生成比较报告
        self._generate_comparison_report(cv_results, dl_results, integrated_results, output_dir)
        
        return {
            'cv_results': cv_results,
            'dl_results': dl_results,
            'integrated_results': integrated_results
        }
    
    def _generate_comparison_report(self, cv_results: List[Dict], dl_results: List[Dict], 
                                  integrated_results: List[Dict], output_dir: str):
        """生成方法比较报告"""
        
        # 创建图片名到结果的映射
        cv_map = {r['imgname']: r['Ispass'] for r in cv_results}
        dl_map = {r['imgname']: r['Ispass'] for r in dl_results}
        integrated_map = {r['imgname']: r['Ispass'] for r in integrated_results}
        
        # 统计各种情况
        all_images = set(cv_map.keys()) | set(dl_map.keys()) | set(integrated_map.keys())
        
        agreement_count = 0  # 三种方法一致的数量
        cv_dl_agreement = 0  # CV和DL一致的数量
        cv_integrated_agreement = 0  # CV和集成一致的数量
        dl_integrated_agreement = 0  # DL和集成一致的数量
        
        detailed_results = []
        
        for img in all_images:
            cv_pass = cv_map.get(img, 0)
            dl_pass = dl_map.get(img, 0)
            integrated_pass = integrated_map.get(img, 0)
            
            if cv_pass == dl_pass == integrated_pass:
                agreement_count += 1
            
            if cv_pass == dl_pass:
                cv_dl_agreement += 1
            
            if cv_pass == integrated_pass:
                cv_integrated_agreement += 1
            
            if dl_pass == integrated_pass:
                dl_integrated_agreement += 1
            
            detailed_results.append({
                'image': img,
                'cv': cv_pass,
                'dl': dl_pass,
                'integrated': integrated_pass,
                'all_agree': cv_pass == dl_pass == integrated_pass
            })
        
        # 生成报告
        report = f"""
# 漂浮窗检测方法比较报告

## 总体统计
- 总图片数: {len(all_images)}
- 三种方法完全一致: {agreement_count} ({agreement_count/len(all_images)*100:.1f}%)
- CV与深度学习一致: {cv_dl_agreement} ({cv_dl_agreement/len(all_images)*100:.1f}%)
- CV与集成方法一致: {cv_integrated_agreement} ({cv_integrated_agreement/len(all_images)*100:.1f}%)
- 深度学习与集成方法一致: {dl_integrated_agreement} ({dl_integrated_agreement/len(all_images)*100:.1f}%)

## 各方法通过率
- 传统CV方法: {sum(cv_map.values())/len(cv_map)*100:.1f}%
- 深度学习方法: {sum(dl_map.values())/len(dl_map)*100:.1f}%
- 集成方法: {sum(integrated_map.values())/len(integrated_map)*100:.1f}%

## 方法特点分析
- **传统CV方法**: 基于图像处理和OCR，不需要训练数据，但对复杂场景适应性较差
- **深度学习方法**: 基于YOLO目标检测，需要标注数据训练，但泛化性好
- **集成方法**: 结合两种方法的优势，通过多重验证提高准确性

## 建议
1. 对于标注数据充足的场景，推荐使用深度学习方法
2. 对于快速部署和解释性要求高的场景，可使用传统CV方法
3. 对于准确性要求最高的场景，推荐使用集成方法
        """
        
        # 保存报告
        report_file = os.path.join(output_dir, "comparison_report.md")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 保存详细结果
        detailed_file = os.path.join(output_dir, "detailed_comparison.json")
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"比较报告已生成: {report_file}")
        self.logger.info(f"详细比较结果: {detailed_file}")