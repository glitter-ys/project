import cv2
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Any
import logging
from PIL import Image, ImageDraw
import easyocr
from paddleocr import PaddleOCR
import imutils
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

class FloatingWindowDetector:
    """
    漂浮窗无障碍合规性检测器
    
    基于GB/T 37668-2019和WCAG标准，检测移动应用页面中的漂浮窗是否满足无障碍要求
    """
    
    def __init__(self, use_paddle_ocr=True, debug_mode=False):
        """
        初始化检测器
        
        Args:
            use_paddle_ocr: 是否使用PaddleOCR（默认True，也支持EasyOCR）
            debug_mode: 是否开启调试模式，会保存中间结果图片
        """
        self.debug_mode = debug_mode
        self.use_paddle_ocr = use_paddle_ocr
        
        # 初始化OCR引擎
        if use_paddle_ocr:
            self.ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
        else:
            self.ocr = easyocr.Reader(['ch_sim', 'en', 'ja'])
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 关闭按钮模板（常见的X按钮样式）
        self.close_button_templates = self._create_close_button_templates()
        
    def _create_close_button_templates(self) -> List[np.ndarray]:
        """创建关闭按钮的模板"""
        templates = []
        
        # 创建不同大小的X形状模板
        for size in [16, 20, 24, 28, 32]:
            template = np.zeros((size, size), dtype=np.uint8)
            # 画X
            cv2.line(template, (2, 2), (size-3, size-3), 255, 2)
            cv2.line(template, (2, size-3), (size-3, 2), 255, 2)
            templates.append(template)
            
            # 圆形背景的X
            template_circle = np.zeros((size, size), dtype=np.uint8)
            cv2.circle(template_circle, (size//2, size//2), size//2-1, 255, 1)
            cv2.line(template_circle, (4, 4), (size-5, size-5), 255, 2)
            cv2.line(template_circle, (4, size-5), (size-5, 4), 255, 2)
            templates.append(template_circle)
        
        return templates
    
    def detect_floating_windows(self, image: np.ndarray) -> List[Dict]:
        """
        检测图像中的漂浮窗
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            检测到的漂浮窗列表，每个元素包含边界框和置信度
        """
        height, width = image.shape[:2]
        
        # 转换为HSV进行颜色分析
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 检测潜在的漂浮窗区域
        floating_windows = []
        
        # 方法1: 基于颜色突出度检测
        floating_windows.extend(self._detect_by_color_prominence(image, hsv))
        
        # 方法2: 基于边缘和轮廓检测
        floating_windows.extend(self._detect_by_contours(image))
        
        # 方法3: 基于区域亮度差异检测
        floating_windows.extend(self._detect_by_brightness_difference(image))
        
        # 去重和过滤
        floating_windows = self._filter_and_merge_windows(floating_windows, width, height)
        
        return floating_windows
    
    def _detect_by_color_prominence(self, image: np.ndarray, hsv: np.ndarray) -> List[Dict]:
        """基于颜色突出度检测漂浮窗"""
        windows = []
        
        # 检测饱和度高的区域（通常漂浮窗颜色比较鲜艳）
        saturation = hsv[:, :, 1]
        high_sat_mask = cv2.threshold(saturation, 100, 255, cv2.THRESH_BINARY)[1]
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        high_sat_mask = cv2.morphologyEx(high_sat_mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找连通组件
        contours, _ = cv2.findContours(high_sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:  # 面积阈值
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.3 < aspect_ratio < 3:  # 长宽比合理
                    windows.append({
                        'bbox': (x, y, w, h),
                        'confidence': min(area / 50000, 1.0),
                        'method': 'color_prominence'
                    })
        
        return windows
    
    def _detect_by_contours(self, image: np.ndarray) -> List[Dict]:
        """基于轮廓检测漂浮窗"""
        windows = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 8000:
                # 计算轮廓的凸包
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                
                if hull_area > 0:
                    solidity = area / hull_area
                    if solidity > 0.7:  # 实心度较高，可能是矩形窗口
                        x, y, w, h = cv2.boundingRect(contour)
                        windows.append({
                            'bbox': (x, y, w, h),
                            'confidence': solidity,
                            'method': 'contours'
                        })
        
        return windows
    
    def _detect_by_brightness_difference(self, image: np.ndarray) -> List[Dict]:
        """基于亮度差异检测漂浮窗"""
        windows = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 自适应阈值化
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2
        )
        
        # 查找连通组件
        contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.5 < aspect_ratio < 2.5:
                    windows.append({
                        'bbox': (x, y, w, h),
                        'confidence': 0.6,
                        'method': 'brightness_difference'
                    })
        
        return windows
    
    def _filter_and_merge_windows(self, windows: List[Dict], width: int, height: int) -> List[Dict]:
        """过滤和合并重叠的窗口"""
        if not windows:
            return []
        
        # 过滤太小或太大的窗口
        filtered_windows = []
        for window in windows:
            x, y, w, h = window['bbox']
            area = w * h
            image_area = width * height
            
            # 面积应该在图像的5%-80%之间
            if 0.05 * image_area < area < 0.8 * image_area:
                # 不能是全屏或接近全屏
                if not (x < 20 and y < 20 and w > width * 0.9 and h > height * 0.9):
                    filtered_windows.append(window)
        
        # 使用非极大值抑制合并重叠窗口
        if len(filtered_windows) > 1:
            boxes = np.array([w['bbox'] for w in filtered_windows])
            scores = np.array([w['confidence'] for w in filtered_windows])
            
            # 转换为 [x1, y1, x2, y2] 格式
            boxes_nms = np.column_stack([
                boxes[:, 0],  # x1
                boxes[:, 1],  # y1
                boxes[:, 0] + boxes[:, 2],  # x2
                boxes[:, 1] + boxes[:, 3]   # y2
            ])
            
            indices = cv2.dnn.NMSBoxes(
                boxes_nms.tolist(), scores.tolist(), 0.3, 0.4
            )
            
            if len(indices) > 0:
                indices = indices.flatten()
                filtered_windows = [filtered_windows[i] for i in indices]
        
        return filtered_windows
    
    def detect_close_buttons(self, image: np.ndarray, window_bbox: Tuple[int, int, int, int]) -> List[Dict]:
        """
        在指定窗口区域内检测关闭按钮
        
        Args:
            image: 原始图像
            window_bbox: 窗口边界框 (x, y, w, h)
            
        Returns:
            检测到的关闭按钮列表
        """
        x, y, w, h = window_bbox
        window_roi = image[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(window_roi, cv2.COLOR_BGR2GRAY)
        
        close_buttons = []
        
        # 方法1: 模板匹配
        close_buttons.extend(self._detect_close_by_template_matching(gray_roi, x, y))
        
        # 方法2: 检测X形状
        close_buttons.extend(self._detect_close_by_x_shape(gray_roi, x, y))
        
        # 方法3: 检测圆形按钮中的符号
        close_buttons.extend(self._detect_close_by_circle_symbol(gray_roi, x, y))
        
        return close_buttons
    
    def _detect_close_by_template_matching(self, gray_roi: np.ndarray, offset_x: int, offset_y: int) -> List[Dict]:
        """使用模板匹配检测关闭按钮"""
        buttons = []
        
        for template in self.close_button_templates:
            result = cv2.matchTemplate(gray_roi, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= 0.6)
            
            for pt in zip(*locations[::-1]):
                buttons.append({
                    'bbox': (offset_x + pt[0], offset_y + pt[1], template.shape[1], template.shape[0]),
                    'confidence': result[pt[1], pt[0]],
                    'type': 'close_button'
                })
        
        return buttons
    
    def _detect_close_by_x_shape(self, gray_roi: np.ndarray, offset_x: int, offset_y: int) -> List[Dict]:
        """检测X形状的关闭按钮"""
        buttons = []
        
        # 边缘检测
        edges = cv2.Canny(gray_roi, 50, 150)
        
        # 霍夫直线检测
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=10, maxLineGap=5)
        
        if lines is not None:
            # 寻找交叉的直线对
            for i, line1 in enumerate(lines):
                x1, y1, x2, y2 = line1[0]
                for j, line2 in enumerate(lines[i+1:], i+1):
                    x3, y3, x4, y4 = line2[0]
                    
                    # 计算两条直线的交点
                    intersection = self._line_intersection((x1, y1, x2, y2), (x3, y3, x4, y4))
                    if intersection:
                        ix, iy = intersection
                        # 检查是否形成X形状
                        if self._is_x_shape((x1, y1, x2, y2), (x3, y3, x4, y4), intersection):
                            # 创建边界框
                            min_x = min(x1, x2, x3, x4)
                            max_x = max(x1, x2, x3, x4)
                            min_y = min(y1, y2, y3, y4)
                            max_y = max(y1, y2, y3, y4)
                            
                            buttons.append({
                                'bbox': (offset_x + min_x, offset_y + min_y, 
                                        max_x - min_x, max_y - min_y),
                                'confidence': 0.8,
                                'type': 'close_button'
                            })
        
        return buttons
    
    def _detect_close_by_circle_symbol(self, gray_roi: np.ndarray, offset_x: int, offset_y: int) -> List[Dict]:
        """检测圆形背景中的关闭符号"""
        buttons = []
        
        # 检测圆形
        circles = cv2.HoughCircles(
            gray_roi, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=30, minRadius=10, maxRadius=50
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (cx, cy, r) in circles:
                # 提取圆形区域
                mask = np.zeros(gray_roi.shape, dtype=np.uint8)
                cv2.circle(mask, (cx, cy), r, 255, -1)
                circle_roi = cv2.bitwise_and(gray_roi, mask)
                
                # 在圆形区域内检测X形状
                edges = cv2.Canny(circle_roi, 50, 150)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, minLineLength=5, maxLineGap=3)
                
                if lines is not None and len(lines) >= 2:
                    buttons.append({
                        'bbox': (offset_x + cx - r, offset_y + cy - r, 2*r, 2*r),
                        'confidence': 0.7,
                        'type': 'close_button'
                    })
        
        return buttons
    
    def _line_intersection(self, line1: Tuple, line2: Tuple) -> Tuple[float, float] or None:
        """计算两条直线的交点"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            return (ix, iy)
        
        return None
    
    def _is_x_shape(self, line1: Tuple, line2: Tuple, intersection: Tuple) -> bool:
        """判断两条直线是否形成X形状"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        ix, iy = intersection
        
        # 计算角度
        import math
        angle1 = math.atan2(y2 - y1, x2 - x1)
        angle2 = math.atan2(y4 - y3, x4 - x3)
        
        angle_diff = abs(angle1 - angle2)
        angle_diff = min(angle_diff, math.pi - angle_diff)
        
        # X形状的角度差应该接近45度或135度
        return math.pi/4 - 0.5 < angle_diff < math.pi/4 + 0.5 or \
               3*math.pi/4 - 0.5 < angle_diff < 3*math.pi/4 + 0.5
    
    def detect_text_buttons(self, image: np.ndarray, window_bbox: Tuple[int, int, int, int]) -> List[Dict]:
        """
        在指定窗口区域内检测文字按钮
        
        Args:
            image: 原始图像
            window_bbox: 窗口边界框 (x, y, w, h)
            
        Returns:
            检测到的文字按钮列表
        """
        x, y, w, h = window_bbox
        window_roi = image[y:y+h, x:x+w]
        
        text_buttons = []
        
        # 使用OCR检测文字
        if self.use_paddle_ocr:
            ocr_results = self.ocr.ocr(window_roi, cls=True)
            if ocr_results and ocr_results[0]:
                for line in ocr_results[0]:
                    text_bbox = line[0]
                    text_content = line[1][0]
                    confidence = line[1][1]
                    
                    if confidence > 0.5 and len(text_content.strip()) > 0:
                        # 检查是否是按钮文字
                        if self._is_button_text(text_content):
                            # 转换坐标到原图
                            x_min = int(min([p[0] for p in text_bbox])) + x
                            y_min = int(min([p[1] for p in text_bbox])) + y
                            x_max = int(max([p[0] for p in text_bbox])) + x
                            y_max = int(max([p[1] for p in text_bbox])) + y
                            
                            # 检测文字周围是否有按钮边界
                            button_bbox = self._detect_button_boundary(
                                window_roi, text_bbox, (x_min-x, y_min-y, x_max-x, y_max-y)
                            )
                            
                            if button_bbox:
                                text_buttons.append({
                                    'bbox': (button_bbox[0] + x, button_bbox[1] + y, 
                                            button_bbox[2], button_bbox[3]),
                                    'text': text_content,
                                    'confidence': confidence,
                                    'type': 'text_button'
                                })
        else:
            # 使用EasyOCR
            ocr_results = self.ocr.readtext(window_roi)
            for (bbox, text, confidence) in ocr_results:
                if confidence > 0.5 and len(text.strip()) > 0:
                    if self._is_button_text(text):
                        x_min = int(min([p[0] for p in bbox])) + x
                        y_min = int(min([p[1] for p in bbox])) + y
                        x_max = int(max([p[0] for p in bbox])) + x
                        y_max = int(max([p[1] for p in bbox])) + y
                        
                        button_bbox = self._detect_button_boundary(
                            window_roi, bbox, (x_min-x, y_min-y, x_max-x, y_max-y)
                        )
                        
                        if button_bbox:
                            text_buttons.append({
                                'bbox': (button_bbox[0] + x, button_bbox[1] + y, 
                                        button_bbox[2], button_bbox[3]),
                                'text': text,
                                'confidence': confidence,
                                'type': 'text_button'
                            })
        
        return text_buttons
    
    def _is_button_text(self, text: str) -> bool:
        """判断文字是否可能是按钮文字"""
        text = text.strip().lower()
        
        # 常见的按钮关键词
        button_keywords = [
            '确定', '取消', '关闭', '退出', '返回', '跳过', '忽略', 
            '继续', '下一步', '上一步', '完成', '提交', '保存',
            'ok', 'cancel', 'close', 'exit', 'back', 'skip', 'ignore',
            'continue', 'next', 'previous', 'finish', 'submit', 'save',
            'はい', 'いいえ', 'キャンセル', '閉じる', '戻る', 'スキップ',
            '次へ', '前へ', '完了', '送信', '保存', 'チェック', 'クリック'
        ]
        
        # 检查是否包含按钮关键词
        for keyword in button_keywords:
            if keyword in text:
                return True
        
        # 检查文字长度（按钮文字通常较短）
        if len(text) <= 8 and len(text) >= 1:
            return True
            
        return False
    
    def _detect_button_boundary(self, roi: np.ndarray, text_bbox: List, text_region: Tuple) -> Tuple or None:
        """检测文字周围的按钮边界"""
        x_min, y_min, x_max, y_max = text_region
        
        # 扩展文字区域来寻找按钮边界
        padding = 10
        search_x1 = max(0, x_min - padding)
        search_y1 = max(0, y_min - padding)
        search_x2 = min(roi.shape[1], x_max + padding)
        search_y2 = min(roi.shape[0], y_max + padding)
        
        search_roi = roi[search_y1:search_y2, search_x1:search_x2]
        gray_search = cv2.cvtColor(search_roi, cv2.COLOR_BGR2GRAY)
        
        # 检测边缘
        edges = cv2.Canny(gray_search, 50, 150)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # 最小按钮面积
                x, y, w, h = cv2.boundingRect(contour)
                
                # 检查文字是否在这个轮廓内
                text_center_x = (x_min + x_max) / 2 - search_x1
                text_center_y = (y_min + y_max) / 2 - search_y1
                
                if x <= text_center_x <= x + w and y <= text_center_y <= y + h:
                    return (search_x1 + x, search_y1 + y, w, h)
        
        # 如果没有找到明显的边界，返回扩展的文字区域
        return (search_x1, search_y1, search_x2 - search_x1, search_y2 - search_y1)
    
    def check_accessibility_compliance(self, image_path: str) -> Dict[str, Any]:
        """
        检查单张图片的漂浮窗无障碍合规性
        
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
        
        self.logger.info(f"正在检测图片: {image_path}")
        
        # 检测漂浮窗
        floating_windows = self.detect_floating_windows(image)
        
        if not floating_windows:
            # 没有检测到漂浮窗，认为通过（没有漂浮窗就不需要检查）
            return {
                'image_path': image_path,
                'is_pass': 1,
                'reason': 'No floating windows detected',
                'floating_windows': [],
                'close_buttons': [],
                'text_buttons': []
            }
        
        # 对每个漂浮窗检测关闭按钮和文字按钮
        all_close_buttons = []
        all_text_buttons = []
        
        for window in floating_windows:
            window_bbox = window['bbox']
            
            # 检测关闭按钮
            close_buttons = self.detect_close_buttons(image, window_bbox)
            all_close_buttons.extend(close_buttons)
            
            # 检测文字按钮
            text_buttons = self.detect_text_buttons(image, window_bbox)
            all_text_buttons.extend(text_buttons)
        
        # 判断是否通过测试
        has_close_button = len(all_close_buttons) > 0
        has_text_button = len(all_text_buttons) > 0
        
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
            'close_buttons': all_close_buttons,
            'text_buttons': all_text_buttons
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
            cv2.rectangle(debug_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(debug_image, f"Window({window['confidence']:.2f})", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
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
            cv2.putText(debug_image, f"Text: {button['text'][:10]}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 保存调试图片
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        debug_path = f"debug_{base_name}_result.jpg"
        cv2.imwrite(debug_path, debug_image)
        self.logger.info(f"调试图片已保存: {debug_path}")
    
    def batch_check(self, image_folder: str, output_file: str = "pred.json"):
        """
        批量检测图片文件夹中的所有图片
        
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
                    result = self.check_accessibility_compliance(image_path)
                    
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
        
        self.logger.info(f"检测完成，结果已保存到: {output_file}")
        self.logger.info(f"总共检测 {len(results)} 张图片")
        
        # 统计结果
        pass_count = sum(1 for r in results if r['Ispass'] == 1)
        fail_count = len(results) - pass_count
        self.logger.info(f"通过: {pass_count}, 不通过: {fail_count}")
        
        return results