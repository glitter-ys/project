# 漂浮窗无障碍合规性检测系统

基于GB/T 37668-2019《信息技术 互联网内容无障碍可访问性技术要求与测试方法》和WCAG标准的移动应用漂浮窗无障碍合规性自动检测工具。

## 项目概述

本项目旨在自动检测移动应用页面中的漂浮窗是否符合无障碍合规性要求。根据标准要求，漂浮窗需要满足以下条件之一才能通过测试：

1. **存在关闭方式**：漂浮窗或悬浮窗存在明确的关闭按钮（如叉号、关闭图标等）
2. **存在文字按钮**：漂浮窗内包含可点击的文字按钮，能够关闭窗口或跳转到其他页面

## 技术方案

本项目提供三种检测方案：

### 1. 传统计算机视觉方案
- **漂浮窗检测**：基于颜色突出度、边缘轮廓、亮度差异等特征
- **关闭按钮检测**：模板匹配、X形状检测、圆形符号检测
- **文字按钮检测**：OCR文字识别 + 按钮边界检测
- **优点**：无需训练数据，部署简单，解释性强
- **缺点**：对复杂场景适应性较差

### 2. 深度学习方案
- **基础模型**：YOLOv8目标检测框架
- **检测类别**：floating_window, close_button, text_button
- **训练数据**：支持COCO格式和自定义格式标注
- **优点**：准确率高，泛化性好
- **缺点**：需要标注数据和训练时间

### 3. 集成方案
- **策略**：结合传统CV和深度学习方法
- **融合方法**：非极大值抑制(NMS)去重合并
- **优点**：准确率最高，鲁棒性强
- **缺点**：计算复杂度较高

## 安装说明

### 环境要求
- Python 3.7+
- OpenCV 4.5+
- PyTorch 1.9+ (用于深度学习方案)

### 安装依赖
```bash
pip install -r requirements.txt
```

### 安装PaddleOCR (推荐)
```bash
pip install paddlepaddle paddleocr
```

### 安装EasyOCR (备选)
```bash
pip install easyocr
```

## 使用方法

### 1. 命令行使用

#### 检测单张图片
```bash
python main.py --image path/to/image.jpg
```

#### 批量检测文件夹
```bash
python main.py --folder path/to/images/
```

#### 开启调试模式
```bash
python main.py --folder path/to/images/ --debug
```

#### 指定输出文件
```bash
python main.py --folder path/to/images/ --output results.json
```

#### 选择OCR引擎
```bash
python main.py --folder path/to/images/ --ocr-engine paddle  # 使用PaddleOCR
python main.py --folder path/to/images/ --ocr-engine easy    # 使用EasyOCR
```

### 2. Python API使用

#### 传统CV方法
```python
from floating_window_detector import FloatingWindowDetector

detector = FloatingWindowDetector(debug_mode=True)
result = detector.check_accessibility_compliance("image.jpg")
print(f"是否通过: {result['is_pass']}")
```

#### 深度学习方法
```python
from deep_learning_detector import DeepLearningFloatingWindowDetector

detector = DeepLearningFloatingWindowDetector()
result = detector.check_accessibility_compliance_dl("image.jpg")
print(f"是否通过: {result['is_pass']}")
```

#### 集成方法
```python
from integrated_detector import IntegratedFloatingWindowDetector

detector = IntegratedFloatingWindowDetector(debug_mode=True)
result = detector.check_accessibility_compliance_integrated("image.jpg")
print(f"是否通过: {result['is_pass']}")
```

### 3. 批量检测
```python
# 批量检测并保存结果
detector = FloatingWindowDetector()
results = detector.batch_check("images_folder/", "pred.json")
```

## 输出格式

检测结果以JSON格式保存，格式如下：

```json
[
  {
    "imgname": "image1.jpg",
    "Ispass": 1
  },
  {
    "imgname": "image2.jpg", 
    "Ispass": 0
  }
]
```

其中：
- `imgname`: 图片文件名
- `Ispass`: 检测结果，1表示通过，0表示不通过

## 训练自定义模型

### 1. 准备训练数据

#### 数据格式
支持两种标注格式：

**自定义格式**（推荐）：
```
filename,class,x,y,w,h
image1.jpg,floating_window,100,150,300,400
image1.jpg,close_button,350,160,30,30
image1.jpg,text_button,120,480,80,40
```

**COCO格式**：
标准COCO JSON格式，包含images、annotations、categories字段。

#### 类别说明
- `floating_window`: 漂浮窗区域
- `close_button`: 关闭按钮（X按钮、关闭图标等）
- `text_button`: 文字按钮（包含可点击文字的按钮区域）

### 2. 转换数据格式
```python
from deep_learning_detector import DeepLearningFloatingWindowDetector

detector = DeepLearningFloatingWindowDetector()
detector.create_training_dataset(
    annotation_file="annotations.txt",
    images_dir="images/",
    output_dir="yolo_dataset/"
)
```

### 3. 训练模型
```python
detector.train_model(
    dataset_yaml="yolo_dataset/dataset.yaml",
    epochs=100,
    imgsz=640,
    batch_size=16
)
```

## 项目结构

```
├── main.py                          # 主程序入口
├── floating_window_detector.py      # 传统CV检测器
├── deep_learning_detector.py        # 深度学习检测器
├── integrated_detector.py           # 集成检测器
├── test_detectors.py               # 测试脚本
├── requirements.txt                # 依赖包列表
├── README.md                      # 项目说明
├── pred.json                      # 检测结果输出
└── test_images/                   # 测试图像文件夹
    ├── sample_pass.jpg           # 通过测试的示例
    ├── sample_fail.jpg           # 不通过测试的示例
    └── sample_no_floating.jpg    # 无漂浮窗的示例
```

## 快速测试

运行完整测试：
```bash
python test_detectors.py
```

该脚本将：
1. 创建示例测试图像
2. 测试所有三种检测方法
3. 生成比较报告
4. 创建标准格式的预测结果文件

## 性能特点

### 准确性
- **传统CV方法**：对简单场景准确率较高，复杂场景可能误检
- **深度学习方法**：在有充足训练数据的情况下准确率最高
- **集成方法**：结合两种方法优势，整体准确率最佳

### 速度
- **传统CV方法**：最快，适合实时检测
- **深度学习方法**：中等，依赖GPU加速
- **集成方法**：最慢，但可通过并行化优化

### 资源需求
- **传统CV方法**：CPU即可，内存需求较少
- **深度学习方法**：推荐GPU，需要较大内存
- **集成方法**：综合两者需求

## 技术优化

### 1. 降低误检率的策略
- 多重检测方法验证
- 基于上下文的后处理
- 置信度阈值自适应调整
- 非极大值抑制去重

### 2. 提高检测精度的方法
- 细粒度特征提取
- 多尺度检测
- 数据增强和迁移学习
- 集成学习策略

### 3. 性能优化
- 模型量化和剪枝
- 多线程并行处理
- 缓存优化
- GPU加速

## 应用场景

1. **移动应用测试**：自动化UI测试中的无障碍合规性检查
2. **质量保证**：产品发布前的无障碍合规性验证
3. **监管合规**：协助企业满足无障碍法规要求
4. **用户体验优化**：提升应用的无障碍用户体验

## 扩展性

本系统设计具有良好的扩展性：

1. **新检测类型**：可轻松添加其他UI元素的检测
2. **新检测方法**：模块化设计便于集成新的检测算法
3. **多平台支持**：可扩展支持Web、桌面应用等平台
4. **云端部署**：支持RESTful API和微服务架构

## 贡献指南

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 许可证

本项目采用MIT许可证。详见 [LICENSE](LICENSE) 文件。

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交Issue
- 发送邮件至项目维护者
- 在讨论区参与讨论

## 致谢

感谢以下开源项目和标准：

- [OpenCV](https://opencv.org/) - 计算机视觉库
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - OCR工具
- [YOLOv8](https://github.com/ultralytics/ultralytics) - 目标检测框架
- GB/T 37668-2019 标准
- [WCAG 2.1](https://www.w3.org/WAI/WCAG21/Understanding/) 指南