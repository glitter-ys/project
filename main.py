#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
漂浮窗无障碍合规性检测主程序

基于GB/T 37668-2019和WCAG标准的移动应用漂浮窗无障碍检测工具
"""

import argparse
import os
import sys
from floating_window_detector import FloatingWindowDetector

def main():
    parser = argparse.ArgumentParser(
        description='移动应用漂浮窗无障碍合规性检测工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py --image single_image.jpg                    # 检测单张图片
  python main.py --folder test_images/                       # 批量检测文件夹
  python main.py --folder test_images/ --debug               # 开启调试模式
  python main.py --folder test_images/ --output results.json # 指定输出文件
        """
    )
    
    # 输入参数
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', '-i', 
                           help='检测单张图片的路径')
    input_group.add_argument('--folder', '-f', 
                           help='批量检测图片文件夹路径')
    
    # 输出参数
    parser.add_argument('--output', '-o', 
                       default='pred.json',
                       help='输出结果文件路径 (默认: pred.json)')
    
    # 其他参数
    parser.add_argument('--debug', '-d', 
                       action='store_true',
                       help='开启调试模式，保存标注图片')
    parser.add_argument('--ocr-engine', 
                       choices=['paddle', 'easy'],
                       default='paddle',
                       help='选择OCR引擎 (默认: paddle)')
    
    args = parser.parse_args()
    
    # 检查输入路径
    if args.image:
        if not os.path.exists(args.image):
            print(f"错误: 图片文件不存在: {args.image}")
            sys.exit(1)
    elif args.folder:
        if not os.path.exists(args.folder):
            print(f"错误: 文件夹不存在: {args.folder}")
            sys.exit(1)
    
    # 初始化检测器
    use_paddle_ocr = (args.ocr_engine == 'paddle')
    detector = FloatingWindowDetector(
        use_paddle_ocr=use_paddle_ocr,
        debug_mode=args.debug
    )
    
    print("漂浮窗无障碍合规性检测工具")
    print("=" * 50)
    print(f"OCR引擎: {'PaddleOCR' if use_paddle_ocr else 'EasyOCR'}")
    print(f"调试模式: {'开启' if args.debug else '关闭'}")
    print("=" * 50)
    
    try:
        if args.image:
            # 单张图片检测
            print(f"正在检测图片: {args.image}")
            result = detector.check_accessibility_compliance(args.image)
            
            # 输出结果
            print(f"\n检测结果:")
            print(f"图片: {result['image_path']}")
            print(f"是否通过: {'通过' if result['is_pass'] else '不通过'}")
            print(f"原因: {result.get('reason', '')}")
            
            if result.get('floating_windows'):
                print(f"检测到漂浮窗数量: {len(result['floating_windows'])}")
            if result.get('close_buttons'):
                print(f"检测到关闭按钮数量: {len(result['close_buttons'])}")
            if result.get('text_buttons'):
                print(f"检测到文字按钮数量: {len(result['text_buttons'])}")
                for btn in result['text_buttons']:
                    print(f"  - 文字按钮: '{btn['text']}'")
            
            # 保存单张图片结果
            import json
            single_result = [{
                "imgname": os.path.basename(args.image),
                "Ispass": result['is_pass']
            }]
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(single_result, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到: {args.output}")
            
        else:
            # 批量检测
            print(f"正在批量检测文件夹: {args.folder}")
            results = detector.batch_check(args.folder, args.output)
            
            print(f"\n批量检测完成!")
            print(f"总共检测图片: {len(results)}")
            pass_count = sum(1 for r in results if r['Ispass'] == 1)
            fail_count = len(results) - pass_count
            print(f"通过: {pass_count} 张")
            print(f"不通过: {fail_count} 张")
            print(f"通过率: {pass_count/len(results)*100:.1f}%")
            
    except KeyboardInterrupt:
        print("\n\n检测已中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()