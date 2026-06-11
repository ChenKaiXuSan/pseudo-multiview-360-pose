#!/usr/bin/env python
"""
360° 等距柱投影图片 → VLM 人物检测（bounding box）测试

使用 Qwen2.5-VL-7B 本地运行，完全免费，无需 API Key。
支持输入：图片、视频帧、立方体面视角

输出：在原图上绘制检测框 + φ/θ 球面坐标
"""

import argparse
import os
import sys
import json
import re
from typing import List, Dict, Any, Optional
from PIL import Image

import cv2
import numpy as np


# ========== 等距柱投影工具函数 ==========

def equirectangular_to_face(equirect_img: np.ndarray, face_size: int = 512, face: int = 0) -> np.ndarray:
    """等距柱投影 → 立方体一个面"""
    h, w = equirect_img.shape[:2]
    face_coords = np.zeros((face_size, face_size, 2), dtype=np.float32)
    u = (np.arange(face_size) / face_size - 0.5) * 2
    v = -(np.arange(face_size) / face_size - 0.5) * 2
    u, v = np.meshgrid(u, v)

    if face == 0: x3d, y3d, z3d = u, v, np.ones_like(u)
    elif face == 1: x3d, y3d, z3d = np.ones_like(u), v, -u
    elif face == 2: x3d, y3d, z3d = -u, v, -np.ones_like(u)
    elif face == 3: x3d, y3d, z3d = -np.ones_like(u), v, u
    elif face == 4: x3d, y3d, z3d = u, np.ones_like(v), v
    elif face == 5: x3d, y3d, z3d = u, -np.ones_like(v), -v

    r = np.sqrt(x3d**2 + y3d**2 + z3d**2)
    phi = np.arctan2(y3d, x3d)
    theta = np.arccos(z3d / r)
    face_coords[..., 0] = (phi + np.pi) / (2 * np.pi) * w
    face_coords[..., 1] = theta / np.pi * h
    return cv2.remap(equirect_img, face_coords, None, cv2.INTER_LINEAR)


def crop_center_strip(frame: np.ndarray, strip_height: int = 600) -> np.ndarray:
    """裁剪赤道条（人物集中区域）"""
    h = frame.shape[0]
    y1 = max(0, h // 2 - strip_height // 2)
    y2 = min(h, h // 2 + strip_height // 2)
    return frame[y1:y2, :]


def convert_bbox_to_sphere(x1, y1, x2, y2, w_img: int, h_img: int) -> tuple:
    """等距柱投影图片的 bbox → 球面坐标 (φ, θ)（弧度）"""
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    phi = 2.0 * (cx / w_img - 0.5) * np.pi
    theta = cy / h_img * np.pi
    return phi, theta


# ========== Qwen2.5-VL 加载和检测 ==========

def load_qwen_vl():
    """加载 Qwen2.5-VL-7B 模型"""
    try:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        import torch
    except ImportError:
        print("[!] 需要安装: pip install transformers accelerate qwen-vl-utils")
        return None, None

    model_name = 'Qwen/Qwen2.5-VL-7B-Instruct'
    print(f"\n正在加载 Qwen2.5-VL 模型（首次会下载约16GB）...")

    processor = AutoProcessor.from_pretrained(model_name)

    # BF16 节省显存（RTX 3090 支持）
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    model.eval()

    print(f"模型加载完成，设备: {next(model.parameters()).device}")
    return model, processor


def detect_persons_with_qwen(model, processor, image: np.ndarray) -> List[Dict]:
    """用 Qwen2.5-VL 检测人物 bbox"""
    from qwen_vl_utils.vision_process import extract_vision_information
    import torch

    # numpy → PIL Image (RGB)
    pil_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(pil_image)

    prompt = """你是一个人物检测器。请直接返回 JSON 格式的人物 bounding box 列表，不要任何解释文字。

要求：
1. 找出图中所有清晰可见的滑雪者/人物（衣服颜色鲜明、轮廓完整）
2. bbox 坐标 (x1, y1, x2, y2) 必须是图片像素坐标（左上角为 0,0）
3. bbox 框住全身（头到脚）
4. conf 是 0-1 的置信度

返回格式（严格遵守，不要 markdown、不要代码块标记、不要额外文字）：
[{"bbox": [x1, y1, x2, y2], "label": "person", "conf": 0.95}, ...]

如果图中没有人或人物不可见，返回空列表 []。"""

    messages = [{
        'role': 'user',
        'content': [
            {'type': 'image', 'image': pil_img},
            {'type': 'text', 'text': prompt}
        ]
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = extract_vision_information(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors='pt',
    )
    inputs = inputs.to('cuda').to(torch.bfloat16)

    generated_ids = model.generate(**inputs, max_new_tokens=512)
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 提取 JSON（去除 prompt 和代码块标记）
    json_match = re.search(r'\[.*\]', output_text, re.DOTALL)
    if json_match:
        try:
            detections = json.loads(json_match.group())
            return detections
        except json.JSONDecodeError as e:
            print(f"[!] JSON 解析失败: {e}")
            # 尝试修复常见格式问题
            fixed = output_text.replace('\n', ' ')
            json_match2 = re.search(r'\[.*\]', fixed, re.DOTALL)
            if json_match2:
                try:
                    return json.loads(json_match2.group())
                except:
                    pass

    print(f"[!] Qwen-VL 未返回有效 bbox")
    print(f"   完整输出:\n{output_text}")
    return []


# ========== 可视化 ==========

def draw_bbox_on_frame(frame: np.ndarray, detections: List[Dict], w_orig: int, h_orig: int) -> np.ndarray:
    """在等距柱投影图上绘制 bbox + φ/θ 标签"""
    vis = frame.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0)]

    for i, det in enumerate(detections):
        bbox_raw = det.get('bbox', [])
        if len(bbox_raw) != 4:
            continue

        # 判断是否是归一化坐标
        max_val = max(abs(v) for v in bbox_raw)
        if max_val <= 1.0:
            x1, y1, x2, y2 = [int(c * (w_orig if j % 2 == 0 else h_orig)) for j, c in enumerate(bbox_raw)]
        else:
            x1, y1, x2, y2 = [int(v) for v in bbox_raw]

        # 裁剪到画面范围
        x1, y1 = max(0, min(x1, w_orig-1)), max(0, min(y1, h_orig-1))
        x2, y2 = max(0, min(x2, w_orig-1)), max(0, min(y2, h_orig-1))

        color = colors[i % len(colors)]
        conf = det.get('conf', 0.0)
        phi_deg = np.degrees(convert_bbox_to_sphere(x1, y1, x2, y2, w_orig, h_orig)[0])
        theta_deg = np.degrees(convert_bbox_to_sphere(x1, y1, x2, y2, w_orig, h_orig)[1])

        label = f"P{i} conf={conf:.2f} φ={phi_deg:.1f}° θ={theta_deg:.1f}°"

        # 等距柱投影 bbox 可能跨越边界（左/右），需要特殊处理
        if abs(x2 - x1) > w_orig / 2:
            cv2.rectangle(vis, (x1, y1), (w_orig-1, y2), color, 2)
            cv2.rectangle(vis, (0, y1), (x2-x1-w_orig, y2), color, 2)
        else:
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # 标签（自动选择不会遮挡的位置）
        text_y = y1 - 5 if y1 > 30 else min(y2 + 20, h_orig - 20)
        cv2.putText(vis, label, (x1, max(10, text_y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return vis


# ========== 主流程 ==========

def main():
    parser = argparse.ArgumentParser(description='360° VLM Person Detection (Qwen2.5-VL-7B 本地免费)')
    parser.add_argument('--image', type=str, default=None, help='输入图片路径')
    parser.add_argument('--video_frame', type=int, default=10, help='从视频提取第 N 帧')
    parser.add_argument('--output_dir', type=str, default='/mnt/dataset/skiing/360PoseFusion/vlm_test_output',
                       help='输出目录')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取测试图片
    frame = None
    image_path = None

    if args.image:
        image_path = args.image
        frame = cv2.imread(image_path)
    elif args.video_frame is not None:
        video_path = '/mnt/dataset/skiing/raw_new/kimura2_360.mp4'
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.video_frame)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"[!] 无法读取帧 {args.video_frame} from {video_path}")
            sys.exit(1)
        image_path = os.path.join(args.output_dir, f'frame_{args.video_frame}.jpg')
        cv2.imwrite(image_path, frame)

    if frame is None:
        print("[!] 未提供图片路径")
        parser.print_help()
        sys.exit(1)

    h_orig, w_orig = frame.shape[:2]
    print(f"输入: {image_path or 'video_frame'} ({w_orig}x{h_orig})\n")

    # ========== 加载模型 ==========
    model, processor = load_qwen_vl()
    if model is None:
        sys.exit(1)

    # ========== 测试多种视角输入 ==========
    views = [
        ('original', frame, '等距柱投影原图', None),
        ('equator_strip', crop_center_strip(frame), '赤道条带（人物区）', (300, w_orig)),
    ]
    for face_id in range(6):
        names = ['Front', 'Right', 'Back', 'Left', 'Top', 'Bottom']
        view_img = equirectangular_to_face(frame, face_size=512, face=face_id)
        views.append((f'face_{face_id}', view_img, f'立方体面({names[face_id]})', (512, 512)))

    all_results = {}
    best_view = None
    best_count = 0

    for view_name, img, title, resize_shape in views:
        print(f"\n--- {title} ({img.shape[1]}x{img.shape[0]}) ---")

        # Resize（加速推理）
        if resize_shape:
            img_resized = cv2.resize(img, resize_shape)
        else:
            img_resized = img

        # 保存输入图片
        cv2.imwrite(os.path.join(args.output_dir, f'{view_name}_input.jpg'), img_resized)

        # VLM 检测
        detections = detect_persons_with_qwen(model, processor, img_resized)

        # 如果需要映射回原图尺寸（resize 过的需要反缩放）
        if resize_shape and (resize_shape[0] != img.shape[0] or resize_shape[1] != img.shape[1]):
            scale_x = img.shape[1] / resize_shape[1]
            scale_y = img.shape[0] / resize_shape[0]
            for det in detections:
                bbox = det.get('bbox', [0, 0, 0, 0])
                if len(bbox) == 4:
                    det['bbox_scaled'] = [
                        bbox[0] * scale_x, bbox[1] * scale_y,
                        bbox[2] * scale_x, bbox[3] * scale_y
                    ]

        all_results[view_name] = detections
        print(f"  → {len(detections)} persons detected")

        if len(detections) > best_count:
            best_count = len(detections)
            best_view = view_name

    # ========== 用最佳视角的结果在原图上绘制 ==========
    if best_view and best_view in all_results:
        final_dets = all_results[best_view]

        # 映射 bbox 回原图尺寸
        for det in final_dets:
            if 'bbox_scaled' in det:
                det['bbox'] = det['bbox_scaled']

        vis = draw_bbox_on_frame(frame, final_dets, w_orig, h_orig)

        output_path = os.path.join(args.output_dir, f'vlm_result_best_{best_view}.jpg')
        cv2.imwrite(output_path, vis)
        print(f"\n✓ 最佳视角: {best_view} ({best_count} persons)")
        print(f"  结果已保存: {output_path}")

        # 打印详细结果
        print("\n检测详情:")
        for i, det in enumerate(final_dets):
            bbox = det.get('bbox', [0, 0, 0, 0])
            phi_deg = np.degrees(2.0 * ((bbox[0]+bbox[2])/4 / w_orig - 0.5) * np.pi) % 360 - 180
            theta_deg = np.degrees((bbox[1]+bbox[3]) / (4*h_orig) * np.pi)
            print(f"  #{i}: conf={det.get('conf',0):.2f} bbox=({bbox[0]:.0f},{bbox[1]:.0f})-({bbox[2]:.0f},{bbox[3]:.0f}) "
                  f"φ={phi_deg:.1f}° θ={theta_deg:.1f}°")

    # 保存各视角的检测结果对比表
    summary_path = os.path.join(args.output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("VLM Person Detection Summary\n" + "="*40 + "\n")
        for view_name, dets in all_results.items():
            f.write(f"{view_name:20s}: {len(dets)} persons\n")
    print(f"\n完整结果保存在: {args.output_dir}/")

    return best_view, best_count


if __name__ == '__main__':
    main()
