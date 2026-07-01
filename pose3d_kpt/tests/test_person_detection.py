#!/usr/bin/env python
"""
测试视频中的人物检测
使用 YOLOv8 检测视频每一帧中的人物
"""

import cv2
import torch
from ultralytics import YOLO
import os
from pathlib import Path

def test_video_detection(video_path, output_path=None, frame_skip=30, save_frames=False, frames_output_dir=None):
    """
    测试视频中的人物检测

    Args:
        video_path: 视频文件路径
        output_path: 输出视频路径（可选）
        frame_skip: 跳帧数，每N帧检测一次
        save_frames: 是否保存检测后的帧为图片
        frames_output_dir: 保存帧的目录路径
    """
    # 检查CUDA是否可用
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # 加载YOLOv8模型（预训练的COCO数据集，包含person类别）
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')  # 使用轻量级的yolov8n模型

    # 打开视频
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return False

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info: {width}x{height}, {fps} fps, {total_frames} frames")

    # 输出视频设置
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    detected_frames = 0
    person_counts = []

    print(f"\nProcessing video (detecting every {frame_skip} frames)...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 每frame_skip帧检测一次
        if frame_count % frame_skip == 0:
            # 使用YOLO进行预测（降低置信度阈值到0.3）
            results = model(frame, verbose=False, conf=0.3)

            # 绘制检测结果
            annotated_frame = results[0].plot()

            # 统计检测到的人物数量
            person_count = 0
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                if cls_name == 'person':
                    person_count += 1

            person_counts.append(person_count)
            if person_count > 0:
                detected_frames += 1
                print(f"Frame {frame_count}: Detected {person_count} person(s)")

            # 写入输出视频
            if out:
                out.write(annotated_frame)

            # 保存检测帧为图片
            if save_frames and frames_output_dir:
                # 确保目录存在
                os.makedirs(frames_output_dir, exist_ok=True)
                frame_filename = os.path.join(frames_output_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_filename, annotated_frame)
                print(f"  -> Saved {frame_filename}")
        else:
            # 未检测的帧直接写入（或跳过）
            if out:
                out.write(frame)

        # 显示进度
        progress = (frame_count / total_frames) * 100
        print(f"\rProgress: {progress:.1f}%", end='', flush=True)

    cap.release()
    if out:
        out.release()

    print(f"\n\n=== Summary ===")
    print(f"Total frames processed: {frame_count}")
    print(f"Frames with person detected: {detected_frames}")
    print(f"Total person detections: {sum(person_counts)}")
    print(f"Average persons per detection: {sum(person_counts) / len(person_counts) if person_counts else 0:.2f}")

    if output_path:
        print(f"\nOutput video saved to: {output_path}")

    if save_frames and frames_output_dir:
        print(f"\nDetected frames saved to: {frames_output_dir}")

    return True

if __name__ == "__main__":
    video_path = "/mnt/dataset/skiing/raw_new/kimura2_360.mp4"
    output_path = "/mnt/dataset/skiing/raw_new/kimura2_360_detected.mp4"
    frames_output_dir = "/mnt/dataset/skiing/raw_new/kimura2_360_frames"

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
    else:
        test_video_detection(video_path, output_path, frame_skip=10, save_frames=True, frames_output_dir=frames_output_dir)
