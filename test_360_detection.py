#!/usr/bin/env python
"""
360度视频人物检测
使用滑动窗口方式处理360视频，通过立方体面投影消除畸变

流程：等距柱投影帧 → 转6个立方体面 → 每个面上滑动窗口YOLO检测 → 坐标映射回原图 → NMS去重
"""

import cv2
import torch
from ultralytics import YOLO
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt


def cubemap_face_xyz(u, v, face):
    """将 cubemap 面内归一化坐标映射到 3D 方向；z 为前方，y 为上方。"""
    if face == 0:  # front (z+)
        return u, v, np.ones_like(u)
    if face == 1:  # right (x+)
        return np.ones_like(u), v, -u
    if face == 2:  # back (z-)
        return -u, v, -np.ones_like(u)
    if face == 3:  # left (x-)
        return -np.ones_like(u), v, u
    if face == 4:  # top (y+)
        return u, np.ones_like(v), v
    if face == 5:  # bottom (y-)
        return u, -np.ones_like(v), -v
    raise ValueError("face must be 0-5")


def xyz_to_equirectangular(x, y, z, width, height):
    """3D 方向转等距柱投影坐标，返回 lon、lat、x、y。"""
    lon = np.arctan2(x, z)
    lat = np.arctan2(y, np.sqrt(x**2 + z**2))

    x_e = (lon / (2 * np.pi) + 0.5) * width
    y_e = (0.5 - lat / np.pi) * height

    x_e = np.mod(x_e, width)
    y_e = np.clip(y_e, 0, height - 1)
    return lon, lat, x_e, y_e


def equirectangular_to_face(equirect_img, face_size=1024, face=0):
    """
    将等距柱投影的360图片转换为立方体的一个面

    Args:
        equirect_img: 等距柱投影图片 (H, W, 3)
        face_size: 每个立方体面的尺寸
        face: 立方体面编号 (0-5): 0=front, 1=right, 2=back, 3=left, 4=top, 5=bottom

    Returns:
        cubemap_face: 对应面的图片
    """
    h, w = equirect_img.shape[:2]

    # 立方体面的映射坐标生成
    face_coords = np.zeros((face_size, face_size, 2), dtype=np.float32)

    # 归一化坐标范围 [-1, 1]
    u = (np.arange(face_size) / face_size - 0.5) * 2
    v = -(np.arange(face_size) / face_size - 0.5) * 2  # 反转v使其向上为正
    u, v = np.meshgrid(u, v)

    x, y, z = cubemap_face_xyz(u, v, face)
    _, _, u_equirect, v_equirect = xyz_to_equirectangular(x, y, z, w, h)

    face_coords[..., 0] = u_equirect
    face_coords[..., 1] = v_equirect

    # 重映射
    cubemap_face = cv2.remap(equirect_img, face_coords, None, cv2.INTER_LINEAR)

    return cubemap_face


def face_to_equirectangular_coords(face_size, face=0):
    """
    获取立方体面到等距柱投影的坐标映射

    Args:
        face_size: 立方体面尺寸
        face: 立方体面编号 (0-5)

    Returns:
        lon: 经度数组，范围 [-pi, pi]
        lat: 纬度数组，范围 [-pi/2, pi/2]
    """
    # 归一化坐标范围 [-1, 1]
    u = (np.arange(face_size) / face_size - 0.5) * 2
    v = -(np.arange(face_size) / face_size - 0.5) * 2
    u, v = np.meshgrid(u, v)

    x, y, z = cubemap_face_xyz(u, v, face)
    lon, lat, _, _ = xyz_to_equirectangular(x, y, z, 1, 1)

    return lon, lat


def visualize_cube_mapping(equirect_img, face_size=512, output_path=None):
    """
    可视化验证立方体面映射：将6个面按立方体展开图排列

    Args:
        equirect_img: 等距柱投影图片
        face_size: 每个立方体面的尺寸
        output_path: 输出可视化图片的路径

    Returns:
        visualization: 立方体展开图可视化
    """
    # 创建2x3的子图布局
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    faces = [
        (0, 'Front (Z+)', (0, 0)),
        (1, 'Right (X+)', (0, 1)),
        (2, 'Back (Z-)', (0, 2)),
        (3, 'Left (X-)', (1, 0)),
        (4, 'Top (Y+)', (1, 1)),
        (5, 'Bottom (Y-)', (1, 2))
    ]

    # 原始等距柱投影
    axes[0, 0].imshow(cv2.cvtColor(equirect_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Equirectangular')
    axes[0, 0].axis('off')

    # 可视化每个立方体面
    for face_id, title, (row, col) in faces:
        face_img = equirectangular_to_face(equirect_img, face_size=face_size, face=face_id)

        ax = axes[row, col]
        ax.imshow(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Cube mapping visualization saved to: {output_path}")
    else:
        plt.show()

    return fig


def sphere_distance(phi1, lat1, phi2, lat2):
    """两个点在单位球面上的角距离（度），使用经纬度输入"""
    # 笛卡尔坐标（经度phi + 纬度lat）
    cos_lat1 = np.cos(lat1)
    x1 = cos_lat1 * np.cos(phi1)
    y1 = cos_lat1 * np.sin(phi1)
    z1 = np.sin(lat1)

    cos_lat2 = np.cos(lat2)
    x2 = cos_lat2 * np.cos(phi2)
    y2 = cos_lat2 * np.sin(phi2)
    z2 = np.sin(lat2)

    # 叉积的模（sin of angle）与点积（cos of angle）
    cross_x = y1*z2 - z1*y2
    cross_y = z1*x2 - x1*z2
    cross_z = x1*y2 - y1*x2
    sin_angle = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)
    cos_angle = x1*x2 + y1*y2 + z1*z2

    # atan2 在 [-π, π] 上连续，不会受 φ±π jump 影响（因为用了笛卡尔坐标而非 φ）
    return float(np.degrees(np.arctan2(sin_angle, np.clip(cos_angle, -1.0, 1.0))))


def non_max_suppression_sphere(
    detections,
    angle_threshold_deg=5.0,
    iou_threshold=0.35,
    containment_threshold=0.65,
):
    """混合 NMS：球面中心距离 + 回投矩形重叠，减少滑窗重复框。"""
    if not detections:
        return []

    classes = set(d["class"] for d in detections)
    kept = []

    for cls in classes:
        cls_dets = [d for d in detections if d["class"] == cls]
        cls_dets.sort(key=lambda x: x["conf"], reverse=True)

        while cls_dets:
            current = cls_dets.pop(0)
            kept.append(current)

            remaining = []
            for det in cls_dets:
                angle_match = False
                if "_phi" in det and "_lat" in det and "_phi" in current and "_lat" in current:
                    angle_dist = sphere_distance(
                        current["_phi"], current["_lat"],
                        det["_phi"], det["_lat"]
                    )
                    angle_match = angle_dist < angle_threshold_deg

                iou = compute_iou(current, det)
                containment = compute_containment(current, det)
                box_match = iou > iou_threshold or containment > containment_threshold

                if not (angle_match or box_match):
                    remaining.append(det)

            cls_dets = remaining

    return kept


def compute_iou(det1, det2):
    """计算两个检测框的IoU"""
    x1 = max(det1["x1"], det2["x1"])
    y1 = max(det1["y1"], det2["y1"])
    x2 = min(det1["x2"], det2["x2"])
    y2 = min(det1["y2"], det2["y2"])

    if x1 >= x2 or y1 >= y2:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = max(0, det1["x2"] - det1["x1"]) * max(0, det1["y2"] - det1["y1"])
    area2 = max(0, det2["x2"] - det2["x1"]) * max(0, det2["y2"] - det2["y1"])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def compute_containment(det1, det2):
    """交集占较小框面积的比例，用来删除大框套小框的重复检测。"""
    x1 = max(det1["x1"], det2["x1"])
    y1 = max(det1["y1"], det2["y1"])
    x2 = min(det1["x2"], det2["x2"])
    y2 = min(det1["y2"], det2["y2"])

    if x1 >= x2 or y1 >= y2:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = max(0, det1["x2"] - det1["x1"]) * max(0, det1["y2"] - det1["y1"])
    area2 = max(0, det2["x2"] - det2["x1"]) * max(0, det2["y2"] - det2["y1"])
    min_area = min(area1, area2)

    return intersection / min_area if min_area > 0 else 0.0


def cubemap_sliding_detection(
    model,
    frame,
    window_width_ratio=0.25,
    step_size=100,
    conf=0.7,
    face_size=512,
    edge_samples=9,
    nms_angle_threshold_deg=5.0,
    nms_iou_threshold=0.35,
    nms_containment_threshold=0.65,
):
    """
    在立方体面投影上滑动窗口检测（进行畸变矫正）

    Args:
        model: YOLO模型
        frame: 原始帧
        window_width_ratio: 窗口宽度占帧宽度的比例（每个面的窗口比例）
        step_size: 滑动步长
        conf: 置信度阈值
        face_size: 立方体面尺寸
        edge_samples: bbox 回投时每条边的采样点数
        nms_angle_threshold_deg: 球面中心距离NMS阈值（度）
        nms_iou_threshold: 平面IoU NMS阈值
        nms_containment_threshold: 小框包含率NMS阈值

    Returns:
        detections: NMS后的检测结果
        visualization: 绘制了检测框的帧
    """
    h, w = frame.shape[:2]

    # 检测的窗口配置（每个立方体面的配置）
    num_faces = 6  # 立方体6个面
    window_w_per_face = int(face_size * window_width_ratio)
    step_w = step_size

    all_detections = []

    # 遍历每个立方体面
    for face_id in range(num_faces):
        # 转换为立方体面
        face_img = equirectangular_to_face(frame, face_size=face_size, face=face_id)

        # 滑动窗口检测
        x_start = 0
        while x_start + window_w_per_face <= face_size:
            window = face_img[:, x_start:x_start + window_w_per_face]
            results = model(window, verbose=False, conf=conf)

            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]

                if cls_name == 'person':
                    x1_rel, y1_rel, x2_rel, y2_rel = box.xyxy[0].cpu().numpy()
                    conf_score = box.conf[0].cpu().numpy()

                    x1_face = x1_rel + x_start
                    x2_face = x2_rel + x_start

                    # bbox 回投是非线性的；沿四条边多采样，比只投四个角稳定得多。
                    xs = np.linspace(x1_face, x2_face, edge_samples)
                    ys = np.linspace(y1_rel, y2_rel, edge_samples)
                    sample_points = []
                    sample_points.extend((x, y1_rel) for x in xs)
                    sample_points.extend((x, y2_rel) for x in xs)
                    sample_points.extend((x1_face, y) for y in ys)
                    sample_points.extend((x2_face, y) for y in ys)

                    phis = []
                    lats = []
                    x_vals = []
                    y_vals = []
                    for x_face, y_face in sample_points:
                        u_c = 2.0 * x_face / face_size - 1.0
                        v_c = 1.0 - 2.0 * y_face / face_size
                        cx, cy, cz = cubemap_face_xyz(u_c, v_c, face_id)
                        phi_val, lat_val, x_e, y_e = xyz_to_equirectangular(cx, cy, cz, w, h)

                        phis.append(float(phi_val))
                        lats.append(float(lat_val))
                        x_vals.append(float(x_e))
                        y_vals.append(float(y_e))

                    x_vals = np.array(x_vals, dtype=np.float32)
                    y_vals = np.array(y_vals, dtype=np.float32)
                    if x_vals.max() - x_vals.min() > w * 0.5:
                        x_vals[x_vals < w * 0.5] += w

                    x_equirect_min = int(np.clip(np.floor(x_vals.min()), 0, w - 1))
                    x_equirect_max = int(np.clip(np.ceil(x_vals.max()), 0, w - 1))
                    y_equirect_min = int(np.clip(np.floor(y_vals.min()), 0, h - 1))
                    y_equirect_max = int(np.clip(np.ceil(y_vals.max()), 0, h - 1))

                    x_center_face = (x1_face + x2_face) * 0.5
                    y_center_face = (y1_rel + y2_rel) * 0.5
                    u_center = 2.0 * x_center_face / face_size - 1.0
                    v_center = 1.0 - 2.0 * y_center_face / face_size
                    cx, cy, cz = cubemap_face_xyz(u_center, v_center, face_id)
                    phi_center, lat_center, x_center, y_center = xyz_to_equirectangular(cx, cy, cz, w, h)
                    x_equirect_c = int(np.clip(round(float(x_center)), 0, w - 1))
                    y_equirect_c = int(np.clip(round(float(y_center)), 0, h - 1))

                    all_detections.append({
                        'x1': max(0, min(x_equirect_min, w-1)),
                        'y1': max(0, min(y_equirect_min, h-1)),
                        'x2': max(0, min(x_equirect_max, w-1)),
                        'y2': max(0, min(y_equirect_max, h-1)),
                        'conf': float(conf_score),
                        'class': cls_name,
                        'face_id': face_id,
                        '_phi': phi_center,
                        '_lat': lat_center,
                        '_x_c': x_equirect_c,
                        '_y_c': y_equirect_c,
                    })

            x_start += step_w

    # After ALL faces — call NMS once on all detections
    nms_detections = non_max_suppression_sphere(
        all_detections,
        angle_threshold_deg=nms_angle_threshold_deg,
        iou_threshold=nms_iou_threshold,
        containment_threshold=nms_containment_threshold,
    )
    face_ids_set = set(d['face_id'] for d in all_detections)
    print(f"[debug] Total raw: {len(all_detections)}, After hybrid NMS: {len(nms_detections)} detections, faces={face_ids_set}")

    visualization = frame.copy()
    color = (255, 165, 0)

    if nms_detections:
        for det in nms_detections:
            cv2.rectangle(visualization, (det['x1'], det['y1']), (det['x2'], det['y2']), color, 2)
            cv2.putText(visualization, f"{det['class']} {det['conf']:.2f} face{det['face_id']}",
                       (det['x1'], det['y1'] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cx = (det['x1'] + det['x2']) // 2
            cy = (det['y1'] + det['y2']) // 2
            cv2.circle(visualization, (cx, cy), 3, color, -1)

    else:
        # default green drawing
        for det in all_detections:
            cv2.rectangle(visualization, (det['x1'], det['y1']), (det['x2'], det['y2']), (0, 255, 0), 2)
            cv2.putText(visualization, f"person {det['conf']:.2f}",
                       (det['x1'], det['y1'] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return nms_detections, visualization


def process_360_video(
    video_path,
    output_path=None,
    frames_output_dir=None,
    model_path="yolov8n.pt",
    window_width_ratio=0.25,
    step_size=100,
    face_size=512,
    frame_skip=10,
    conf=0.7,
    edge_samples=9,
    nms_angle_threshold_deg=5.0,
    nms_iou_threshold=0.35,
    nms_containment_threshold=0.65,
    save_frames=False,
    debug_vis=False,           # 是否启用调试可视化
    debug_dir=None            # 调试可视化输出目录
):
    """
    处理360视频（立方体面投影 + 滑动窗口YOLO检测）

    Args:
        video_path: 视频路径
        output_path: 输出视频路径
        frames_output_dir: 保存帧的目录
        model_path: YOLO模型路径
        window_width_ratio: 每个面的滑动窗口占面宽比例
        step_size: 滑动步长
        face_size: 立方体面尺寸
        conf: 置信度阈值
        edge_samples: bbox 回投时每条边的采样点数
        nms_angle_threshold_deg: 球面中心距离NMS阈值（度）
        nms_iou_threshold: 平面IoU NMS阈值
        nms_containment_threshold: 小框包含率NMS阈值
        save_frames: 是否保存帧
        debug_vis: 是否启用调试可视化（立方体面映射验证）
        debug_dir: 调试可视化输出目录
    """
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    print(f"Loading YOLOv8 model: {model_path}")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info: {width}x{height}, {fps} fps, {total_frames} frames")
    print(f"Sliding window (per face): width={int(face_size * window_width_ratio)}, step={step_size}, conf={conf}")
    print(f"BBox edge samples: {edge_samples}")
    print(f"NMS: angle={nms_angle_threshold_deg}, iou={nms_iou_threshold}, containment={nms_containment_threshold}")
    print(f"Debug visualization: {debug_vis}")

    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if save_frames and frames_output_dir:
        os.makedirs(frames_output_dir, exist_ok=True)

    frame_count = 0
    detected_frames = 0
    all_person_counts = []
    total_detections = 0

    print(f"\nProcessing video...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % frame_skip == 0:
            # 调试可视化：在第一个检测帧时保存立方体映射可视化
            if debug_vis and debug_dir and frame_count == frame_skip:
                print(f"Saving cube mapping visualization...")
                vis_path = os.path.join(debug_dir, "cube_mapping.jpg")
                visualize_cube_mapping(frame, face_size=face_size, output_path=vis_path)

            detections, annotated_frame = cubemap_sliding_detection(
                model,
                frame,
                window_width_ratio=window_width_ratio,
                step_size=step_size,
                conf=conf,
                face_size=face_size,
                edge_samples=edge_samples,
                nms_angle_threshold_deg=nms_angle_threshold_deg,
                nms_iou_threshold=nms_iou_threshold,
                nms_containment_threshold=nms_containment_threshold,
            )

            person_count = len([d for d in detections if d['class'] == 'person'])
            all_person_counts.append(person_count)
            total_detections += len(detections)

            if person_count > 0:
                detected_frames += 1
                print(f"Frame {frame_count}: Detected {person_count} person(s), total det={len(detections)}")

            if out:
                out.write(annotated_frame)

            if save_frames and frames_output_dir:
                frame_filename = os.path.join(frames_output_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_filename, annotated_frame)
                print(f"  -> Saved {frame_filename}")

        progress = (frame_count / total_frames) * 100
        print(f"\rProgress: {progress:.1f}%", end='', flush=True)

    cap.release()
    if out:
        out.release()

    print(f"\n\n=== Summary ===")
    print(f"Total frames processed: {frame_count}")
    print(f"Frames with person detected: {detected_frames}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per frame: {total_detections / len(all_person_counts) if all_person_counts else 0:.2f}")

    if output_path:
        print(f"\nOutput video saved to: {output_path}")

    if save_frames and frames_output_dir:
        print(f"\nDetected frames saved to: {frames_output_dir}")

    if debug_vis and debug_dir:
        print(f"\nDebug visualizations saved to: {debug_dir}")

    return True


if __name__ == "__main__":
    CONFIG = {
        # 路径
        "video_path": "/mnt/dataset/skiing/raw_new/kimura2_360.mp4",
        "output_path": "/mnt/dataset/skiing/raw_new/kimura2_360_cubemap_detected.mp4",
        "frames_output_dir": "/mnt/dataset/skiing/raw_new/kimura2_360_cubemap_frames",
        "debug_dir": "/mnt/dataset/skiing/raw_new/kimura2_360_debug_vis",
        "model_path": "yolov8n.pt",

        # 检测参数
        "window_width_ratio": 0.4,
        "step_size": 50,
        "face_size": 512,
        "frame_skip": 10,
        "conf": 0.7,

        # bbox 回投和去重参数
        "edge_samples": 9,
        "nms_angle_threshold_deg": 5.0,
        "nms_iou_threshold": 0.35,
        "nms_containment_threshold": 0.65,

        # 输出/调试开关
        "save_frames": True,
        "debug_vis": True,
    }

    if not os.path.exists(CONFIG["video_path"]):
        print("Error: Video file not found: " + CONFIG["video_path"])
    else:
        if CONFIG["debug_vis"] and CONFIG["debug_dir"]:
            os.makedirs(CONFIG["debug_dir"], exist_ok=True)

        process_360_video(**CONFIG)
