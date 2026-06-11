#!/usr/bin/env python3
"""
视频 VLM 分析脚本
===================
从视频中按时间均匀采样帧，送入 VLM（视觉语言模型）进行逐帧理解。

用法:
    conda run -n vlminference python vlm_video_analyze.py
    conda run -n vlminference python vlm_video_analyze.py --frames 8
    conda run -n vlminference python vlm_video_analyze.py --model Qwen/Qwen2.5-VL-72B-Instruct --gpus 0,1

配置项在脚本顶部的 CONFIG 字典中，修改 prompt / 采样帧数 / 模型名即可。
"""

# ==============================================================================
# ⬇️ 在这里修改你的配置
# ==============================================================================
CONFIG = {
    # ----- 视频路径 -----
    "video_path": "/mnt/dataset/skiing/360PoseFusion/kimura2_360_half.mp4",

    # ----- 模型选择 (HuggingFace repo) -----
    # 推荐:
    #   Qwen/Qwen2.5-VL-72B-Instruct      — 最强中文理解，需要 ~144GB VRAM (API/多卡)
    #   Qwen/Qwen2.5-VL-32B-Instruct      — 次强，~64GB VRAM
    #   Qwen/Qwen2.5-VL-7B-Instruct       — 单张 RTX 3090 可跑 ✅
    #     （如果显存不够就选这个）
    #   Alibaba-NLP/Qwen2.5-VL-7B-Instruct — 同上的 HuggingFace 路径
    "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",

    # ----- GPU 配置 -----
    # "auto" = 用所有可见 GPU；也可以指定 "0" / "0,1" / "cpu"
    # 单张 RTX 3090 建议用 "0"，多卡可尝试 "auto"
    "device_map": "auto",

    # ----- VLM 生成参数 -----
    "max_new_tokens": 1024,
    "temperature": 0.7,

    # ----- 帧缩放比例 (节省 GPU 显存) -----
    # 将每帧的宽高都乘以此系数后再送入 VLM。默认 1.0（原图）。
    # 缩小到一半: 0.5，再小: 0.25。越小越省显存，但细节会丢失。
    "frame_scale": 0.5,

    # ----- 全局默认值 ============
    "default_num_frames": 6,   # 每段（或无分段时）默认的采样帧数
    "default_prompt": (        # 如果没有 per-segment prompt，使用此默认提示词
        "请仔细观察以下视频帧，逐帧描述你所看到的内容，"
        "特别注意：\n"
        "1. 每个帧中人物的姿态、动作和位置\n"
        "2. 人物的服装特征\n"
        "3. 场景环境（室内/室外、背景等）\n"
        "4. 帧与帧之间的动作变化\n\n"
        "请用中文输出你的详细分析。"
    ),

    # ----- 分段配置（三选一，优先级从高到低）==========

    # A. segments = 自定义段列表（支持每段不同 num_frames / prompt）
    #    segments: [{"name": "准备", "start_sec": 0, "end_sec": 3}, ...]

    # B. segments_by = [秒数边界列表]，自动在边界间切段
    #    segments_by: [0, 5, 12, 16, 20, 24, 30] → 5段: [0-5], [5-12], [12-16] ...

    # C. segment_duration = 固定间隔秒数，视频从 0 开始按此间隔切到底
    #    segment_duration: 1 → 30s 视频切成 30 段，每段 1s
    "segment_duration": None,   # 改为 1 即按 1s 切；设为 None 关闭

    "segments_by": None,        # 改为 [0, 5, 12, 16, ...] 即可启用

    "segments": None,           # 优先级最高：[{"name": "...", "start_sec": ..., "end_sec": ...}, ...]

    # ----- 输出保存路径 -----
    # 空字符串 "" 表示不保存到文件，仅打印到控制台
    "output_path": "/mnt/dataset/skiing/360PoseFusion/vlm_analysis_result.txt",
}
# ==============================================================================
# ⬆️ 配置结束
# ==============================================================================

import os
import sys
import math
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import decord
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def select_device(device_map: str):
    """解析 device_map 参数，返回目标设备字符串。"""
    if device_map.lower() == "cpu":
        return "cpu"
    if device_map.lower() == "auto":
        n_gpus = torch.cuda.device_count()
        if n_gpus > 0:
            # auto 模式下用多卡自动分配
            return "auto"
        print("[!] 无 GPU，回退到 cpu")
        return "cpu"
    # 处理 "0", "cuda:0", "0,1", "cuda:0,cuda:1" 等格式
    parts = [x.strip() for x in device_map.split(",")]
    gpu_ids = []
    for p in parts:
        if p.isdigit():
            gpu_ids.append(int(p))
        elif p.lower().startswith("cuda:"):
            try:
                gpu_ids.append(int(p.split(":")[1]))
            except (ValueError, IndexError):
                pass
    n_gpus = torch.cuda.device_count()
    if not gpu_ids or any(g >= n_gpus for g in gpu_ids):
        print(f"[!] 无法解析 device_map '{device_map}'，使用 cuda:0")
        return "cuda:0"
    # 多卡时返回 auto（让 transformers 自动分配）
    if len(gpu_ids) > 1:
        return "auto"
    return f"cuda:{gpu_ids[0]}"


def load_model(model_name: str, target_device: str):
    """加载 VLM 模型和处理器。用 device_map={""} 直接指定 GPU 避免加速库分片。"""
    print(f"\n{'='*60}")
    print(f"📦 正在加载模型: {model_name}")
    print(f"   target_device: {target_device}")
    print(f"{'='*60}\n")

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # device_map={"": target_device} 直接加载到指定 GPU（避免 accelerate 分片）
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"   使用精度: {dtype}")

    # auto → 用 accelarate 自动分配；否则取 index
    if target_device.lower() == "auto":
        dev_map = "auto"
    else:
        device_idx = int(target_device.split(":")[1]) if ":" in target_device else 0
        dev_map = {"": device_idx}

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=dev_map,
        trust_remote_code=True,
    )
    model.eval()

    return model, processor


def open_video(video_path: str):
    """打开视频，返回 (VideoReader, fps, total_frames, duration)。"""
    if not os.path.exists(video_path):
        print(f"[!] 视频不存在: {video_path}")
        sys.exit(1)

    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    duration = total_frames / fps
    print(f"   总帧数: {total_frames}, 时长: {duration:.2f}s, FPS: {fps:.1f}")
    return vr, fps, total_frames, duration


def extract_frames_from_range(vr, fps, start_sec: float, end_sec: float, num_frames: int):
    """在 [start_sec, end_sec] 范围内均匀采样帧，返回 [(timestamp_sec, PIL.Image)] 列表。"""
    start_idx = max(0, int(start_sec * fps))
    end_idx = min(len(vr) - 1, int(end_sec * fps))

    if start_idx == end_idx and num_frames > 1:
        # 区间太短无法采样，往两端扩展
        mid = (start_idx + end_idx) // 2
        start_idx = max(0, mid - num_frames // 2)
        end_idx = min(len(vr) - 1, mid + num_frames // 2)

    indices = np.linspace(start_idx, end_idx, num_frames, dtype=int)
    indices = np.unique(indices)

    frames = []
    for i in indices:
        img_pil = vr[i].asnumpy()  # HWC, RGB
        ts = i / fps
        frames.append((ts, Image.fromarray(img_pil)))
        print(f"   [{start_sec:.1f}-{end_sec:.1f}s] 采样: idx={i}, t={ts:.2f}s")

    return frames


def build_chat_prompt(prompt: str, frames_times: list[float], segment_name: str = "") -> list[dict]:
    """
    构建 Qwen2.5-VL 的对话消息格式。
    每帧关联一个时间戳 ref，使模型理解各帧来自视频的哪个时刻。
    """
    image_refs = [{'type': 'image', 'image': '', 'index': i} for i, _ in enumerate(frames_times)]
    if segment_name:
        prompt = f"[阶段：{segment_name}]\n" + prompt
    messages = [
        {
            "role": "user",
            "content": [*image_refs, {"type": "text", "text": prompt}],
        }
    ]
    return messages


@torch.no_grad()
def analyze_frames(model, processor, frames_times: list[float], frames_images: list[Image.Image], config: dict):
    """将帧送入 VLM 并返回文本输出。"""
    messages = build_chat_prompt(config["prompt"], frames_times)

    # 处理文本 + 图像
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 缩放帧以节省显存
    scale = config.get("frame_scale", 1.0)
    if scale < 1.0:
        print(f"\n🔍 帧缩放比例: {scale}（原图 → 缩小以节省显存）")
        scaled_images = []
        for img in frames_images:
            w, h = img.size
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            scaled_images.append(img.resize((new_w, new_h), Image.LANCZOS))
            print(f"   {w}x{h} → {new_w}x{new_h}")
    else:
        scaled_images = frames_images

    inputs = processor(
        text=[text],
        images=scaled_images,
        video_frames=None,
        padding=True,
        return_tensors="pt",
    )

    # 移到正确设备
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print(f"\n🧠 正在推理模型 (输出长度上限={config['max_new_tokens']})...")

    output_ids = model.generate(
        **inputs,
        max_new_tokens=config["max_new_tokens"],
        temperature=config["temperature"],
        do_sample=config["temperature"] > 0,
    )

    # 解码（去掉 prompt 部分，只保留生成的）
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]

    return generated_text


def main():
    parser = argparse.ArgumentParser(description="VLM 视频逐帧分析")
    parser.add_argument("--config", default=None, help="JSON 配置文件路径（覆盖 CONFIG）")
    args, _ = parser.parse_known_args()

    # 支持命令行覆盖
    cli_parser = argparse.ArgumentParser()
    for key in CONFIG:
        flag = f"--{key}"
        if isinstance(CONFIG[key], int):
            cli_parser.add_argument(flag, type=int)
        elif isinstance(CONFIG[key], float):
            cli_parser.add_argument(flag, type=float)
        else:
            cli_parser.add_argument(flag, type=str)
    cli_args, _ = cli_parser.parse_known_args()

    # 合并配置: CONFIG > args.config (JSON) > CLI overrides
    config = dict(CONFIG)
    if args.config and os.path.exists(args.config):
        import json
        with open(args.config) as f:
            config.update(json.load(f))
    for key in config:
        val = getattr(cli_args, key.replace("-", "_"), None)
        if val is not None:
            config[key] = val

    # ========== 分段配置（二选一）============
    default_num_frames = config.get("default_num_frames", 6)
    default_prompt   = config.get("default_prompt", (
        "请仔细观察以下视频帧，逐帧描述你所看到的内容，"
        "特别注意：\n"
        "1. 每个帧中人物的姿态、动作和位置\n"
        "2. 人物的服装特征\n"
        "3. 场景环境（室内/室外、背景等）\n"
        "4. 帧与帧之间的动作变化\n\n"
        "请用中文输出你的详细分析。"
    ))

    segments = config.get("segments", None)
    segments_by = config.get("segments_by", None)      # [秒数边界列表] → 自动切段

    # A: segments 已存在 → 直接用；否则在 open_video 后根据 B/C 生成

    # 1. 选择设备
    target_device = select_device(config["device_map"])
    print(f"🔧 GPU 设备: {target_device}")

    # 2. 加载模型
    model, processor = load_model(config["model_name"], target_device)

    # 3. 打开视频（只打开一次，所有段复用）
    vr, fps, total_frames, duration = open_video(config["video_path"])
    print(f"\n{'='*60}")

    # 如果 segments 未设置，根据 segment_duration 或 segments_by 自动生成
    if segments is None:
        segment_duration = config.get("segment_duration", None)
        segments_by = config.get("segments_by", None)
        if segments_by is not None and len(segments_by) >= 2:
            # B: [0, 5, 12] → [(0,5), (5,12)]
            segments = [{"start_sec": segments_by[i], "end_sec": segments_by[i+1]} for i in range(len(segments_by)-1)]
        elif segment_duration is not None and segment_duration > 0:
            # C: 固定间隔秒数切段：duration=30s, segment_duration=1 → 30段 [0-1), [1-2), ...
            num_segs = int(duration // segment_duration)
            segments = [{"start_sec": i * segment_duration, "end_sec": (i+1) * segment_duration} for i in range(num_segs)]
        # else: 仍为 None，走全局均匀采样

    if segments is None or len(segments) == 0:
        # ========== 无分段：全局均匀采样（原有行为）==========
        num_frames = config.get("num_frames", default_num_frames)
        prompt = config.get("prompt", default_prompt)
        all_frames = extract_frames_from_range(vr, fps, 0, duration, num_frames)

        if not all_frames:
            print("[!] 未提取到任何帧，退出")
            return

        frames_times, frames_images = zip(*all_frames)

        result = analyze_frames(model, processor, list(frames_times), list(frames_images), {
            **config, "prompt": prompt
        })

        results = [{"segment_name": "全部", "times": list(frames_times), "result": result}]

    else:
        # ========== 有分段：每段独立采样、独立推理 ==========
        results = []
        for seg_idx, seg in enumerate(segments):
            seg_name  = seg.get("name", f"阶段{seg_idx+1}")
            start_sec = float(seg["start_sec"])
            end_sec   = float(seg["end_sec"])
            seg_frames_num = seg.get("num_frames", default_num_frames)
            seg_prompt   = seg.get("prompt", None)
            prompt       = seg_prompt if seg_prompt else default_prompt

            print(f"\n{'='*60}")
            print(f"📌 段 [{seg_idx+1}/{len(segments)}]: {seg_name}  "
                  f"[{start_sec:.1f}s - {end_sec:.1f}s], 采样 {seg_frames_num} 帧")
            print(f"{'='*60}")

            all_frames = extract_frames_from_range(vr, fps, start_sec, end_sec, seg_frames_num)

            if not all_frames:
                print(f"[!] 段 '{seg_name}' 未提取到任何帧，跳过")
                continue

            frames_times, frames_images = zip(*all_frames)

            seg_result = analyze_frames(model, processor, list(frames_times), list(frames_images), {
                **config, "prompt": prompt
            })

            results.append({
                "segment_name": seg_name,
                "times": list(frames_times),
                "result": seg_result,
            })

    # 4. 输出结果（逐段独立输出）
    print(f"\n{'='*60}")
    print("📝 VLM 分析结果:")
    print(f"{'='*60}\n")

    output_parts = []
    for r in results:
        if len(results) > 1:
            print(f"\n>>> === {r['segment_name']} ===")
        print(r["result"])
        print()
        output_parts.append(f"--- {r['segment_name']} ---\n{r['result']}\n")

    # 5. 第二轮推理：将所有段的分析结果喂给模型，生成整体总结
    if len(results) > 1:
        all_text = "\n\n".join([f"[{r['segment_name']}] ({r['times'][0]:.1f}s–{r['times'][-1]:.1f}s):\n{r['result']}" for r in results])

        summary_prompt = (
            "以上是一个视频按时间段分段后，VLM 对每段帧的分析结果。"
            "请基于这些片段分析，做以下整合：\n\n"
            "1. **整体动作流程总结**：人物从开始到结束做了什么？\n"
            "2. **关键姿态变化**：各段之间人物的姿态/动作有什么连续变化？\n"
            "3. **服装与环境一致性**：各段观察到的服装、背景是否一致？有无场景转换？\n"
            "4. **帧间连贯性判断**：是否存在不合理或断裂的动作跳跃？\n\n"
            f"{all_text}\n\n请综合以上信息，给出完整的整体分析报告。"
        )

        # 构造一个"虚拟帧"来复用 analyze_frames — 只传文本不传图像，省显存
        summary_config = {**config, "prompt": summary_prompt, "summary_mode": True}

        print(f"\n🔗 正在生成整体总结（基于以上 {len(results)} 段分析）...")
        overall_summary = _analyze_text_only(model, processor, summary_prompt, summary_config)

        # 输出整体总结
        print(f"\n>>> === 整体总结 ===")
        print(overall_summary)
        output_parts.append(f"\n--- 整体总结 ---\n{overall_summary}\n")
    else:
        overall_summary = None

    # 6. 保存结果到文件
    output_path = config.get("output_path", "")
    if output_path:
        Path(output_path).write_text("\n\n".join(output_parts), encoding="utf-8")
        print(f"\n💾 结果已保存到: {output_path}\n")


def _analyze_text_only(model, processor, prompt: str, config: dict):
    """纯文本模式的 VLM 推理（不传图像），用于整体总结。"""
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=[text], return_tensors="pt")
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=config["max_new_tokens"],
        temperature=config.get("temperature", 0.7),
        do_sample=config.get("temperature", 0.7) > 0,
    )

    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]


def write_log(msg: str, filepath: str = "/tmp/vlm_progress.log"):
    with open(filepath, "a") as f:
        f.write(msg + "\n")
        f.flush()


if __name__ == "__main__":
    try:
        write_log("=== VLM video analyze started ===")
        main()
        write_log("=== VLM video analyze finished ===")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        write_log(f"ERROR:\n{tb}")
