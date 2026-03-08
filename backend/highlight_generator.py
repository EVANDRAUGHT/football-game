#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
精彩视频生成模块 — 足球智能视频分析系统
========================================

【核心算法原理】

一、精彩时刻检测（HighlightDetector）
--------------------------------------
本模块采用"运动强度 × 人群密度"双特征融合算法，从原始视频中智能识别
进球、射门、庆祝、传球等关键时刻，具体步骤如下：

1. **稀疏采样**（每 N 帧取1帧，默认 N=20）
   - 对于大视频（1GB+），全帧分析耗时过长
   - 采用均匀采样策略，将分析帧数压缩至原始的 1/20
   - 采样间隔越小精度越高，但耗时越长；默认 N=20 可将速度提升约4倍

2. **光流分析（Farneback稠密光流）**
   - 算法：cv2.calcOpticalFlowFarneback
   - 原理：对相邻两帧灰度图计算每个像素的运动速度矢量场
   - 特征：magnitude = sqrt(Vx² + Vy²)，即各像素运动幅度均值
   - 归一化：将幅度均值除以阈值10.0，映射到 [0, 1]
   - 足球场景中，射门、突破、奔跑冲刺会产生高运动强度

3. **人群密度估计（YOLOv8目标检测）**
   - 对采样帧运行 YOLOv8 推理，统计 class=0（person）的边界框数量
   - 归一化公式：density = min(1.0, person_count / 10.0)
   - 进球后庆祝聚集、角球/任意球墙等场景密度高

4. **多特征融合评分**
   - combined_score = motion × 0.6 + density × 0.4
   - 运动权重高于密度权重，因为运动是精彩动作的更直接指标

5. **时间序列平滑（滑动平均）**
   - 窗口大小 = fps × 3 帧（约3秒）
   - 目的：消除单帧噪声，使评分曲线更平滑稳定

6. **峰值检测**
   - 条件：评分 > 阈值(0.5)，且为局部极大值
   - 最小间距：fps × 5秒（避免连续时刻重叠）

7. **时刻类型分类（规则引擎）**
   ┌─────────────────────────────────────────┐
   │  运动>0.7 AND 密度>0.6  →  goal  (进球) │
   │  运动>0.7               →  shot  (射门) │
   │  密度>0.7               →  celebration  │
   │  运动>0.5               →  pass  (传球) │
   │  其他                   →  action       │
   └─────────────────────────────────────────┘

二、视频特效处理（VideoEffectsProcessor）
------------------------------------------
- 慢动作：ffmpeg setpts=2*PTS（0.5倍速播放）
- 特写镜头：OpenCV中心区域裁剪 + INTER_CUBIC双三次插值放大
- 画中画：ffmpeg overlay滤镜，右下角0.25尺寸，0.3倍慢放

三、精彩视频合成（HighlightVideoGenerator）
--------------------------------------------
- 按分数贪心选取片段：优先选入进球时刻（最多5个），再按分数填满目标时长
- ffmpeg concat协议无损合并，保持原始画质

【技术选型】
- 视频处理：OpenCV + ffmpeg
- 目标检测：YOLOv8n（轻量模型，推理速度快）
- 光流计算：Farneback稠密光流（精度与速度均衡）
- 后端框架：FastAPI（异步高性能）
- 前端框架：原生HTML/JS + Chart.js
"""

import cv2
import numpy as np
import os
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import timedelta
import subprocess

# ── 获取 ffmpeg 可执行文件路径（兼容 Windows 未配置 PATH 的情况）─────────────
def _get_ffmpeg_exe() -> str:
    """优先使用 imageio_ffmpeg 内置的 ffmpeg，回退到系统 ffmpeg"""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return 'ffmpeg'

FFMPEG_EXE = _get_ffmpeg_exe()


def _build_atempo_chain(speed_factor: float) -> str:
    """
    构建 ffmpeg atempo 滤镜链（每级范围 [0.5, 2.0]）。
    例：speed_factor=0.35 → 'atempo=0.5,atempo=0.700'（0.5×0.7=0.35）
    """
    if speed_factor <= 0 or speed_factor > 1.0:
        return f'atempo={speed_factor:.3f}'
    chain_parts = []
    remaining = speed_factor
    while remaining < 0.5:
        chain_parts.append('atempo=0.500')
        remaining /= 0.5
    chain_parts.append(f'atempo={remaining:.3f}')
    return ','.join(chain_parts)


# ── BGM 文件路径（内置生成，无需外部文件）────────────────────────────────────
import math, struct, wave, tempfile

def _generate_bgm_wav(duration: float = 300.0) -> str:
    """
    程序化生成激昂对抗风格背景音乐 WAV（无需外部素材）。
    风格：电子硬核 + 重金属鼓组，BPM=155，高度紧张对抗感。
    多层合成：
      - 超重低音踢鼓（深沉下扫，每拍起始）
      - 军鼓 + 开放 hi-hat（反拍强打击）
      - 电吉他失真和弦（Am-F-C-G 循环，锯齿波近似）
      - 贝斯线（根音 + 八度交替，推进感强）
      - 高频合成器刮弦（每节拍前段）
    """
    sample_rate = 44100
    num_samples = int(sample_rate * duration)
    bpm = 155.0
    beat_interval = 60.0 / bpm
    beat_samples  = int(beat_interval * sample_rate)
    bar_samples   = beat_samples * 4   # 一小节 = 4拍

    # 和弦根音序列：Am-F-C-E (各占一小节，循环)
    chord_roots = [220.0, 174.6, 261.6, 329.6]

    data = []
    for i in range(num_samples):
        t = i / sample_rate
        beat_phase = (i % beat_samples) / beat_samples
        bar_phase  = (i % bar_samples)  / bar_samples

        # ── 层1：超重低音踢鼓（每拍强起 + 每拍3/4处加花）──────────────────
        kick = 0.0
        bp2 = beat_phase if beat_phase < 0.75 else beat_phase - 0.75
        for bp_trigger, amp in [(beat_phase, 0.70), (bp2 if beat_phase >= 0.72 else -1, 0.40)]:
            if 0 <= bp_trigger < 0.06:
                env = math.exp(-bp_trigger * 55)
                fsweep = 100 * math.exp(-bp_trigger * 45)
                kick += amp * env * math.sin(2 * math.pi * fsweep * t)

        # ── 层2：军鼓（反拍：每小节第2、4拍）─────────────────────────────
        snare = 0.0
        if 0.48 < bar_phase < 0.52 or 0.98 < bar_phase <= 1.0 or bar_phase < 0.01:
            env_s = math.exp(-((bar_phase % 0.5) * 200))
            snare = 0.45 * env_s * (
                math.sin(2 * math.pi * 2800 * t) * 0.35 +
                math.sin(2 * math.pi * 4200 * t) * 0.35 +
                math.sin(2 * math.pi * 6800 * t) * 0.30
            )

        # ── 层3：hi-hat（每8分音符，偶数强/奇数弱）────────────────────────
        hihat = 0.0
        eighth_phase = (i % (beat_samples // 2)) / (beat_samples // 2)
        eighth_count = (i // (beat_samples // 2)) % 2
        if eighth_phase < 0.018:
            env_h = math.exp(-eighth_phase * 500)
            amp_h = 0.20 if eighth_count == 0 else 0.10
            hihat = amp_h * env_h * math.sin(2 * math.pi * 9500 * t)

        # ── 层4：电吉他失真和弦（锯齿波 + 软削波近似失真）─────────────────
        chord_idx = int(t / (beat_interval * 4)) % 4
        root = chord_roots[chord_idx]
        # 锯齿波：谐波叠加（基音+3度+5度+8度+泛音）
        guitar = 0.0
        for harmonic, amp_g, detune in [
            (1.0,   0.30, 0.0),
            (1.259, 0.22, 0.003),   # 小三度
            (1.498, 0.18, -0.002),  # 纯五度
            (2.0,   0.12, 0.001),   # 八度
            (2.52,  0.08, 0.002),   # 八度+三度（泛音）
        ]:
            freq = root * harmonic * (1 + detune)
            guitar += amp_g * math.sin(2 * math.pi * freq * t)
        # 失真（软限幅）
        guitar = math.tanh(guitar * 3.5) * 0.28
        # 吉他节奏门控：每拍起始打弦，之后渐弱
        gate = math.exp(-beat_phase * 2.5) * 0.6 + 0.4
        guitar *= gate

        # ── 层5：贝斯线（根音+八度交替，每拍切换）────────────────────────
        bass_root = chord_roots[chord_idx] * 0.5   # 低八度
        bass_oct  = chord_roots[chord_idx]          # 原八度
        bass_freq = bass_root if beat_phase < 0.5 else bass_oct
        bass = 0.32 * math.sin(2 * math.pi * bass_freq * t)
        # 贝斯包络（每拍紧凑衰减）
        bass *= math.exp(-beat_phase * 3.0) * 0.75 + 0.25

        # ── 层6：高频合成器刮弦（每拍前段制造紧张感）──────────────────────
        synth = 0.0
        if beat_phase < 0.12:
            sweep_freq = root * 4 * (1 + beat_phase * 2)
            env_syn = math.exp(-beat_phase * 30)
            synth = 0.10 * env_syn * math.sin(2 * math.pi * sweep_freq * t)

        # ── 混合 & 整体音量渐入渐出 ─────────────────────────────────────────
        fade_in  = min(1.0, t / 2.0)
        fade_out = min(1.0, (duration - t) / 2.0)
        envelope = fade_in * fade_out

        sample = (kick + snare + hihat + guitar + bass + synth) * envelope
        # 整体软削波防爆音
        sample = math.tanh(sample * 1.5) * 0.82
        data.append(max(-32767, min(32767, int(sample * 32767))))

    # 写 WAV
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp_path = tmp.name
    tmp.close()
    with wave.open(tmp_path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f'<{len(data)}h', *data))
    return tmp_path


# ============================================================================
#  数据结构定义
# ============================================================================

@dataclass
class HighlightMoment:
    """精彩时刻数据结构"""
    start_time: float  # 开始时间（秒）
    end_time: float    # 结束时间（秒）
    moment_type: str   # 类型：'goal', 'shot', 'pass', 'save', 'tackle'
    score: float       # 精彩度评分 (0-1)
    description: str   # 描述
    metadata: dict     # 额外信息


@dataclass
class VideoClip:
    """视频片段数据结构"""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    moment_type: str
    score: float
    apply_slowmo: bool = False  # 是否应用慢动作
    slowmo_factor: float = 0.5  # 慢动作速度（0.5 = 一半速度）
    apply_zoom: bool = False    # 是否应用特写


# ============================================================================
#  精彩时刻检测器
# ============================================================================

class HighlightDetector:
    """
    精彩时刻检测器
    通过分析视频内容识别关键时刻
    """
    
    def __init__(self, yolo_model=None):
        self.yolo_model = yolo_model
        self.frame_history = []  # 帧历史记录
        self.motion_history = []  # 运动历史
        
    def detect_highlights(self, video_path: str, progress_callback=None) -> List[HighlightMoment]:
        """
        核心方法：检测视频中的精彩时刻

        算法流程：
          1. 打开视频，获取帧率/总帧数/时长
          2. 逐帧（每20帧采样1帧）计算：
             a. 光流运动强度（Farneback 稠密光流 → 幅度均值归一化）
             b. 人群密度（YOLOv8 检测 person 数量归一化）
          3. 调用 _identify_highlights 识别峰值时刻
          4. 返回 HighlightMoment 列表，按精彩度降序排列

        Args:
            video_path: 视频文件绝对路径
            progress_callback: 进度回调函数 callback(percent: int)，范围 [0, 50]

        Returns:
            List[HighlightMoment]，每个元素代表一个精彩时刻区间
        """
        
        try:
            print(f"[HIGHLIGHT] 开始分析视频: {video_path}")
            
            # 🔥 立即通知进度开始
            if progress_callback:
                progress_callback(5)
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[ERROR] 无法打开视频: {video_path}")
                return []
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            print(f"[HIGHLIGHT] 视频信息: {duration:.1f}秒, {fps:.1f} FPS, {total_frames} 帧")
            
            # 🔥 再次通知进度
            if progress_callback:
                progress_callback(8)
            
            highlights = []
            frame_idx = 0
            prev_frame = None
            motion_scores = []
            density_scores = []
            motion_delta_scores = []  # 运动突变分数（用于捕捉射门/过人瞬间）
            last_motion = 0.0

            # 采样间隔：每 SAMPLE_INTERVAL 帧分析一次
            # 10帧采样（约3帧/秒@30fps），可捕捉 0.3秒内的快速射门/过人动作
            SAMPLE_INTERVAL = 10

            # 分析采样帧
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                # ── 关键修复：先做跳帧判断，非采样帧只保存 prev_frame ──
                if frame_idx % SAMPLE_INTERVAL != 0:
                    # 仅更新 prev_frame，不做任何耗时计算
                    prev_frame = frame
                    continue

                # ── 进度回调（每个采样帧更新一次）────────────────────────
                if progress_callback:
                    raw_progress = (frame_idx / total_frames) * 42
                    progress = int(8 + raw_progress)
                    progress = min(50, progress)
                    progress_callback(progress)

                # 1. 运动强度分析（光流法）
                motion_score = self._calculate_motion_intensity(frame, prev_frame)
                motion_scores.append(motion_score)

                # 2. 运动突变检测（射门/过人特征：运动强度骤然升高）
                # 若当前帧运动比上一采样帧高出 0.3 以上，视为突变事件
                delta = max(0.0, motion_score - last_motion)
                motion_delta_scores.append(delta)
                last_motion = motion_score

                # 3. 人群密度分析（YOLO）
                density_score = self._calculate_crowd_density(frame)
                density_scores.append(density_score)

                prev_frame = frame

            cap.release()
            
            print(f"[HIGHLIGHT] 运动分析完成，开始识别精彩时刻...")
            
            # 识别精彩时刻（基于运动和密度评分）
            highlights = self._identify_highlights(
                motion_scores, 
                density_scores,
                motion_delta_scores,
                fps, 
                duration
            )
            
            print(f"[HIGHLIGHT] 识别到 {len(highlights)} 个精彩时刻")
            
            return highlights
            
        except Exception as e:
            print(f"[HIGHLIGHT ERROR] 视频分析过程出错: {e}")
            import traceback
            traceback.print_exc()
            return []  # 返回空列表而不是崩溃
    
    def _calculate_motion_intensity(self, frame, prev_frame) -> float:
        """
        计算相邻两帧的运动强度（光流法）

        原理：
          使用 Farneback 稠密光流算法，对灰度图计算每个像素的运动矢量 (Vx, Vy)，
          取各像素运动幅度 magnitude = sqrt(Vx²+Vy²) 的均值，
          再归一化到 [0, 1]（除以经验阈值10.0）。

        参数说明（Farneback）：
          pyr_scale=0.5  图像金字塔缩放比，0.5表示每层缩小一半
          levels=3       金字塔层数，越大捕捉大范围运动
          winsize=15     平滑窗口大小，越大越鲁棒但越模糊
          iterations=3   每层迭代次数
          poly_n=5       多项式拟合邻域大小
          poly_sigma=1.2 高斯标准差

        Returns:
            float in [0, 1]，0表示无运动，1表示极高运动强度
        """
        if prev_frame is None:
            return 0.0
        
        try:
            # 转换为灰度图
            gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 计算光流（Farneback 方法）
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # 计算运动幅度
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_intensity = np.mean(magnitude)
            
            # 归一化到 0-1
            return min(1.0, motion_intensity / 10.0)
        
        except Exception as e:
            return 0.0
    
    def _calculate_crowd_density(self, frame) -> float:
        """
        估计画面中的人群密度

        原理：
          使用 YOLOv8 对当前帧进行目标检测，统计类别 class=0（person）的边界框数量。
          归一化公式：density = min(1.0, person_count / 10.0)
          即10人及以上为满分1.0，视为高密度场景（进球庆祝、角球人墙等）。

        无模型时：返回中性值 0.5，不影响整体评分趋势。

        Returns:
            float in [0, 1]，0表示无人，1表示高密度人群
        """
        if self.yolo_model is None:
            return 0.5  # 无模型时返回中等密度
        
        try:
            # YOLO 检测
            results = self.yolo_model(frame, verbose=False)
            
            # 统计人数（类别 0 = person）
            person_count = 0
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls == 0:  # person
                        person_count += 1
            
            # 归一化密度分数（10人以上 = 高密度）
            return min(1.0, person_count / 10.0)
        
        except Exception as e:
            return 0.5
    
    def _identify_highlights(
        self,
        motion_scores: List[float],
        density_scores: List[float],
        motion_delta_scores: List[float],
        fps: float,
        duration: float
    ) -> List[HighlightMoment]:
        """
        基于运动、密度、运动突变评分序列识别精彩时刻

        完整算法流程：
          Step 1: 多特征融合（含运动突变）
            combined = motion×0.5 + density×0.3 + delta×0.2
            突变权重用于捕捉射门/过人等瞬间高速动作

          Step 2: 时间序列平滑（窗口15，保留瞬间特征）

          Step 3: 双轨峰值检测
            A. 高分峰值（阈值0.45）：持续精彩动作
            B. 突变峰值（delta>0.35）：射门/过人瞬间

          Step 4: 合并去重（任意两峰值间距 ≥ 5秒）

          Step 5: 差异化片段窗口
            - goal/shot：前3秒+后5秒（含起脚到庆祝）
            - dribble/过人：前2秒+后4秒
            - 其他：前2秒+后3秒
        """

        n = len(motion_scores)
        if n == 0:
            return []

        # 补齐 delta 长度
        delta_scores = motion_delta_scores if len(motion_delta_scores) == n else [0.0] * n

        # Step 1: 多特征融合
        combined_scores = [
            motion_scores[i] * 0.5 + density_scores[i] * 0.3 + delta_scores[i] * 0.2
            for i in range(n)
        ]

        # Step 2: 平滑（窗口15）
        smoothed_scores = self._smooth_scores(combined_scores, window_size=15)

        # Step 3A: 常规峰值（持续高运动/高密度）
        SAMPLE_INTERVAL = 10  # 与采集时保持一致
        min_dist_regular = max(10, int(fps * 3 / SAMPLE_INTERVAL))
        peaks_regular = self._find_peaks(
            smoothed_scores, threshold=0.45, min_distance=min_dist_regular
        )

        # Step 3B: 突变峰值（射门/过人：delta骤升）
        smoothed_delta = self._smooth_scores(delta_scores, window_size=5)
        min_dist_delta = max(8, int(fps * 2 / SAMPLE_INTERVAL))
        peaks_delta = self._find_peaks(
            smoothed_delta, threshold=0.35, min_distance=min_dist_delta
        )

        # Step 4: 合并去重（间距 ≥ 5秒对应采样帧数）
        MIN_GAP = max(15, int(fps * 5 / SAMPLE_INTERVAL))
        all_peaks = sorted(set(peaks_regular + peaks_delta))
        merged_peaks = []
        last_peak = -999
        for pk in all_peaks:
            if pk - last_peak >= MIN_GAP:
                merged_peaks.append(pk)
                last_peak = pk

        # Step 5: 生成精彩时刻（差异化窗口）
        highlights = []
        for peak_idx in merged_peaks:
            peak_time = (peak_idx * SAMPLE_INTERVAL) / fps

            moment_type = self._classify_moment(
                peak_idx, motion_scores, density_scores, delta_scores
            )

            if moment_type in ('goal', 'shot'):
                pre, post = 3.0, 5.0   # 8秒：起脚→入网→庆祝
            elif moment_type == 'dribble':
                pre, post = 2.0, 4.0   # 6秒：接球→过人→传球
            else:
                pre, post = 2.0, 3.0   # 5秒：通用

            start_time = max(0, peak_time - pre)
            end_time   = min(duration, peak_time + post)
            score = smoothed_scores[peak_idx]

            highlight = HighlightMoment(
                start_time=start_time,
                end_time=end_time,
                moment_type=moment_type,
                score=score,
                description=self._generate_description(moment_type, score),
                metadata={
                    'peak_time': peak_time,
                    'motion_score': motion_scores[peak_idx] if peak_idx < n else 0,
                    'density_score': density_scores[peak_idx] if peak_idx < n else 0,
                    'delta_score': delta_scores[peak_idx] if peak_idx < n else 0,
                }
            )
            highlights.append(highlight)

        # 按分数降序排列
        highlights.sort(key=lambda h: h.score, reverse=True)

        return highlights
    
    def _smooth_scores(self, scores: List[float], window_size: int = 30) -> List[float]:
        """平滑分数曲线"""
        if len(scores) < window_size:
            return scores
        
        smoothed = []
        for i in range(len(scores)):
            start = max(0, i - window_size // 2)
            end = min(len(scores), i + window_size // 2)
            smoothed.append(np.mean(scores[start:end]))
        
        return smoothed
    
    def _find_peaks(
        self, 
        scores: List[float], 
        threshold: float = 0.5, 
        min_distance: int = 150
    ) -> List[int]:
        """找到局部峰值"""
        peaks = []
        
        for i in range(1, len(scores) - 1):
            if scores[i] > threshold:
                # 检查是否是局部最大值
                if scores[i] > scores[i-1] and scores[i] > scores[i+1]:
                    # 检查与之前的峰值距离
                    if not peaks or (i - peaks[-1]) > min_distance:
                        peaks.append(i)
        
        return peaks
    
    def _classify_moment(
        self,
        peak_idx: int,
        motion_scores: List[float],
        density_scores: List[float],
        delta_scores: List[float] = None
    ) -> str:
        """分类精彩时刻类型"""

        motion  = motion_scores[peak_idx]  if peak_idx < len(motion_scores)  else 0
        density = density_scores[peak_idx] if peak_idx < len(density_scores) else 0
        delta   = (delta_scores[peak_idx]  if delta_scores and peak_idx < len(delta_scores) else 0)

        # 分类逻辑（优先级从高到低）
        # ── 进球：高运动 + 高密度（起脚→入网→球员聚集庆祝）────────────────
        if motion > 0.7 and density > 0.6:
            return 'goal'
        # ── 射门：高运动 + 突然加速（起脚瞬间）────────────────────────────
        elif motion > 0.65 and delta > 0.3:
            return 'shot'
        # ── 抢球/铲球：高 delta + 中等密度（两人对抗碰撞瞬间）──────────────
        elif delta > 0.35 and 0.3 <= density <= 0.65 and motion > 0.5:
            return 'tackle'
        # ── 过人：强突变 + 低密度（1v1 快速突破）───────────────────────────
        elif delta > 0.4 and density < 0.5:
            return 'dribble'
        # ── 仅高运动（带球推进/远射）────────────────────────────────────────
        elif motion > 0.7:
            return 'shot'
        # ── 高密度（任意球人墙/角球）────────────────────────────────────────
        elif density > 0.7:
            return 'celebration'
        # ── 中等运动（配合传球）─────────────────────────────────────────────
        elif motion > 0.5:
            return 'pass'
        else:
            return 'action'
    
    def _generate_description(self, moment_type: str, score: float) -> str:
        """生成时刻描述"""
        descriptions = {
            'goal': '⚽ 进球时刻',
            'shot': '🎯 精彩射门',
            'celebration': '🎉 激情庆祝',
            'pass': '⚡ 精妙传球',
            'dribble': '🏃 精彩过人',
            'save': '🧤 神扑救险',
            'tackle': '💪 关键拦截',
            'action': '✨ 精彩瞬间'
        }
        
        intensity = '精彩' if score > 0.7 else '不错'
        return f"{descriptions.get(moment_type, '✨ 精彩瞬间')} ({intensity})"


# ============================================================================
#  视频特效处理器
# ============================================================================

class VideoEffectsProcessor:
    """视频特效处理器"""
    
    def __init__(self, output_width: int = 1920, output_height: int = 1080):
        self.output_width = output_width
        self.output_height = output_height
    
    def apply_slowmotion(
        self,
        video_path: str,
        output_path: str,
        start_time: float,
        end_time: float,
        speed_factor: float = 0.5
    ) -> bool:
        """
        应用慢动作效果（视频减速，音频同步拉伸）。

        参数:
            speed_factor: 速度因子，0.5 = 一半速度（2倍慢放），0.35 = 约3倍慢放
        注意：
            慢动作片段的音频在最终成品中会被 BGM 替换，
            此处仍保留音频轨道结构，便于 ffmpeg concat 时对齐。
        """
        try:
            pts_factor = 1.0 / speed_factor
            atempo_chain = _build_atempo_chain(speed_factor)

            cmd = [
                FFMPEG_EXE, '-y',
                '-i', video_path,
                '-ss', str(start_time),
                '-to', str(end_time),
                '-filter_complex',
                f'[0:v]setpts={pts_factor:.4f}*PTS[vout];'
                f'[0:a]{atempo_chain}[aout]',
                '-map', '[vout]',
                '-map', '[aout]',
                '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
                '-c:a', 'aac',
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if result.returncode == 0:
                return True
            # 降级：若音频流不存在，则静音输出
            cmd_fallback = [
                FFMPEG_EXE, '-y',
                '-i', video_path,
                '-ss', str(start_time),
                '-to', str(end_time),
                '-filter:v', f'setpts={pts_factor:.4f}*PTS',
                '-an',
                output_path
            ]
            result2 = subprocess.run(cmd_fallback, capture_output=True, text=True, timeout=120)
            return result2.returncode == 0

        except subprocess.TimeoutExpired:
            print(f"[ERROR] 慢动作处理超时")
            return False
        except Exception as e:
            print(f"[ERROR] 慢动作处理失败: {e}")
            return False

    def add_background_music(
        self,
        video_path: str,
        output_path: str,
        bgm_path: str,
        bgm_volume: float = 0.85,
        video_volume: float = 0.0
    ) -> bool:
        """
        为视频去除原音频并混入激昂背景音乐。

        参数:
            bgm_path:     背景音乐文件路径（WAV/MP3）
            bgm_volume:   BGM 音量系数（0~1）
            video_volume: 原视频音量（0.0 = 完全静音）
        """
        try:
            # 尝试 ffprobe 获取视频时长
            ffprobe_exe = FFMPEG_EXE.replace('ffmpeg', 'ffprobe')
            if not os.path.exists(ffprobe_exe):
                ffprobe_exe = 'ffprobe'
            try:
                r = subprocess.run(
                    [ffprobe_exe, '-v', 'error', '-show_entries', 'format=duration',
                     '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
                    capture_output=True, text=True, timeout=30
                )
                video_dur = float(r.stdout.strip()) if r.returncode == 0 else 300.0
            except Exception:
                video_dur = 300.0

            print(f"    🎵 混入背景音乐（视频时长: {video_dur:.1f}s, BGM音量: {bgm_volume}）")

            filter_complex = (
                f'[1:a]aloop=loop=-1:size=2e+09,atrim=duration={video_dur:.3f},'
                f'volume={bgm_volume:.2f}[bgm];'
            )
            if video_volume > 0:
                filter_complex += (
                    f'[0:a]volume={video_volume:.2f}[orig];'
                    f'[orig][bgm]amix=inputs=2:duration=first[aout]'
                )
            else:
                filter_complex += f'[bgm]acopy[aout]'
            audio_map = '[aout]'

            cmd = [
                FFMPEG_EXE, '-y',
                '-i', video_path,
                '-i', bgm_path,
                '-filter_complex', filter_complex,
                '-map', '0:v',
                '-map', audio_map,
                '-c:v', 'copy',
                '-c:a', 'aac', '-b:a', '192k',
                '-shortest',
                output_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                print(f"    ✅ 背景音乐混音完成")
                return True
            else:
                print(f"    [WARN] BGM 混音失败: {result.stderr[-300:]}")
                return False

        except subprocess.TimeoutExpired:
            print(f"    [ERROR] BGM 混音超时")
            return False
        except Exception as e:
            print(f"    [ERROR] BGM 混音异常: {e}")
            return False
    
    def apply_zoom_effect(
        self,
        video_path: str,
        output_path: str,
        start_time: float,
        end_time: float,
        zoom_factor: float = 1.5
    ) -> bool:
        """应用特写镜头效果（数字变焦）"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (self.output_width, self.output_height))
            
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            for frame_idx in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 应用数字变焦
                zoomed = self._zoom_frame(frame, zoom_factor)
                out.write(zoomed)
            
            cap.release()
            out.release()
            return True
        
        except Exception as e:
            print(f"[ERROR] 特写处理失败: {e}")
            return False
    
    def _zoom_frame(self, frame: np.ndarray, zoom_factor: float) -> np.ndarray:
        """对单帧应用数字变焦"""
        h, w = frame.shape[:2]
        
        # 计算裁剪区域（中心区域）
        new_w = int(w / zoom_factor)
        new_h = int(h / zoom_factor)
        
        x = (w - new_w) // 2
        y = (h - new_h) // 2
        
        # 裁剪并缩放回原始尺寸
        cropped = frame[y:y+new_h, x:x+new_w]
        zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)
        
        return zoomed
    
    def create_picture_in_picture(
        self,
        main_video: str,
        pip_video: str,
        output_path: str,
        pip_position: str = 'top-right',  # top-left, top-right, bottom-left, bottom-right
        pip_scale: float = 0.3
    ) -> bool:
        """
        创建画中画效果
        
        用于进球时刻：主画面显示进球，小窗显示慢动作回放
        """
        try:
            # 使用 ffmpeg 实现画中画
            # 计算小窗位置
            positions = {
                'top-right': 'main_w-overlay_w-10:10',
                'top-left': '10:10',
                'bottom-right': 'main_w-overlay_w-10:main_h-overlay_h-10',
                'bottom-left': '10:main_h-overlay_h-10'
            }
            
            pos = positions.get(pip_position, positions['top-right'])
            
            cmd = [
                FFMPEG_EXE, '-y',
                '-i', main_video,
                '-i', pip_video,
                '-filter_complex',
                f'[1:v]scale=iw*{pip_scale}:ih*{pip_scale}[pip];'
                f'[0:v][pip]overlay={pos}',
                '-c:a', 'copy',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            return result.returncode == 0
        
        except subprocess.TimeoutExpired:
            print(f"[ERROR] 画中画处理超时")
            return False
        except Exception as e:
            print(f"[ERROR] 画中画处理失败: {e}")
            return False


# ============================================================================
#  精彩视频生成器（主控制器）
# ============================================================================

class HighlightVideoGenerator:
    """精彩视频生成器 - 主控制器"""
    
    def __init__(self, yolo_model=None):
        self.detector = HighlightDetector(yolo_model)
        self.effects = VideoEffectsProcessor()
        self.temp_dir = 'temp_highlights'
        
        # 创建临时目录
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def generate_highlight_video(
        self,
        input_video: str,
        output_video: str,
        target_duration: float = 180.0,  # 目标3分钟
        enable_slowmo: bool = True,
        enable_zoom: bool = True,
        enable_pip: bool = True,
        enable_bgm: bool = True,         # 启用背景音乐（去除原音频）
        bgm_volume: float = 0.88,        # BGM 音量
        progress_callback=None
    ) -> Dict:
        """
        生成精彩视频
        
        返回:
            {
                'success': bool,
                'output_path': str,
                'duration': float,
                'highlight_count': int,
                'moments': List[HighlightMoment]
            }
        """
        
        print("\n" + "="*70)
        print("  🎬 精彩视频生成开始")
        print("="*70)
        
        try:
            # 步骤1: 检测精彩时刻 (0-50%)
            print("\n[1/5] 🔍 分析视频，识别精彩时刻...")
            highlights = self.detector.detect_highlights(
                input_video, 
                progress_callback=progress_callback
            )
            
            if not highlights:
                print("[HIGHLIGHT] ⚠️ 未检测到精彩时刻，尝试均匀采样降级处理...")
                # 降级：均匀抽取几个片段凑满目标时长
                import random
                cap_tmp = cv2.VideoCapture(input_video)
                total_sec = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap_tmp.get(cv2.CAP_PROP_FPS), 1))
                cap_tmp.release()
                if total_sec < 10:
                    return {
                        'success': False,
                        'error': '视频过短，无法生成精彩视频（至少需要10秒）',
                        'highlight_count': 0
                    }
                clip_len = 8  # 每段8秒
                num_clips = max(1, int(target_duration / clip_len))
                step = max(1, (total_sec - clip_len) // num_clips)
                highlights = [
                    HighlightMoment(
                        start_time=i * step,
                        end_time=min(i * step + clip_len, total_sec),
                        moment_type='action',
                        score=0.5,
                        description='自动采样片段',
                        metadata={}
                    )
                    for i in range(num_clips)
                ]
                print(f"[HIGHLIGHT] 降级生成 {len(highlights)} 个均匀采样片段")
            
            if progress_callback:
                progress_callback(50)
            
            # 步骤2: 选择最佳片段 (50-60%)
            print(f"\n[2/5] ✂️ 从 {len(highlights)} 个时刻中选择精彩片段...")
            selected_clips = self._select_best_clips(highlights, target_duration)
            print(f"[INFO] 选中 {len(selected_clips)} 个片段，总时长约 {sum([c.end_time - c.start_time for c in selected_clips]):.1f}秒")
            
            if progress_callback:
                progress_callback(60)
            
            # 步骤3: 应用特效 (60-80%)
            print("\n[3/5] ✨ 应用特效（慢动作、特写、画中画）...")
            processed_clips = self._process_clips_with_effects(
                input_video,
                selected_clips,
                enable_slowmo,
                enable_zoom,
                enable_pip,
                progress_callback
            )
            
            if progress_callback:
                progress_callback(80)
            
            # 步骤4: 合并视频 (80-90%)
            print("\n[4/5] 🎞️ 合并片段，生成最终视频...")
            # 先合并到临时文件，再做 BGM 混音
            merged_tmp = output_video.replace('.mp4', '_merged_tmp.mp4')
            success = self._merge_clips(processed_clips, merged_tmp)

            if progress_callback:
                progress_callback(90)

            # 步骤5: 背景音乐混音 (90-100%)
            if success and enable_bgm:
                print("\n[5/5] 🎵 生成激昂背景音乐并混入视频（去除原音频）...")
                bgm_wav = None
                try:
                    bgm_wav = _generate_bgm_wav(duration=target_duration + 30)
                    bgm_ok  = self.effects.add_background_music(
                        merged_tmp, output_video,
                        bgm_path=bgm_wav,
                        bgm_volume=bgm_volume,
                        video_volume=0.0    # 完全去除原视频音频
                    )
                    if not bgm_ok:
                        print("[WARN] BGM 混音失败，将使用无 BGM 版本")
                        import shutil
                        shutil.copy2(merged_tmp, output_video)
                except Exception as bgm_err:
                    print(f"[WARN] BGM 生成/混音异常: {bgm_err}，使用无 BGM 版本")
                    import shutil
                    shutil.copy2(merged_tmp, output_video)
                finally:
                    # 清理临时 WAV 和中间 MP4
                    if bgm_wav and os.path.exists(bgm_wav):
                        try: os.remove(bgm_wav)
                        except: pass
                    if os.path.exists(merged_tmp):
                        try: os.remove(merged_tmp)
                        except: pass
            elif success:
                # BGM 关闭，直接重命名
                import shutil
                shutil.move(merged_tmp, output_video)

            if progress_callback:
                progress_callback(100)

            if success:
                # 获取输出视频信息
                cap = cv2.VideoCapture(output_video)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                final_duration = frame_count / fps
                cap.release()
                
                print("\n" + "="*70)
                print("  ✅ 精彩视频生成完成！")
                print("="*70)
                print(f"  输出文件: {output_video}")
                print(f"  视频时长: {final_duration:.1f} 秒")
                print(f"  精彩片段: {len(selected_clips)} 个")
                print("="*70 + "\n")
                
                return {
                    'success': True,
                    'output_path': output_video,
                    'duration': final_duration,
                    'highlight_count': len(selected_clips),
                    'moments': [
                        {
                            'type': clip.moment_type,
                            'start': clip.start_time,
                            'end': clip.end_time,
                            'score': clip.score
                        }
                        for clip in selected_clips
                    ]
                }
            else:
                return {
                    'success': False,
                    'error': '视频合并失败'
                }
        
        except Exception as e:
            print(f"\n[ERROR] 精彩视频生成失败: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e)
            }
        
        finally:
            # 清理临时文件
            self._cleanup_temp_files()
    
    def _select_best_clips(
        self, 
        highlights: List[HighlightMoment], 
        target_duration: float
    ) -> List[VideoClip]:
        """
        选择最佳精彩片段
        
        策略：
        1. 优先选择高分片段
        2. 进球时刻必选
        3. 确保总时长接近目标时长
        """
        
        # 各类型精彩度权重与慢动作速度因子配置
        # ─────────────────────────────────────────────────────────────────────
        #  类型         慢动作速度   说明
        #  goal         0.40×       进球：约2.5倍慢放，完整呈现起脚→入网→庆祝
        #  shot         0.35×       射门：约2.9倍慢放，突出起脚瞬间力量感
        #  tackle       0.40×       抢球成功：约2.5倍慢放，凸显拼抢激烈
        #  dribble      0.45×       过人：约2.2倍慢放，展示盘带技术
        #  pass/action  0.55×       其他：约1.8倍慢放，自然流畅
        # ─────────────────────────────────────────────────────────────────────
        SLOWMO_CONFIG = {
            'goal':        {'factor': 0.40, 'zoom': True,  'pip': True},
            'shot':        {'factor': 0.35, 'zoom': True,  'pip': False},
            'tackle':      {'factor': 0.40, 'zoom': True,  'pip': False},
            'dribble':     {'factor': 0.45, 'zoom': False, 'pip': False},
            'celebration': {'factor': 0.55, 'zoom': False, 'pip': False},
            'pass':        {'factor': 0.55, 'zoom': False, 'pip': False},
            'action':      {'factor': 0.55, 'zoom': False, 'pip': False},
            'save':        {'factor': 0.40, 'zoom': True,  'pip': False},
        }

        # 按类型和分数排序
        goal_moments    = [h for h in highlights if h.moment_type == 'goal']
        shot_moments    = [h for h in highlights if h.moment_type == 'shot']
        tackle_moments  = [h for h in highlights if h.moment_type == 'tackle']
        dribble_moments = [h for h in highlights if h.moment_type == 'dribble']
        other_moments   = [h for h in highlights if h.moment_type not in ('goal', 'shot', 'tackle', 'dribble')]

        # 优先级：进球(≤5) → 射门(≤5) → 抢球(≤4) → 过人(≤4)，再按分数填满目标时长
        selected = goal_moments[:5] + shot_moments[:5] + tackle_moments[:4] + dribble_moments[:4]

        # 计算已选时长
        current_duration = sum([m.end_time - m.start_time for m in selected])

        # 添加其他精彩时刻直到达到目标时长
        for moment in other_moments:
            clip_duration = moment.end_time - moment.start_time

            if current_duration + clip_duration <= target_duration * 1.1:  # 允许超10%
                selected.append(moment)
                current_duration += clip_duration

            if current_duration >= target_duration:
                break

        # 转换为 VideoClip，根据类型分配精确慢动作速度因子
        clips = []
        for i, moment in enumerate(selected):
            cfg = SLOWMO_CONFIG.get(moment.moment_type, SLOWMO_CONFIG['action'])
            clip = VideoClip(
                start_frame=0,
                end_frame=0,
                start_time=moment.start_time,
                end_time=moment.end_time,
                moment_type=moment.moment_type,
                score=moment.score,
                apply_slowmo=True,               # 所有精彩片段均应用慢动作
                slowmo_factor=cfg['factor'],
                apply_zoom=cfg['zoom']
            )
            clips.append(clip)
        
        # 按时间排序
        clips.sort(key=lambda c: c.start_time)
        
        return clips
    
    def _process_clips_with_effects(
        self,
        input_video: str,
        clips: List[VideoClip],
        enable_slowmo: bool,
        enable_zoom: bool,
        enable_pip: bool,
        progress_callback
    ) -> List[str]:
        """处理片段并应用特效"""
        
        processed_files = []
        
        for i, clip in enumerate(clips):
            print(f"\n  处理片段 {i+1}/{len(clips)}: {clip.moment_type} ({clip.start_time:.1f}s - {clip.end_time:.1f}s)")
            
            # 基础片段路径
            clip_path = os.path.join(self.temp_dir, f'clip_{i:03d}.mp4')
            
            # 提取片段
            extract_ok = self._extract_clip(input_video, clip_path, clip.start_time, clip.end_time)
            if not extract_ok or not os.path.exists(clip_path) or os.path.getsize(clip_path) == 0:
                print(f"  [SKIP] 片段 {i+1} 提取失败，跳过该片段")
                if progress_callback:
                    progress = 60 + int((i + 1) / len(clips) * 20)
                    progress_callback(progress)
                continue

            current_file = clip_path
            
            # 应用慢动作（如果需要）
            if enable_slowmo and clip.apply_slowmo:
                print(f"    ⏱️ 应用慢动作效果 ({clip.slowmo_factor}x)")
                slowmo_path = os.path.join(self.temp_dir, f'clip_{i:03d}_slowmo.mp4')
                
                if self.effects.apply_slowmotion(
                    current_file, slowmo_path,
                    0, clip.end_time - clip.start_time,
                    clip.slowmo_factor
                ):
                    current_file = slowmo_path
            
            # 应用特写（如果需要）
            if enable_zoom and clip.apply_zoom:
                print(f"    🔍 应用特写镜头")
                zoom_path = os.path.join(self.temp_dir, f'clip_{i:03d}_zoom.mp4')
                
                if self.effects.apply_zoom_effect(
                    current_file, zoom_path,
                    0, clip.end_time - clip.start_time,
                    zoom_factor=1.3
                ):
                    current_file = zoom_path
            
            # 进球/射门：创建画中画（超慢动作回放小窗）
            pip_types = ['goal', 'shot', 'save']
            if enable_pip and clip.moment_type in pip_types:
                print(f"    📺 创建画中画效果（超慢动作回放）")
                pip_path = os.path.join(self.temp_dir, f'clip_{i:03d}_pip.mp4')

                # 画中画小窗：比主画面更慢（0.3×），增强戏剧性
                pip_slowmo_factor = min(0.3, clip.slowmo_factor * 0.7)
                slowmo_for_pip = os.path.join(self.temp_dir, f'clip_{i:03d}_pip_slowmo.mp4')
                self.effects.apply_slowmotion(
                    clip_path, slowmo_for_pip,
                    0, clip.end_time - clip.start_time,
                    speed_factor=pip_slowmo_factor
                )

                # 合并为画中画
                if self.effects.create_picture_in_picture(
                    clip_path, slowmo_for_pip, pip_path,
                    pip_position='bottom-right', pip_scale=0.25
                ):
                    current_file = pip_path
            
            processed_files.append(current_file)
            
            # 更新进度
            if progress_callback:
                progress = 60 + int((i + 1) / len(clips) * 20)
                progress_callback(progress)
        
        return processed_files
    
    def _extract_clip(
        self,
        input_video: str,
        output_path: str,
        start_time: float,
        end_time: float
    ) -> bool:
        """
        提取视频片段。
        优先使用"输入前置 -ss + stream copy"（快速精确），若输出文件为空则
        自动降级为"重新编码"模式，彻底解决关键帧对齐导致空文件的问题。
        """
        duration = end_time - start_time
        if duration <= 0:
            print(f"[WARN] _extract_clip: 无效时间范围 {start_time:.1f}→{end_time:.1f}")
            return False

        def _run(cmd: list) -> bool:
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                return r.returncode == 0
            except subprocess.TimeoutExpired:
                print(f"[ERROR] ffmpeg 提取超时（>120s）")
                return False
            except Exception as exc:
                print(f"[ERROR] 提取片段失败: {exc}")
                return False

        # ── 方案A：输入前置 -ss（最快，stream copy）───────────────────────
        cmd_fast = [
            FFMPEG_EXE, '-y',
            '-ss', str(start_time),          # 放在 -i 前：快速 seek 到关键帧
            '-i', input_video,
            '-t', str(duration),             # 用 -t 而非 -to，避免时间戳偏移
            '-c', 'copy',
            '-avoid_negative_ts', 'make_zero',
            output_path
        ]
        if _run(cmd_fast):
            # 确保输出文件有内容（非空）
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
                return True
            print(f"[WARN] stream-copy 输出为空，降级为重新编码...")

        # ── 方案B：重新编码（兼容性最强，速度稍慢）─────────────────────────
        cmd_encode = [
            FFMPEG_EXE, '-y',
            '-ss', str(start_time),
            '-i', input_video,
            '-t', str(duration),
            '-vcodec', 'libx264',
            '-preset', 'ultrafast',          # 极速编码
            '-crf', '23',
            '-acodec', 'aac',
            '-strict', 'experimental',
            output_path
        ]
        if _run(cmd_encode):
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
                return True

        print(f"[ERROR] 无法提取片段 {start_time:.1f}s–{end_time:.1f}s，两种方案均失败")
        return False
    
    def _merge_clips(self, clip_files: List[str], output_path: str) -> bool:
        """合并视频片段"""
        try:
            # 过滤掉不存在或空文件
            valid_files = [
                f for f in clip_files
                if os.path.exists(f) and os.path.getsize(f) > 1024
            ]
            if not valid_files:
                print(f"[ERROR] 没有有效的片段文件可合并")
                return False

            print(f"[INFO] 合并 {len(valid_files)}/{len(clip_files)} 个有效片段")

            # 创建 ffmpeg 输入文件列表
            concat_file = os.path.join(self.temp_dir, 'concat_list.txt')
            with open(concat_file, 'w', encoding='utf-8') as f:
                for clip_file in valid_files:
                    abs_path = os.path.abspath(clip_file)
                    f.write(f"file '{abs_path}'\n")
            
            # 使用 ffmpeg concat 合并
            cmd = [
                FFMPEG_EXE, '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c', 'copy',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"[ERROR] ffmpeg 合并失败: {result.stderr[-500:]}")
                return False
            
            return True
        
        except subprocess.TimeoutExpired:
            print(f"[ERROR] 合并视频超时（>300s）")
            return False
        except Exception as e:
            print(f"[ERROR] 合并视频失败: {e}")
            return False
    
    def _cleanup_temp_files(self):
        """清理临时文件"""
        try:
            if os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir)
                os.makedirs(self.temp_dir, exist_ok=True)
        except Exception as e:
            print(f"[WARNING] 清理临时文件失败: {e}")


# ============================================================================
#  快速测试函数
# ============================================================================

def test_highlight_generation(video_path: str):
    """测试精彩视频生成"""
    
    print("\n🎬 精彩视频生成测试")
    print("="*70)
    
    # 初始化生成器
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("[OK] YOLO 模型加载成功")
    except:
        model = None
        print("[WARNING] YOLO 模型未加载，将使用简化检测")
    
    generator = HighlightVideoGenerator(yolo_model=model)
    
    # 生成精彩视频
    output_path = video_path.replace('.mp4', '_highlights.mp4')
    
    result = generator.generate_highlight_video(
        input_video=video_path,
        output_video=output_path,
        target_duration=180.0,
        enable_slowmo=True,
        enable_zoom=True,
        enable_pip=True,
        progress_callback=lambda p: print(f"  进度: {p}%")
    )
    
    print("\n" + "="*70)
    if result['success']:
        print("✅ 测试成功！")
        print(f"输出文件: {result['output_path']}")
        print(f"视频时长: {result['duration']:.1f} 秒")
        print(f"精彩片段: {result['highlight_count']} 个")
    else:
        print("❌ 测试失败")
        print(f"错误: {result.get('error', '未知错误')}")
    print("="*70 + "\n")
    
    return result


if __name__ == '__main__':
    # 测试代码
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        test_highlight_generation(video_path)
    else:
        print("用法: python highlight_generator.py <video_path>")
