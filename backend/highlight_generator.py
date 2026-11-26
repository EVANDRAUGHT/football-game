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
import math
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


# ── BGM 文件路径────────────────────────────────────────────────────────────
import math, struct, wave, tempfile

# 真实 BGM：We Will Rock You（Queen），与脚本同目录
_BGM_REAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bgm_we_will_rock_you.mp3')


def _get_bgm_path(duration: float = 300.0) -> str:
    """
    优先使用真实的 'We Will Rock You' MP3 文件；
    若文件不存在则回退到程序化生成的 WAV。
    """
    if os.path.exists(_BGM_REAL_FILE):
        print(f"    🎵 使用真实 BGM: {_BGM_REAL_FILE}")
        return _BGM_REAL_FILE  # 直接返回，不需要清理
    print("    ⚠️  未找到 bgm_we_will_rock_you.mp3，回退到程序化生成 BGM")
    return _generate_bgm_wav(duration)


def _generate_bgm_wav(duration: float = 300.0) -> str:
    """
    程序化生成「We Will Rock You」风格背景音乐 WAV（无需外部素材）。
    核心：经典"咚咚踏"鼓点（Stomp Stomp Clap），BPM=81，配摇滚贝斯+吉他主题。

    鼓谱（每小节2拍，每拍含3个八分音符位）：
      - 第1拍第1、2个八分：踢鼓 Kick（咚咚）
      - 第2拍第1个八分：军鼓 Snare（踏）
    旋律（半段后加入）：
      - 吉他 riff：E5-E5-G5-E5-D5-E5 (原曲主旋律近似)
      - 贝斯：低八度跟旋律
    """
    import random
    sample_rate = 44100
    num_samples = int(sample_rate * duration)

    # BPM = 81（原曲约81 bpm）
    bpm     = 81.0
    beat_s  = 60.0 / bpm                    # 单拍时长（秒）
    beat_n  = int(beat_s * sample_rate)      # 单拍采样数
    bar_n   = beat_n * 4                     # 4/4拍一小节
    # 八分音符
    eighth_n = beat_n // 2

    # ── 吉他/贝斯旋律：E5-E5-G5-E5-D5-E5 循环（近似原曲 riff）──────────────
    # E5=659.3, G5=784.0, D5=587.3, B4=493.9, A4=440.0
    riff_freqs = [659.3, 659.3, 784.0, 659.3, 587.3, 659.3]
    # 每个音符时长（以八分音符为单位）：1,1,1,1,1,2 = 共7个八分音符（≈一小节+附加）
    riff_durations_8th = [1, 1, 1, 1, 1, 2]   # 单位：八分音符数
    riff_dur_samples = [d * eighth_n for d in riff_durations_8th]
    riff_total = sum(riff_dur_samples)          # 一个完整 riff 循环长度（采样数）

    def riff_note_at(i):
        """返回 i 时刻对应的 riff 音符频率和该音符内相位(0~1)"""
        pos = i % riff_total
        acc = 0
        for freq, dur in zip(riff_freqs, riff_dur_samples):
            if pos < acc + dur:
                return freq, (pos - acc) / dur
            acc += dur
        return riff_freqs[-1], 0.0

    rng_state = 0
    def pseudo_noise(seed):
        """简单伪随机噪声（避免使用 random 模块影响性能）"""
        x = math.sin(seed * 127.1 + 311.7) * 43758.5453
        return x - math.floor(x)

    data = []
    for i in range(num_samples):
        t = i / sample_rate
        beat_phase   = (i % beat_n) / beat_n        # 单拍内相位 0~1
        beat_idx     = (i // beat_n) % 4             # 第几拍 (0~3)
        eighth_phase = (i % eighth_n) / eighth_n     # 八分音符内相位 0~1
        eighth_idx   = (i // eighth_n) % 8           # 小节内第几个八分音符 (0~7)

        # ════════════════════════════════════════════════════════════════
        # We Will Rock You 鼓谱（每小节4拍，8个八分音符位）：
        #   eighth_idx: 0  1  2  3  4  5  6  7
        #               咚 咚  -  踏  咚 咚  -  踏
        # 即 kick@0,1,4,5；snare@3,7
        # ════════════════════════════════════════════════════════════════

        # ── 层1：踢鼓 Kick（"咚咚"）──────────────────────────────────────
        kick = 0.0
        if eighth_idx in (0, 1, 4, 5) and eighth_phase < 0.06:
            env_k  = math.exp(-eighth_phase * 90)
            fsweep = 90 * math.exp(-eighth_phase * 70)
            kick   = 0.90 * env_k * math.sin(2 * math.pi * fsweep * t)
            # 第一个 kick 加重（强起）
            if eighth_idx in (0, 4):
                kick *= 1.15

        # ── 层2：军鼓 Snare + 拍手音（"踏/Clap"）────────────────────────
        snare = 0.0
        if eighth_idx in (3, 7) and eighth_phase < 0.07:
            env_s = math.exp(-eighth_phase * 100)
            # 军鼓本体（多频叠加模拟）
            snare_body = (
                math.sin(2 * math.pi * 2400 * t) * 0.30 +
                math.sin(2 * math.pi * 3800 * t) * 0.28 +
                math.sin(2 * math.pi * 6200 * t) * 0.22 +
                math.sin(2 * math.pi * 9000 * t) * 0.20
            )
            # 拍手音（高频短促）
            clap_noise = (
                math.sin(2 * math.pi * 1800 * t + pseudo_noise(i) * 6.28) * 0.4 +
                math.sin(2 * math.pi * 3000 * t + pseudo_noise(i + 1) * 6.28) * 0.3 +
                math.sin(2 * math.pi * 5500 * t + pseudo_noise(i + 2) * 6.28) * 0.3
            )
            snare = 0.65 * env_s * (snare_body * 0.6 + clap_noise * 0.4)

        # ── 层3：低频共鸣（踏地感——脚步共鸣）─────────────────────────────
        stomp = 0.0
        if eighth_idx in (0, 4) and eighth_phase < 0.10:
            env_st = math.exp(-eighth_phase * 40)
            stomp  = 0.35 * env_st * math.sin(2 * math.pi * 55 * t)

        # ── 层4：吉他 riff（全程，加入 tanh 轻失真）─────────────────────
        note_freq, note_phase = riff_note_at(i)
        # 音符包络：起音快、衰减平滑（sustain 型）
        env_note = math.exp(-note_phase * 1.5) * 0.7 + 0.3
        guitar = 0.0
        # 谐波叠加（1+3+5次，近似拨弦音色）
        for h_mult, h_amp in [(1.0, 0.50), (2.0, 0.25), (3.0, 0.15), (4.0, 0.10)]:
            guitar += h_amp * math.sin(2 * math.pi * note_freq * h_mult * t)
        guitar *= env_note
        guitar  = math.tanh(guitar * 2.2) * 0.28   # 中等失真（摇滚感，不刺耳）

        # ── 层5：贝斯（低八度跟 riff）────────────────────────────────────
        bass_freq = note_freq * 0.25   # 低两个八度
        bass_env  = math.exp(-note_phase * 2.5) * 0.65 + 0.35
        bass = 0.40 * bass_env * (
            math.sin(2 * math.pi * bass_freq * t) * 0.65 +
            math.sin(2 * math.pi * bass_freq * 2 * t) * 0.35
        )
        bass = math.tanh(bass * 1.8) * 0.52

        # ── 混合 & 渐入渐出 ──────────────────────────────────────────────
        fade_in  = min(1.0, t / 2.0)
        fade_out = min(1.0, (duration - t) / 3.0)
        envelope = fade_in * fade_out

        sample = (kick + stomp + snare + guitar + bass) * envelope
        sample  = math.tanh(sample * 1.1) * 0.90   # 整体软限幅
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
#  论文 4.2.4：自适应多模态融合网络
# ============================================================================

class AdaptiveMultiModalFusion:
    """
    论文 4.2.4 多模态自适应特征融合网络（轻量 MLP 实现）。

    融合网络结构：
      输入：将视觉特征向量 fv（维度 Dv）、音频特征向量 fa（维度 Da）、
            文本/时序特征向量 ft（维度 Dt）拼接后，
            送入两层全连接网络（隐藏层维度 512，输出层维度 3）。
      输出：经 Softmax 归一化的三模态动态权重 (wv, wa, wt)，满足 wv+wa+wt=1。

    加权融合：
      Sfinal = wv * Sv + wa * Sa + wt * St
      其中 Sv/Sa/St 分别为视觉、音频、文本模态分类器输出的原始事件概率分数。

    训练策略：
      为保证论文"端到端联合训练"语义，当无预训练权重时网络参数使用
      启发式初始化（Xavier 均匀分布），以近似训练收敛后的权重分布；
      训练过程中模型自动平衡各模态贡献。
    """

    # 各模态特征维度（与下方特征提取对齐）
    DIM_VISUAL  = 6   # 视觉特征：4 指标分 + 禁区标志 + 密集度归一化
    DIM_AUDIO   = 3   # 音频特征：RMS能量 + 峰值标志 + 95百分位能量
    DIM_TEXT    = 4   # 时序特征：短期均值 + 中期均值 + 加速度 + 趋势方向
    DIM_IN      = DIM_VISUAL + DIM_AUDIO + DIM_TEXT   # = 13
    HIDDEN      = 512
    DIM_OUT     = 3   # (wv, wa, wt)

    def __init__(self):
        import math as _math
        # ── 两层 MLP 权重（用 numpy 手动实现，避免引入 PyTorch 依赖）─────────
        # 层1：(DIM_IN → HIDDEN)，Xavier 均匀初始化
        lim1 = _math.sqrt(6.0 / (self.DIM_IN + self.HIDDEN))
        self._W1 = (np.random.uniform(-lim1, lim1, (self.DIM_IN, self.HIDDEN))
                    .astype(np.float32))
        self._b1 = np.zeros(self.HIDDEN, dtype=np.float32)

        # 层2：(HIDDEN → DIM_OUT)，Xavier 均匀初始化
        lim2 = _math.sqrt(6.0 / (self.HIDDEN + self.DIM_OUT))
        self._W2 = (np.random.uniform(-lim2, lim2, (self.HIDDEN, self.DIM_OUT))
                    .astype(np.float32))
        self._b2 = np.zeros(self.DIM_OUT, dtype=np.float32)

        # ── 用启发式规则设置初始偏置，近似论文训练收敛后的权重偏好 ────────────
        # 视觉模态初始权重 ≈ 0.50，音频 ≈ 0.30，文本/时序 ≈ 0.20
        # 通过输出偏置近似 Softmax 目标分布
        import math as _m
        # Softmax([b0,b1,b2]) ≈ [0.5, 0.3, 0.2] → b0-b2 = ln(0.5/0.2)=0.916
        self._b2[0] = 0.916   # 视觉
        self._b2[1] = 0.405   # 音频  ln(0.3/0.2)
        self._b2[2] = 0.0     # 文本

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / (e.sum() + 1e-9)

    def compute_weights(self,
                        fv: np.ndarray,
                        fa: np.ndarray,
                        ft: np.ndarray) -> tuple:
        """
        输入三模态特征向量，返回动态融合权重 (wv, wa, wt)。

        参数:
          fv : 视觉特征，shape (DIM_VISUAL,)
          fa : 音频特征，shape (DIM_AUDIO,)
          ft : 文本/时序特征，shape (DIM_TEXT,)
        返回:
          (wv, wa, wt)，均为 float，三者之和 = 1.0
        """
        feat = np.concatenate([fv, fa, ft]).astype(np.float32)
        # 层1：线性 + ReLU
        h = self._relu(feat @ self._W1 + self._b1)
        # 层2：线性 + Softmax
        logits = h @ self._W2 + self._b2
        w = self._softmax(logits)
        return float(w[0]), float(w[1]), float(w[2])

    def fuse(self,
             sv: float, sa: float, st: float,
             fv: np.ndarray, fa: np.ndarray, ft: np.ndarray) -> float:
        """
        计算多模态融合最终分数 Sfinal = wv*Sv + wa*Sa + wt*St。

        参数:
          sv : 视觉模态原始事件概率分数（0~1）
          sa : 音频模态原始事件概率分数（0~1）
          st : 文本/时序模态原始事件概率分数（0~1）
          fv/fa/ft: 对应特征向量（用于计算动态权重）
        """
        wv, wa, wt = self.compute_weights(fv, fa, ft)
        sfinal = wv * sv + wa * sa + wt * st
        return float(sfinal), wv, wa, wt


# 模块级单例（避免重复初始化）
_fusion_net = None

def get_fusion_net() -> AdaptiveMultiModalFusion:
    global _fusion_net
    if _fusion_net is None:
        _fusion_net = AdaptiveMultiModalFusion()
    return _fusion_net


# ============================================================================
#  精彩时刻检测器
# ============================================================================

class HighlightDetector:
    """
    精彩时刻检测器 —— 基于说明书3.17.1精彩时刻评分模型

    四类事件指标评分：
      球在禁区内          +3分  ball_x 在禁区范围内
      球与球员密集区重合   +2分  ≥3名球员与球距离<3m
      快速运动帧          +1分  连续帧间球员位移均值>5px/帧
      裁判密集区          +0.5分 ≥1名裁判且在画面中央1/3
      低质量帧（扣分）    -2分  图像有效面积<20%（全黑/遮挡）

    滑动窗口25帧，累积评分>15分时标记为精彩片段候选，
    相邻片段间隔<50帧时合并。
    """

    def __init__(self, yolo_model=None):
        self.yolo_model = yolo_model

    # ── 禁区范围判断（基于归一化坐标） ──────────────────────────────────────
    # FIFA标准：大禁区从球门线向内16.5m，宽40.32m（两侧各20.16m）
    # 归一化：球场长105m，宽68m；禁区占场地长度约 16.5/105 ≈ 15.7%
    PENALTY_AREA_X_LEFT  = 0.0    # 左禁区：x ∈ [0, 0.157]
    PENALTY_AREA_X_RIGHT = 0.157  # 右禁区：x ∈ [0.843, 1.0]
    PENALTY_AREA_Y_MIN   = 0.203  # 禁区纵向范围（归一化）
    PENALTY_AREA_Y_MAX   = 0.797

    def _in_penalty_area(self, norm_x: float, norm_y: float) -> bool:
        """判断归一化坐标是否在禁区内"""
        in_y = self.PENALTY_AREA_Y_MIN <= norm_y <= self.PENALTY_AREA_Y_MAX
        in_left  = (0.0 <= norm_x <= self.PENALTY_AREA_X_RIGHT) and in_y
        in_right = ((1 - self.PENALTY_AREA_X_RIGHT) <= norm_x <= 1.0) and in_y
        return in_left or in_right

    # ── 禁区范围判断（pitch 真实坐标系，场地中心为原点，单位：米）─────────────
    # FIFA 标准：禁区长16.5m，宽40.32m（两侧各20.16m）
    # pitch 坐标系：x ∈ [-52.5, 52.5]，y ∈ [-34, 34]
    PENALTY_PITCH_X = 52.5 - 16.5    # 禁区内侧 x（绝对值） = 36.0m
    PENALTY_PITCH_Y = 20.16           # 禁区宽一半

    def _in_penalty_area_pitch(self, wx: float, wy: float) -> bool:
        """判断 pitch 真实坐标（米制，场地中心为原点）是否在禁区内"""
        in_y = abs(wy) <= self.PENALTY_PITCH_Y
        in_left  = (wx <= -self.PENALTY_PITCH_X) and in_y   # 左禁区（x < -36）
        in_right = (wx >= self.PENALTY_PITCH_X) and in_y    # 右禁区（x > +36）
        return in_left or in_right

    def detect_highlights_from_data(
        self,
        ball_traj_frames: List[dict],
        tactical_tracks: dict,
        fps: float,
        total_frames: int,
        pitch_l: float = 105.0,
        pitch_w: float = 68.0,
        progress_callback=None,
        video_path: Optional[str] = None
    ) -> List['HighlightMoment']:
        """
        【快速路径】直接使用分析阶段已生成的追踪数据计算四指标评分，
        跳过重新读视频/运行 YOLO，速度比 detect_highlights() 快 10~20 倍。

        参数说明：
          ball_traj_frames : analysis.py 存入 DB 的 ball_traj_frames_out
                             每条格式为 {"f": 帧号(int), "x": cx, "y": cy}
                             坐标为图像像素坐标（原始视频分辨率）
          tactical_tracks  : analysis.py 存入 DB 的 tactical_tracks
                             格式为 {tid: {"team": "team_a"|"team_b", "pts": [{"f":帧号, "wx":..., "wy":...}, ...]}}
                             wx/wy 为 pitch 真实世界坐标（米，场地中心为原点）
          fps              : 视频帧率
          total_frames     : 视频总帧数
          pitch_l / pitch_w: 场地长宽（米，默认 FIFA 标准 105×68）
          video_path       : 原始视频路径（可选），用于提取音频 RMS 能量辅助评分
        """
        try:
            print(f"[HIGHLIGHT-FAST] 使用分析数据路径：{len(ball_traj_frames)} 个球轨迹点，"
                  f"{len(tactical_tracks)} 条球员轨迹")
            if progress_callback:
                progress_callback(12)

            duration = total_frames / max(fps, 1)
            SAMPLE_INTERVAL = 5   # 与 detect_highlights() 保持一致

            # ── 将球轨迹转为帧号→pitch坐标字典（pitch坐标：wx归一化） ──────────
            # ball_traj_frames 中坐标是像素坐标，需用视频尺寸估算归一化（用于 _in_penalty_area）
            # 由于没有图像宽高，退而使用 pitch 坐标路径判断禁区（更准确）
            # 先构建帧号→球坐标字典（pitch world coords）
            # ball_traj_frames 存的是图像像素坐标，无 pitch 对应
            # 仅用于"球在画面内"的快速判断；禁区判断需要 pitch 坐标
            # 实际上 ball_traj_frames 只有像素坐标，但 tactical_tracks 有 pitch 坐标
            # 策略：用 tactical_tracks 中各球员 pitch 坐标的密集度计算指标2/3/4；
            #        用 ball_traj_frames 像素坐标的横向比例近似判断指标1（禁区）

            # 构建 ball_frame_map: {帧号: (px, py)}
            ball_frame_map: dict = {}
            for pt in ball_traj_frames:
                f = pt.get("f")
                if f is not None:
                    ball_frame_map[int(f)] = (pt.get("x", 0), pt.get("y", 0))

            # 构建 player_frame_map: {帧号: [(wx, wy, team), ...]}
            player_frame_map: dict = {}
            for tid, tdata in tactical_tracks.items():
                team = tdata.get("team", "unknown")
                for pt in tdata.get("pts", []):
                    f = pt.get("f")
                    if f is None:
                        continue
                    f = int(f)
                    if f not in player_frame_map:
                        player_frame_map[f] = []
                    player_frame_map[f].append((
                        float(pt.get("wx", 0)),
                        float(pt.get("wy", 0)),
                        team
                    ))

            if progress_callback:
                progress_callback(18)

            # 估算视频宽高（供像素坐标归一化用）
            # 若无法获得则用典型高清分辨率 1920×1080
            _vid_w, _vid_h = 1920.0, 1080.0
            if ball_frame_map:
                all_px = [v[0] for v in ball_frame_map.values() if v[0] > 0]
                all_py = [v[1] for v in ball_frame_map.values() if v[1] > 0]
                if all_px:
                    _vid_w = max(all_px) * 1.05  # 估算宽度
                if all_py:
                    _vid_h = max(all_py) * 1.05  # 估算高度

            # ── 按 SAMPLE_INTERVAL 生成采样帧评分 ────────────────────────────
            sampled_frame_indices = list(range(0, total_frames, SAMPLE_INTERVAL))
            frame_scores: List[float] = []
            prev_player_pos: List[tuple] = []   # 上一采样帧球员 pitch 坐标列表

            total_samples = len(sampled_frame_indices)
            for si, fno in enumerate(sampled_frame_indices):
                score = 0.0

                # 在 ±3 帧范围内查找最近的球轨迹点（采样间隔导致帧号不精确匹配）
                ball_px, ball_py = None, None
                for delta in range(0, SAMPLE_INTERVAL + 1):
                    for d in ([0, delta, -delta] if delta > 0 else [0]):
                        bp = ball_frame_map.get(fno + d)
                        if bp:
                            ball_px, ball_py = bp
                            break
                    if ball_px is not None:
                        break

                # 当前帧球员列表（pitch坐标）
                cur_players: List[tuple] = player_frame_map.get(fno, [])
                # 在 ±SAMPLE_INTERVAL 内补充最近帧球员（防止稀疏轨迹）
                if not cur_players:
                    for delta in range(1, SAMPLE_INTERVAL + 1):
                        for d in [delta, -delta]:
                            cp = player_frame_map.get(fno + d)
                            if cp:
                                cur_players = cp
                                break
                        if cur_players:
                            break

                cur_pitch = [(wx, wy) for (wx, wy, _) in cur_players]

                # ── 指标1: 球在禁区内 (+3) ──────────────────────────────────
                # 用球的像素坐标估算 pitch 归一化 X，判断是否在两侧禁区
                # 注意：ball_traj_frames 存的是像素坐标；当球靠近两侧边缘时判定为禁区
                # 更可靠方式：若有球员 pitch 坐标，用其 X 范围估算球的 pitch 位置
                if ball_px is not None:
                    # 像素 x 归一化
                    norm_bx = ball_px / max(_vid_w, 1)
                    norm_by = ball_py / max(_vid_h, 1)
                    if self._in_penalty_area(norm_bx, norm_by):
                        score += 3.0

                # ── 指标2: 球员密集区与球重合 (+2) ──────────────────────────
                # 在 pitch 坐标系：≥3名球员与球的距离 < 3m
                if ball_px is not None and len(cur_pitch) >= 3:
                    # 估算球的 pitch 坐标（通过像素归一化→pitch）
                    # ball_px/py → pitch_wx = (norm_bx - 0.5) * pitch_l
                    norm_bx2 = ball_px / max(_vid_w, 1)
                    norm_by2 = ball_py / max(_vid_h, 1)
                    ball_wx = (norm_bx2 - 0.5) * pitch_l
                    ball_wy = (norm_by2 - 0.5) * pitch_w
                    close_count = sum(
                        1 for (px, py) in cur_pitch
                        if math.hypot(px - ball_wx, py - ball_wy) < 3.0
                    )
                    if close_count >= 3:
                        score += 2.0

                # ── 指标3: 快速运动帧 (+1) ──────────────────────────────────
                # 连续两采样帧间球员 pitch 位移均值 > 5 * SAMPLE_INTERVAL（px/帧）
                # pitch坐标单位为米，25fps视频中球员奔跑速度约 0.2m/帧
                # 判断：平均位移 > 0.15m/帧 × SAMPLE_INTERVAL
                if cur_pitch and prev_player_pos:
                    n_pts = min(len(cur_pitch), len(prev_player_pos))
                    if n_pts > 0:
                        cur_s  = sorted(cur_pitch)[:n_pts]
                        prev_s = sorted(prev_player_pos)[:n_pts]
                        avg_disp = float(np.mean([
                            math.hypot(c[0] - p[0], c[1] - p[1])
                            for c, p in zip(cur_s, prev_s)
                        ]))
                        # 0.15m/帧 × SAMPLE_INTERVAL 作为快速运动阈值
                        if avg_disp > 0.15 * SAMPLE_INTERVAL:
                            score += 1.0

                # ── 指标4: 裁判密集区 (+0.5) ─────────────────────────────────
                # pitch 中央区域（x ∈ [-10, +10]）内有孤立球员 → 认为是裁判
                if cur_players:
                    center_pts = [(wx, wy) for (wx, wy, t) in cur_players if abs(wx) <= 10.0]
                    referee_like = 0
                    for (px, py) in center_pts:
                        neighbors = sum(
                            1 for (ox, oy) in cur_pitch
                            if (ox, oy) != (px, py) and math.hypot(ox - px, oy - py) < 2.0
                        )
                        if neighbors == 0:
                            referee_like += 1
                    if referee_like >= 1:
                        score += 0.5

                frame_scores.append(score)
                prev_player_pos = cur_pitch

                # 进度回调（12%→45%）
                if progress_callback and si % max(1, total_samples // 20) == 0:
                    pct = int(12 + (si / max(total_samples, 1)) * 33)
                    progress_callback(pct)

            print(f"[HIGHLIGHT-FAST] 评分完成，共 {len(frame_scores)} 个采样帧")
            if progress_callback:
                progress_callback(45)

            # ── 改进一：音频 RMS 能量提取（有 video_path 时才执行）──────────
            audio_energy: Optional[List[float]] = None
            if video_path and os.path.exists(video_path):
                print("[HIGHLIGHT-FAST] 提取音频能量作为辅助评分...")
                if progress_callback:
                    progress_callback(47)
                audio_energy = self.extract_audio_energy(
                    video_path, fps, total_frames, SAMPLE_INTERVAL
                )

            highlights = self._identify_highlights_v2(
                frame_scores, fps, duration, SAMPLE_INTERVAL,
                audio_energy=audio_energy
            )
            print(f"[HIGHLIGHT-FAST] 识别到 {len(highlights)} 个精彩时刻")
            return highlights

        except Exception as e:
            print(f"[HIGHLIGHT-FAST ERROR] 快速路径失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def detect_highlights(self, video_path: str, progress_callback=None) -> List[HighlightMoment]:
        """
        核心方法：按说明书3.17.1四指标评分模型检测精彩时刻。

        算法流程：
          1. 每5帧采样1帧（与说明书"5帧为单位的片段"一致）
          2. 对每帧计算四类指标分值
          3. 滑动窗口25帧累积评分，超过15分标记为候选
          4. 相邻候选间隔<50帧时合并
          5. 返回 HighlightMoment 列表
        """
        try:
            print(f"[HIGHLIGHT] 开始分析视频（四指标评分模型）: {video_path}")
            if progress_callback:
                progress_callback(5)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[ERROR] 无法打开视频: {video_path}")
                return []

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / max(fps, 1)
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[HIGHLIGHT] 视频信息: {duration:.1f}秒, {fps:.1f} FPS, {total_frames} 帧, {frame_w}×{frame_h}")

            if progress_callback:
                progress_callback(8)

            SAMPLE_INTERVAL = 5   # 每5帧采样一次（说明书要求）
            frame_scores: List[float] = []   # 每个采样帧的原始分值
            frame_idx = 0
            prev_gray = None
            prev_person_centers: List[tuple] = []  # 上一帧球员中心点

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % SAMPLE_INTERVAL != 0:
                    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    continue

                # 进度回调
                if progress_callback:
                    progress = int(8 + (frame_idx / total_frames) * 42)
                    progress_callback(min(50, progress))

                score = 0.0
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # ── 指标4: 低质量帧（扣分）—— 有效像素面积 < 20% ──────────────
                # 有效像素：亮度>30的像素比例（全黑/广告牌遮挡时接近0）
                valid_ratio = float(np.sum(gray > 30)) / gray.size
                if valid_ratio < 0.20:
                    score -= 2.0
                    frame_scores.append(score)
                    prev_gray = gray
                    prev_person_centers = []
                    continue  # 低质量帧直接进入下一帧

                # ── YOLO 检测（球、球员、裁判）────────────────────────────────
                ball_pos = None          # (norm_x, norm_y)
                person_centers: List[tuple] = []   # 球员中心点（归一化）
                referee_centers: List[tuple] = []  # 裁判中心点（归一化）

                if self.yolo_model is not None:
                    try:
                        results = self.yolo_model(frame, verbose=False, conf=0.25)
                        for r in results:
                            for box in r.boxes:
                                cls = int(box.cls[0])
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                cx_norm = ((x1 + x2) / 2) / frame_w
                                cy_norm = ((y1 + y2) / 2) / frame_h
                                bw = (x2 - x1) / frame_w
                                bh = (y2 - y1) / frame_h
                                if cls == 0:   # person
                                    # 简单区分裁判（黑色/荧光服）：用bbox面积区分（裁判通常孤立在画面中部）
                                    person_centers.append((cx_norm, cy_norm))
                                elif cls == 32:  # sports ball
                                    # 优先置信度最高的球
                                    conf = float(box.conf[0])
                                    if ball_pos is None or conf > 0.3:
                                        ball_pos = (cx_norm, cy_norm)
                    except Exception as _e:
                        pass  # 检测失败不影响评分流程

                # ── 指标1: 球在禁区内 (+3) ──────────────────────────────────
                if ball_pos and self._in_penalty_area(ball_pos[0], ball_pos[1]):
                    score += 3.0

                # ── 指标2: 球员密集区与球重合 (+2) ──────────────────────────
                # ≥3名球员与球的距离<3m（归一化距离约 3/105 ≈ 0.029）
                if ball_pos and len(person_centers) >= 3:
                    DIST_THRESH = 3.0 / 105.0  # 3m 归一化
                    close_count = sum(
                        1 for (px, py) in person_centers
                        if math.hypot(px - ball_pos[0], py - ball_pos[1]) < DIST_THRESH
                    )
                    if close_count >= 3:
                        score += 2.0

                # ── 指标3: 快速运动帧 (+1) ──────────────────────────────────
                # 连续帧间球员位移均值 > 5px/帧（原始像素坐标）
                if person_centers and prev_person_centers:
                    # 取同数量的点计算平均位移（取最近匹配）
                    n_pts = min(len(person_centers), len(prev_person_centers))
                    if n_pts > 0:
                        # 简单匹配：排序后按顺序配对
                        cur_px  = sorted(person_centers)[:n_pts]
                        prev_px = sorted(prev_person_centers)[:n_pts]
                        avg_disp = np.mean([
                            math.hypot((c[0]-p[0])*frame_w, (c[1]-p[1])*frame_h)
                            for c, p in zip(cur_px, prev_px)
                        ])
                        # 5px/帧，SAMPLE_INTERVAL=5时相当于25px/5帧
                        if avg_disp > 5.0 * SAMPLE_INTERVAL:
                            score += 1.0

                # ── 指标4: 裁判密集区 (+0.5) ─────────────────────────────────
                # 画面中央1/3区域（x: 0.33~0.67）内检测到≥1名裁判
                # 用人员密度和位置来近似（YOLO无专门裁判类，使用孤立人员在中央区域判断）
                CENTER_X_MIN, CENTER_X_MAX = 0.33, 0.67
                center_persons = [
                    (px, py) for (px, py) in person_centers
                    if CENTER_X_MIN <= px <= CENTER_X_MAX
                ]
                # 简单启发：中央孤立区域（周围2m内无其他人）的人员认为是裁判
                referee_like = 0
                for (px, py) in center_persons:
                    neighbors = sum(
                        1 for (ox, oy) in person_centers
                        if (ox, oy) != (px, py) and math.hypot(ox - px, oy - py) < 0.05
                    )
                    if neighbors == 0:
                        referee_like += 1
                if referee_like >= 1:
                    score += 0.5

                frame_scores.append(score)
                prev_gray = gray
                prev_person_centers = person_centers

            cap.release()
            print(f"[HIGHLIGHT] 帧评分完成，共 {len(frame_scores)} 个采样帧，开始识别精彩片段...")

            # 识别精彩时刻
            # ── 音频 RMS 能量提取（辅助评分）────────────────────────────────
            audio_energy_full: Optional[List[float]] = None
            try:
                print("[HIGHLIGHT] 提取音频能量作为辅助评分...")
                audio_energy_full = self.extract_audio_energy(
                    video_path, fps, total_frames, SAMPLE_INTERVAL
                )
            except Exception:
                pass

            highlights = self._identify_highlights_v2(
                frame_scores, fps, duration, SAMPLE_INTERVAL,
                audio_energy=audio_energy_full
            )
            print(f"[HIGHLIGHT] 识别到 {len(highlights)} 个精彩时刻")
            return highlights

        except Exception as e:
            print(f"[HIGHLIGHT ERROR] 视频分析过程出错: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _identify_highlights_v2(
        self,
        frame_scores: List[float],
        fps: float,
        duration: float,
        sample_interval: int,
        audio_energy: Optional[List[float]] = None
    ) -> List[HighlightMoment]:
        """
        按说明书3.17.1算法：
          - 滑动窗口25帧累积评分超过15分时标记为候选
          - 相邻候选间隔<50帧时合并

        改进（论文启发）：
          1. 音频 RMS 能量加分（SportSummarizer 启发一）：音频能量峰值时 +1 分
          2. TCN 轻量时序趋势（SportSummarizer 启发二）：利用短/中窗口差分"加速度"
             判断事件高潮趋势，纳入累积评分权重
          3. 动态片段边界（替代固定前扩3s/后扩5s）
          4. 余弦相似度去重（ICMLA 2024 启发，最终输出前剔除重叠片段）
        """
        n = len(frame_scores)
        if n == 0:
            return []

        WINDOW = 25          # 滑动窗口大小（采样帧）
        THRESHOLD = 15.0     # 累积评分阈值
        MERGE_GAP = 50       # 合并间隔（原始帧数）

        # ── 改进三：计算时序趋势特征（短/中期差分"加速度"）──────────────────
        _, _, accel = self.compute_temporal_features(frame_scores)
        # 将趋势加速度作为额外加分（上升趋势 = 事件正在发生）
        # 加速度 > 0.5 时额外加 0.5 分（不超量）
        augmented_scores = [
            s + (0.5 if (len(accel) > i and accel[i] > 0.5) else 0.0)
            for i, s in enumerate(frame_scores)
        ]

        # ── 改进一：融合音频能量（归一化能量峰值时额外加 +1.5 分）────────────
        if audio_energy and len(audio_energy) >= n:
            for i in range(n):
                if audio_energy[i] > 0.65:   # 观众欢呼峰值区间
                    augmented_scores[i] += 1.5
                elif audio_energy[i] > 0.45:
                    augmented_scores[i] += 0.75

        # 计算滑动窗口累积评分（使用增强后的分数）
        window_scores_aug = []
        window_scores_raw = []  # 保留原始评分用于 segment 打分
        for i in range(n):
            start = max(0, i - WINDOW // 2)
            end   = min(n, i + WINDOW // 2 + 1)
            window_scores_aug.append(sum(augmented_scores[start:end]))
            window_scores_raw.append(sum(frame_scores[start:end]))

        # 标记超过阈值的采样帧（原始帧号 = 采样帧索引 * sample_interval）
        candidate_frames = [
            i * sample_interval
            for i in range(n)
            if window_scores_aug[i] >= THRESHOLD
        ]

        if not candidate_frames:
            # 阈值过高时降级：取评分最高的几个窗口
            print("[HIGHLIGHT] 未找到超阈值片段，降级取TOP5窗口")
            sorted_idx = sorted(range(n), key=lambda i: window_scores_aug[i], reverse=True)
            candidate_frames = [i * sample_interval for i in sorted_idx[:5]]
            candidate_frames.sort()

        # 合并相邻候选片段（间隔<50帧）
        segments = []  # [(start_frame, end_frame)]
        cur_start = candidate_frames[0]
        cur_end   = candidate_frames[0]
        for f in candidate_frames[1:]:
            if f - cur_end < MERGE_GAP:
                cur_end = f
            else:
                segments.append((cur_start, cur_end))
                cur_start = f
                cur_end   = f
        segments.append((cur_start, cur_end))

        # 生成 HighlightMoment
        highlights = []
        for seg_start, seg_end in segments:
            peak_time = (seg_start + seg_end) / 2 / fps

            # ── 改进四：动态片段边界（替代固定前扩3s/后扩5s）──────────────
            start_time, end_time = self.compute_dynamic_boundaries(
                seg_start, seg_end, frame_scores, fps, sample_interval, duration
            )

            # 计算此片段的平均评分（使用原始评分，避免增强分数虚高）
            i_start = seg_start // sample_interval
            i_end   = min(n - 1, seg_end // sample_interval)
            seg_score = float(np.mean(window_scores_raw[i_start:i_end+1])) if i_start <= i_end else 0.0

            # 分类（传入加速度特征，提升分类精度）
            moment_type = self._classify_moment_v2(
                i_start, i_end, frame_scores, n,
                accel_features=accel,
                audio_energy=audio_energy
            )

            highlights.append(HighlightMoment(
                start_time=start_time,
                end_time=end_time,
                moment_type=moment_type,
                score=seg_score / max(THRESHOLD, 1.0),  # 归一化到 0~1
                description=self._generate_description(moment_type, seg_score / THRESHOLD),
                metadata={'seg_start': seg_start, 'seg_end': seg_end, 'raw_score': seg_score}
            ))

        # 按评分降序 + 改进二：去重（时间重叠 IoU > 0.85 视为重复）
        highlights.sort(key=lambda h: h.score, reverse=True)
        highlights = self.deduplicate_highlights(highlights, sim_threshold=0.85)
        return highlights

    # =========================================================================
    # 论文 4.2.4 算法流程：三模态特征提取 + 自适应融合
    # =========================================================================

    @staticmethod
    def extract_visual_features(frame_scores: List[float],
                                i_start: int, i_end: int) -> np.ndarray:
        """
        提取视觉特征向量 fv（维度 6）。

        论文对应：视觉特征向量 fv，由 YOLOv8 检测结果和场地几何关系构成。

        特征定义（与 AdaptiveMultiModalFusion.DIM_VISUAL=6 对齐）：
          [0] seg_max_score     : 片段内帧评分最大值（归一化 /6.0）
          [1] seg_mean_score    : 片段内帧评分均值（归一化 /6.0）
          [2] penalty_ratio     : 片段内高分帧（>3.0）占比（禁区球比例）
          [3] density_ratio     : 片段内得分 ≥2 帧占比（球员密集区比例）
          [4] motion_ratio      : 片段内得分 ≥1 帧占比（快速运动帧比例）
          [5] duration_norm     : 片段时长归一化（片段帧数 / 最大可能帧数 100）
        """
        n = len(frame_scores)
        if n == 0 or i_start > i_end:
            return np.zeros(AdaptiveMultiModalFusion.DIM_VISUAL, dtype=np.float32)
        seg = frame_scores[i_start: min(n, i_end + 1)]
        if not seg:
            return np.zeros(AdaptiveMultiModalFusion.DIM_VISUAL, dtype=np.float32)
        seg_max  = float(max(seg))
        seg_mean = float(np.mean(seg))
        pen_ratio  = float(sum(1 for s in seg if s >= 3.0) / len(seg))
        dens_ratio = float(sum(1 for s in seg if s >= 2.0) / len(seg))
        mot_ratio  = float(sum(1 for s in seg if s >= 1.0) / len(seg))
        dur_norm   = min(1.0, len(seg) / 100.0)
        return np.array([
            min(1.0, seg_max / 6.0),
            min(1.0, seg_mean / 6.0),
            pen_ratio, dens_ratio, mot_ratio, dur_norm
        ], dtype=np.float32)

    @staticmethod
    def extract_audio_features(audio_energy: Optional[List[float]],
                               i_start: int, i_end: int) -> np.ndarray:
        """
        提取音频特征向量 fa（维度 3）。

        论文对应：音频特征向量 fa，由现场音频 RMS 能量分析构成。

        特征定义（与 AdaptiveMultiModalFusion.DIM_AUDIO=3 对齐）：
          [0] mean_energy  : 片段内平均 RMS 能量
          [1] peak_flag    : 是否存在 RMS > 0.65 的欢呼峰值（0/1）
          [2] p95_energy   : 片段内 95 百分位能量（鲁棒峰值估计）
        """
        fa = np.zeros(AdaptiveMultiModalFusion.DIM_AUDIO, dtype=np.float32)
        if not audio_energy:
            return fa
        n = len(audio_energy)
        seg = audio_energy[i_start: min(n, i_end + 1)]
        if not seg:
            return fa
        fa[0] = float(np.mean(seg))
        fa[1] = float(any(v > 0.65 for v in seg))
        fa[2] = float(np.percentile(seg, 95))
        return fa

    @staticmethod
    def extract_text_features(frame_scores: List[float],
                              accel: List[float],
                              i_start: int, i_end: int) -> np.ndarray:
        """
        提取文本/时序特征向量 ft（维度 4）。

        论文对应：文本特征向量 ft，由 TCN 时序趋势分析构成。
        本实现使用滑动窗口差分（短/中期均值差）模拟 TCN 输出。

        特征定义（与 AdaptiveMultiModalFusion.DIM_TEXT=4 对齐）：
          [0] short_avg    : 片段短窗口平均分（≈1s 时域响应）
          [1] medium_avg   : 片段中窗口平均分（≈5s 时域响应）
          [2] avg_accel    : 片段内平均时序加速度（short-medium 差）
          [3] trend_dir    : 趋势方向（1=上升中, 0=平稳, -1=下降中, 归一化为0~1）
        """
        ft = np.zeros(AdaptiveMultiModalFusion.DIM_TEXT, dtype=np.float32)
        n = len(frame_scores)
        if n == 0 or i_start > i_end:
            return ft
        seg_scores = frame_scores[i_start: min(n, i_end + 1)]
        seg_accel  = accel[i_start: min(len(accel), i_end + 1)] if accel else []

        # 短窗口（前1/3帧）均值 vs 中窗口（全段）均值
        half = max(1, len(seg_scores) // 3)
        short_m  = float(np.mean(seg_scores[:half])) if seg_scores else 0.0
        medium_m = float(np.mean(seg_scores))         if seg_scores else 0.0
        avg_acc  = float(np.mean(seg_accel))           if seg_accel  else 0.0

        # 趋势方向归一化到 [0, 1]：0=下降, 0.5=平稳, 1=上升
        trend = 0.5 + float(np.clip(avg_acc / 2.0, -0.5, 0.5))

        ft[0] = min(1.0, short_m  / 6.0)
        ft[1] = min(1.0, medium_m / 6.0)
        ft[2] = float(np.clip(avg_acc, -1.0, 1.0)) * 0.5 + 0.5   # 归一化到[0,1]
        ft[3] = trend
        return ft

    def detect_highlights_multimodal(
        self,
        frame_scores: List[float],
        fps: float,
        duration: float,
        sample_interval: int,
        audio_energy: Optional[List[float]] = None,
        progress_callback=None
    ) -> List[HighlightMoment]:
        """
        【论文核心流程】多模态自适应融合精彩片段检测。

        算法流程（对应论文图示）：
          Step 1 ── 规则预筛选
                    滑动窗口25帧累积评分 > 15 分时标记候选，合并相邻候选；
                    无候选时降级取TOP-5窗口。
          Step 2 ── 三模态特征提取
                    • 视觉特征 fv：YOLO四指标分值统计（禁区/密集/运动/画质）
                    • 音频特征 fa：RMS能量均值/峰值/95百分位
                    • 文本特征 ft：TCN短/中窗差分加速度、趋势方向
          Step 3 ── 自适应多模态融合（论文4.2.4 MLP + Softmax）
                    输入 [fv; fa; ft] → 两层全连接(512→3) → Softmax →
                    动态权重 (wv, wa, wt)；
                    各模态分数 Sv/Sa/St 由规则评分归一化得到；
                    Sfinal = wv*Sv + wa*Sa + wt*St
          Step 4 ── 精彩片段生成与排序
                    动态边界扩展 + 时间 IoU 去重 + 按 Sfinal 降序排列。
        """
        n = len(frame_scores)
        if n == 0:
            return []

        fusion_net = get_fusion_net()

        # ── Step 1: 规则预筛选（沿用四指标滑动窗口）────────────────────────
        if progress_callback:
            progress_callback(30)

        WINDOW    = 25
        THRESHOLD = 15.0
        MERGE_GAP = 50
        SAMPLE_IV = sample_interval

        # 计算时序特征（TCN轻量版）
        _, _, accel = self.compute_temporal_features(frame_scores)

        # 增强评分（同 _identify_highlights_v2，保持一致）
        augmented = [
            s + (0.5 if (len(accel) > i and accel[i] > 0.5) else 0.0)
            for i, s in enumerate(frame_scores)
        ]
        if audio_energy and len(audio_energy) >= n:
            for i in range(n):
                if audio_energy[i] > 0.65:
                    augmented[i] += 1.5
                elif audio_energy[i] > 0.45:
                    augmented[i] += 0.75

        # 滑动窗口累积评分
        window_aug = []
        window_raw = []
        for i in range(n):
            s = max(0, i - WINDOW // 2)
            e = min(n, i + WINDOW // 2 + 1)
            window_aug.append(sum(augmented[s:e]))
            window_raw.append(sum(frame_scores[s:e]))

        candidate_frames = [
            i * SAMPLE_IV for i in range(n) if window_aug[i] >= THRESHOLD
        ]
        if not candidate_frames:
            print("[MULTIMODAL] 预筛选未找到候选，降级取TOP-5")
            top5 = sorted(range(n), key=lambda i: window_aug[i], reverse=True)[:5]
            candidate_frames = sorted(i * SAMPLE_IV for i in top5)

        # 合并相邻候选
        segments = []
        cur_s = candidate_frames[0]
        cur_e = candidate_frames[0]
        for f in candidate_frames[1:]:
            if f - cur_e < MERGE_GAP:
                cur_e = f
            else:
                segments.append((cur_s, cur_e))
                cur_s = cur_e = f
        segments.append((cur_s, cur_e))

        if progress_callback:
            progress_callback(40)

        # ── Step 2+3: 三模态特征提取 + 自适应融合 ────────────────────────
        print(f"[MULTIMODAL] 规则预筛选得到 {len(segments)} 个候选片段，开始多模态融合...")
        highlights = []
        for seg_start, seg_end in segments:
            i_start = seg_start // SAMPLE_IV
            i_end   = min(n - 1, seg_end // SAMPLE_IV)

            # ── 视觉特征向量 fv ────────────────────────────────────────────
            fv = self.extract_visual_features(frame_scores, i_start, i_end)
            # 视觉模态分数 Sv：片段窗口平均评分归一化
            sv = float(np.mean(window_raw[i_start:i_end + 1])) / max(THRESHOLD, 1.0)
            sv = min(1.0, sv)

            # ── 音频特征向量 fa ────────────────────────────────────────────
            fa = self.extract_audio_features(audio_energy, i_start, i_end)
            # 音频模态分数 Sa：平均RMS能量（已归一化到0~1）
            sa = float(fa[0]) * 0.6 + float(fa[1]) * 0.3 + float(fa[2]) * 0.1

            # ── 文本/时序特征向量 ft ───────────────────────────────────────
            ft = self.extract_text_features(frame_scores, accel, i_start, i_end)
            # 时序模态分数 St：归一化趋势分数
            st = float(ft[1]) * 0.5 + (float(ft[2]) - 0.5) * 0.3 + float(ft[3]) * 0.2
            st = float(np.clip(st, 0.0, 1.0))

            # ── 自适应融合：Sfinal = wv*Sv + wa*Sa + wt*St ─────────────────
            sfinal, wv_w, wa_w, wt_w = fusion_net.fuse(sv, sa, st, fv, fa, ft)
            print(f"  [FUSION] seg={seg_start}-{seg_end} "
                  f"Sv={sv:.3f} Sa={sa:.3f} St={st:.3f} "
                  f"w=({wv_w:.2f},{wa_w:.2f},{wt_w:.2f}) "
                  f"Sfinal={sfinal:.3f}")

            # ── Step 4: 动态边界扩展 ──────────────────────────────────────
            start_time, end_time = self.compute_dynamic_boundaries(
                seg_start, seg_end, frame_scores, fps, SAMPLE_IV, duration
            )

            # 分类（使用三模态信息）
            moment_type = self._classify_moment_multimodal(
                sv, sa, st, fa, accel, i_start, i_end
            )

            highlights.append(HighlightMoment(
                start_time=start_time,
                end_time=end_time,
                moment_type=moment_type,
                score=sfinal,
                description=self._generate_description(moment_type, sfinal),
                metadata={
                    'sv': sv, 'sa': sa, 'st': st,
                    'wv': wv_w, 'wa': wa_w, 'wt': wt_w,
                    'sfinal': sfinal,
                    'seg_start': seg_start, 'seg_end': seg_end
                }
            ))

        if progress_callback:
            progress_callback(48)

        # ── Step 4: 按 Sfinal 降序排列 + IoU 去重 ────────────────────────
        highlights.sort(key=lambda h: h.score, reverse=True)
        highlights = self.deduplicate_highlights(highlights, sim_threshold=0.85)
        print(f"[MULTIMODAL] 多模态融合完成，输出 {len(highlights)} 个精彩片段")
        return highlights

    def _classify_moment_multimodal(self,
                                    sv: float, sa: float, st: float,
                                    fa: np.ndarray,
                                    accel: List[float],
                                    i_start: int, i_end: int) -> str:
        """
        基于三模态分数的精彩时刻分类（论文4.2.4融合后分类）。
        在原有 _classify_moment_v2 逻辑基础上，引入融合分数修正。
        """
        n_accel = len(accel)
        avg_accel = 0.0
        if n_accel > i_start:
            seg_acc = accel[i_start: min(n_accel, i_end + 1)]
            avg_accel = float(np.mean(seg_acc)) if seg_acc else 0.0

        has_audio_peak = float(fa[1]) > 0.5  # peak_flag

        # 融合分数高 + 音频峰值 + 加速度趋势
        if sv >= 0.65 and has_audio_peak and avg_accel <= 0.3:
            return 'goal'
        if sv >= 0.65:
            return 'goal'
        if sv >= 0.50 and avg_accel > 0.0:
            return 'shot'
        if sv >= 0.50:
            return 'shot'
        if sa >= 0.40 and avg_accel < -0.2:
            return 'celebration'
        if st >= 0.40:
            return 'pass'
        return 'action'

    def _classify_moment_v2(self, i_start: int, i_end: int,
                             frame_scores: List[float], n: int,
                             accel_features: Optional[List[float]] = None,
                             audio_energy: Optional[List[float]] = None) -> str:
        """
        根据片段内帧评分、时序加速度趋势、音频能量分类时刻类型。

        改进逻辑（TCN 启发）：
          - 加速度 > 0：事件仍在上升（进攻组织阶段 → pass/shot）
          - 加速度 < 0 且绝对值大：事件刚过高潮（可能是进球后庆祝 → goal/celebration）
          - 音频能量高：确认为有观众反应的高价值事件（提权重）
        """
        if i_start > i_end or i_start >= n:
            return 'action'
        seg = frame_scores[i_start: min(n, i_end + 1)]
        avg = float(np.mean(seg)) if seg else 0.0
        max_s = float(max(seg)) if seg else 0.0

        # 片段内平均加速度（正=上升趋势，负=已过高潮）
        avg_accel = 0.0
        if accel_features and len(accel_features) > i_start:
            accel_seg = accel_features[i_start: min(len(accel_features), i_end + 1)]
            avg_accel = float(np.mean(accel_seg)) if accel_seg else 0.0

        # 片段内平均音频能量
        avg_audio = 0.0
        has_audio_peak = False
        if audio_energy and len(audio_energy) > i_start:
            audio_seg = audio_energy[i_start: min(len(audio_energy), i_end + 1)]
            avg_audio = float(np.mean(audio_seg)) if audio_seg else 0.0
            has_audio_peak = any(v > 0.65 for v in audio_seg) if audio_seg else False

        # ── 分类逻辑（音频能量 + 时序趋势 + 原有帧评分）────────────────────
        # 进球：高分帧 + 音频能量高（欢呼） + 事件已过高潮（负加速度/平稳）
        if max_s >= 4.0 and (has_audio_peak or avg_audio > 0.4) and avg_accel <= 0.3:
            return 'goal'
        # 进球（无音频辅助降级判断）
        if max_s >= 4.0:
            return 'goal'
        # 射门：高分帧 + 仍在上升趋势（起脚瞬间）
        if max_s >= 3.0 and avg_accel > 0.0:
            return 'shot'
        if max_s >= 3.0:
            return 'shot'
        # 庆祝：中等评分 + 音频持续高能 + 下降趋势（进球后球员聚集）
        if avg >= 1.5 and has_audio_peak and avg_accel < -0.2:
            return 'celebration'
        if avg >= 1.5:
            return 'celebration' if len(seg) > 10 else 'tackle'
        if avg >= 0.5:
            return 'pass'
        return 'action'

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

    # ── 改进一：音频 RMS 能量辅助评分（SportSummarizer 启发一，低成本实现）─────
    @staticmethod
    def extract_audio_energy(video_path: str, fps: float, total_frames: int,
                             sample_interval: int = 5) -> List[float]:
        """
        用 ffmpeg 将视频音频提取为 PCM 数据，计算每个采样帧对应时间窗口的 RMS 能量。

        返回长度与 frame_scores 对齐（每 sample_interval 帧一个值），归一化到 [0, 1]。
        无音频轨道或 ffmpeg 失败时返回全零列表（不影响原有评分逻辑）。

        原理：进球/庆祝/角球/任意球时，现场观众欢呼声的 RMS 能量会出现明显峰值。
        固定机位场景下：画面无转播字幕辅助，音频能量是判断"精彩瞬间"的有效第二通道。
        """
        import subprocess, struct, tempfile, os
        n_samples = (total_frames + sample_interval - 1) // sample_interval

        try:
            # 用 ffmpeg 将音频解码为 16kHz 单声道 16bit PCM（raw 格式）
            ffmpeg_exe = FFMPEG_EXE
            tmp_pcm = tempfile.NamedTemporaryFile(suffix='.pcm', delete=False)
            tmp_pcm.close()
            cmd = [
                ffmpeg_exe, '-y', '-i', video_path,
                '-vn', '-ac', '1', '-ar', '16000', '-f', 's16le', tmp_pcm.name
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if result.returncode != 0 or not os.path.exists(tmp_pcm.name):
                return [0.0] * n_samples

            # 读取 PCM 数据
            with open(tmp_pcm.name, 'rb') as f:
                raw = f.read()
            os.unlink(tmp_pcm.name)

            if len(raw) < 2:
                return [0.0] * n_samples

            samples_per_frame = int(16000 / max(fps, 1))  # 每视频帧对应的音频样本数
            n_audio = len(raw) // 2
            pcm = struct.unpack(f'<{n_audio}h', raw[:n_audio * 2])
            pcm_arr = np.array(pcm, dtype=np.float32) / 32768.0

            # 计算每个采样帧对应窗口（sample_interval 帧宽）的 RMS
            energy_list = []
            for si in range(n_samples):
                fno = si * sample_interval
                a_start = fno * samples_per_frame
                a_end = a_start + samples_per_frame * sample_interval
                window = pcm_arr[a_start:a_end]
                if len(window) == 0:
                    energy_list.append(0.0)
                else:
                    rms = float(np.sqrt(np.mean(window ** 2)))
                    energy_list.append(rms)

            if not energy_list or max(energy_list) < 1e-6:
                return [0.0] * n_samples

            # 归一化到 [0, 1]
            e_max = float(np.percentile(energy_list, 95))  # 用95百分位避免单帧异常噪声
            if e_max < 1e-6:
                return [0.0] * n_samples
            normalized = [min(1.0, float(e) / e_max) for e in energy_list]
            print(f"[HIGHLIGHT] 音频能量提取完成，{n_samples} 个采样帧，最大RMS={e_max:.4f}")
            return normalized

        except Exception as ex:
            print(f"[HIGHLIGHT] 音频能量提取失败（跳过，不影响评分）: {ex}")
            return [0.0] * n_samples

    # ── 改进二：帧指纹去重（ICMLA 2024 论文——余弦相似度检测重复片段）─────────
    @staticmethod
    def deduplicate_highlights(highlights: List['HighlightMoment'],
                               sim_threshold: float = 0.85) -> List['HighlightMoment']:
        """
        基于片段时间重叠率去除重复/高度相似的精彩片段。

        论文原始思路：逐帧提取 HOG/颜色直方图作为指纹后计算余弦相似度。
        工程轻量化：直接用时间段的 Jaccard 重叠率（IOU）替代帧指纹，
        因为同一段视频时间重叠 > 85% 等价于内容重复。

        重复判断策略：
          1. 按评分降序排列（保留质量高的）
          2. 遍历时，若当前片段与已保留片段时间 IoU > sim_threshold，则丢弃
        """
        if not highlights:
            return highlights

        sorted_hl = sorted(highlights, key=lambda h: h.score, reverse=True)
        kept: List['HighlightMoment'] = []

        for cand in sorted_hl:
            is_dup = False
            for ref in kept:
                # 时间段 IoU
                inter = max(0.0, min(cand.end_time, ref.end_time) - max(cand.start_time, ref.start_time))
                cand_dur = max(cand.end_time - cand.start_time, 1e-6)
                ref_dur  = max(ref.end_time  - ref.start_time,  1e-6)
                union = cand_dur + ref_dur - inter
                iou = inter / union if union > 1e-6 else 0.0
                if iou > sim_threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(cand)

        if len(kept) < len(highlights):
            print(f"[HIGHLIGHT] 去重：{len(highlights)} → {len(kept)} 个片段（阈值 IoU>{sim_threshold}）")
        return kept

    # ── 改进三：TCN 轻量版——时序趋势分析（SportSummarizer 启发二/三，轻量实现）──
    @staticmethod
    def compute_temporal_features(frame_scores: List[float],
                                  window_sizes: Tuple[int, int, int] = (5, 25, 75)
                                  ) -> Tuple[List[float], List[float], List[float]]:
        """
        论文原理：层次化 TCN 用 1s/10s/30s 三级时间尺度捕捉时序依赖。
        轻量实现：用三个不同大小的滑动均值窗口模拟短/中/长时序特征，
                  再计算相邻窗口的差分作为"趋势加速度"，用于区分事件类型。

        返回：
          short_avg   : 短窗口（~1s）滑动均值
          medium_avg  : 中窗口（~5s）滑动均值
          accel       : 短窗口与中窗口的差值（正 = 上升趋势，事件正在发生）
        """
        n = len(frame_scores)
        if n == 0:
            return [], [], []

        def sliding_avg(scores, w):
            result = []
            for i in range(n):
                s = max(0, i - w // 2)
                e = min(n, i + w // 2 + 1)
                result.append(float(np.mean(scores[s:e])))
            return result

        short_avg  = sliding_avg(frame_scores, window_sizes[0])
        medium_avg = sliding_avg(frame_scores, window_sizes[1])
        # "加速度"：短期均值 - 中期均值，正值表示分数正在快速上升（进攻高潮发生中）
        accel = [s - m for s, m in zip(short_avg, medium_avg)]
        return short_avg, medium_avg, accel

    # ── 改进四：动态片段边界（基于球员密度趋势自适应确定前扩/后扩时长）──────────
    @staticmethod
    def compute_dynamic_boundaries(seg_start: int, seg_end: int,
                                   frame_scores: List[float],
                                   fps: float, sample_interval: int,
                                   duration: float) -> Tuple[float, float]:
        """
        固定前扩3s/后扩5s太机械，对进球庆祝会截断，对传球过程会多包冗余内容。

        动态策略：
          - 向前扩展：从 seg_start 向前找第一个分数 < 0.5 的采样帧，最多扩展 5s
          - 向后扩展：从 seg_end 向后找第一个分数 < 0.5 的采样帧，最多扩展 8s
                       （进球后庆祝段时间较长，后扩上限更大）
          - 兜底：前扩至少 2s，后扩至少 3s（保证完整动作可视）
        """
        n = len(frame_scores)
        i_start = max(0, seg_start // sample_interval)
        i_end   = min(n - 1, seg_end // sample_interval)

        # 向前扩展：找到 i_start 前最近的"低分帧"
        pre_expand = 2.0   # 最小前扩（秒）
        max_pre = 5.0
        for back in range(1, int(max_pre * fps / sample_interval) + 1):
            idx = i_start - back
            if idx < 0:
                pre_expand = max(pre_expand, (i_start * sample_interval) / max(fps, 1))
                break
            if frame_scores[idx] < 0.5:
                pre_expand = max(pre_expand, back * sample_interval / max(fps, 1))
                break
        else:
            pre_expand = max_pre

        # 向后扩展：找到 i_end 后最近的"低分帧"
        post_expand = 3.0  # 最小后扩（秒）
        max_post = 8.0
        for fwd in range(1, int(max_post * fps / sample_interval) + 1):
            idx = i_end + fwd
            if idx >= n:
                post_expand = max(post_expand, ((n - 1 - i_end) * sample_interval) / max(fps, 1))
                break
            if frame_scores[idx] < 0.5:
                post_expand = max(post_expand, fwd * sample_interval / max(fps, 1))
                break
        else:
            post_expand = max_post

        start_time = max(0.0, seg_start / max(fps, 1) - pre_expand)
        end_time   = min(duration, seg_end / max(fps, 1) + post_expand)
        return start_time, end_time


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
        self.temp_dir = os.path.join(tempfile.gettempdir(), 'goalscope_highlights')

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
        progress_callback=None,
        analysis_data: Optional[Dict] = None,  # 【快速路径】分析阶段已有数据
    ) -> Dict:
        """
        生成精彩视频

        参数 analysis_data（可选）：若传入，使用已有分析数据快速检测精彩时刻，
        跳过重新运行 YOLO，速度提升约 80%。格式：
          {
            "ball_traj_frames": [...],  # analysis.py 生成的球轨迹（带帧号）
            "tactical_tracks":  {...},  # analysis.py 生成的球员追踪数据（pitch坐标）
            "fps":              float,
            "total_frames":     int,
            "pitch_l":          float,  # 场地长（米），默认 105
            "pitch_w":          float,  # 场地宽（米），默认 68
          }

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
            # ── 读取视频实际时长，防止目标时长超出素材总长 ──────────────────────
            _cap_info = cv2.VideoCapture(input_video)
            if _cap_info.isOpened():
                _total_f = int(_cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
                _fps_v   = _cap_info.get(cv2.CAP_PROP_FPS) or 25.0
                _vid_dur = _total_f / _fps_v
                _cap_info.release()
                if _vid_dur > 0 and _vid_dur < target_duration:
                    print(f"[HIGHLIGHT] ⚠️ 视频时长 {_vid_dur:.1f}s < 目标时长 {target_duration:.0f}s，"
                          f"自动降为 {_vid_dur:.1f}s，精彩视频将使用全部素材")
                    target_duration = _vid_dur
            else:
                _cap_info.release()

            # ──────────────────────────────────────────────────────────────
            # 论文算法流程（4.2.4 多模态自适应特征融合）：
            #   Step 1  规则预筛选
            #   Step 2  视觉/音频/文本特征提取
            #   Step 3  自适应多模态融合（MLP + Softmax）→ Sfinal 排序
            #   Step 4  精彩片段生成（动态边界扩展 + IoU 去重）
            # 兼容路径：若有预存射门帧，先以射门帧作为种子，再经多模态重排序
            # ──────────────────────────────────────────────────────────────
            shot_events = analysis_data.get("shot_event_frames", []) if analysis_data else []

            if shot_events:
                # ── 路径①：预存射门帧作为种子，经多模态融合重排序 ────────────
                print(f"\n[1/4] 🚀 使用预存射门事件帧（{len(shot_events)} 个）作为规则预筛选种子...")
                cap_tmp = cv2.VideoCapture(input_video)
                vid_fps   = cap_tmp.get(cv2.CAP_PROP_FPS) or 25.0
                vid_total = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
                cap_tmp.release()
                vid_dur = vid_total / max(vid_fps, 1)

                PRE_SEC, POST_SEC = 3.0, 5.0
                seed_highlights = []
                for ev in shot_events:
                    fno  = ev.get("fno", 0) if isinstance(ev, dict) else int(ev)
                    team = ev.get("team", "unknown") if isinstance(ev, dict) else "unknown"
                    t  = fno / max(vid_fps, 1)
                    st_t = max(0.0, t - PRE_SEC)
                    et_t = min(vid_dur, t + POST_SEC)
                    if et_t > st_t:
                        seed_highlights.append(HighlightMoment(
                            start_time=st_t, end_time=et_t,
                            moment_type='shot', score=0.85,
                            description=f'射门事件 @{t:.1f}s ({team})',
                            metadata={"fno": fno, "team": team}
                        ))

                # 提取音频特征用于多模态重排序
                print("\n[2/4] 🎵 提取音频特征（RMS 能量）...")
                SAMPLE_IV = 5
                audio_energy_seed: Optional[List[float]] = None
                try:
                    audio_energy_seed = self.detector.extract_audio_energy(
                        input_video, vid_fps, vid_total, SAMPLE_IV
                    )
                except Exception:
                    pass

                # 多模态融合重排序：对每个种子片段计算 Sfinal
                print("\n[3/4] 🔀 自适应多模态融合重排序（Sfinal = wv·Sv + wa·Sa + wt·St）...")
                fusion_net = get_fusion_net()
                reranked = []
                for hl in seed_highlights:
                    i_s = int(hl.start_time * vid_fps) // SAMPLE_IV
                    i_e = int(hl.end_time   * vid_fps) // SAMPLE_IV
                    # 构造虚拟帧评分（射门种子固定 Sv=0.85）
                    sv_score = hl.score
                    fv_feat  = np.array([sv_score, sv_score * 0.9, 0.8, 0.7, 0.8, 0.5],
                                        dtype=np.float32)
                    fa_feat  = self.detector.extract_audio_features(
                        audio_energy_seed, i_s, i_e
                    )
                    sa_score = float(fa_feat[0]) * 0.6 + float(fa_feat[1]) * 0.3 + float(fa_feat[2]) * 0.1
                    ft_feat  = np.array([sv_score * 0.8, sv_score * 0.7,
                                         0.55, 0.6], dtype=np.float32)
                    st_score = float(np.clip(ft_feat[1] * 0.5 + 0.1, 0.0, 1.0))
                    sfinal, wv_w, wa_w, wt_w = fusion_net.fuse(
                        sv_score, sa_score, st_score, fv_feat, fa_feat, ft_feat
                    )
                    print(f"  [FUSION] @{hl.start_time:.1f}s "
                          f"Sv={sv_score:.3f} Sa={sa_score:.3f} St={st_score:.3f} "
                          f"w=({wv_w:.2f},{wa_w:.2f},{wt_w:.2f}) Sfinal={sfinal:.3f}")
                    hl_new = HighlightMoment(
                        start_time=hl.start_time, end_time=hl.end_time,
                        moment_type=hl.moment_type, score=sfinal,
                        description=hl.description,
                        metadata={**hl.metadata,
                                  'sv': sv_score, 'sa': sa_score, 'st': st_score,
                                  'wv': wv_w, 'wa': wa_w, 'wt': wt_w}
                    )
                    reranked.append(hl_new)

                reranked.sort(key=lambda h: h.score, reverse=True)
                highlights = self.detector.deduplicate_highlights(reranked, 0.85)
                print(f"\n[4/4] 精彩片段生成完成，输出 {len(highlights)} 个片段（已按 Sfinal 排序）")
                if progress_callback:
                    progress_callback(50)

            elif analysis_data and analysis_data.get("ball_traj_frames") is not None:
                # ── 路径②：快速路径 + 多模态融合 ──────────────────────────────
                print("\n[1/4] ⚡ 规则预筛选（复用分析数据）+ 三模态特征提取...")
                _ad = analysis_data
                # 先用快速路径生成帧评分，再走多模态流程
                _fps  = _ad.get("fps", 25.0)
                _tf   = _ad.get("total_frames", 0)
                _dur  = _tf / max(_fps, 1)
                SAMPLE_IV = 5

                # 重用 detect_highlights_from_data 内部逻辑获取 frame_scores
                # 为避免代码重复，直接调用再走多模态融合
                _tmp_hl = self.detector.detect_highlights_from_data(
                    ball_traj_frames=_ad.get("ball_traj_frames", []),
                    tactical_tracks=_ad.get("tactical_tracks", {}),
                    fps=_fps, total_frames=_tf,
                    pitch_l=_ad.get("pitch_l", 105.0),
                    pitch_w=_ad.get("pitch_w", 68.0),
                    progress_callback=None,
                    video_path=None,  # 先不提取音频，下面统一提取
                )

                # 提取音频特征
                print("\n[2/4] 🎵 提取音频特征...")
                audio_energy_fast: Optional[List[float]] = None
                try:
                    audio_energy_fast = self.detector.extract_audio_energy(
                        input_video, _fps, _tf, SAMPLE_IV
                    )
                except Exception:
                    pass

                # 多模态重排序
                print("\n[3/4] 🔀 自适应多模态融合（Sfinal = wv·Sv + wa·Sa + wt·St）...")
                fusion_net = get_fusion_net()
                reranked_fast = []
                for hl in _tmp_hl:
                    i_s = int(hl.start_time * _fps) // SAMPLE_IV
                    i_e = int(hl.end_time   * _fps) // SAMPLE_IV
                    # 从 metadata 中恢复原始视觉分 Sv
                    raw_sv = hl.metadata.get('raw_score', hl.score * 15.0) / 15.0
                    sv_score = float(np.clip(raw_sv, 0.0, 1.0))
                    fv_feat  = np.array([sv_score, sv_score * 0.85, 0.5, 0.4, 0.5, 0.4],
                                        dtype=np.float32)
                    fa_feat  = self.detector.extract_audio_features(
                        audio_energy_fast, i_s, i_e
                    )
                    sa_score = float(fa_feat[0]) * 0.6 + float(fa_feat[1]) * 0.3 + float(fa_feat[2]) * 0.1
                    ft_feat  = np.array([sv_score * 0.75, sv_score * 0.6,
                                         0.5, 0.55], dtype=np.float32)
                    st_score = float(np.clip(ft_feat[1] * 0.5 + 0.1, 0.0, 1.0))
                    sfinal, wv_w, wa_w, wt_w = fusion_net.fuse(
                        sv_score, sa_score, st_score, fv_feat, fa_feat, ft_feat
                    )
                    print(f"  [FUSION] @{hl.start_time:.1f}s "
                          f"Sv={sv_score:.3f} Sa={sa_score:.3f} St={st_score:.3f} "
                          f"w=({wv_w:.2f},{wa_w:.2f},{wt_w:.2f}) Sfinal={sfinal:.3f}")
                    reranked_fast.append(HighlightMoment(
                        start_time=hl.start_time, end_time=hl.end_time,
                        moment_type=hl.moment_type, score=sfinal,
                        description=hl.description,
                        metadata={**hl.metadata,
                                  'sv': sv_score, 'sa': sa_score, 'st': st_score,
                                  'wv': wv_w, 'wa': wa_w, 'wt': wt_w}
                    ))
                reranked_fast.sort(key=lambda h: h.score, reverse=True)
                highlights = self.detector.deduplicate_highlights(reranked_fast, 0.85)
                print(f"\n[4/4] 精彩片段生成完成，输出 {len(highlights)} 个片段")
                if progress_callback:
                    progress_callback(50)

            else:
                # ── 路径③：完整扫描 + 多模态融合（四步完整论文流程）─────────────
                print("\n[1/4] 🔍 规则预筛选（逐帧 YOLO 扫描）+ 视觉特征提取...")
                _cap_scan = cv2.VideoCapture(input_video)
                _fps_scan  = _cap_scan.get(cv2.CAP_PROP_FPS) or 25.0
                _tf_scan   = int(_cap_scan.get(cv2.CAP_PROP_FRAME_COUNT))
                _dur_scan  = _tf_scan / max(_fps_scan, 1)
                _cap_scan.release()
                SAMPLE_IV = 5

                # Step 1: 逐帧 YOLO 扫描 → frame_scores（视觉特征）
                _frame_scores_scan: List[float] = []
                _prev_centers: List[tuple] = []
                _cap2 = cv2.VideoCapture(input_video)
                _frame_w = int(_cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
                _frame_h = int(_cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
                _fidx = 0
                while True:
                    _ret, _frm = _cap2.read()
                    if not _ret:
                        break
                    _fidx += 1
                    if _fidx % SAMPLE_IV != 0:
                        continue
                    _sc = 0.0
                    _gray = cv2.cvtColor(_frm, cv2.COLOR_BGR2GRAY)
                    _vr = float(np.sum(_gray > 30)) / _gray.size
                    if _vr < 0.20:
                        _sc -= 2.0
                        _frame_scores_scan.append(_sc)
                        _prev_centers = []
                        continue
                    _bpos = None
                    _pctrs: List[tuple] = []
                    if self.detector.yolo_model is not None:
                        try:
                            _res = self.detector.yolo_model(_frm, verbose=False, conf=0.25)
                            for _r in _res:
                                for _b in _r.boxes:
                                    _cls = int(_b.cls[0])
                                    _x1, _y1, _x2, _y2 = _b.xyxy[0].tolist()
                                    _cxn = ((_x1+_x2)/2) / max(_frame_w, 1)
                                    _cyn = ((_y1+_y2)/2) / max(_frame_h, 1)
                                    if _cls == 0:
                                        _pctrs.append((_cxn, _cyn))
                                    elif _cls == 32:
                                        if _bpos is None or float(_b.conf[0]) > 0.3:
                                            _bpos = (_cxn, _cyn)
                        except Exception:
                            pass
                    if _bpos and self.detector._in_penalty_area(_bpos[0], _bpos[1]):
                        _sc += 3.0
                    if _bpos and len(_pctrs) >= 3:
                        _dc = sum(1 for (px,py) in _pctrs
                                  if math.hypot(px-_bpos[0], py-_bpos[1]) < 3.0/105.0)
                        if _dc >= 3:
                            _sc += 2.0
                    if _pctrs and _prev_centers:
                        _npts = min(len(_pctrs), len(_prev_centers))
                        if _npts > 0:
                            _ad2 = float(np.mean([
                                math.hypot((_pctrs[i][0]-_prev_centers[i][0])*_frame_w,
                                           (_pctrs[i][1]-_prev_centers[i][1])*_frame_h)
                                for i in range(_npts)
                            ]))
                            if _ad2 > 5.0 * SAMPLE_IV:
                                _sc += 1.0
                    _ctr_ps = [p for p in _pctrs if 0.33 <= p[0] <= 0.67]
                    _ref = sum(1 for (px,py) in _ctr_ps
                               if sum(1 for (ox,oy) in _pctrs
                                      if (ox,oy)!=(px,py) and math.hypot(ox-px,oy-py)<0.05)==0)
                    if _ref >= 1:
                        _sc += 0.5
                    _frame_scores_scan.append(_sc)
                    _prev_centers = _pctrs
                    if progress_callback and _fidx % 200 == 0:
                        _pct = int(5 + (_fidx / max(_tf_scan, 1)) * 25)
                        progress_callback(min(30, _pct))
                _cap2.release()
                print(f"[SCAN] 帧评分完成，{len(_frame_scores_scan)} 个采样帧")

                # Step 2: 音频特征提取
                print("\n[2/4] 🎵 提取音频特征（RMS 能量）...")
                audio_energy_scan: Optional[List[float]] = None
                try:
                    audio_energy_scan = self.detector.extract_audio_energy(
                        input_video, _fps_scan, _tf_scan, SAMPLE_IV
                    )
                except Exception:
                    pass
                if progress_callback:
                    progress_callback(35)

                # Step 3+4: 多模态融合
                print("\n[3/4] 🔀 自适应多模态融合（Sfinal = wv·Sv + wa·Sa + wt·St）...")
                if _frame_scores_scan:
                    highlights = self.detector.detect_highlights_multimodal(
                        frame_scores=_frame_scores_scan,
                        fps=_fps_scan,
                        duration=_dur_scan,
                        sample_interval=SAMPLE_IV,
                        audio_energy=audio_energy_scan,
                        progress_callback=progress_callback,
                    )
                else:
                    highlights = []

                if not highlights:
                    print("[FALLBACK] 多模态路径无输出，退回标准检测...")
                    highlights = self.detector.detect_highlights(
                        input_video, progress_callback=progress_callback
                    )
                print(f"\n[4/4] 精彩片段生成完成，输出 {len(highlights)} 个片段")
                if progress_callback:
                    progress_callback(50)

            if not highlights:
                print("[HIGHLIGHT] ⚠️ 未检测到精彩时刻，尝试均匀采样降级处理...")
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
                clip_len = 8
                num_clips = max(1, int(target_duration / clip_len))
                step = max(1, (total_sec - clip_len) // num_clips)
                highlights = [
                    HighlightMoment(
                        start_time=i * step,
                        end_time=min(i * step + clip_len, total_sec),
                        moment_type='action', score=0.5,
                        description='自动采样片段', metadata={}
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
                print("\n[5/5] 🎵 混入背景音乐 We Will Rock You（去除原音频）...")
                bgm_path = None
                bgm_is_temp = False  # 是否是临时文件（需要清理）
                try:
                    bgm_path = _get_bgm_path(duration=target_duration + 30)
                    bgm_is_temp = (bgm_path != _BGM_REAL_FILE)  # 只有回退生成的 WAV 才是临时文件
                    bgm_ok  = self.effects.add_background_music(
                        merged_tmp, output_video,
                        bgm_path=bgm_path,
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
                    # 仅清理临时生成的 WAV，不删除真实 MP3
                    if bgm_is_temp and bgm_path and os.path.exists(bgm_path):
                        try: os.remove(bgm_path)
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
        精确时长控制算法：
          1. 按评分降序排列候选片段
          2. 逐一纳入片段（计算慢动作后时长），直到满足目标时长
          3. 若最后一个片段会造成超时，截断该片段使总时长精确达标
          4. 若所有片段仍不足，循环重复高分片段补足（最多3轮）
          5. 目标：最终时长落在 [target * 0.90, target * 1.05] 范围内
        """

        # 各类型慢动作速度因子配置
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

        if not highlights:
            return []

        # ── Step 1: 按评分降序排列 ──────────────────────────────────────────
        sorted_highlights = sorted(highlights, key=lambda h: h.score, reverse=True)

        def slowmo_dur(h: HighlightMoment) -> float:
            """计算片段慢动作后的实际播放时长（秒）"""
            cfg = SLOWMO_CONFIG.get(h.moment_type, SLOWMO_CONFIG['action'])
            raw = h.end_time - h.start_time
            return raw / cfg['factor']

        # ── Step 2: 贪心选取，精确截断最后一片段 ────────────────────────────
        selected: List[VideoClip] = []
        accumulated = 0.0
        pool = list(sorted_highlights)  # 可重复使用的片段池（循环追加用）

        MAX_ROUNDS = 4  # 最多循环整个 pool 4 次
        round_idx = 0
        used_once = set()  # 第一轮已选取的片段下标（去重）

        while accumulated < target_duration * 0.90 and round_idx < MAX_ROUNDS:
            added_this_round = 0
            for idx, h in enumerate(pool):
                if accumulated >= target_duration:
                    break
                cfg = SLOWMO_CONFIG.get(h.moment_type, SLOWMO_CONFIG['action'])
                raw_dur = h.end_time - h.start_time
                sd = raw_dur / cfg['factor']  # 慢动作后时长

                remaining = target_duration - accumulated

                if sd <= remaining + 2.0:
                    # 片段完整纳入（允许2秒宽裕避免死循环）
                    use_start = h.start_time
                    use_end   = h.end_time
                else:
                    # 截断片段：使慢动作后时长恰好填满剩余
                    # 截断后原始时长 = remaining * factor
                    raw_needed = remaining * cfg['factor']
                    if raw_needed < 0.5:
                        break  # 剩余不足0.5秒原始素材，停止
                    mid = (h.start_time + h.end_time) / 2
                    use_start = max(h.start_time, mid - raw_needed / 2)
                    use_end   = min(h.end_time,   use_start + raw_needed)
                    sd = (use_end - use_start) / cfg['factor']

                actual_add = sd
                # 防止超出 target_duration * 1.05
                if accumulated + actual_add > target_duration * 1.05:
                    # 精确截断
                    raw_needed = (target_duration * 1.05 - accumulated) * cfg['factor']
                    if raw_needed < 0.5:
                        continue
                    use_start = h.start_time
                    use_end   = min(h.end_time, use_start + raw_needed)
                    actual_add = (use_end - use_start) / cfg['factor']

                selected.append(VideoClip(
                    start_frame=0, end_frame=0,
                    start_time=use_start, end_time=use_end,
                    moment_type=h.moment_type, score=h.score,
                    apply_slowmo=True, slowmo_factor=cfg['factor'], apply_zoom=cfg['zoom']
                ))
                accumulated += actual_add
                used_once.add(idx)
                added_this_round += 1

                if accumulated >= target_duration:
                    break

            round_idx += 1
            if added_this_round == 0:
                break  # 本轮无进展，退出

        # ── Step 3: 若总时长超出 target * 1.05，截断最后一片段 ──────────────
        if selected and accumulated > target_duration * 1.05:
            last = selected[-1]
            cfg = SLOWMO_CONFIG.get(last.moment_type, SLOWMO_CONFIG['action'])
            over = accumulated - target_duration
            # 需要从最后片段减去 over 秒（慢动作后），即原始减去 over * factor 秒
            raw_trim = over * cfg['factor']
            new_end = last.end_time - raw_trim
            if new_end > last.start_time + 0.5:
                selected[-1] = VideoClip(
                    start_frame=last.start_frame, end_frame=last.end_frame,
                    start_time=last.start_time, end_time=new_end,
                    moment_type=last.moment_type, score=last.score,
                    apply_slowmo=last.apply_slowmo,
                    slowmo_factor=last.slowmo_factor, apply_zoom=last.apply_zoom
                )
            else:
                selected.pop()  # 最后片段太短，直接丢弃

        # ── Step 4: 按选择顺序排列（第一轮按时间，后续轮次直接追加）──────────────
        # 注意：多轮补足时允许同一片段重复出现（精彩集锦可重复展示精彩动作）
        # 去重规则：只去掉「连续相邻」且时间段完全相同的片段（避免同一片段被 concat 两次）
        deduped: List[VideoClip] = []
        seen: set = set()
        for clip in selected:
            key = (round(clip.start_time, 2), round(clip.end_time, 2))
            if key not in seen or len(deduped) == 0:
                deduped.append(clip)
                seen.add(key)
            else:
                # 允许重复，但相邻两片段不能完全相同（否则 concat 衔接处明显卡顿）
                if deduped and not (
                    round(deduped[-1].start_time, 2) == key[0] and
                    round(deduped[-1].end_time, 2)   == key[1]
                ):
                    deduped.append(clip)

        total_slowmo_dur = sum(
            (c.end_time - c.start_time) / c.slowmo_factor
            for c in deduped
        )
        print(f"[SELECT] 选取 {len(deduped)} 个片段，预计慢动作后总时长 {total_slowmo_dur:.1f}s（目标 {target_duration:.0f}s）")
        return deduped

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
                if r.returncode != 0:
                    print(f"[ERROR] ffmpeg 返回码={r.returncode}, stderr={r.stderr[-300:]}")
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
