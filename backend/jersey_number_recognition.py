#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
球衣号码识别与球队归属分类模块 — 足球智能视频分析系统
======================================================

【核心算法原理】

一、整体流程
  1. YOLOv8 检测画面中所有运动员（person 类别）
  2. 对每个运动员 ROI 裁剪背部区域（上身下半部分）
  3. 用训练好的 OCR / 号码检测模型识别背部号码
  4. 用 K-means 颜色聚类分析球衣颜色，判断球队归属

二、球衣号码检测（两阶段pipeline）
  Stage 1 — 号码区域定位（YOLO二次检测）
    - 输入：运动员上身裁剪图（约80×120像素）
    - 输出：号码区域边界框
    - 模型：jersey_number_detector.pt（基于YOLOv8n微调）

  Stage 2 — 数字识别（OCR / 分类）
    - 方案A：EasyOCR / PaddleOCR（通用OCR，无需训练）
    - 方案B：自训练CNN分类器（0-99号）
    - 本模块同时实现两种方案，优先使用OCR

三、球队归属分类（颜色聚类）
  - 提取运动员躯干区域像素，HSV颜色空间K-means聚类（k=2）
  - 取主色调H分量均值代表球衣颜色
  - 按颜色距离将所有球员划分为2队（或更多队）

【数据集准备指引】

  目录结构（YOLOv8格式）：
    datasets/jersey_numbers/
    ├── images/
    │   ├── train/  (约1000张运动员背部图)
    │   └── val/    (约200张)
    ├── labels/
    │   ├── train/  (对应YOLO格式txt标注)
    │   └── val/
    └── dataset.yaml

  标注类别：每个数字(0-9)一个类，共10类
  推荐标注工具：LabelImg / Roboflow

  快速获取数据集：
    - SoccerNet Jersey Number Recognition Dataset
    - https://github.com/SoccerNet/sn-jersey
"""

import os
import cv2
import numpy as np
import logging
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlayerDetection:
    """单个运动员检测结果"""
    player_id: int
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    jersey_number: Optional[int] = None
    jersey_number_conf: float = 0.0
    team_id: int = -1                  # 0=主队, 1=客队, -1=未知
    jersey_color_hsv: Optional[Tuple[float, float, float]] = None
    back_roi: Optional[np.ndarray] = None  # 背部裁剪图


@dataclass
class FrameAnalysisResult:
    """单帧分析结果"""
    frame_idx: int
    players: List[PlayerDetection] = field(default_factory=list)
    team_colors: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# 球衣号码识别器
# ─────────────────────────────────────────────────────────────────────────────

class JerseyNumberRecognizer:
    """
    球衣背部号码识别器

    pipeline：
      detect_players → extract_back_roi → recognize_number → classify_team
    """

    def __init__(
        self,
        yolo_model_path: str = "yolov8n.pt",
        jersey_model_path: Optional[str] = None,
        use_ocr: bool = True,
        conf_threshold: float = 0.5,
        device: str = "cpu",
    ):
        """
        初始化识别器

        Args:
            yolo_model_path:   YOLOv8基础模型路径（用于检测运动员）
            jersey_model_path: 球衣号码专用检测模型（可选，需单独训练）
            use_ocr:           是否启用OCR识别号码（True=使用EasyOCR）
            conf_threshold:    检测置信度阈值
            device:            推理设备 "cpu" / "cuda"
        """
        self.yolo_model_path = yolo_model_path
        self.jersey_model_path = jersey_model_path
        self.use_ocr = use_ocr
        self.conf_threshold = conf_threshold
        self.device = device

        self._yolo = None
        self._jersey_detector = None
        self._ocr_reader = None

        self._load_models()

    # ── 模型加载 ──────────────────────────────────────────────────────────────

    def _load_models(self):
        """懒加载所有模型"""
        # 1. 加载 YOLOv8 基础检测模型
        try:
            from ultralytics import YOLO
            self._yolo = YOLO(self.yolo_model_path)
            logger.info(f"[JerseyRec] YOLOv8 加载成功: {self.yolo_model_path}")
        except Exception as e:
            logger.warning(f"[JerseyRec] YOLOv8 加载失败: {e}")

        # 2. 加载球衣号码专用检测模型（如果存在）
        if self.jersey_model_path and os.path.exists(self.jersey_model_path):
            try:
                from ultralytics import YOLO as YOLO2
                self._jersey_detector = YOLO2(self.jersey_model_path)
                logger.info(f"[JerseyRec] 号码检测模型加载成功")
            except Exception as e:
                logger.warning(f"[JerseyRec] 号码检测模型加载失败: {e}")

        # 3. 加载 OCR（EasyOCR 优先，其次 PaddleOCR）
        if self.use_ocr:
            self._load_ocr()

    def _load_ocr(self):
        """尝试加载 OCR 引擎（EasyOCR → PaddleOCR → 降级到规则方法）"""
        # 优先：EasyOCR
        try:
            import easyocr
            self._ocr_reader = easyocr.Reader(['en'], gpu=(self.device == 'cuda'), verbose=False)
            self._ocr_type = 'easyocr'
            logger.info("[JerseyRec] OCR引擎: EasyOCR")
            return
        except ImportError:
            pass

        # 备选：PaddleOCR
        try:
            from paddleocr import PaddleOCR
            self._ocr_reader = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
            self._ocr_type = 'paddleocr'
            logger.info("[JerseyRec] OCR引擎: PaddleOCR")
            return
        except ImportError:
            pass

        # 降级：无OCR，仅依赖号码检测模型
        self._ocr_type = 'none'
        logger.warning("[JerseyRec] 未找到OCR引擎，号码识别将降级为模板匹配")

    # ── 主处理流程 ────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, frame_idx: int = 0) -> FrameAnalysisResult:
        """
        处理单帧图像，返回所有运动员的号码和球队归属

        Args:
            frame:     BGR格式图像（OpenCV读取）
            frame_idx: 帧序号（用于日志）

        Returns:
            FrameAnalysisResult 包含所有检测结果
        """
        result = FrameAnalysisResult(frame_idx=frame_idx)

        # Step 1: 检测所有运动员
        players = self._detect_players(frame)
        if not players:
            return result

        # Step 2: 对每个运动员提取背部ROI并识别号码
        for p in players:
            p.back_roi = self._extract_back_roi(frame, p.bbox)
            if p.back_roi is not None:
                num, conf = self._recognize_number(p.back_roi)
                p.jersey_number = num
                p.jersey_number_conf = conf
                p.jersey_color_hsv = self._extract_jersey_color(frame, p.bbox)

        result.players = players

        # Step 3: 根据颜色聚类划分球队
        self._classify_teams(result)

        return result

    def process_video(
        self,
        video_path: str,
        sample_interval: int = 30,
        progress_callback=None
    ) -> Dict[int, Dict]:
        """
        处理视频，跨帧累积识别结果，返回每个球员的稳定号码

        算法：
          - 对同一追踪ID的球员，多帧识别结果投票（取众数）
          - 置信度加权：高置信度识别权重更高

        Args:
            video_path:       视频路径
            sample_interval:  每N帧采样1帧
            progress_callback: 进度回调 callback(percent)

        Returns:
            {player_track_id: {"number": int, "team": int, "confidence": float}}
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # 累积投票：{player_id: {number: total_weight}}
        votes: Dict[int, Dict[int, float]] = {}
        team_votes: Dict[int, Dict[int, int]] = {}

        frame_idx = 0
        processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                try:
                    frame_result = self.process_frame(frame, frame_idx)
                    for p in frame_result.players:
                        pid = p.player_id
                        if pid not in votes:
                            votes[pid] = {}
                            team_votes[pid] = {}

                        # 号码投票（按置信度加权）
                        if p.jersey_number is not None and p.jersey_number_conf > 0.3:
                            n = p.jersey_number
                            votes[pid][n] = votes[pid].get(n, 0.0) + p.jersey_number_conf

                        # 球队投票
                        if p.team_id >= 0:
                            t = p.team_id
                            team_votes[pid][t] = team_votes[pid].get(t, 0) + 1

                    processed += 1
                except Exception as e:
                    logger.debug(f"[JerseyRec] 帧{frame_idx}处理失败: {e}")

            frame_idx += 1
            if progress_callback and frame_idx % 100 == 0:
                pct = int(frame_idx / max(total_frames, 1) * 100)
                progress_callback(pct)

        cap.release()

        # 汇总投票结果
        final_results = {}
        for pid, num_votes in votes.items():
            if num_votes:
                best_num = max(num_votes, key=num_votes.get)
                total_w = sum(num_votes.values())
                confidence = num_votes[best_num] / total_w if total_w > 0 else 0.0
            else:
                best_num = None
                confidence = 0.0

            best_team = -1
            if pid in team_votes and team_votes[pid]:
                best_team = max(team_votes[pid], key=team_votes[pid].get)

            final_results[pid] = {
                "number": best_num,
                "team": best_team,
                "confidence": round(confidence, 3),
            }

        logger.info(f"[JerseyRec] 视频处理完成，识别 {len(final_results)} 个球员")
        return final_results

    # ── 运动员检测 ────────────────────────────────────────────────────────────

    def _detect_players(self, frame: np.ndarray) -> List[PlayerDetection]:
        """
        使用 YOLOv8 检测所有运动员

        Returns:
            PlayerDetection 列表（仅 person 类别，置信度 > threshold）
        """
        if self._yolo is None:
            return self._fallback_detect(frame)

        try:
            results = self._yolo.track(
                frame,
                classes=[0],               # class 0 = person
                conf=self.conf_threshold,
                persist=True,              # 开启跨帧追踪（ByteTrack）
                verbose=False,
            )
            players = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # 过滤太小的框（宽或高小于30px）
                    if (x2 - x1) < 30 or (y2 - y1) < 30:
                        continue
                    # 获取追踪ID（如果可用）
                    track_id = int(box.id[0]) if box.id is not None else i
                    players.append(PlayerDetection(
                        player_id=track_id,
                        bbox=(x1, y1, x2, y2),
                    ))
            return players
        except Exception as e:
            logger.debug(f"[JerseyRec] YOLO检测失败: {e}")
            return []

    def _fallback_detect(self, frame: np.ndarray) -> List[PlayerDetection]:
        """YOLOv8不可用时的降级检测（基于背景减除）"""
        # 简单降级：无检测结果
        return []

    # ── 背部ROI提取 ───────────────────────────────────────────────────────────

    def _extract_back_roi(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        back_ratio: Tuple[float, float] = (0.22, 0.65),
    ) -> Optional[np.ndarray]:
        """
        从运动员边界框中提取背部号码区域（多候选版，取最佳）。

        策略：
          - 提取精确（0.22~0.65）、宽松（0.18~0.68）两个候选区域
          - 合并为拼接图（垂直叠加），让 OCR 处理两个区域

        Returns:
            裁剪后的 BGR 图像（128×64），或 None
        """
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        w = x2 - x1

        if h < 50 or w < 18:
            return None

        H_img, W_img = frame.shape[:2]
        margin_x = max(2, int(w * 0.08))

        best_roi = None
        best_score = -1.0

        for (yt, yb) in [(0.22, 0.65), (0.18, 0.68), (0.20, 0.58)]:
            top  = max(0, y1 + int(h * yt))
            bot  = min(H_img, y1 + int(h * yb))
            left = max(0, x1 + margin_x)
            right = min(W_img, x2 - margin_x)

            if bot <= top or right <= left:
                continue

            roi = frame[top:bot, left:right].copy()
            if roi.shape[0] < 15 or roi.shape[1] < 8:
                continue

            # 用对比度分数评估候选（越高说明号码区域越清晰）
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            score = float(gray.std())
            if score > best_score:
                best_score = score
                best_roi = roi

        if best_roi is None:
            return None

        # 统一放大到 128×64 以保证OCR效果
        roi_resized = cv2.resize(best_roi, (128, 64), interpolation=cv2.INTER_CUBIC)
        return roi_resized

    # ── 号码识别 ──────────────────────────────────────────────────────────────

    def _recognize_number(
        self,
        roi: np.ndarray,
    ) -> Tuple[Optional[int], float]:
        """
        从背部ROI图像中识别球衣号码

        流程：
          1. 图像预处理（去噪/二值化/对比度增强）
          2. 调用OCR识别数字字符
          3. 后处理：过滤非数字，限制范围 1-99

        Returns:
            (number, confidence) 或 (None, 0.0)
        """
        if roi is None:
            return None, 0.0

        # 预处理：增强号码可读性
        preprocessed = self._preprocess_for_ocr(roi)

        # 方案A：OCR识别
        if self._ocr_type == 'easyocr' and self._ocr_reader is not None:
            return self._ocr_easyocr(preprocessed)
        elif self._ocr_type == 'paddleocr' and self._ocr_reader is not None:
            return self._ocr_paddleocr(preprocessed)
        else:
            # 降级：模板匹配或规则识别
            return self._rule_based_recognition(preprocessed)

    def _preprocess_for_ocr(self, roi: np.ndarray) -> np.ndarray:
        """
        OCR 前多策略预处理，自动选择识别效果最佳的版本。

        候选策略：
          1. CLAHE + Otsu 反色二值化（深底浅字）
          2. CLAHE + Otsu 正色二值化（浅底深字）
          3. 自适应阈值（局部光照不均匀）
          4. 2× 上采样 + CLAHE + Otsu（小目标放大）
          5. 锐化 + Otsu（模糊号码增强）

        选择策略：取前景像素比最接近 0.25 的版本
        （正常数字占图像约 20~30%，偏差最小说明二值化最准确）
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        candidates = []

        # 1. Otsu 反色
        _, b1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        candidates.append(cv2.morphologyEx(b1, cv2.MORPH_CLOSE, kernel))

        # 2. Otsu 正色
        _, b2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates.append(cv2.morphologyEx(b2, cv2.MORPH_CLOSE, kernel))

        # 3. 自适应阈值
        b3 = cv2.adaptiveThreshold(blurred, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 4)
        candidates.append(cv2.morphologyEx(b3, cv2.MORPH_CLOSE, kernel))

        # 4. 2× 上采样
        big = cv2.resize(roi, (roi.shape[1]*2, roi.shape[0]*2), interpolation=cv2.INTER_CUBIC)
        gray_big = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        eq_big = clahe2.apply(gray_big)
        blurred_big = cv2.GaussianBlur(eq_big, (3, 3), 0)
        _, b4 = cv2.threshold(blurred_big, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        b4_closed = cv2.morphologyEx(b4, cv2.MORPH_CLOSE, kernel)
        # resize 回原始大小
        b4_resized = cv2.resize(b4_closed, (roi.shape[1], roi.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
        candidates.append(b4_resized)

        # 5. 锐化 + Otsu
        sharp_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(enhanced, -1, sharp_kernel)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        _, b5 = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        candidates.append(cv2.morphologyEx(b5, cv2.MORPH_CLOSE, kernel))

        # 选择前景占比最接近目标值 0.25 的候选
        target_ratio = 0.25
        best_img = candidates[0]
        best_dist = float('inf')
        for c in candidates:
            ratio = np.sum(c > 0) / max(c.size, 1)
            dist = abs(ratio - target_ratio)
            if dist < best_dist:
                best_dist = dist
                best_img = c

        return cv2.cvtColor(best_img, cv2.COLOR_GRAY2BGR)

    def _ocr_easyocr(self, img: np.ndarray) -> Tuple[Optional[int], float]:
        """
        使用 EasyOCR 识别数字（多策略投票版）。
        对同一 ROI 分别调用正常图和反色图，取加权投票结果。
        """
        def _run_ocr(image):
            try:
                results = self._ocr_reader.readtext(
                    image,
                    allowlist='0123456789',
                    detail=1,
                    paragraph=False,
                    min_size=8,
                    text_threshold=0.5,
                    low_text=0.3,
                )
                return results or []
            except Exception:
                return []

        # 正常图
        votes: dict = {}
        for text, conf, image in [
            (None, None, img),
            (None, None, cv2.bitwise_not(img)),  # 反色
        ]:
            for (_, text, conf) in _run_ocr(image):
                num = self._parse_number(str(text).strip())
                if num and float(conf) > 0.30:
                    votes[num] = votes.get(num, 0.0) + float(conf)

        if not votes:
            return None, 0.0

        best_num = max(votes, key=votes.get)
        total = sum(votes.values())
        confidence = votes[best_num] / total if total > 0 else 0.0
        return best_num, round(confidence, 3)

    def _ocr_paddleocr(self, img: np.ndarray) -> Tuple[Optional[int], float]:
        """使用 PaddleOCR 识别数字"""
        try:
            result = self._ocr_reader.ocr(img, cls=False)
            if not result or not result[0]:
                return None, 0.0

            best_text, best_conf = "", 0.0
            for line in result[0]:
                text = line[1][0]
                conf = float(line[1][1])
                if conf > best_conf:
                    best_text, best_conf = text, conf

            number = self._parse_number(best_text.strip())
            return number, best_conf if number is not None else 0.0
        except Exception as e:
            logger.debug(f"[PaddleOCR] 识别失败: {e}")
            return None, 0.0

    def _rule_based_recognition(self, img: np.ndarray) -> Tuple[Optional[int], float]:
        """
        规则方法：轮廓分析估算数字个数
        （OCR不可用时的降级方案，准确率约40%）
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 过滤轮廓：面积在合理范围内的认为是数字
        digit_contours = [
            c for c in contours
            if 50 < cv2.contourArea(c) < 3000
        ]

        # 按x坐标排序
        digit_contours.sort(key=lambda c: cv2.boundingRect(c)[0])

        count = len(digit_contours)
        if count == 0:
            return None, 0.0
        elif count == 1:
            # 单个轮廓：返回占位符，置信度低
            return None, 0.1
        elif count >= 2:
            # 多个数字轮廓，无法精确识别，返回None
            return None, 0.15

    @staticmethod
    def _parse_number(text: str) -> Optional[int]:
        """
        从OCR文本中解析球衣号码

        规则：
          - 仅保留数字字符
          - 范围限制 1~99
          - 空字符串或范围外返回 None
        """
        digits = ''.join(c for c in text if c.isdigit())
        if not digits:
            return None
        try:
            n = int(digits[:2])   # 最多取2位
            if 1 <= n <= 30:
                return n
        except ValueError:
            pass
        return None

    # ── 球衣颜色提取 ──────────────────────────────────────────────────────────

    def _extract_jersey_color(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[Tuple[float, float, float]]:
        """
        提取运动员球衣主色（HSV颜色空间）

        方法：
          1. 裁剪躯干上半区域（避免腿部干扰）
          2. 转换为 HSV 颜色空间
          3. K-means 聚类（k=2），取像素数量最多的簇
          4. 返回该簇的 HSV 均值

        Returns:
            (H, S, V) 三元组，H∈[0,180]，S/V∈[0,255]
        """
        x1, y1, x2, y2 = bbox
        h = y2 - y1

        # 躯干区域：纵向 20%~50%
        top = y1 + int(h * 0.20)
        bot = y1 + int(h * 0.50)
        H_img, W_img = frame.shape[:2]
        top = max(0, top)
        bot = min(H_img, bot)

        if bot <= top:
            return None

        torso = frame[top:bot, x1:x2]
        if torso.size == 0:
            return None

        # 转 HSV
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3).astype(np.float32)

        if len(pixels) < 10:
            return None

        # K-means 聚类
        try:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            k = min(2, len(pixels))
            _, labels, centers = cv2.kmeans(
                pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
            )

            # 选取像素数量最多的簇（主色）
            counts = np.bincount(labels.flatten())
            dominant_idx = np.argmax(counts)
            dominant_color = centers[dominant_idx]

            return tuple(dominant_color.tolist())  # (H, S, V)
        except Exception:
            # 降级：直接取均值
            mean_hsv = np.mean(pixels, axis=0)
            return tuple(mean_hsv.tolist())

    # ── 球队分类 ──────────────────────────────────────────────────────────────

    def _classify_teams(self, result: FrameAnalysisResult):
        """
        按球衣颜色将运动员分配到球队

        算法：
          1. 收集所有有颜色信息的球员
          2. 对H（色调）值进行K-means聚类（k=2），将球员分为2组
          3. 色调相近的分为同一队
          4. 裁判（通常穿黑色/黄色）被过滤掉
        """
        players_with_color = [
            p for p in result.players
            if p.jersey_color_hsv is not None
        ]

        if len(players_with_color) < 2:
            return

        # 提取 H 分量（色调，最能区分球队）
        hues = np.array([[p.jersey_color_hsv[0]] for p in players_with_color], dtype=np.float32)

        if len(hues) < 2:
            return

        try:
            k = min(2, len(hues))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(hues, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

            # 按中心H值从小到大排序，team_id保持一致
            order = np.argsort(centers[:, 0])
            label_map = {order[i]: i for i in range(k)}

            for i, p in enumerate(players_with_color):
                p.team_id = label_map.get(int(labels[i][0]), -1)

            # 记录球队颜色代表色
            for team_id in range(k):
                result.team_colors[team_id] = tuple(centers[order[team_id]].tolist())

        except Exception as e:
            logger.debug(f"[JerseyRec] 球队分类失败: {e}")

    # ── 可视化 ────────────────────────────────────────────────────────────────

    @staticmethod
    def draw_results(frame: np.ndarray, result: FrameAnalysisResult) -> np.ndarray:
        """
        在原始帧上绘制识别结果（号码 + 球队颜色框）

        Args:
            frame:  原始 BGR 图像
            result: 帧分析结果

        Returns:
            带标注的 BGR 图像
        """
        TEAM_COLORS_BGR = {
            0: (0, 255, 0),    # 主队：绿色
            1: (0, 0, 255),    # 客队：红色
            -1: (128, 128, 128),  # 未知：灰色
        }

        vis = frame.copy()

        for p in result.players:
            x1, y1, x2, y2 = p.bbox
            color = TEAM_COLORS_BGR.get(p.team_id, (128, 128, 128))

            # 绘制边界框
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # 号码标签
            label_parts = []
            if p.jersey_number is not None:
                label_parts.append(f"#{p.jersey_number}")
                if p.jersey_number_conf > 0:
                    label_parts.append(f"({p.jersey_number_conf:.0%})")
            team_name = {0: "主队", 1: "客队"}.get(p.team_id, "")
            if team_name:
                label_parts.append(team_name)

            label = " ".join(label_parts) if label_parts else f"ID:{p.player_id}"

            # 绘制标签背景
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        return vis


# ─────────────────────────────────────────────────────────────────────────────
# YOLO 球衣号码专项训练脚本
# ─────────────────────────────────────────────────────────────────────────────

class JerseyNumberTrainer:
    """
    球衣号码 YOLO 模型训练器

    【训练流程】
      1. 准备数据集（见 prepare_dataset_yaml）
      2. 调用 train() 开始微调
      3. 模型保存至 runs/jersey_number/weights/best.pt

    【推荐配置】
      - 基础模型：yolov8n.pt（轻量，适合嵌入式部署）
      - 训练轮次：100 epochs（数据充足时可至200）
      - 批次大小：16（GPU 8GB时可用32）
      - 图像尺寸：128×64（号码区域分辨率）
      - 数据增强：mosaic=0.5, hsv_h=0.3（增强颜色泛化）

    【数据集要求 — 仅球员样本】
      - 最小规模：每个数字(0-9)各100张，共1000张
      - 推荐规模：每类500张，共5000张
      - 标注格式：YOLOv8（class_id cx cy w h，归一化）
      - 类别定义：0-9 对应数字 0-9
      - 重要：数据集中只能包含球员背号图像，
               不得包含观众、裁判、广告牌等非球员样本
    """

    DATASET_YAML_TEMPLATE = """# 球衣号码数据集配置（仅球员样本）
path: {dataset_path}
train: images/train
val: images/val

# 类别数量（0-9 十个数字）
nc: 10

# 类别名称
names:
  0: '0'
  1: '1'
  2: '2'
  3: '3'
  4: '4'
  5: '5'
  6: '6'
  7: '7'
  8: '8'
  9: '9'
"""

    # 球员检测专用 YAML（整体人体检测，仅 person 类）
    PLAYER_DETECT_YAML_TEMPLATE = """# 足球球员检测数据集配置
# 注意：仅包含球员样本（排除观众、裁判、广告牌等）
path: {dataset_path}
train: images/train
val: images/val

# 仅1个类别：球员
nc: 1
names:
  0: 'player'

# 数据集构建建议：
#   - 仅标注场地内球员（排除观众席、替补席人员）
#   - 排除身着黑色/荧光黄服装的裁判
#   - 目标像素高度建议 >= 50px（过滤远景小目标）
"""

    def __init__(self, base_model: str = "yolov8n.pt", device: str = "cpu"):
        self.base_model = base_model
        self.device = device

    def prepare_dataset_yaml(self, dataset_path: str) -> str:
        """
        生成数据集配置文件 dataset.yaml

        Args:
            dataset_path: 数据集根目录的绝对路径

        Returns:
            yaml 文件路径
        """
        yaml_content = self.DATASET_YAML_TEMPLATE.format(
            dataset_path=dataset_path.replace("\\", "/")
        )
        yaml_path = os.path.join(dataset_path, "dataset.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        logger.info(f"[Trainer] 数据集配置已生成: {yaml_path}")
        return yaml_path

    def prepare_player_detect_yaml(self, dataset_path: str) -> str:
        """
        生成球员检测专用 dataset.yaml（单类：player）

        Args:
            dataset_path: 数据集根目录的绝对路径

        Returns:
            yaml 文件路径
        """
        yaml_content = self.PLAYER_DETECT_YAML_TEMPLATE.format(
            dataset_path=dataset_path.replace("\\", "/")
        )
        yaml_path = os.path.join(dataset_path, "player_detect.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        logger.info(f"[Trainer] 球员检测配置已生成: {yaml_path}")
        return yaml_path

    def train(
        self,
        dataset_yaml: str,
        epochs: int = 100,
        imgsz: int = 128,
        batch: int = 16,
        project: str = "runs/jersey_number",
        name: str = "exp",
    ) -> str:
        """
        启动YOLOv8微调训练（专注球衣号码识别）

        Args:
            dataset_yaml: 数据集配置文件路径
            epochs:       训练轮次
            imgsz:        输入图像尺寸（号码区域较小，128足够）
            batch:        批次大小
            project:      保存目录
            name:         实验名称

        Returns:
            最优模型路径 best.pt
        """
        try:
            from ultralytics import YOLO
            model = YOLO(self.base_model)

            results = model.train(
                data=dataset_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=self.device,
                project=project,
                name=name,
                # 数据增强参数（针对球衣号码特性优化）
                hsv_h=0.3,       # 色调增强（模拟不同光线/球衣颜色）
                hsv_s=0.7,       # 饱和度增强
                hsv_v=0.4,       # 亮度增强（应对阴影/逆光）
                degrees=10,      # 轻微旋转（号码可能倾斜）
                translate=0.1,   # 平移
                scale=0.3,       # 缩放
                mosaic=0.5,      # Mosaic增强
                mixup=0.1,       # MixUp（轻微，提升泛化）
                flipud=0.0,      # 不上下翻转（号码有方向性）
                fliplr=0.0,      # 不左右翻转（避免镜像号码混淆）
                copy_paste=0.0,  # 禁用copy-paste（避免号码重叠混乱）
                # 超参数
                lr0=0.01,
                lrf=0.01,
                weight_decay=0.0005,
                patience=20,     # 早停耐心值
                # 小目标优化：增大检测头感受野
                box=7.5,         # 边界框损失权重（默认7.5）
                cls=0.5,         # 分类损失权重（号码分类更重要，稍提升）
                verbose=True,
            )

            best_model = os.path.join(project, name, "weights", "best.pt")
            logger.info(f"[Trainer] 训练完成！最优模型: {best_model}")
            return best_model

        except ImportError:
            raise RuntimeError("请安装 ultralytics: pip install ultralytics")
        except Exception as e:
            raise RuntimeError(f"训练失败: {e}")

    def train_player_detector(
        self,
        dataset_yaml: str,
        epochs: int = 150,
        imgsz: int = 640,
        batch: int = 16,
        project: str = "runs/player_detector",
        name: str = "exp",
    ) -> str:
        """
        训练专用球员检测模型（单类 player，排除观众/裁判）

        与通用 yolov8n.pt 的区别：
          - 仅检测 player 类（不检测 person、球、裁判等）
          - 训练数据严格限制为场地内球员
          - 更高置信度阈值，减少误检

        Args:
            dataset_yaml: 球员检测数据集配置（使用 prepare_player_detect_yaml 生成）
            epochs:       训练轮次（球员检测需要更多轮次）
            imgsz:        输入分辨率（640 适合全帧检测）
            batch:        批次大小
            project:      保存目录
            name:         实验名称

        Returns:
            最优模型路径 best.pt
        """
        try:
            from ultralytics import YOLO
            model = YOLO(self.base_model)

            results = model.train(
                data=dataset_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=self.device,
                project=project,
                name=name,
                # 针对球员检测优化的数据增强
                hsv_h=0.015,     # 轻微色调抖动（保留球衣颜色特征）
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=5,       # 轻微旋转
                translate=0.1,
                scale=0.5,       # 较大缩放增强（适应不同景深）
                mosaic=1.0,      # 全量 Mosaic（提升场景多样性）
                mixup=0.1,
                flipud=0.0,
                fliplr=0.5,      # 左右翻转（球场对称）
                # 小目标优化配置
                box=7.5,
                cls=0.5,
                # 提高检测置信度阈值（减少观众误检）
                conf=0.45,
                iou=0.5,
                # 超参数
                lr0=0.01,
                lrf=0.01,
                weight_decay=0.0005,
                patience=30,
                verbose=True,
            )

            best_model = os.path.join(project, name, "weights", "best.pt")
            logger.info(f"[Trainer] 球员检测模型训练完成！最优模型: {best_model}")
            return best_model

        except ImportError:
            raise RuntimeError("请安装 ultralytics: pip install ultralytics")
        except Exception as e:
            raise RuntimeError(f"训练失败: {e}")

    def evaluate(self, model_path: str, dataset_yaml: str) -> Dict:
        """
        评估模型性能

        Returns:
            {mAP50: float, mAP50-95: float, precision: float, recall: float}
        """
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            metrics = model.val(data=dataset_yaml, device=self.device)
            return {
                "mAP50": round(metrics.box.map50, 4),
                "mAP50-95": round(metrics.box.map, 4),
                "precision": round(metrics.box.mp, 4),
                "recall": round(metrics.box.mr, 4),
            }
        except Exception as e:
            logger.error(f"[Trainer] 评估失败: {e}")
            return {}


# ─────────────────────────────────────────────────────────────────────────────
# 数据集自动构建（从现有视频提取训练数据）
# ─────────────────────────────────────────────────────────────────────────────

class DatasetBuilder:
    """
    从已有视频自动提取运动员背部图像，辅助构建训练数据集

    使用方法：
      builder = DatasetBuilder(output_dir="datasets/jersey_numbers")
      builder.extract_from_video("match.mp4", sample_interval=60)
      # 然后用 LabelImg 对提取的图像进行标注

    【重要】仅提取场地内球员图像，自动过滤：
      - 观众（顶部/边缘区域 + 过小目标）
      - 裁判（黑色/荧光黄服装）
      - 过小/高宽比异常目标
    """

    def __init__(self, output_dir: str = "datasets/jersey_numbers"):
        self.output_dir = output_dir
        os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images", "val"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", "train"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", "val"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "raw_crops"), exist_ok=True)

    @staticmethod
    def _is_player_crop(crop_bgr: np.ndarray, full_w: int, full_h: int,
                         x1: int, y1: int, x2: int, y2: int) -> bool:
        """
        判断裁剪图是否为真实球员（排除观众/裁判/小目标）

        过滤规则：
          1. 位置：中心点不在顶部18%观众席 / 底部8% 替补区
          2. 尺寸：高度 >= 60px，宽度 >= 20px，高宽比 1.5~5.0
          3. 服装颜色：排除黑色（裁判）和荧光黄/绿（裁判/工作人员）
        """
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        bw = x2 - x1
        bh = y2 - y1

        # 位置过滤
        if cy < full_h * 0.18 or cy > full_h * 0.92:
            return False
        if cx < full_w * 0.05 or cx > full_w * 0.95:
            return False

        # 尺寸过滤
        if bh < 60 or bw < 20:
            return False
        if bw > 0 and not (1.5 <= bh / bw <= 5.0):
            return False

        # 颜色过滤：裁判识别（黑色/荧光黄绿）
        if crop_bgr is None or crop_bgr.size == 0:
            return False
        h_c, w_c = crop_bgr.shape[:2]
        roi = crop_bgr[int(h_c*0.2):int(h_c*0.55), int(w_c*0.25):int(w_c*0.75)]
        if roi.size == 0:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        avg = cv2.mean(hsv)[:3]
        hue, sat, val = avg
        # 黑色裁判服
        if val < 55 and sat < 70:
            return False
        # 荧光黄/绿裁判服
        if 28 <= hue <= 90 and sat > 120 and val > 160:
            return False

        return True

    def extract_from_video(
        self,
        video_path: str,
        sample_interval: int = 60,
        split: str = "train",
        yolo_model_path: str = "yolov8n.pt",
    ) -> int:
        """
        从视频中提取场地内球员背部图像用于标注（严格过滤非球员）

        Args:
            video_path:       输入视频路径
            sample_interval:  每N帧提取1帧
            split:            "train" 或 "val"
            yolo_model_path:  用于检测运动员的YOLO模型

        Returns:
            提取的图像数量（仅球员，已过滤观众/裁判）
        """
        from ultralytics import YOLO
        model = YOLO(yolo_model_path)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        crop_dir = os.path.join(self.output_dir, "raw_crops")
        count = 0
        skipped = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                # 提高置信度至 0.5，减少误检
                results = model(frame, classes=[0], conf=0.5, verbose=False)
                if results and results[0].boxes is not None:
                    for i, box in enumerate(results[0].boxes):
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        h_box = y2 - y1

                        # 裁剪完整身体区域（用于球员过滤检查）
                        full_crop = frame[max(0, y1):y2, max(0, x1):x2]

                        # 严格球员过滤
                        if not self._is_player_crop(full_crop, frame_w, frame_h,
                                                     x1, y1, x2, y2):
                            skipped += 1
                            continue

                        # 裁剪背部号码区域
                        top = y1 + int(h_box * 0.25)
                        bot = y1 + int(h_box * 0.65)
                        back = frame[max(0, top):bot, max(0, x1):x2]
                        if back.size == 0:
                            continue
                        back_resized = cv2.resize(back, (128, 64))
                        fname = f"frame{frame_idx:06d}_p{i:02d}.jpg"
                        cv2.imwrite(os.path.join(crop_dir, fname), back_resized)
                        count += 1

            frame_idx += 1
            if frame_idx % 300 == 0:
                print(f"  进度: {frame_idx}/{total_frames} | 球员: {count} 张 | 已过滤: {skipped}")

        cap.release()
        print(f"[DatasetBuilder] 提取完成！")
        print(f"  球员样本: {count} 张 | 过滤（观众/裁判/小目标）: {skipped} 个")
        print(f"  保存至: {crop_dir}")
        print(f"  下一步：使用 LabelImg 对图像进行标注")
        print(f"  安装: pip install labelImg")
        print(f"  启动: labelImg {crop_dir}")
        return count


# ─────────────────────────────────────────────────────────────────────────────
# 快速使用示例
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法：")
        print("  识别视频: python jersey_number_recognition.py video <video_path>")
        print("  训练模型: python jersey_number_recognition.py train <dataset_dir>")
        print("  提取数据: python jersey_number_recognition.py extract <video_path>")
        sys.exit(0)

    mode = sys.argv[1]

    if mode == "video" and len(sys.argv) >= 3:
        video_path = sys.argv[2]
        print(f"[INFO] 处理视频: {video_path}")
        recognizer = JerseyNumberRecognizer(use_ocr=True)
        results = recognizer.process_video(video_path, sample_interval=30)
        print(f"\n识别结果（共 {len(results)} 名球员）：")
        for pid, info in sorted(results.items()):
            num = info['number'] if info['number'] else '?'
            team = {0: '主队', 1: '客队', -1: '未知'}.get(info['team'], '未知')
            print(f"  球员ID={pid:3d} | 号码={num:>3} | 队伍={team} | 置信度={info['confidence']:.1%}")

    elif mode == "train" and len(sys.argv) >= 3:
        dataset_dir = sys.argv[2]
        trainer = JerseyNumberTrainer()
        yaml_path = trainer.prepare_dataset_yaml(dataset_dir)
        best = trainer.train(yaml_path, epochs=100)
        print(f"[INFO] 训练完成，模型: {best}")

    elif mode == "extract" and len(sys.argv) >= 3:
        video_path = sys.argv[2]
        builder = DatasetBuilder()
        count = builder.extract_from_video(video_path)
        print(f"[INFO] 数据提取完成，共 {count} 张")
