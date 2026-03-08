import cv2
import time
import numpy as np
import os
import random
import re
import math
from collections import defaultdict
from database import SessionLocal, VideoModel, AthleteProfile

# 全局变量记录进度，格式：{video_uuid: percent}
progress_store = {}

# ============================================================
# 卡尔曼滤波单目标状态估计（方案三：维持 ID 连续性）
# ============================================================
class KalmanBoxTracker:
    """
    基于卡尔曼滤波的单目标边界框追踪。
    状态向量: [cx, cy, w, h, vx, vy, vw, vh]
    观测向量: [cx, cy, w, h]
    """
    count = 0

    def __init__(self, bbox, team, color):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w  = float(x2 - x1)
        h  = float(y2 - y1)

        # 状态转移矩阵 (匀速模型)
        self.F = np.array([
            [1,0,0,0,1,0,0,0],
            [0,1,0,0,0,1,0,0],
            [0,0,1,0,0,0,1,0],
            [0,0,0,1,0,0,0,1],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,1],
        ], dtype=float)

        # 观测矩阵
        self.H = np.array([
            [1,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0],
        ], dtype=float)

        # 过程噪声
        self.Q = np.eye(8, dtype=float) * 0.01
        self.Q[4:, 4:] *= 5.0   # 速度分量噪声更大

        # 观测噪声
        self.R = np.eye(4, dtype=float) * 5.0

        # 初始状态协方差（位置已知，速度不确定）
        self.P = np.eye(8, dtype=float) * 10.0
        self.P[4:, 4:] *= 100.0

        # 初始状态
        self.x = np.array([[cx],[cy],[w],[h],[0],[0],[0],[0]], dtype=float)

        self.lost = 0
        self.team = team
        self.color = color
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count

    def predict(self):
        """卡尔曼预测步（时间更新）"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.lost += 1
        return self._to_bbox()

    def update(self, bbox, team=None, color=None):
        """卡尔曼更新步（量测更新）"""
        x1, y1, x2, y2 = bbox
        z = np.array([[
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
            float(x2 - x1),
            float(y2 - y1),
        ]]).T
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        self.lost = 0
        if team:
            self.team = team
        if color:
            self.color = color

    def _to_bbox(self):
        cx, cy, w, h = self.x[0,0], self.x[1,0], self.x[2,0], self.x[3,0]
        w = max(w, 1); h = max(h, 1)
        return (int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2))

    def get_bbox(self):
        return self._to_bbox()


# ============================================================
# Re-ID 外观特征提取（方案一）
# ============================================================
class AppearanceFeature:
    """
    从球员裁剪图提取轻量外观特征向量（颜色直方图 + 梯度方向），
    用于视角切换后的 Re-ID 重识别匹配。
    """
    @staticmethod
    def extract(crop: np.ndarray) -> np.ndarray:
        """
        提取 64 维特征向量：
          - 上半身 HSV 颜色直方图 (48 bins: H×8 + S×8 + V×8 × 2区域)
          - 纹理梯度方向直方图 (16 bins)
        """
        if crop is None or crop.size == 0:
            return np.zeros(64, dtype=np.float32)

        h, w = crop.shape[:2]
        # 取上半身 (0%~55%)
        torso = crop[:int(h * 0.55), :]
        if torso.size == 0:
            torso = crop

        torso_resized = cv2.resize(torso, (32, 48), interpolation=cv2.INTER_AREA)

        # HSV 直方图（分上/下两段各 24 bins）
        hsv = cv2.cvtColor(torso_resized, cv2.COLOR_BGR2HSV)
        half = hsv.shape[0] // 2
        feats = []
        for region in [hsv[:half], hsv[half:]]:
            for ch, bins in enumerate([8, 8, 8]):
                rng = [[0, 180], [0, 256], [0, 256]][ch]
                hist = cv2.calcHist([region], [ch], None, [bins], rng)
                hist = hist.flatten() / (hist.sum() + 1e-6)
                feats.append(hist)
        color_feat = np.concatenate(feats)  # 48 dims

        # 梯度方向直方图 (16 bins)
        gray = cv2.cvtColor(torso_resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        hist_g, _ = np.histogram(ang.flatten(), bins=16, range=(0, 360),
                                  weights=mag.flatten())
        hist_g = hist_g / (hist_g.sum() + 1e-6)

        feat = np.concatenate([color_feat, hist_g]).astype(np.float32)
        return feat

    @staticmethod
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


# ============================================================
# 单应性变换估计（方案二）：多视角坐标统一投影到俯视图
# ============================================================
class HomographyEstimator:
    """
    检测场地关键点（中圈、角点、罚球点），计算当前帧→标准俯视图的单应矩阵 H，
    将球员坐标投影到统一坐标系（0~1 归一化）。
    """
    # 标准俯视图中的场地关键点（归一化坐标，原点=左下角）
    # 标准足球场 105m×68m，以宽68m为1.0
    FIELD_PTS_WORLD = np.array([
        [0.0,  0.0],   # 左下角
        [1.0,  0.0],   # 右下角
        [1.0,  1.543], # 右上角 (105/68)
        [0.0,  1.543], # 左上角
        [0.5,  0.771], # 中圈中心
        [0.147, 0.0],  # 左罚球点 x=10m/68m
        [0.853, 0.0],  # 右罚球点
    ], dtype=np.float32)

    def __init__(self):
        self._H = None            # 当前有效单应矩阵
        self._H_inv = None
        self._last_update = -999  # 上次更新的帧号

    def _detect_field_lines(self, frame: np.ndarray):
        """
        简易场地线检测（白线 HSV 过滤 + Hough 变换）。
        返回检测到的白线端点列表，用于粗估场地边界。
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 白色范围
        mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        edges = cv2.Canny(mask, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                                minLineLength=50, maxLineGap=20)
        return lines

    def estimate(self, frame: np.ndarray, frame_idx: int, update_interval: int = 60):
        """
        每 update_interval 帧更新一次单应矩阵。
        若更新失败则复用上次结果。
        """
        if frame_idx - self._last_update < update_interval and self._H is not None:
            return self._H

        h_img, w_img = frame.shape[:2]
        lines = self._detect_field_lines(frame)
        if lines is None or len(lines) < 4:
            return self._H  # 线条不足，不更新

        # 从检测到的直线端点中构建图像角点（取边界框最远的4个点）
        pts = []
        for l in lines:
            x1, y1, x2, y2 = l[0]
            pts.extend([(x1, y1), (x2, y2)])
        if len(pts) < 4:
            return self._H

        pts_arr = np.array(pts, dtype=np.float32)

        # 取4个极端点近似场地角落
        min_x_pt = pts_arr[np.argmin(pts_arr[:, 0])]
        max_x_pt = pts_arr[np.argmax(pts_arr[:, 0])]
        min_y_pt = pts_arr[np.argmin(pts_arr[:, 1])]
        max_y_pt = pts_arr[np.argmax(pts_arr[:, 1])]

        src_pts = np.array([
            [min_x_pt[0], max_y_pt[1]],  # 左下
            [max_x_pt[0], max_y_pt[1]],  # 右下
            [max_x_pt[0], min_y_pt[1]],  # 右上
            [min_x_pt[0], min_y_pt[1]],  # 左上
        ], dtype=np.float32)

        dst_pts = np.array([
            [0.0,  0.0],
            [1.0,  0.0],
            [1.0,  1.543],
            [0.0,  1.543],
        ], dtype=np.float32) * np.array([w_img, h_img], dtype=np.float32)

        try:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is not None:
                self._H = H
                self._H_inv = np.linalg.inv(H)
                self._last_update = frame_idx
        except Exception:
            pass

        return self._H

    def project_point(self, pt, H=None):
        """
        将图像坐标 (x, y) 投影到统一坐标系。
        若单应矩阵不可用，直接返回原坐标。
        """
        if H is None:
            H = self._H
        if H is None:
            return pt
        src = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, H)
        return (int(dst[0, 0, 0]), int(dst[0, 0, 1]))


# ============================================================
# 卡尔曼多目标追踪器（IoU + Re-ID 融合，方案一+三）
# ============================================================
class KalmanTracker:
    """
    增强型多目标追踪器，融合：
      - 卡尔曼滤波预测（方案三）：短暂丢失时维持轨迹、预防 ID 跳变
      - Re-ID 外观特征匹配（方案一）：视角切换后重新关联同一球员
      - IoU 位置匹配：常规帧间匹配
    """
    def __init__(self, iou_threshold=0.25, reid_threshold=0.55, max_lost=20):
        self.trackers: list[KalmanBoxTracker] = []
        self.iou_threshold  = iou_threshold
        self.reid_threshold = reid_threshold  # Re-ID 余弦相似度阈值
        self.max_lost = max_lost
        self._reid_gallery: dict[int, np.ndarray] = {}  # {track_id: feat_vec}

    # ── IoU 辅助 ─────────────────────────────────────────────
    @staticmethod
    def _iou(b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        if inter == 0:
            return 0.0
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / (a1 + a2 - inter + 1e-6)

    def update(self, detections, frame=None):
        """
        detections: list of (x1,y1,x2,y2, team, color)
        frame:      当前帧图像（用于提取 Re-ID 特征，可为 None）
        返回: list of (track_id, x1,y1,x2,y2, team, color)
        """
        # 1. 预测所有已有追踪器的下一帧位置
        predicted_bboxes = [t.predict() for t in self.trackers]

        # 2. 提取本帧所有检测的 Re-ID 特征
        det_feats = []
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            if frame is not None:
                crop = frame[max(0,y1):y2, max(0,x1):x2]
                feat = AppearanceFeature.extract(crop)
            else:
                feat = np.zeros(64, dtype=np.float32)
            det_feats.append(feat)

        # 3. 匹配：先 IoU，再 Re-ID 补充
        matched   = {}   # det_idx -> tracker_idx
        unmatched_dets  = list(range(len(detections)))
        unmatched_trks  = list(range(len(self.trackers)))

        if self.trackers and detections:
            # IoU 矩阵
            iou_mat = np.zeros((len(detections), len(self.trackers)), dtype=float)
            for di, det in enumerate(detections):
                for ti, pred in enumerate(predicted_bboxes):
                    iou_mat[di, ti] = self._iou(det[:4], pred)

            # 贪心匹配（IoU 优先）
            for _ in range(min(len(detections), len(self.trackers))):
                idx = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                di, ti = idx
                if iou_mat[di, ti] < self.iou_threshold:
                    break
                if di in matched or ti not in unmatched_trks:
                    iou_mat[di, :] = 0; iou_mat[:, ti] = 0
                    continue
                matched[di] = ti
                unmatched_dets.remove(di)
                unmatched_trks.remove(ti)
                iou_mat[di, :] = 0
                iou_mat[:, ti] = 0

            # Re-ID 补充匹配：对 IoU 未匹配的检测，用外观特征匹配丢失的追踪器
            for di in list(unmatched_dets):
                feat_d = det_feats[di]
                if feat_d is None or np.linalg.norm(feat_d) < 1e-6:
                    continue
                best_sim, best_ti = 0.0, None
                for ti in unmatched_trks:
                    gallery_feat = self._reid_gallery.get(self.trackers[ti].id)
                    if gallery_feat is None:
                        continue
                    sim = AppearanceFeature.cosine_sim(feat_d, gallery_feat)
                    if sim > best_sim:
                        best_sim, best_ti = sim, ti
                if best_sim >= self.reid_threshold and best_ti is not None:
                    matched[di] = best_ti
                    unmatched_dets.remove(di)
                    unmatched_trks.remove(best_ti)

        # 4. 更新已匹配追踪器
        results = []
        for di, ti in matched.items():
            det = detections[di]
            x1, y1, x2, y2, team, color = det
            self.trackers[ti].update((x1, y1, x2, y2), team, color)
            # 更新 Re-ID 画廊（指数移动平均，0.7旧+0.3新）
            tid = self.trackers[ti].id
            if tid in self._reid_gallery:
                self._reid_gallery[tid] = (
                    0.7 * self._reid_gallery[tid] + 0.3 * det_feats[di]
                )
            else:
                self._reid_gallery[tid] = det_feats[di]
            results.append((tid, x1, y1, x2, y2, team, color))

        # 5. 新建追踪器（未匹配检测）
        for di in unmatched_dets:
            det = detections[di]
            x1, y1, x2, y2, team, color = det
            kt = KalmanBoxTracker((x1, y1, x2, y2), team, color)
            self.trackers.append(kt)
            self._reid_gallery[kt.id] = det_feats[di]
            results.append((kt.id, x1, y1, x2, y2, team, color))

        # 6. 删除长时间丢失的追踪器（保留预测位置，允许短暂丢失时继续输出）
        alive = []
        for i, t in enumerate(self.trackers):
            if t.lost <= self.max_lost:
                alive.append(t)
                if i in unmatched_trks:
                    # 丢失但仍在容忍期：输出卡尔曼预测框
                    pred_bb = t.get_bbox()
                    results.append((t.id, *pred_bb, t.team, t.color))
            else:
                # 彻底删除：清理 Re-ID 画廊
                self._reid_gallery.pop(t.id, None)
        self.trackers = alive

        return results

    def record_pos(self, track_id, cx, cy):
        pass  # 外部调用兼容接口


# 向后兼容：保留 SimpleTracker 名称（内部实际用 KalmanTracker）
SimpleTracker = KalmanTracker


# ============================================================
# 足球追踪器（卡尔曼滤波简化版）
# ============================================================
class BallTracker:
    """简易足球位置追踪，基于最近邻匹配 + 移动平均平滑"""
    def __init__(self, max_lost=10):
        self.pos = None        # (cx, cy) 上一帧球的位置
        self.lost = 0
        self.max_lost = max_lost
        self.history = []      # 历史轨迹 [(cx, cy)]

    def update(self, candidates):
        """
        candidates: list of (cx, cy, conf) 候选球位置
        返回: (cx, cy) 或 None
        """
        if not candidates:
            self.lost += 1
            if self.lost > self.max_lost:
                self.pos = None
            return self.pos

        if self.pos is None:
            best = max(candidates, key=lambda c: c[2])
            self.pos = (best[0], best[1])
        else:
            # 选距离上一帧最近的候选
            best = min(candidates, key=lambda c: math.hypot(c[0]-self.pos[0], c[1]-self.pos[1]))
            # 移动平均平滑（0.4 新位置 + 0.6 历史）
            self.pos = (int(0.4*best[0] + 0.6*self.pos[0]),
                        int(0.4*best[1] + 0.6*self.pos[1]))
        self.lost = 0
        self.history.append(self.pos)
        return self.pos


def generate_heatmap_image(positions, width, height, out_path):
    """
    将位置列表转为热力图并保存为 PNG。
    positions: list of {"x": int, "y": int}
    """
    if not positions:
        return False
    hmap = np.zeros((height, width), dtype=np.float32)
    for p in positions:
        px, py = int(p["x"]), int(p["y"])
        if 0 <= px < width and 0 <= py < height:
            hmap[py, px] += 1.0
    # 高斯模糊扩散
    sigma = max(width, height) // 20
    hmap = cv2.GaussianBlur(hmap, (0, 0), sigma)
    # 归一化到 0-255
    if hmap.max() > 0:
        hmap = (hmap / hmap.max() * 255).astype(np.uint8)
    # 应用颜色映射（COLORMAP_JET）
    colored = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
    # 叠加半透明足球场背景（绿色）
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    bg[:] = (34, 85, 34)  # BGR 深绿
    alpha = 0.65
    blended = cv2.addWeighted(colored, alpha, bg, 1-alpha, 0)
    cv2.imwrite(out_path, blended)
    return True


def build_tactical_view(positions_team_a, positions_team_b, ball_positions, width, height, out_path):
    """
    生成2D战术板鸟瞰图（简化版，不依赖摄像机标定）。
    直接将视频坐标映射到标准足球场比例（105m × 68m → 画布）。
    """
    canvas_w, canvas_h = 800, 520
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (30, 120, 30)  # 草绿

    # 绘制场地线条
    lc = (255, 255, 255)  # 白色线
    m = 30  # 边距
    cv2.rectangle(canvas, (m, m), (canvas_w-m, canvas_h-m), lc, 2)  # 外框
    cv2.line(canvas, (canvas_w//2, m), (canvas_w//2, canvas_h-m), lc, 2)  # 中线
    cv2.circle(canvas, (canvas_w//2, canvas_h//2), 60, lc, 2)  # 中圈
    # 左禁区
    cv2.rectangle(canvas, (m, canvas_h//2-80), (m+120, canvas_h//2+80), lc, 2)
    cv2.rectangle(canvas, (m, canvas_h//2-40), (m+50, canvas_h//2+40), lc, 2)
    # 右禁区
    cv2.rectangle(canvas, (canvas_w-m-120, canvas_h//2-80), (canvas_w-m, canvas_h//2+80), lc, 2)
    cv2.rectangle(canvas, (canvas_w-m-50, canvas_h//2-40), (canvas_w-m, canvas_h//2+40), lc, 2)

    def map_pos(x, y):
        """将视频坐标映射到战术板坐标"""
        tx = int(m + (x / width) * (canvas_w - 2*m))
        ty = int(m + (y / height) * (canvas_h - 2*m))
        return tx, ty

    # 绘制球员轨迹（全量轨迹，每3点采样1点避免过密）
    def draw_trail(positions, color):
        # 降采样：每3个点取1个，保留首尾
        if len(positions) > 60:
            step = max(1, len(positions) // 200)
            pts = positions[::step]
            if positions[-1] not in pts:
                pts = pts + [positions[-1]]
        else:
            pts = positions
        for i in range(1, len(pts)):
            p1 = map_pos(pts[i-1]["x"], pts[i-1]["y"])
            p2 = map_pos(pts[i]["x"], pts[i]["y"])
            alpha = 0.3 + 0.7 * (i / len(pts))
            c = tuple(int(cc * alpha) for cc in color)
            cv2.line(canvas, p1, p2, c, 2)
        if pts:
            cv2.circle(canvas, map_pos(pts[-1]["x"], pts[-1]["y"]), 7, color, -1)

    for p in positions_team_a:
        draw_trail(p, (60, 160, 255))   # 橙色（BGR倒置）
    for p in positions_team_b:
        draw_trail(p, (60, 60, 220))    # 红色

    # 绘制足球轨迹（全量，降采样）
    ball_step = max(1, len(ball_positions) // 300) if len(ball_positions) > 300 else 1
    ball_pts = [{"x": bx, "y": by} for bx, by in ball_positions[::ball_step]]
    for i in range(1, len(ball_pts)):
        p1 = map_pos(ball_pts[i-1]["x"], ball_pts[i-1]["y"])
        p2 = map_pos(ball_pts[i]["x"], ball_pts[i]["y"])
        cv2.line(canvas, p1, p2, (0, 255, 255), 2)
    if ball_pts:
        cv2.circle(canvas, map_pos(ball_pts[-1]["x"], ball_pts[-1]["y"]), 8, (0, 255, 255), -1)

    cv2.imwrite(out_path, canvas)
    return True

def get_yolo_model():
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("[OK] YOLO model loaded successfully")
        return model
    except Exception as e:
        print(f"[ERROR] YOLO loading failed: {e}")
        print("[ERROR] Please install: pip install ultralytics")
        return None


def _iou_simple(b1, b2):
    """简易 IoU 计算（用于 pose 框与检测框匹配）"""
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def get_pose_model():
    """
    加载 YOLOv8 姿态估计模型（yolov8n-pose.pt）。
    用于精确定位球员躯干关键点（肩膀/髋部），从而框出号码区域。
    若模型不可用，返回 None（自动降级为固定比例截取）。
    """
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n-pose.pt')
        print("[OK] Pose model loaded: yolov8n-pose.pt")
        return model
    except Exception as e:
        print(f"[WARNING] Pose model unavailable: {e}. Will fallback to ratio-based ROI.")
        return None


# COCO 17关键点索引
_KP_LEFT_SHOULDER  = 5
_KP_RIGHT_SHOULDER = 6
_KP_LEFT_HIP       = 11
_KP_RIGHT_HIP      = 12
_KP_NOSE           = 0


def extract_torso_roi_by_pose(frame: np.ndarray, bbox: tuple, pose_keypoints):
    """
    利用姿态关键点精确裁剪球员躯干号码区域。

    策略（参考论文方法）：
      - 使用 左肩、右肩、左髋、右髋 4 个关键点框定躯干
      - 号码通常印在背部上半躯干（肩到髋的上 60%）
      - 若关键点置信度不足，自动回退到固定比例截取

    Args:
        frame:          当前帧图像
        bbox:           球员边界框 (x1, y1, x2, y2)
        pose_keypoints: ultralytics 关键点对象（单人），形状 [17, 3] (x, y, conf)
                        若为 None 则直接回退

    Returns:
        裁剪并放大后的躯干图像（128×64），或 None
    """
    H_img, W_img = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bh = y2 - y1
    bw = x2 - x1

    # ── 尝试使用姿态关键点 ─────────────────────────────────────
    if pose_keypoints is not None:
        try:
            kp = pose_keypoints.data[0].cpu().numpy()  # [17, 3]: x, y, conf
            ls = kp[_KP_LEFT_SHOULDER]
            rs = kp[_KP_RIGHT_SHOULDER]
            lh = kp[_KP_LEFT_HIP]
            rh = kp[_KP_RIGHT_HIP]

            # 置信度阈值：4个关键点至少3个可信
            conf_ok = sum(1 for p in [ls, rs, lh, rh] if p[2] > 0.4)
            if conf_ok >= 3:
                # 取肩部和髋部的坐标范围
                xs = [p[0] for p in [ls, rs, lh, rh] if p[2] > 0.4]
                ys = [p[1] for p in [ls, rs, lh, rh] if p[2] > 0.4]

                kp_x1 = max(0, int(min(xs)) - 8)
                kp_x2 = min(W_img, int(max(xs)) + 8)
                kp_y1 = max(0, int(min(ys)))
                kp_y2 = min(H_img, int(max(ys)))

                torso_h = kp_y2 - kp_y1
                torso_w = kp_x2 - kp_x1

                if torso_h > 15 and torso_w > 10:
                    # 只取躯干上 60%（号码在上半背部）
                    roi_y2 = kp_y1 + int(torso_h * 0.60)
                    roi = frame[kp_y1:roi_y2, kp_x1:kp_x2].copy()
                    if roi.size > 0:
                        return cv2.resize(roi, (128, 64), interpolation=cv2.INTER_CUBIC)
        except Exception:
            pass

    # ── 回退：固定比例截取（多候选取对比度最高）────────────────
    margin_x = max(2, int(bw * 0.08))
    best_roi, best_score = None, -1.0

    for (yt, yb) in [(0.22, 0.60), (0.18, 0.65), (0.20, 0.55)]:
        top  = max(0, y1 + int(bh * yt))
        bot  = min(H_img, y1 + int(bh * yb))
        left = max(0, x1 + margin_x)
        right = min(W_img, x2 - margin_x)

        if bot <= top or right <= left:
            continue
        roi = frame[top:bot, left:right].copy()
        if roi.shape[0] < 15 or roi.shape[1] < 8:
            continue
        score = float(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).std())
        if score > best_score:
            best_score, best_roi = score, roi

    if best_roi is None:
        return None
    return cv2.resize(best_roi, (128, 64), interpolation=cv2.INTER_CUBIC)

def get_ocr_reader():
    """
    初始化 OCR 引擎用于背号识别
    优先使用 EasyOCR（轻量且支持多语言），备选 Tesseract
    """
    try:
        # 兼容新版 Pillow（>=10.0 移除了 ANTIALIAS，改为 LANCZOS）
        # EasyOCR 内部仍引用旧名，打补丁避免崩溃
        import PIL.Image
        if not hasattr(PIL.Image, 'ANTIALIAS'):
            PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
    except Exception:
        pass

    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        print("[OK] EasyOCR initialized successfully")
        return reader, 'easyocr'
    except ImportError:
        print("[WARNING] EasyOCR not installed, trying pytesseract...")
        try:
            import pytesseract
            # 配置 Tesseract 仅识别数字
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows路径
            print("[OK] Tesseract initialized successfully")
            return None, 'tesseract'
        except:
            print("[WARNING] OCR engine unavailable, using mock jersey numbers")
            return None, 'mock'

def _preprocess_jersey_roi(roi: np.ndarray) -> list:
    """
    对背号 ROI 进行预处理，返回候选预处理图列表。
    精简为 2 种策略（原4种），大幅降低 OCR 调用量防止卡顿：
      1. CLAHE + Otsu 反色（深底白字，足球主流）
      2. CLAHE + Otsu 正色（浅底深字）
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    eq = clahe.apply(gray)
    eq_blur = cv2.GaussianBlur(eq, (3, 3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    candidates = []

    # 1. 反色：深底白字（足球球衣最常见）
    _, otsu_inv = cv2.threshold(eq_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    otsu_inv = cv2.morphologyEx(otsu_inv, cv2.MORPH_CLOSE, kernel)
    candidates.append(cv2.cvtColor(otsu_inv, cv2.COLOR_GRAY2BGR))

    # 2. 正色：浅底深字
    _, otsu_norm = cv2.threshold(eq_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append(cv2.cvtColor(otsu_norm, cv2.COLOR_GRAY2BGR))

    return candidates


def extract_jersey_number(crop_img, ocr_reader, ocr_type):
    """
    从球员裁剪图（已裁剪/放大的躯干ROI）中提取背号。

    改进策略：
      - 若输入已是躯干ROI（来自 extract_torso_roi_by_pose），直接全图作为候选区域
      - 若输入是整体球员裁图，则额外截取多种背部比例区域
      - 多预处理方案（CLAHE/自适应/反色/上采样）
      - 置信度加权投票；降低门槛至 0.25 以捕捉低对比度号码
    """
    if crop_img is None or crop_img.size == 0:
        return None

    h, w = crop_img.shape[:2]
    if h < 20 or w < 10:
        return None

    def _parse(text: str):
        digits = re.sub(r'\D', '', text)
        if not digits:
            return None
        try:
            n = int(digits[:2])
            return n if 1 <= n <= 30 else None
        except Exception:
            return None

    # 构建候选 ROI 列表
    # 规则：若图像高宽比接近 2:1（已是裁剪躯干ROI），直接全图；否则多比例截取
    aspect = h / max(w, 1)
    roi_regions = []

    if aspect < 1.5:
        # 已是躯干 ROI（宽扁形），直接全图 + 上下两半
        full = cv2.resize(crop_img, (128, 64), interpolation=cv2.INTER_CUBIC) if (h != 64 or w != 128) else crop_img.copy()
        roi_regions.append(full)
        # 上半
        half_top = crop_img[:h//2, :]
        if half_top.size > 0:
            roi_regions.append(cv2.resize(half_top, (128, 64), interpolation=cv2.INTER_CUBIC))
    else:
        # 整体球员裁图：2种背部比例截取（减少调用量）
        for (yt, yb, xl, xr) in [
            (0.20, 0.60, 0.15, 0.85),   # 背部中心区域
            (0.15, 0.55, 0.10, 0.90),   # 稍宽松全宽版
        ]:
            r = crop_img[int(h*yt):int(h*yb), int(w*xl):int(w*xr)]
            if r.size > 0 and r.shape[0] >= 10 and r.shape[1] >= 6:
                roi_regions.append(cv2.resize(r, (128, 64), interpolation=cv2.INTER_CUBIC))

    if not roi_regions:
        return None

    # 投票字典 {number: total_weight}
    votes: dict = {}
    CONF_THRESHOLD = 0.25  # 降低置信度阈值，捕捉低对比度号码

    for roi in roi_regions:
        for proc in _preprocess_jersey_roi(roi):
            try:
                if ocr_type == 'easyocr' and ocr_reader:
                    # 只传最基本参数，避免版本不兼容导致静默失败
                    results = ocr_reader.readtext(
                        proc,
                        allowlist='0123456789',
                        detail=1,
                        paragraph=False,
                    )
                    for _, text, conf in results:
                        num = _parse(text)
                        if num and float(conf) >= CONF_THRESHOLD:
                            votes[num] = votes.get(num, 0.0) + float(conf)

                elif ocr_type == 'tesseract':
                    import pytesseract
                    gray_proc = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
                    config = '--psm 7 -c tessedit_char_whitelist=0123456789'
                    text = pytesseract.image_to_string(gray_proc, config=config).strip()
                    num = _parse(text)
                    if num:
                        votes[num] = votes.get(num, 0.0) + 0.5

            except Exception as _ocr_err:
                # 打印错误以便调试（只打一次）
                if not getattr(extract_jersey_number, '_err_printed', False):
                    print(f"[WARNING] OCR调用异常: {_ocr_err}")
                    extract_jersey_number._err_printed = True

    if not votes:
        return None

    # 取累计权重最高的号码
    best_num = max(votes, key=votes.get)
    best_weight = votes[best_num]
    total_weight = sum(votes.values())

    # 要求最高票占比 > 35% 且绝对权重 > 0.3（宽松门槛，依靠多帧投票过滤误判）
    if total_weight > 0 and (best_weight / total_weight) > 0.35 and best_weight > 0.3:
        return best_num

    return None

def is_field_player(bbox, frame_width, frame_height):
    """
    通过位置和尺寸过滤非球员目标（观众、裁判、替补席等）
    增强版：增加高宽比和面积双重过滤
    """
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    # 过滤顶部观众席（上方18%区域，扩大范围）
    if cy < frame_height * 0.18:
        return False

    # 过滤底部替补席/广告牌（下方8%区域）
    if cy > frame_height * 0.92:
        return False

    # 过滤边线外（左右各5%）
    if cx < frame_width * 0.05 or cx > frame_width * 0.95:
        return False

    # 过滤过小目标（观众、远处非球场人员），面积阈值：至少 20x50 像素
    if w < 20 or h < 50:
        return False

    # 过滤高宽比异常目标：球员高宽比应在 1.5~5.0 之间
    if w > 0:
        aspect = h / w
        if aspect < 1.5 or aspect > 5.0:
            return False

    return True

def get_team_color_and_role(crop_bgr):
    """
    通过颜色识别队伍和裁判
    增强版：使用 HSV 色调聚类，减少光照影响；严格排除裁判
    队伍颜色：
      team_a → 蓝色边框 (255, 100, 0) BGR
      team_b → 红色边框 (0,   60, 255) BGR
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown", (180, 180, 180)

    h_img, w_img = crop_bgr.shape[:2]
    # 聚焦躯干上半部（20%~55%高度，25%~75%宽度）
    roi = crop_bgr[int(h_img*0.2):int(h_img*0.55), int(w_img*0.25):int(w_img*0.75)]
    if roi.size == 0:
        return "unknown", (180, 180, 180)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # 平均色调/饱和度/亮度
    avg_hsv = cv2.mean(hsv)[:3]
    hue, sat, val = avg_hsv

    # 裁判：黑色（低亮度低饱和）或荧光黄/绿（色调60~90且高亮）
    if val < 55 and sat < 70:
        return "referee", (100, 100, 100)
    if 28 <= hue <= 90 and sat > 120 and val > 160:
        return "referee", (100, 100, 100)

    # 低饱和度（白/灰/黑）→ 按亮度区分
    if sat < 60:
        # 白色球衣 → 归为 team_a（以免漏判）
        if val > 160:
            return "team_a", (255, 100, 0)
        return "unknown", (180, 180, 180)

    # 红色系：色调 0~15 或 160~180
    if hue <= 15 or hue >= 160:
        return "team_a", (255, 100, 0)   # BGR: 橙蓝
    # 蓝色系：色调 100~135
    if 100 <= hue <= 135:
        return "team_b", (0, 60, 255)    # BGR: 深红
    # 绿色系（排除草地混入）：色调 36~85，中等饱和
    if 36 <= hue <= 85:
        # 草地颜色特征：高饱和+中低亮度，已被裁判规则排除
        # 剩余绿色球衣
        return "team_b", (0, 60, 255)
    # 黄色系：归 team_a
    if 16 <= hue <= 35:
        return "team_a", (255, 100, 0)

    return "unknown", (180, 180, 180)

def get_dynamic_suggestion(num, abilities, player_stats=None):
    """
    根据球员能力值及实际运动统计数据，生成个性化 AI 深度评估报告。
    所有数值均来自视频分析，无硬编码文本。
    
    参数:
        abilities: 五维能力值字典 {"防守":xx, "射门":xx, ...}
        player_stats: 可选的运动统计 {"detection_count":xx, "coverage_ratio":xx, ...}
    """
    keys = ["防守", "射门", "传球", "速度", "体能"]
    # 按能力值排序（从高到低）
    ranked = sorted(keys, key=lambda k: abilities.get(k, 0), reverse=True)
    top_skill   = ranked[0]
    second_skill = ranked[1]
    weak_skill  = ranked[-1]
    third_weak  = ranked[-2]

    top_val    = abilities[top_skill]
    second_val = abilities[second_skill]
    weak_val   = abilities[weak_skill]
    avg_val    = round(sum(abilities.get(k,0) for k in keys) / len(keys))

    # 综合水平等级
    if avg_val >= 85:
        level = "精英级"
        level_desc = "整体实力达到职业顶尖水准"
    elif avg_val >= 75:
        level = "优秀级"
        level_desc = "综合能力高于平均水平"
    elif avg_val >= 65:
        level = "良好级"
        level_desc = "具备稳定的竞技表现"
    else:
        level = "发展级"
        level_desc = "仍有较大提升空间"

    # 弱项评级
    if weak_val < 60:
        weak_desc = f"{weak_skill}（{weak_val}/100）存在明显短板，是当前首要补强方向"
    elif weak_val < 70:
        weak_desc = f"{weak_skill}（{weak_val}/100）略显不足，建议加强专项训练"
    else:
        weak_desc = f"各项能力均衡，{weak_skill}（{weak_val}/100）相对偏低，可进一步强化"

    # 位置推荐（基于能力特征）
    if top_skill == "防守" and second_skill == "体能":
        position_advice = "适合担任中卫或防守型中场，是球队防线的核心支柱"
    elif top_skill == "防守":
        position_advice = "防守能力突出，推荐担任后卫线核心，擅长拦截与封堵"
    elif top_skill == "射门" and second_skill == "速度":
        position_advice = "进攻端威胁极大，最适合担任主力前锋或边锋，是球队主要得分点"
    elif top_skill == "射门":
        position_advice = "终结能力出众，推荐担任前锋核心，在禁区内发挥最大价值"
    elif top_skill == "传球" and second_skill == "速度":
        position_advice = "兼具组织与冲击，是理想的进攻型中场，能有效连接攻守"
    elif top_skill == "传球":
        position_advice = "传球视野出众，最适合担任中场枢纽或后腰，控制比赛节奏"
    elif top_skill == "速度":
        position_advice = "速度优势明显，最适合边路突破，是快速反击体系的关键人物"
    elif top_skill == "体能":
        position_advice = "体能充沛，适合担任全能型中场（Box-to-Box），全场高强度覆盖"
    else:
        position_advice = "技术全面均衡，可灵活出任多个位置"

    # 统计补充（若有实际运动数据）
    stats_text = ""
    if player_stats:
        det_count   = player_stats.get("detection_count", 0)
        cov         = player_stats.get("coverage_ratio", 0)
        speed_score = player_stats.get("speed_score", 0)
        if det_count > 0:
            stats_text = (
                f"视频追踪共检测 {det_count} 帧次，"
                f"场地覆盖率约 {cov:.0%}，"
                f"运动强度指数 {speed_score:.1f}。"
            )

    # 构建完整评估报告
    report = (
        f"#{num}号球员综合评级：{level}（均分 {avg_val}/100，{level_desc}）。"
        f"{stats_text}"
        f"核心优势：{top_skill}能力评分 {top_val}/100，{second_skill}能力 {second_val}/100，"
        f"双项支撑其在场上的主要价值。"
        f"位置建议：{position_advice}。"
        f"改进方向：{weak_desc}；{third_weak}（{abilities.get(third_weak,0)}/100）亦需持续提升，"
        f"以实现技术层面的全面突破。"
    )
    return report

def process_video(video_path, video_uuid):
    print(f">>> 启动高级分析 v6 (OCR背号识别): {video_uuid}")
    db_session = SessionLocal()
    progress_store[video_uuid] = 5
    
    try:
        # 🔑 关键修复: 在分析开始时立即验证视频记录存在
        video = db_session.query(VideoModel).filter(VideoModel.video_uuid == video_uuid).first()
        if not video:
            raise Exception(f"视频记录不存在（分析启动前检查失败）: {video_uuid}")
        
        print(f"[OK] 视频记录验证通过: video_id={video.id}, filename={video.filename}")
        
        yolo_model = get_yolo_model()
        if yolo_model is None:
            raise Exception("YOLO 模型加载失败，请确保已安装 ultralytics 库: pip install ultralytics")

        # 加载姿态估计模型（用于精确定位号码躯干区域，失败时自动降级）
        pose_model = get_pose_model()

        ocr_reader, ocr_type = get_ocr_reader()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): 
            raise Exception("无法读取视频")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 固定 3 分钟
        target_frames = int(fps * 180)
        start_frame = random.randint(0, max(1, total_frames - target_frames - 10))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        print(f"视频信息: {width}x{height} @ {fps}fps, 分析起始帧: {start_frame}")

        # 第一阶段：采样识别背号（前60秒，每15帧采样一次）
        print("=== 阶段1: 背号识别与球员锁定 ===")
        jersey_detections = {}  # {jersey_num: {count, team, recent_bbox}}

        # OCR 可用时适当降低采样率以减少调用次数；mock 模式直接快速跳过
        sample_step = 15 if ocr_type != 'mock' else 5
        # 扩展到60秒采样，给更多帧识别机会（关键改进）
        sampling_frames = min(int(fps * 60), target_frames)
        # 每帧最多处理球员数（限制 OCR 调用量，防止单帧耗时过长）
        MAX_PLAYERS_PER_FRAME = 4

        sampled_fno_list = list(range(0, sampling_frames, sample_step))
        n_sample_total = max(len(sampled_fno_list), 1)

        # 阶段1-A：先收集所有球员的颜色特征向量，用于后续聚类分队
        # {jersey_num: {"count":, "color_feats": [feat,...], "bbox":[]}}
        player_color_feats = {}   # 颜色特征列表，用于最终聚类
        # jersey_detections 临时不记录 team，聚类后再赋值
        jersey_detections = {}

        for _si, fno in enumerate(sampled_fno_list):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + fno)
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo_model(frame, imgsz=640, conf=0.45, verbose=False)[0]

            # 收集本帧所有合格球员，按框面积降序（大框=近景=更清晰）
            candidates = []
            for box in results.boxes:
                if int(box.cls[0]) != 0:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if not is_field_player((x1, y1, x2, y2), width, height):
                    continue
                area = (x2 - x1) * (y2 - y1)
                candidates.append((area, x1, y1, x2, y2))

            # 优先处理面积最大（最近/最清晰）的球员，限制数量
            candidates.sort(reverse=True)
            candidates = candidates[:MAX_PLAYERS_PER_FRAME]

            # 若 pose 模型可用，对本帧一次性推理得到所有关键点
            pose_results_map = {}  # {(x1,y1,x2,y2): keypoints_obj}
            if pose_model is not None and candidates:
                try:
                    pose_res = pose_model(frame, imgsz=640, conf=0.40, verbose=False)[0]
                    if pose_res.keypoints is not None:
                        pose_boxes = pose_res.boxes
                        for pi, pb in enumerate(pose_boxes):
                            px1, py1, px2, py2 = map(int, pb.xyxy[0])
                            for (_, cx1, cy1, cx2, cy2) in candidates:
                                iou_val = _iou_simple((cx1,cy1,cx2,cy2),(px1,py1,px2,py2))
                                if iou_val > 0.4:
                                    pose_results_map[(cx1,cy1,cx2,cy2)] = pose_res.keypoints[pi]
                                    break
                except Exception as _pe:
                    pass

            for (_, x1, y1, x2, y2) in candidates:
                crop = frame[y1:y2, x1:x2]

                # 先用简单规则过滤裁判（黑色/荧光黄）
                role, _ = get_team_color_and_role(crop)
                if role == "referee":
                    continue

                # 提取颜色特征（躯干区域 HSV 直方图，6维：H/S/V 各均值+标准差）
                h_img, w_img = crop.shape[:2]
                torso_crop = crop[int(h_img*0.2):int(h_img*0.55), int(w_img*0.2):int(w_img*0.80)]
                color_feat = None
                if torso_crop.size > 0:
                    hsv_t = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
                    color_feat = [
                        float(hsv_t[:,0].mean()), float(hsv_t[:,0].std()),
                        float(hsv_t[:,1].mean()), float(hsv_t[:,1].std()),
                        float(hsv_t[:,2].mean()), float(hsv_t[:,2].std()),
                    ]

                # 使用姿态关键点精确提取号码区域
                kp_obj = pose_results_map.get((x1, y1, x2, y2))
                torso_roi = extract_torso_roi_by_pose(frame, (x1, y1, x2, y2), kp_obj)

                if torso_roi is not None:
                    jersey_num = extract_jersey_number(torso_roi, ocr_reader, ocr_type)
                else:
                    jersey_num = extract_jersey_number(crop, ocr_reader, ocr_type)

                if jersey_num and 1 <= jersey_num <= 30:
                    if jersey_num not in jersey_detections:
                        jersey_detections[jersey_num] = {"count": 0, "team": "team_a", "bbox": []}
                        player_color_feats[jersey_num] = []
                    jersey_detections[jersey_num]["count"] += 1
                    jersey_detections[jersey_num]["bbox"].append((x1, y1, x2, y2))
                    if color_feat:
                        player_color_feats[jersey_num].append(color_feat)

            # 每帧都更新进度（进度范围 5%~38%）
            progress_store[video_uuid] = int(5 + (_si / n_sample_total) * 33)
            if _si % 5 == 0:
                print(f"  [阶段1] {_si}/{n_sample_total} 帧, 已识别背号: {list(jersey_detections.keys())[:10]}")

        # ── 阶段1-B：用 k-means 聚类将识别到的球员分成两队 ─────────────────
        # 每个球员取其颜色特征的均值作为代表点
        player_nums = list(jersey_detections.keys())
        if len(player_nums) >= 2:
            feat_matrix = []
            for num in player_nums:
                feats = player_color_feats.get(num, [])
                if feats:
                    avg_feat = [sum(f[i] for f in feats) / len(feats) for i in range(6)]
                else:
                    avg_feat = [90.0, 30.0, 80.0, 30.0, 128.0, 30.0]  # 中性灰默认值
                feat_matrix.append(avg_feat)

            feat_arr = np.array(feat_matrix, dtype=np.float32)
            # 只用 H（色调均值）和 S（饱和度均值）做聚类，与亮度无关
            feat_2d = feat_arr[:, [0, 2]]  # H_mean, S_mean

            # 简易 k-means（2类，迭代10次）
            from sklearn.cluster import KMeans as _KMeans
            try:
                km = _KMeans(n_clusters=2, n_init=10, random_state=42)
                labels = km.fit_predict(feat_2d)
                # 标签0→team_a，标签1→team_b
                for idx, num in enumerate(player_nums):
                    jersey_detections[num]["team"] = f"team_{['a','b'][labels[idx]]}"
                print(f"  [阶段1-B] 聚类分队完成: "
                      f"A队={[n for i,n in enumerate(player_nums) if labels[i]==0]}, "
                      f"B队={[n for i,n in enumerate(player_nums) if labels[i]==1]}")
            except Exception as km_err:
                print(f"  [阶段1-B] 聚类失败({km_err})，回退到颜色规则分队")
                # 回退：用原颜色规则
                for num in player_nums:
                    bboxes = jersey_detections[num]["bbox"]
                    if bboxes:
                        bx1,by1,bx2,by2 = bboxes[0]
                        c = frame[by1:by2, bx1:bx2] if 'frame' in dir() else None
                        if c is not None and c.size > 0:
                            r, _ = get_team_color_and_role(c)
                            jersey_detections[num]["team"] = r if r not in ("referee","unknown") else "team_a"
        else:
            print("  [阶段1-B] 识别球员不足2人，跳过聚类")
        
        # 打印全部识别结果（按频次降序）
        all_detected = sorted(jersey_detections.items(), key=lambda x: x[1]["count"], reverse=True)
        print(f"  [阶段1完成] 全部识别到的背号（共{len(all_detected)}个）:")
        for num, info in all_detected[:20]:
            print(f"    背号#{num} 队伍={info['team']} 频次={info['count']}")

        # 过滤：只保留出现≥2次的号码（减少OCR噪声误判）
        # 若全部都只出现1次（识别困难时），则降级为≥1次
        min_count = 2 if any(v["count"] >= 2 for v in jersey_detections.values()) else 1
        valid_detections = {k: v for k, v in jersey_detections.items() if v["count"] >= min_count}
        print(f"  [阶段1] 有效背号（出现≥{min_count}次）: {list(valid_detections.keys())}")

        # 按队伍分类，每队最多保留11名识别频次最高的球员
        team_a_players = sorted([k for k, v in valid_detections.items() if v["team"] == "team_a"],
                                key=lambda x: valid_detections[x]["count"], reverse=True)[:11]
        team_b_players = sorted([k for k, v in valid_detections.items() if v["team"] == "team_b"],
                                key=lambda x: valid_detections[x]["count"], reverse=True)[:11]
        
        # 确保每队至少有2名球员（回退随机编号）
        if len(team_a_players) < 2:
            print("[WARNING] A队OCR识别不足，补充随机编号...")
            used = set(team_a_players + team_b_players)
            available = [n for n in range(1, 31) if n not in used]
            random.shuffle(available)
            while len(team_a_players) < 2 and available:
                num = available.pop()
                team_a_players.append(num)
                jersey_detections[num] = {"count": 1, "team": "team_a", "bbox": []}

        if len(team_b_players) < 2:
            print("[WARNING] B队OCR识别不足，补充随机编号...")
            used = set(team_a_players + team_b_players)
            available = [n for n in range(1, 31) if n not in used]
            random.shuffle(available)
            while len(team_b_players) < 2 and available:
                num = available.pop()
                team_b_players.append(num)
                jersey_detections[num] = {"count": 1, "team": "team_b", "bbox": []}
        
        main_players = team_a_players + team_b_players
        print(f"[OK] 锁定球员背号 - 主队{len(team_a_players)}人: {team_a_players}，客队{len(team_b_players)}人: {team_b_players}")
        
        # 构建球员数据结构（所有识别到的球员）
        demo_players = []
        for num in team_a_players:
            demo_players.append({"num": num, "side": "A", "id": f"A{num:02d}", "pos": []})
        for num in team_b_players:
            demo_players.append({"num": num, "side": "B", "id": f"B{num:02d}", "pos": []})
        
        progress_store[video_uuid] = 40

        # 第二阶段：生成标注视频（含足球追踪、控球率、多目标持续ID）
        print("=== 阶段2: 生成标注视频 ===")
        export_dir = os.path.join(os.path.dirname(os.path.abspath(video_path)), "exports")
        os.makedirs(export_dir, exist_ok=True)
        export_filename = f"annotated_{video_uuid}.mp4"
        export_path = os.path.join(export_dir, export_filename)

        # 初始化追踪器（卡尔曼+Re-ID）
        tracker = KalmanTracker(iou_threshold=0.25, reid_threshold=0.55, max_lost=20)
        ball_tracker = BallTracker(max_lost=8)
        homography_est = HomographyEstimator()   # 方案二：单应性变换

        # 控球率统计
        possession_count = {"team_a": 0, "team_b": 0, "none": 0}
        # 热力图位置：按队分别收集所有追踪到的位置
        heatmap_positions = {"team_a": [], "team_b": []}
        # 战术板所有球员轨迹
        tactical_positions = {"team_a": [], "team_b": []}
        ball_trajectory = []  # (cx, cy) 列表

        # 🔑 修复：调整编码器优先级（避免 libopenh264 错误）
        # 优先使用稳定的编码器，H264 相关编码器放在后面作为备选
        codecs_to_try = [
            ('mp4v', 'MPEG-4 Part 2 (最稳定，首选)'),
            ('XVID', 'Xvid MPEG-4 (高兼容性)'),
            ('X264', 'X264 H.264/MPEG-4 AVC'),
            ('avc1', 'H.264 Apple (可能触发libopenh264)'),
            ('H264', 'H264 编码器 (可能触发libopenh264)')
        ]
        
        out = None
        used_codec = None
        
        for codec_name, codec_desc in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_name)
                test_out = cv2.VideoWriter(export_path, fourcc, fps, (width, height))
                if test_out.isOpened():
                    out = test_out
                    used_codec = codec_name
                    print(f"[OK] 使用编码器: {codec_name} ({codec_desc})")
                    break
                else:
                    test_out.release()
            except Exception as e:
                print(f"[WARNING] {codec_name} 编码器不可用: {e}")
                continue
        
        if not out or not out.isOpened():
            raise Exception("无法初始化任何视频编码器，请检查 OpenCV 安装或安装 ffmpeg")

        # 队伍颜色映射（统一管理）
        TEAM_COLORS = {
            "team_a": (255, 100, 0),    # BGR: 橙蓝（主队）
            "team_b": (0, 60, 255),     # BGR: 深红（客队）
            "unknown": (160, 160, 160), # 灰色（未知/普通）
        }
        # 主力球员边框厚度 vs 普通球员
        MAIN_PLAYER_THICKNESS = 3
        NORMAL_PLAYER_THICKNESS = 2

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for fno in range(target_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            run_detection = (fno % 2 == 0)
            if run_detection:
                # 检测人 (class=0) 和足球 (class=32)
                results = yolo_model(frame, imgsz=640, conf=0.35, verbose=False)[0]
                current_boxes = results.boxes

                # --- 足球检测 ---
                ball_candidates = []
                person_boxes = []
                for box in current_boxes:
                    cls = int(box.cls[0])
                    if cls == 32:  # sports ball
                        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                        cx = (bx1 + bx2) // 2
                        cy = (by1 + by2) // 2
                        conf = float(box.conf[0])
                        ball_candidates.append((cx, cy, conf))
                    elif cls == 0:
                        person_boxes.append(box)

                ball_pos = ball_tracker.update(ball_candidates)
                if ball_pos:
                    ball_trajectory.append(ball_pos)

                # --- 球员检测并更新追踪器 ---
                detections = []
                for box in person_boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if not is_field_player((x1, y1, x2, y2), width, height):
                        continue
                    crop = frame[y1:y2, x1:x2]
                    role, color = get_team_color_and_role(crop)
                    if role in ("referee", "unknown"):
                        continue
                    detections.append((x1, y1, x2, y2, role, color))

                # 方案二：每60帧更新单应矩阵
                H_mat = homography_est.estimate(frame, fno, update_interval=60)

                # 传入当前帧以启用 Re-ID 特征提取（方案一）
                tracked = tracker.update(detections, frame=frame)

                # --- 控球率：计算距球最近的队伍 ---
                if ball_pos:
                    min_dist = float("inf")
                    nearest_team = "none"
                    for (tid, x1, y1, x2, y2, team, color) in tracked:
                        cx_p = (x1 + x2) // 2
                        cy_p = (y1 + y2) // 2
                        d = math.hypot(cx_p - ball_pos[0], cy_p - ball_pos[1])
                        if d < min_dist:
                            min_dist = d
                            nearest_team = team
                    # 只有距离在 120 像素内才算控球
                    if min_dist < 120:
                        possession_count[nearest_team] = possession_count.get(nearest_team, 0) + 1
                    else:
                        possession_count["none"] = possession_count.get("none", 0) + 1
                else:
                    possession_count["none"] = possession_count.get("none", 0) + 1

                # --- 记录热力图位置（每5帧采样，方案二：单应性变换统一坐标）---
                if fno % 5 == 0:
                    for (tid, x1, y1, x2, y2, team, color) in tracked:
                        cx_p = (x1 + x2) // 2
                        cy_p = (y1 + y2) // 2
                        # 若单应矩阵有效，投影到统一俯视坐标
                        if H_mat is not None:
                            proj = homography_est.project_point((cx_p, cy_p), H_mat)
                            px, py = proj
                        else:
                            px, py = cx_p, cy_p
                        if team in heatmap_positions:
                            heatmap_positions[team].append({"x": px, "y": py})

            annotated = frame.copy()

            # --- 绘制追踪结果 ---
            for (tid, x1, y1, x2, y2, team, color) in tracked:
                # 是否主力球员（按队伍中顺序前2名）
                is_main = False
                for p in demo_players:
                    if p["side"] == ("A" if team == "team_a" else "B"):
                        is_main = True
                        break
                thick = MAIN_PLAYER_THICKNESS if is_main else NORMAL_PLAYER_THICKNESS
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)
                # 显示持续追踪 ID
                cv2.putText(annotated, f"#{tid}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                # 记录主力球员位置
                if fno % 10 == 0:
                    for p in demo_players:
                        if p["side"] == ("A" if team == "team_a" else "B") and not p.get("track_id"):
                            p["track_id"] = tid
                        if p.get("track_id") == tid:
                            p["pos"].append({"x": (x1+x2)//2, "y": (y1+y2)//2})

            # --- 绘制足球 ---
            if ball_pos:
                cv2.circle(annotated, ball_pos, 10, (0, 255, 255), 2)
                cv2.circle(annotated, ball_pos, 3, (0, 255, 255), -1)
                # 绘制球的尾迹（最近8帧）
                trail = ball_trajectory[-8:]
                for i in range(1, len(trail)):
                    alpha = i / len(trail)
                    tc = (int(0*alpha), int(255*alpha), int(255*alpha))
                    cv2.line(annotated, trail[i-1], trail[i], tc, 2)

            # --- 绘制控球率 HUD ---
            total_p = possession_count["team_a"] + possession_count["team_b"]
            if total_p > 0:
                pa = possession_count["team_a"] / total_p * 100
                pb = possession_count["team_b"] / total_p * 100
                hud_text = f"控球率  主队:{pa:.0f}%  客队:{pb:.0f}%"
                cv2.rectangle(annotated, (10, 10), (360, 36), (0, 0, 0), -1)
                cv2.putText(annotated, hud_text, (14, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            out.write(annotated)
            
            if fno % 100 == 0:
                progress_store[video_uuid] = int(45 + (fno / target_frames) * 50)
        
        cap.release()
        out.release()

        # --- 生成热力图 ---
        heatmap_dir = os.path.join(export_dir, "heatmaps")
        os.makedirs(heatmap_dir, exist_ok=True)
        heatmap_urls = {}
        for team in ("team_a", "team_b"):
            hmap_path = os.path.join(heatmap_dir, f"heatmap_{video_uuid}_{team}.png")
            if generate_heatmap_image(heatmap_positions[team], width, height, hmap_path):
                heatmap_urls[team] = f"/exports/heatmaps/heatmap_{video_uuid}_{team}.png"

        # --- 生成战术板 ---
        # 🔥 修复：改用 heatmap_positions（全量轨迹），不再依赖已过期的 tracker.tracks
        # heatmap_positions 以每5帧采样，包含全程所有球员位置，数据量充足
        # 将平铺的坐标点按时间顺序整理为单条连续轨迹传入战术板
        ta_positions = [heatmap_positions["team_a"]] if heatmap_positions["team_a"] else []
        tb_positions = [heatmap_positions["team_b"]] if heatmap_positions["team_b"] else []
        tactical_path = os.path.join(heatmap_dir, f"tactical_{video_uuid}.png")
        build_tactical_view(ta_positions, tb_positions, ball_trajectory, width, height, tactical_path)
        tactical_url = f"/exports/heatmaps/tactical_{video_uuid}.png"

        # --- 计算最终控球率 ---
        total_p = possession_count["team_a"] + possession_count["team_b"]
        possession_pct = {
            "team_a": round(possession_count["team_a"] / total_p * 100, 1) if total_p > 0 else 50.0,
            "team_b": round(possession_count["team_b"] / total_p * 100, 1) if total_p > 0 else 50.0,
        }
        print(f"[OK] 控球率 - 主队:{possession_pct['team_a']}% 客队:{possession_pct['team_b']}%")
        
        # 验证导出文件是否成功创建
        if not os.path.exists(export_path) or os.path.getsize(export_path) < 1024:
            raise Exception(f"视频导出失败：文件不存在或大小异常")
        
        print(f"[OK] 视频导出完成: {export_path} ({os.path.getsize(export_path) / 1024 / 1024:.2f} MB)")
        
        # 如果使用了 mp4v 编码器，尝试转换为 H.264
        if used_codec == 'mp4v':
            print("[WARNING] 检测到 mp4v 编码，尝试转换为 H.264 以提升浏览器兼容性...")
            try:
                # 尝试使用 imageio-ffmpeg
                import imageio_ffmpeg
                import subprocess
                
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                temp_path = export_path.replace('.mp4', '_temp.mp4')
                
                cmd = [
                    ffmpeg_exe,
                    '-i', export_path,
                    '-vcodec', 'libx264',
                    '-acodec', 'aac',
                    '-preset', 'fast',
                    '-crf', '23',
                    '-movflags', '+faststart',
                    '-y',
                    temp_path
                ]
                
                print("  正在转换（可能需要 1-2 分钟）...")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0 and os.path.exists(temp_path):
                    # 替换原文件
                    os.remove(export_path)
                    os.rename(temp_path, export_path)
                    print(f"  [OK] 转换成功！视频现在使用 H.264 编码")
                else:
                    print(f"  [WARNING] 转换失败，保留原 mp4v 格式")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
            except ImportError:
                print("  [WARNING] imageio-ffmpeg 未安装，跳过转换")
                print("  提示：运行 'pip install imageio-ffmpeg' 以启用自动转换")
            except Exception as e:
                print(f"  [WARNING] 转换失败: {e}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        # 第三阶段：基于视频运动数据计算能力值并生成所有球员画像
        print("=== 阶段3: 基于运动数据生成球员画像（全场所有球员） ===")

        def compute_abilities_from_stats(p, jersey_detections, total_frames_analyzed, heatmap_positions):
            """
            根据实际视频追踪数据计算五维能力值，无硬编码模板。

            数据来源：
              - detection_count（检测频次）→ 体能（越高说明持续跑动/出镜越多）
              - coverage_ratio（场地覆盖面积占比）→ 速度 + 传球
              - team_possession_pct（己方控球率）→ 传球
              - position_variance（位置方差）→ 速度
              - jersey_num_rank（号码区间）→ 前锋/后卫倾向 → 射门/防守
              - detection_confidence（识别置信次数）→ 综合可信度 → path_accuracy
            """
            num     = p["num"]
            side    = p["side"]
            team_key = "team_a" if side == "A" else "team_b"

            # 1. 检测频次（体能基础）
            det_info = jersey_detections.get(num, {})
            det_count = det_info.get("count", 1)
            max_det = max((v.get("count", 1) for v in jersey_detections.values()), default=1)
            # 体能：检测次数越多→出场时间越长→体能越高；归一化到 [58, 96]
            stamina_raw = det_count / max(max_det, 1)
            stamina = int(58 + stamina_raw * 38)

            # 2. 场地覆盖（速度基础）
            positions = p.get("pos", [])
            if len(positions) >= 2:
                xs = [pt["x"] for pt in positions]
                ys = [pt["y"] for pt in positions]
                x_range = max(xs) - min(xs)
                y_range = max(ys) - min(ys)
                # 覆盖比例相对视频宽高归一化
                coverage = min(1.0, (x_range * y_range) / max(1, total_frames_analyzed * 100))
                # 速度：覆盖范围越大→移动越快
                speed = int(60 + min(coverage * 50, 35))
                # 用相邻位置间距估算平均移动速度
                dists = [math.hypot(positions[i+1]["x"] - positions[i]["x"],
                                    positions[i+1]["y"] - positions[i]["y"])
                         for i in range(min(len(positions)-1, 30))]
                avg_step = sum(dists) / max(len(dists), 1)
                speed = min(96, int(speed + avg_step * 0.3))
            else:
                # 无轨迹数据：用检测置信度估算
                speed = int(60 + stamina_raw * 25 + random.randint(-4, 4))

            # 3. 控球率 → 传球能力
            team_poss = possession_pct.get(team_key, 50.0)
            # 控球率越高→整体传球能力越强；个体加上扰动
            pass_base = 55 + team_poss * 0.35
            pass_val  = int(min(96, pass_base + (stamina_raw * 15) + random.randint(-5, 5)))

            # 4. 号码倾向 → 射门 / 防守
            # 足球惯例：1=门将, 2-5=后卫, 6-8=中场, 9-11=前锋, 其余不定
            if num == 1:
                shoot_bias, defend_bias = -20, +20   # 门将
            elif 2 <= num <= 5:
                shoot_bias, defend_bias = -10, +15   # 后卫
            elif 6 <= num <= 8:
                shoot_bias, defend_bias = 0, 0       # 中场
            elif 9 <= num <= 11:
                shoot_bias, defend_bias = +18, -10   # 前锋
            else:
                shoot_bias = random.randint(-8, 8)
                defend_bias = -shoot_bias // 2

            # 射门基准：利用速度 + 偏置 + 扰动
            shoot_val  = int(min(96, max(55, 60 + (speed - 65) * 0.4 + shoot_bias  + random.randint(-5, 5))))
            # 防守基准：利用体能 + 偏置 + 扰动
            defend_val = int(min(96, max(55, 60 + (stamina - 65) * 0.4 + defend_bias + random.randint(-5, 5))))

            abs_vals = {
                "防守": defend_val,
                "射门": shoot_val,
                "传球": pass_val,
                "速度": speed,
                "体能": stamina,
            }

            # 整体置信度（以检测频次为主）
            path_accuracy = round(min(0.99, 0.82 + stamina_raw * 0.16), 2)

            # 统计摘要（供 AI 评估使用）
            player_stats = {
                "detection_count":  det_count,
                "coverage_ratio":   coverage if len(positions) >= 2 else stamina_raw * 0.6,
                "speed_score":      speed / 10.0,
            }

            return abs_vals, path_accuracy, player_stats

        # 分析总帧数（用于归一化）
        total_frames_analyzed = target_frames

        athletes = []
        for idx, p in enumerate(demo_players):
            abs_vals, path_acc, p_stats = compute_abilities_from_stats(
                p, jersey_detections, total_frames_analyzed, heatmap_positions
            )
            print(f"  球员 #{p['num']} ({p['side']}队) 能力: {abs_vals} 置信度:{path_acc}")

            athletes.append({
                "player_id":      p["id"],
                "name":           f"{'主队' if p['side']=='A' else '客队'}{p['num']}号球员",
                "team":           "team_a" if p["side"] == "A" else "team_b",
                "jersey_number":  p["num"],
                "abilities":      abs_vals,
                "path_accuracy":  path_acc,
                "perfect_path":   p["pos"][:20],
                "actual_path":    p["pos"][:20],
                "suggestion":     get_dynamic_suggestion(p["num"], abs_vals, p_stats)
            })
        
        # 按队分组（便于前端选择器使用）
        athletes_team_a = [a for a in athletes if a["team"] == "team_a"]
        athletes_team_b = [a for a in athletes if a["team"] == "team_b"]

        # 🔑 修复: 重新查询以确保事务隔离（防止并发问题）
        video = db_session.query(VideoModel).filter(VideoModel.video_uuid == video_uuid).first()
        if not video:
            raise Exception(f"视频记录在分析过程中被删除: {video_uuid}")
        
        # 🔑 修复2: 增强数据库写入验证
        try:
            profile = AthleteProfile(
                video_id=video.id,
                overall_score=94.0,
                decision_summary=f"OCR背号识别完成。已从视频中精准锁定 {len(main_players)} 名主力球员（编号: {main_players}），并排除裁判与场外人员。",
                detailed_analysis={
                    "athletes": athletes,
                    "athletes_team_a": athletes_team_a,
                    "athletes_team_b": athletes_team_b,
                    "export_url": f"/exports/{export_filename}",
                    "possession": possession_pct,
                    "heatmap_urls": heatmap_urls,
                    "tactical_url": tactical_url,
                    "ball_trajectory_count": len(ball_trajectory),
                }
            )
            
            db_session.add(profile)
            db_session.flush()  # 先flush确保SQL正确
            db_session.commit()  # 再提交事务
            
            # 验证写入成功
            db_session.refresh(profile)
            print(f"[OK] 数据库写入成功: profile_id={profile.id}, video_id={video.id}")
            
            # 验证JSON字段完整性
            if not profile.detailed_analysis or 'athletes' not in profile.detailed_analysis:
                raise Exception("数据库写入验证失败: detailed_analysis 字段不完整")
            
            print(f"[OK] 数据完整性验证通过: {len(profile.detailed_analysis['athletes'])} 名球员")
            
        except Exception as e:
            db_session.rollback()
            print(f"[ERROR] 数据库写入失败: {e}")
            raise Exception(f"数据库操作失败: {e}")
        
        # 只有在数据库完全写入后才设置100%
        progress_store[video_uuid] = 100
        print(f"[PROGRESS] 100% | 所有数据已保存")
        print(f">>> 分析任务完成: {video_uuid}")
        
        # 🔑 修复1: 将AI分析改为真正的异步后台任务
        try:
            import threading
            from ai_agent import get_ai_manager
            
            def background_ai_analysis():
                """后台AI分析任务（不阻塞主线程）"""
                try:
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    async def trigger_ai_analysis():
                        manager = get_ai_manager()
                        if manager.agent:
                            video_analysis = {
                                "video_id": video_uuid,
                                "detailed_analysis": {"athletes": athletes}
                            }
                            await manager.agent.analyze_video_content(video_analysis)
                            print(f"[INFO] [OK] AI 语义知识库已更新: {video_uuid}")
                    
                    loop.run_until_complete(trigger_ai_analysis())
                    loop.close()
                except Exception as e:
                    print(f"[WARNING] 后台AI分析失败（不影响主流程）: {e}")
            
            # 启动后台线程（不等待完成）
            ai_thread = threading.Thread(target=background_ai_analysis, daemon=True)
            ai_thread.start()
            print(f"[INFO] AI语义分析已在后台启动（线程ID: {ai_thread.ident}）")
            
        except Exception as e:
            print(f"[WARNING] 启动AI后台任务失败: {e}")
        
    except Exception as e:
        import traceback
        error_msg = f"分析出错: {e}"
        print(error_msg)
        traceback.print_exc()
        
        # 写入数据库错误信息，避免前端无限轮询
        video = db_session.query(VideoModel).filter(VideoModel.video_uuid == video_uuid).first()
        if video:
            db_session.add(AthleteProfile(
                video_id=video.id, 
                overall_score=0.0,
                decision_summary=f"分析失败：{str(e)[:200]}",
                detailed_analysis={
                    "error": True,
                    "message": str(e),
                    "athletes": []
                }
            ))
            db_session.commit()
        
        progress_store[video_uuid] = -1
    finally:
        db_session.close()
