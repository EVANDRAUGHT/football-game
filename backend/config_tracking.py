"""
config_tracking.py
论文第5章：多目标跟踪方法 - 参数配置文件

所有超参数均对齐论文设计，可直接用于：
  1. 验证 analysis.py 中的实现是否与论文一致
  2. 作为对比实验的基准配置
  3. 打印权重动态变化表辅助论文写作
"""

# ============================================================================
# §5.2  卡尔曼滤波参数（NSA Kalman）
# ============================================================================
KALMAN_CONFIG = {
    # NSA Kalman 参数（论文第5.2节）
    "NSA_SCALE":           10.0,   # 观测噪声缩放因子 α=10
    "Q_PROCESS_NOISE":     0.01,   # 过程噪声基础值
    "Q_VELOCITY_SCALE":    5.0,    # 速度分量的额外噪声倍数
    "R_MEASUREMENT_BASE":  5.0,    # 基础观测噪声协方差
    # NSA 置信度代理：conf_proxy = clip(框面积/15000, 0.1, 1.0)
    "CONF_AREA_NORM":      15000,
    "CONF_MIN":            0.1,
}

# ============================================================================
# §5.2.1  外观特征参数
# ============================================================================
APPEARANCE_CONFIG = {
    "FEATURE_NETWORK":    "OSNet-AIN x1.0 MSMT17",  # 实际使用（论文ResNet50）
    "FEATURE_DIM":        512,                        # OSNet输出维度（论文2048）
    "INPUT_SIZE":         (256, 128),                 # OSNet标准输入
    "FEATURE_QUEUE_SIZE": 5,                          # 论文：最近N=5帧特征取均值
    "NORMALIZE":          "L2",                       # L2归一化
    "GALLERY_MIN_AREA":   1200,                       # 过小检测框不更新gallery（px²）
}

# ============================================================================
# §5.2.1  运动特征参数（卡尔曼状态空间）
# ============================================================================
MOTION_CONFIG = {
    "STATE_DIM":       8,                    # [cx, cy, w, h, vx, vy, vw, vh]
    "OBSERVATION_DIM": 4,                    # [cx, cy, w, h]
    "MOTION_MODEL":    "constant_velocity",  # 匀速运动模型
}

# ============================================================================
# §5.2.1  时空特征参数
# ============================================================================
SPATIOTEMPORAL_CONFIG = {
    "TRAJECTORY_LENGTH": 30,      # 论文：最近L=30帧轨迹历史
    "PREDICTION_METHOD": "linear", # 线性外推（最近两帧差分）
    "SIMILARITY_METRIC": "IoU",   # 论文：预测bbox与检测bbox的IoU距离
}

# ============================================================================
# §5.2.2  自适应关联代价矩阵参数（论文核心创新）
# ============================================================================
ADAPTIVE_WEIGHTS_CONFIG = {
    # 基础权重（t_occ=0 时）
    "LAMBDA_APP_0": 0.5,    # 外观特征初始权重
    "LAMBDA_MOT_0": 0.3,    # 运动特征权重（保持不变）
    "LAMBDA_ST_0":  0.2,    # 时空特征初始权重

    # 动态调整参数（论文公式）
    "GAMMA": 0.95,           # 外观衰减因子 γ（论文值，原代码0.85已修正）
    "ALPHA": 0.05,           # 时空增益因子 α

    # 边界约束
    "LAMBDA_APP_MIN": 0.05,  # 外观权重下界
    "LAMBDA_ST_MAX":  0.80,  # 时空权重上界
}

# ============================================================================
# §5.2.2  遮挡处理与重识别参数
# ============================================================================
OCCLUSION_CONFIG = {
    "IOU_OCCLUDED_THRESHOLD": 0.10,  # 论文：IOU < 0.1 显式判定遮挡
    "MAX_OCCLUSION_FRAMES":   30,    # 论文：最大遮挡帧数（≈1s@30fps）
    "SEARCH_EXPAND_RATIO":    1.5,   # 论文：搜索框扩大1.5×
    "LOST_THRESHOLD":         3,     # 连续丢失≥3帧才扩大搜索框
    "REID_THRESHOLD":         0.60,  # 论文：余弦相似度>0.6才恢复身份
}

# ============================================================================
# §5.4  评估参数
# ============================================================================
EVAL_CONFIG = {
    "FPS_TARGET":         30,   # 目标实时帧率
    "IOU_MATCH_THRESHOLD": 0.25, # 匈牙利匹配后IOU验证阈值
    # 期望指标（论文报告值）
    "PAPER_MOTA":         68.3,
    "PAPER_MOTP":         82.6,
    # 基线期望指标（论文文献[25] DeepSORT）
    "BASELINE_MOTA":      61.4,
    "BASELINE_MOTP":      79.1,
}

# ============================================================================
# 辅助函数：根据遮挡帧数计算自适应权重
# ============================================================================
def compute_adaptive_weights(t_occ: int) -> dict:
    """
    根据论文公式计算自适应权重（归一化后）。

    论文公式：
        λ_app(t_occ) = λ_app0 × γ^t_occ
        λ_st(t_occ)  = λ_st0  × (1 + α × t_occ)
        λ_mot(t_occ) = λ_mot0（运动权重保持不变）

    Args:
        t_occ: 连续遮挡帧数

    Returns:
        {"app": float, "mot": float, "st": float}  归一化后三项权重
    """
    cfg = ADAPTIVE_WEIGHTS_CONFIG

    raw_app = cfg["LAMBDA_APP_0"] * (cfg["GAMMA"] ** t_occ)
    raw_mot = cfg["LAMBDA_MOT_0"]
    raw_st  = cfg["LAMBDA_ST_0"] * (1.0 + cfg["ALPHA"] * t_occ)

    raw_app = max(cfg["LAMBDA_APP_MIN"], raw_app)
    raw_st  = min(cfg["LAMBDA_ST_MAX"],  raw_st)

    total = raw_app + raw_mot + raw_st
    if total < 1e-8:
        return {"app": 1/3, "mot": 1/3, "st": 1/3}

    return {
        "app": round(raw_app / total, 4),
        "mot": round(raw_mot / total, 4),
        "st":  round(raw_st  / total, 4),
    }


# ============================================================================
# 辅助函数：打印完整配置（辅助论文实验记录）
# ============================================================================
def print_config():
    """打印所有配置参数及权重变化示例，便于论文实验记录。"""
    sep = "=" * 62

    print(sep)
    print("  论文第5章：多目标跟踪方法 - 参数配置")
    print(sep)

    print("\n【卡尔曼滤波参数（NSA Kalman）】")
    k = KALMAN_CONFIG
    print(f"  NSA缩放因子    : {k['NSA_SCALE']}")
    print(f"  过程噪声基础值  : {k['Q_PROCESS_NOISE']}")
    print(f"  速度分量噪声倍数: {k['Q_VELOCITY_SCALE']}")
    print(f"  观测噪声基础值  : {k['R_MEASUREMENT_BASE']}")

    print("\n【外观特征参数】")
    a = APPEARANCE_CONFIG
    print(f"  特征网络   : {a['FEATURE_NETWORK']}")
    print(f"  特征维度   : {a['FEATURE_DIM']} 维")
    print(f"  特征队列长度: N = {a['FEATURE_QUEUE_SIZE']} 帧（取均值作模板）")

    print("\n【时空特征参数】")
    s = SPATIOTEMPORAL_CONFIG
    print(f"  轨迹历史长度: L = {s['TRAJECTORY_LENGTH']} 帧")
    print(f"  预测方法    : {s['PREDICTION_METHOD']}")
    print(f"  相似度指标  : {s['SIMILARITY_METRIC']}")

    print("\n【自适应权重参数（论文公式对齐）】")
    w = ADAPTIVE_WEIGHTS_CONFIG
    print(f"  初始权重  λ_app0={w['LAMBDA_APP_0']}, "
          f"λ_mot0={w['LAMBDA_MOT_0']}, λ_st0={w['LAMBDA_ST_0']}")
    print(f"  衰减因子  γ = {w['GAMMA']}（原代码0.85，已修正）")
    print(f"  增益因子  α = {w['ALPHA']}")

    print("\n【权重动态变化（随遮挡帧数）】")
    print(f"  {'t_occ':>6}  {'λ_app':>7}  {'λ_mot':>7}  {'λ_st':>7}")
    print(f"  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}")
    for t in [0, 5, 10, 15, 20, 25, 30]:
        wt = compute_adaptive_weights(t)
        print(f"  {t:>6}  {wt['app']:>7.4f}  {wt['mot']:>7.4f}  {wt['st']:>7.4f}")

    print("\n【遮挡处理参数】")
    o = OCCLUSION_CONFIG
    print(f"  遮挡判断阈值 : IoU < {o['IOU_OCCLUDED_THRESHOLD']}")
    print(f"  最大遮挡帧数 : {o['MAX_OCCLUSION_FRAMES']} 帧")
    print(f"  搜索框扩大比 : {o['SEARCH_EXPAND_RATIO']}×")
    print(f"  重识别阈值   : 余弦相似度 > {o['REID_THRESHOLD']}")

    print("\n【期望评估指标（论文报告值）】")
    e = EVAL_CONFIG
    print(f"  标准DeepSORT基线 : MOTA={e['BASELINE_MOTA']}%  MOTP={e['BASELINE_MOTP']}%")
    print(f"  本文方法          : MOTA={e['PAPER_MOTA']}%   MOTP={e['PAPER_MOTP']}%")
    print(f"  提升幅度          : ΔMOTA={e['PAPER_MOTA']-e['BASELINE_MOTA']:+.1f}%  "
          f"ΔMOTP={e['PAPER_MOTP']-e['BASELINE_MOTP']:+.1f}%")
    print(sep)


if __name__ == "__main__":
    print_config()
