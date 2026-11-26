// 模式说明弹窗
function showModeInfo() {
    const modal = document.createElement('div');
    modal.style.cssText = 'position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); display: flex; align-items: center; justify-content: center; z-index: 10000;';
    modal.innerHTML = `
        <div style="background: var(--card-bg); border-radius: 16px; padding: 30px; max-width: 600px; max-height: 80vh; overflow-y: auto; position: relative;">
            <button onclick="this.closest('div').parentElement.remove()" style="position: absolute; top: 15px; right: 15px; background: transparent; border: none; color: var(--text-dim); font-size: 1.5rem; cursor: pointer;">&times;</button>
            <h2 style="color: var(--primary); margin-bottom: 20px;"><i class="fas fa-info-circle"></i> 关于演示模式</h2>

            <div style="margin-bottom: 20px;">
                <h3 style="color: var(--accent); font-size: 1.1rem; margin-bottom: 10px;">📊 当前实现功能</h3>
                <ul style="color: var(--text-dim); line-height: 1.8; padding-left: 20px;">
                    <li><strong>深度视觉感知</strong>: YOLO目标检测 + MediaPipe姿态估计</li>
                    <li><strong>多模态特征提取</strong>: 球衣颜色识别 + 运动轨迹分析</li>
                    <li><strong>决策画像生成</strong>: 轨迹分析 + AI评估建议</li>
                    <li><strong>演示数据策略</strong>: 基于真实检测框架 + 模板化能力分布</li>
                </ul>
            </div>

            <div style="margin-bottom: 20px;">
                <h3 style="color: #ffce56; font-size: 1.1rem; margin-bottom: 10px;">🎯 数据生成说明</h3>
                <ul style="color: var(--text-dim); line-height: 1.8; padding-left: 20px;">
                    <li><strong>球员识别</strong>: 真实的YOLO检测 + 颜色分类（排除裁判）</li>
                    <li><strong>编号标注</strong>: 随机分配2-30号，模拟背号识别效果</li>
                    <li><strong>能力评估</strong>: 预设4种差异化模板（防守型/进攻型/组织型/全能型）</li>
                    <li><strong>建议生成</strong>: 基于能力值动态匹配15+专业模板</li>
                </ul>
            </div>

            <div style="background: rgba(255,99,132,0.1); border-left: 3px solid #ff6384; padding: 15px; border-radius: 8px;">
                <p style="color: var(--text-dim); line-height: 1.6; margin: 0;">
                    <strong style="color: #ff6384;">⚠️ 学术说明</strong><br>
                    本演示旨在展示技术框架与功能流程。真实应用中，能力评估需结合历史数据、多场比赛跟踪及专业标注，当前版本侧重可视化与交互体验。
                </p>
            </div>

            <button onclick="this.closest('div').parentElement.remove()" style="width: 100%; margin-top: 20px; padding: 12px; background: var(--primary); color: var(--bg); border: none; border-radius: 8px; cursor: pointer; font-size: 1rem; font-weight: bold;">
                我知道了
            </button>
        </div>
    `;
    document.body.appendChild(modal);
}

console.log('[APP] ===== app.js 模块加载 (v2.2) =====');

document.addEventListener('DOMContentLoaded', () => {
    console.log('[APP] ===== DOMContentLoaded 触发 =====');
    console.log('[APP] 当前时间:', new Date().toISOString());

    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const videoPreview = document.getElementById('video-preview');
    const uploadBtn = document.getElementById('upload-btn');
    const fileInfo = document.getElementById('file-info');
    const filenameDisplay = document.getElementById('filename-display');
    const statusBadge = document.getElementById('status');
    const progressBar = document.getElementById('progress-bar');
    const progressContainer = document.getElementById('progress-container');
    const videoContainer = document.getElementById('video-container');

    const API_BASE = window.location.origin;

    let uploadedVideoId = localStorage.getItem('last_uploaded_video_id') || null;
    window.uploadedVideoId = uploadedVideoId;
    let isAnalyzing = false;
    let heartbeatTimer = null;   // 全局心跳定时器

    // Phase A 全局追踪状态
    let phaseAProgress = -1;   // -1=未开始, 0~99=进行中, 100=完成
    let phaseATimer = null;    // 进度轮询定时器
    let phaseADone = false;    // 是否已完成（cached 或 progress=100）
    let phaseAWaitingOpen = false; // 用户已点击"标注球员"，等待 Phase A 完成后自动开弹窗

    // 启动 Phase A 全局追踪
    async function startPhaseA(videoId) {
        phaseAProgress = 0;
        phaseADone = false;
        try {
            const res = await fetch(`${API_BASE}/pre_track/${videoId}`, { method: 'POST' });
            const d = await res.json();
            if (d.status === 'cached') {
                // 已有缓存，直接标记完成并自动打开弹窗
                phaseAProgress = 100;
                phaseADone = true;
                updatePhaseAButton();
                if (!isAnalyzing) {
                    window._triggerAnalysis(videoId, [], false, [], {}, [], []);
                }
                return;
            }
        } catch (e) {
            console.warn('[PhaseA] /pre_track 请求失败:', e);
        }
        // 开始轮询进度
        if (phaseATimer) clearInterval(phaseATimer);
        phaseATimer = setInterval(() => pollPhaseA(videoId), 2000);
    }

    async function pollPhaseA(videoId) {
        try {
            const res = await fetch(`${API_BASE}/pre_track_progress/${videoId}`);
            const d = await res.json();
            phaseAProgress = typeof d.progress === 'number' ? d.progress : phaseAProgress;
            const done = d.cached || phaseAProgress >= 100;
            if (done) {
                phaseAProgress = 100;
                phaseADone = true;
                if (phaseATimer) { clearInterval(phaseATimer); phaseATimer = null; }
                updatePhaseAButton();
                // Phase A 完成后直接启动 Phase B（无需标注）
                if (!isAnalyzing) {
                    window._triggerAnalysis(videoId, [], false, [], {}, [], []);
                }
            } else {
                updatePhaseAButton();
            }
        } catch (e) {
            // 忽略轮询错误，继续重试
        }
    }

    function updatePhaseAButton() {
        if (isAnalyzing) return;  // 主分析中不改按钮
        if (phaseADone) {
            uploadBtn.textContent = '开始分析';
            uploadBtn.disabled = false;
            uploadBtn.style.opacity = '1';
            statusBadge.textContent = '全局追踪完成，正在启动深度分析...';
        } else if (phaseAProgress >= 0) {
            const pct = phaseAProgress > 0 ? ` ${phaseAProgress}%` : '';
            uploadBtn.textContent = `追踪中${pct}...`;
            uploadBtn.disabled = true;
            uploadBtn.style.opacity = '0.6';
            statusBadge.textContent = `全局追踪进行中${pct}，请稍候...`;
        }
    }

    // 点击上传区域或文件名显示区域都可以触发重新选择
    dropZone.addEventListener('click', () => fileInput.click());

    // 允许在上传后通过点击 filenameDisplay 重新上传
    filenameDisplay.style.cursor = 'pointer';
    filenameDisplay.addEventListener('click', () => {
        if (!isAnalyzing) fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files[0]) handleFile(e.target.files[0]);
    });

    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.style.borderColor = 'var(--primary)'; });
    dropZone.addEventListener('dragleave', () => { dropZone.style.borderColor = 'rgba(255, 255, 255, 0.2)'; });
    dropZone.addEventListener('drop', (e) => { e.preventDefault(); handleFile(e.dataTransfer.files[0]); });

    function handleFile(file) {
        if (file && file.type.startsWith('video/')) {
            // 重置状态
            uploadedVideoId = null;
            window.uploadedVideoId = null;  // 同步到全局
            localStorage.removeItem('last_uploaded_video_id');  // 清除旧持久化ID
            isAnalyzing = false;
            progressContainer.style.display = 'none';
            // 重置 Phase A 状态
            phaseAProgress = -1;
            phaseADone = false;
            phaseAWaitingOpen = false;
            if (phaseATimer) { clearInterval(phaseATimer); phaseATimer = null; }
            const oldOverlay = document.getElementById('phase-a-wait-overlay');
            if (oldOverlay) oldOverlay.remove();

            // 清理旧的视频 URL
            if (videoPreview.src) {
                URL.revokeObjectURL(videoPreview.src);
            }

            filenameDisplay.textContent = `准备上传: ${file.name}`;
            fileInfo.style.display = 'block';

            uploadBtn.disabled = true;
            uploadBtn.textContent = '正在极速上传...';
            uploadBtn.style.opacity = '0.5';

            autoUpload(file);
        }
    }

    async function autoUpload(file) {
        const formData = new FormData();
        formData.append('file', file);

        // 获取当前登录用户名，作为视频所有者
        const authRaw = localStorage.getItem('auth_user');
        const authUser = authRaw ? JSON.parse(authRaw) : null;
        const ownerParam = authUser ? encodeURIComponent(authUser.username) : 'guest';

        // 读取用户选择的分析模式（highlight / tracking / all）
        const analysisMode = window._analysisMode || 'all';

        // 精彩视频在主分析完成后由后端自动触发，上传时不再并行启动
        let uploadUrl = `${API_BASE}/upload?owner=${ownerParam}&analysis_mode=${encodeURIComponent(analysisMode)}`;

        const xhr = new XMLHttpRequest();
        xhr.open('POST', uploadUrl, true);

        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable) {
                const percent = (e.loaded / e.total * 100).toFixed(0);
                statusBadge.textContent = `上传中: ${percent}%`;
                uploadBtn.textContent = `上传中 (${percent}%)`;
                // 工作流步骤 1（上传阶段）
                if (typeof window.showWorkflowSteps === 'function') window.showWorkflowSteps();
                if (typeof window.setWorkflowStep === 'function') window.setWorkflowStep(1, `【阶段1】上传中 ${percent}%`);
            }
        };

        xhr.onload = function() {
            if (xhr.status === 200) {
                const data = JSON.parse(xhr.responseText);
                uploadedVideoId = data.video_id;
                window.uploadedVideoId = data.video_id;  // 同步到全局
                localStorage.setItem('last_uploaded_video_id', data.video_id);  // 持久化

                filenameDisplay.innerHTML = `已就绪: ${file.name} <span style="color:var(--primary); font-size:0.8rem;">(点击可更换)</span>`;

                // 创建一个本地 URL 并显示预览
                const videoURL = URL.createObjectURL(file);
                videoPreview.src = videoURL;
                videoPreview.load(); // 强制加载视频
                videoContainer.style.display = 'block';
                dropZone.style.display = 'none';

                // 确保视频可以播放
                videoPreview.addEventListener('loadeddata', () => {
                    console.log('视频预览加载成功');
                }, { once: true });

                // 显示导出区
                const exportSec = document.getElementById('export-section');
                if (exportSec) exportSec.style.display = 'block';

                const currentMode = window._analysisMode || 'all';
                if (currentMode === 'highlight') {
                    // ── 精彩事件检测模式：跳过 Phase A，直接触发精彩分析 ──
                    uploadBtn.disabled = true;
                    uploadBtn.style.opacity = '0.6';
                    uploadBtn.textContent = '正在深度分析...';
                    statusBadge.textContent = '上传完成，正在提取精彩片段（无需YOLO追踪）...';
                    if (typeof window.setWorkflowStep === 'function') window.setWorkflowStep(2, '【阶段2】视频解码与光流分析...');
                    phaseADone = true;  // 标记跳过 Phase A
                    window._triggerAnalysis(uploadedVideoId, [], false, [], {}, [], []);
                } else {
                    // ── tracking / all 模式：正常启动 Phase A ──
                    uploadBtn.disabled = false;
                    uploadBtn.style.opacity = '0.8';
                    uploadBtn.textContent = '追踪中，点击标注球员';
                    statusBadge.textContent = '上传完成，正在自动启动全局追踪...';
                    if (typeof window.setWorkflowStep === 'function') window.setWorkflowStep(2, '【阶段2】视频解码与帧提取中...');
                    startPhaseA(uploadedVideoId);
                }
            } else {
                statusBadge.textContent = '上传失败，请重试';
                uploadBtn.textContent = '重试上传';
                uploadBtn.disabled = false;
            }
        };
        xhr.send(formData);
    }

    uploadBtn.addEventListener('click', async () => {
        if (!uploadedVideoId || isAnalyzing) return;
        // 若 Phase A 已完成但未自动触发（异常情况），手动触发分析
        if (phaseADone && !isAnalyzing) {
            window._triggerAnalysis(uploadedVideoId, [], false, [], {}, [], []);
        }
    });

    // 显示 Phase A 等待进度浮层
    function showPhaseAWaitOverlay() {
        let overlay = document.getElementById('phase-a-wait-overlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.id = 'phase-a-wait-overlay';
            overlay.style.cssText = [
                'position:fixed;inset:0;z-index:9999;',
                'background:rgba(4,11,24,0.85);backdrop-filter:blur(6px);',
                'display:flex;align-items:center;justify-content:center;'
            ].join('');
            overlay.innerHTML = `
                <div style="background:rgba(255,255,255,0.05);border:1px solid rgba(0,255,136,0.25);
                            border-radius:16px;padding:36px 40px;text-align:center;min-width:320px;max-width:440px;">
                    <div style="font-size:2.2rem;margin-bottom:12px;">
                        <i class="fas fa-satellite-dish" style="color:#00ff88;animation:fa-pulse 1.5s infinite;"></i>
                    </div>
                    <h3 style="color:#fff;margin:0 0 8px;font-size:1.1rem;">全局追踪进行中</h3>
                    <p style="color:rgba(255,255,255,0.5);font-size:0.85rem;margin:0 0 20px;">
                        系统正在对视频进行全局 YOLO 检测与球员追踪，完成后将自动启动深度分析
                    </p>
                    <div style="height:8px;background:rgba(255,255,255,0.1);border-radius:4px;overflow:hidden;margin-bottom:10px;">
                        <div id="phase-a-overlay-bar"
                             style="height:100%;width:0%;background:linear-gradient(90deg,#00ff88,#00d4ff);
                                    border-radius:4px;transition:width 0.5s ease;"></div>
                    </div>
                    <div id="phase-a-overlay-pct" style="color:#00ff88;font-size:1.3rem;font-weight:700;margin-bottom:4px;">0%</div>
                    <div id="phase-a-overlay-hint" style="color:rgba(255,255,255,0.4);font-size:0.8rem;">初始化追踪引擎...</div>
                    <button onclick="document.getElementById('phase-a-wait-overlay').remove();window._phaseAWaitingOpen=false;"
                            style="margin-top:20px;padding:8px 28px;border-radius:20px;border:1px solid rgba(255,255,255,0.2);
                                   background:transparent;color:rgba(255,255,255,0.5);cursor:pointer;font-size:0.85rem;">
                        后台等待
                    </button>
                </div>`;
            document.body.appendChild(overlay);
        }
        overlay.style.display = 'flex';
        // 实时刷新浮层进度
        updatePhaseAOverlay();
    }

    function updatePhaseAOverlay() {
        const overlay = document.getElementById('phase-a-wait-overlay');
        if (!overlay || overlay.style.display === 'none') return;
        const bar = document.getElementById('phase-a-overlay-bar');
        const pct = document.getElementById('phase-a-overlay-pct');
        const hint = document.getElementById('phase-a-overlay-hint');
        const prog = Math.max(0, phaseAProgress);
        if (bar) bar.style.width = prog + '%';
        if (pct) pct.textContent = prog + '%';
        if (hint) hint.textContent = prog < 30 ? '正在初始化 YOLO 检测引擎...'
            : prog < 60 ? '逐帧检测球员目标...'
            : prog < 90 ? '构建跨帧追踪链路...'
            : '生成追踪缓存，即将完成...';
        if (phaseADone) {
            if (bar) bar.style.width = '100%';
            if (pct) pct.textContent = '100%';
            if (hint) hint.textContent = '追踪完成！正在打开标注界面...';
            setTimeout(() => { overlay.remove(); }, 800);
        } else {
            setTimeout(updatePhaseAOverlay, 500);
        }
    }

    // ─── 全局分析触发函数（供弹窗确认/跳过调用） ───
    window._triggerAnalysis = async function (videoId, jerseyAnnotations, teamSwap, nonPlayerList, trackJerseyMap, excludedTrackIds, nonPlayerTrackIds) {
        if (!videoId || isAnalyzing) return;
        uploadedVideoId = videoId;
        window.uploadedVideoId = videoId;
        localStorage.setItem('last_uploaded_video_id', videoId);  // 持久化
        isAnalyzing = true;
        uploadBtn.disabled = true;
        uploadBtn.style.opacity = '0.6';
        uploadBtn.textContent = '正在深度分析中...';
        statusBadge.textContent = 'AI 引擎启动中，请稍候...';
        statusBadge.style.color = 'var(--accent)';

        progressContainer.style.display = 'block';
        progressBar.style.width = '0%';

        // 显示右侧面板内联进度条
        const inlineWrap = document.getElementById('inline-progress-wrap');
        const inlinePct  = document.getElementById('inline-progress-pct');
        const inlineBar  = document.getElementById('inline-progress-bar');
        const inlineHint = document.getElementById('inline-progress-hint');
        const placeholderText = document.getElementById('placeholder-main-text');
        if (inlineWrap) inlineWrap.style.display = 'block';
        if (placeholderText) placeholderText.textContent = '正在深度分析视频...';

        // ── 统一走 v2 接口（Phase B 流程），trackJerseyMap 为空时传 {} ──
        const safeTrackMap = (trackJerseyMap && typeof trackJerseyMap === 'object') ? trackJerseyMap : {};
        const safeExcluded = Array.isArray(excludedTrackIds) ? excludedTrackIds : [];
        const safeNonPlayer = Array.isArray(nonPlayerTrackIds) ? nonPlayerTrackIds : [];
        try {
            const v2Res = await fetch(`${API_BASE}/start_analysis_v2/${videoId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    track_jersey_map: safeTrackMap,
                    excluded_track_ids: safeExcluded,
                    non_player_track_ids: safeNonPlayer,
                    team_swap: !!teamSwap,
                })
            });
            // Phase A 缓存不存在时后端返回 phase_a_not_ready，给用户明确提示
            if (v2Res.ok) {
                const v2Data = await v2Res.json();
                if (v2Data.status === 'phase_a_not_ready') {
                    statusBadge.textContent = '全局追踪缓存未就绪，请等待追踪完成后重试';
                    statusBadge.style.color = '#ff6384';
                    isAnalyzing = false;
                    uploadBtn.disabled = false;
                    uploadBtn.style.opacity = '1';
                    // 重新启动 Phase A 并更新按钮状态
                    phaseADone = false;
                    startPhaseA(videoId);
                    progressContainer.style.display = 'none';
                    return;
                }
            }
        } catch (_) {}

        retryCount = 0;
        lastProgress = -1;
        staleProgressCount = 0;
        pollToken++;
        showResults(videoId, pollToken);
    };

    let retryCount = 0;
    let lastProgress = -1;       // 上一次轮询的进度值
    let staleProgressCount = 0;  // 进度无变化的轮询次数
    let pollToken = 0;           // 每次新分析递增，用于取消旧轮询
    const maxRetries = 3600; // 最多120分钟 (3600次 × 2秒，适应长视频分析)

    function getProgressHint(prog) {
        if (prog < 5)  return '正在初始化 AI 引擎...';
        if (prog < 40) return '阶段一：背号识别与球员锁定...';
        if (prog < 45) return '阶段一完成，准备深度分析...';
        if (prog < 90) return '阶段二：逐帧目标追踪与球员识别...';
        if (prog < 100) return '阶段三：生成分析报告...';
        return '分析完成！';
    }

    async function showResults(videoId, token) {
        // token 不匹配说明已启动新任务或用户切换，放弃此轮询
        if (token !== undefined && token !== pollToken) return;
        try {
            const response = await fetch(`${API_BASE}/results/${videoId}`);
            const data = await response.json();

            console.log(`[POLL] 轮询结果 (第${retryCount + 1}次):`, {
                progress: data.progress,
                hasDetailedAnalysis: !!data.detailed_analysis,
                hasAthletes: !!(data.detailed_analysis && data.detailed_analysis.athletes),
                athletesCount: data.detailed_analysis?.athletes?.length || 0
            });

            // 首先检查进度
            if (data.progress !== undefined) {
                const prog = parseInt(data.progress);
                if (prog >= 0) {
                    progressBar.style.width = `${prog}%`;
                    // 同步更新右侧内联进度条
                    const _inlineBar = document.getElementById('inline-progress-bar');
                    const _inlinePct = document.getElementById('inline-progress-pct');
                    const _inlineHint = document.getElementById('inline-progress-hint');
                    if (_inlineBar) _inlineBar.style.width = `${prog}%`;
                    if (_inlinePct) _inlinePct.textContent = `${prog}%`;

                    // ── 工作流步骤联动 ──────────────────────────────
                    const _statusMsg = data.decision_summary || '';
                    if (typeof window._onProgressUpdate === 'function') {
                        window._onProgressUpdate(prog, _statusMsg);
                    }

                    if (prog < 100) {
                        // 检测进度是否停滞（CPU模式下YOLO推理会较慢）
                        if (prog === lastProgress) {
                            staleProgressCount++;
                            // 停滞超过30次（约1分钟），提示正在处理中
                            if (staleProgressCount > 30 && staleProgressCount % 15 === 0) {
                                const dots = '.'.repeat((staleProgressCount / 15) % 4);
                                const msg = `分析进度: ${prog}%，处理中，请耐心等待${dots}`;
                                statusBadge.textContent = msg;
                                if (_inlineHint) _inlineHint.textContent = `AI 推理中，请耐心等待${dots}`;
                            } else if (staleProgressCount <= 30) {
                                statusBadge.textContent = `分析进度: ${prog}%`;
                                if (_inlineHint) _inlineHint.textContent = getProgressHint(prog);
                            }
                        } else {
                            staleProgressCount = 0;
                            lastProgress = prog;
                            statusBadge.textContent = `分析进度: ${prog}%`;
                            if (_inlineHint) _inlineHint.textContent = getProgressHint(prog);
                        }
                    } else if (prog === 100) {
                        // 进度达到100%，直接跳过后续轮询，渲染结果
                        console.log('[INFO] 分析完成，进度100%，开始渲染结果...');
                        if (_inlineHint) _inlineHint.textContent = '分析完成，正在生成报告...';
                    }
                }
            }

            // 检查是否有错误信息
            if (data.detailed_analysis && data.detailed_analysis.error) {
                statusBadge.textContent = '分析失败：' + (data.detailed_analysis.message || '未知错误');
                statusBadge.style.color = '#ff6384';
                progressContainer.style.display = 'none';
                isAnalyzing = false;
                if (heartbeatTimer) { clearInterval(heartbeatTimer); heartbeatTimer = null; }
                uploadBtn.disabled = false;
                uploadBtn.textContent = '重试分析';

                // 显示详细错误信息到状态栏
                statusBadge.textContent = '分析失败：' + (data.decision_summary || data.detailed_analysis.message || '未知错误');
                return;
            }

            // 🔑 修复3: 完成判断（兼容 highlight_only 模式下 athletes 为空）
            const _da = data.detailed_analysis;
            const _isHighlightOnly = _da && (_da.mode === 'highlight_only' ||
                (_da.highlight_segments && Array.isArray(_da.highlight_segments) && _da.highlight_segments.length > 0 && (!_da.athletes || _da.athletes.length === 0)));
            const isComplete = (
                data.progress === 100 &&
                _da &&
                (
                    _isHighlightOnly ||
                    (_da.athletes && Array.isArray(_da.athletes) && _da.athletes.length > 0)
                )
            );

            if (!isComplete) {
                // 特殊处理：progress=100但数据未准备好（竞态条件）
                if (data.progress === 100 && !data.detailed_analysis) {
                    console.warn('[WARN] 进度100%但数据未就绪，延迟1秒重试...');
                    retryCount++;  // 增加重试计数
                    setTimeout(() => showResults(videoId, token), 1000);
                    return;
                }

                // 分析未完成，继续轮询
                if (data.progress === -1) {
                    statusBadge.textContent = '分析失败：后端处理异常';
                    statusBadge.style.color = '#ff6384';
                    isAnalyzing = false;
                    if (heartbeatTimer) { clearInterval(heartbeatTimer); heartbeatTimer = null; }
                    uploadBtn.disabled = false;
                    uploadBtn.textContent = '重试分析';
                    return;
                }

                // 检查重试次数
                retryCount++;
                if (retryCount >= maxRetries) {
                    statusBadge.textContent = '分析超时（超过120分钟未响应）';
                    statusBadge.style.color = '#ff6384';
                    progressContainer.style.display = 'none';
                    isAnalyzing = false;
                    if (heartbeatTimer) { clearInterval(heartbeatTimer); heartbeatTimer = null; }
                    uploadBtn.disabled = false;
                    uploadBtn.textContent = '重试分析';
                    return;
                }

                // 🔑 优化：根据进度调整轮询频率
                // 进度 < 50%: 每2秒
                // 进度 50-90%: 每1.5秒
                // 进度 = 100%: 每500ms（快速响应竞态条件）
                // 进度 > 90%: 每1秒
                const pollInterval = data.progress < 50 ? 2000 :
                                   data.progress < 90 ? 1500 :
                                   data.progress === 100 ? 500 :  // 100%时更快
                                   1000;

                setTimeout(() => showResults(videoId, token), pollInterval);
                return;
            }

            // ✅ 分析完成，渲染结果
            console.log('[SUCCESS] 分析完成，开始渲染结果页面');
            retryCount = 0;

            // 完成：停止心跳轮询
            if (heartbeatTimer) { clearInterval(heartbeatTimer); heartbeatTimer = null; }
            isAnalyzing = false;
            statusBadge.textContent = '分析任务圆满完成';
            statusBadge.style.color = 'var(--primary)';
            progressContainer.style.display = 'none';
            uploadBtn.textContent = '分析完成';
            uploadBtn.disabled = true;

            // ── 通知工作流步骤进入第5步（导出），激活导出按钮 ──
            window._lastAnalyzedId = videoId;
            localStorage.setItem('last_uploaded_video_id', videoId);  // 覆盖持久化，确保最新
            if (typeof window._onProgressUpdate === 'function') window._onProgressUpdate(100, '分析完成');
            document.dispatchEvent(new CustomEvent('analysisCompleted', { detail: { videoId: videoId } }));

            const _daFinal = data.detailed_analysis || {};
            if (_daFinal.mode === 'highlight_only' ||
                (_daFinal.highlight_segments && _daFinal.highlight_segments.length > 0 && (!_daFinal.athletes || _daFinal.athletes.length === 0))) {
                // 精彩事件检测模式：自动跳到精彩视频页
                console.log('[SUCCESS] highlight_only 模式，跳转精彩视频页');
                statusBadge.textContent = '精彩集锦已生成！';
                if (typeof window.showPage === 'function') {
                    setTimeout(() => window.showPage('highlight'), 600);
                }
            } else {
                drawTracking(data.detailed_analysis);
            }
        } catch (error) {
            console.warn('[POLL] 轮询请求失败:', error.message);
            retryCount++;
            if (retryCount < maxRetries) {
                statusBadge.textContent = `连接后端中... (${retryCount})`;
                setTimeout(() => showResults(videoId, token), 5000);
            } else {
                statusBadge.textContent = '连接超时，请检查后端服务';
                statusBadge.style.color = '#ff6384';
                isAnalyzing = false;
                uploadBtn.disabled = false;
                uploadBtn.textContent = '重试分析';
            }
        }
    }

    function drawTracking(analysis) {
        const canvas = document.getElementById('tracking-overlay');
        const video = document.getElementById('video-preview');
        if (!canvas || !video) return;
        const ctx = canvas.getContext('2d');
        canvas.width = video.clientWidth;
        canvas.height = video.clientHeight;
        analysis.athletes.forEach((player, idx) => {
            const color = idx < 2 ? '#00ff88' : '#2288ff';
            if (player.actual_path && player.actual_path.length > 0) {
                ctx.beginPath();
                ctx.strokeStyle = color;
                ctx.lineWidth = 3;
                ctx.moveTo((player.actual_path[0].x / video.videoWidth) * canvas.width, (player.actual_path[0].y / video.videoHeight) * canvas.height);
                player.actual_path.forEach(p => ctx.lineTo((p.x / video.videoWidth) * canvas.width, (p.y / video.videoHeight) * canvas.height));
                ctx.stroke();
            }
        });
    }

    // ── 全局心跳：每2秒检查后端分析状态，驱动进度条更新 ──────────────
    async function heartbeat() {
        try {
            const r = await fetch(`${API_BASE}/api/analyzing`);
            if (!r.ok) return;
            const data = await r.json();

            if (!data.tasks || data.tasks.length === 0) {
                // 没有进行中的任务
                if (isAnalyzing && uploadedVideoId) {
                    // 可能刚刚完成，触发一次 showResults 渲染最终结果
                    // showResults 内部会判断完成状态
                }
                return;
            }

            // 取进度最高的任务
            const task = data.tasks.sort((a, b) => b.progress - a.progress)[0];

            // ── 情况1: 前端未在分析状态，自动恢复 ──
            // 若标注弹窗正在显示（用户还在标注），跳过自动恢复，不干扰标注流程
            const jerseyModalVisible = document.getElementById('jersey-modal') &&
                document.getElementById('jersey-modal').style.display !== 'none';
            if (!isAnalyzing && !jerseyModalVisible) {
                console.log(`[HEARTBEAT] 检测到进行中的分析任务: ${task.video_id} (${task.progress}%)`);
                uploadedVideoId = task.video_id;
                window.uploadedVideoId = task.video_id;
                localStorage.setItem('last_uploaded_video_id', task.video_id);  // 持久化
                isAnalyzing = true;
                uploadBtn.disabled = true;
                uploadBtn.textContent = '正在深度分析中...';
                statusBadge.textContent = `恢复分析进度: ${task.progress}%`;
                statusBadge.style.color = 'var(--accent)';
                progressContainer.style.display = 'block';
                progressBar.style.width = `${task.progress}%`;
                const inlineWrap = document.getElementById('inline-progress-wrap');
                const inlinePct  = document.getElementById('inline-progress-pct');
                const inlineBar  = document.getElementById('inline-progress-bar');
                const inlineHint = document.getElementById('inline-progress-hint');
                const placeholderText = document.getElementById('placeholder-main-text');
                if (inlineWrap) inlineWrap.style.display = 'block';
                if (inlinePct)  inlinePct.textContent = `${task.progress}%`;
                if (inlineBar)  inlineBar.style.width = `${task.progress}%`;
                if (inlineHint) inlineHint.textContent = getProgressHint(task.progress);
                if (placeholderText) placeholderText.textContent = '正在深度分析视频...';
                retryCount = 0; lastProgress = task.progress; staleProgressCount = 0;
                pollToken++;
                showResults(task.video_id, pollToken);
                return;
            }

            // ── 情况2: 前端已在分析状态，心跳直接更新进度 UI（保底） ──
            if (isAnalyzing && uploadedVideoId) {
                const prog = task.progress;
                progressBar.style.width = `${prog}%`;
                const _inlineBar = document.getElementById('inline-progress-bar');
                const _inlinePct = document.getElementById('inline-progress-pct');
                const _inlineHint = document.getElementById('inline-progress-hint');
                if (_inlineBar) _inlineBar.style.width = `${prog}%`;
                if (_inlinePct) _inlinePct.textContent = `${prog}%`;
                if (_inlineHint) _inlineHint.textContent = getProgressHint(prog);
                if (prog !== lastProgress) {
                    lastProgress = prog;
                    staleProgressCount = 0;
                    statusBadge.textContent = `分析进度: ${prog}%`;
                }
            }
        } catch (e) {
            // 后端未启动或网络问题，静默忽略
        }
    }

    // 立即执行一次，再每2秒循环
    heartbeat();
    heartbeatTimer = setInterval(heartbeat, 2000);
});

// =====================================================================
//  球员号码标注弹窗 —— 独立模块
// =====================================================================
(function () {
    let _videoId    = null;
    let _frameUrl   = null;
    let _detections = [];
    let _frameW     = 1280;
    let _frameH     = 720;
    let _activeDet  = null;
    let _img        = null;
    let _teamColors = { team_a: [128,128,128], team_b: [128,128,128] };
    let _homeTeam   = null;
    // ── 多帧状态 ──
    let _allFrames         = [];   // 后端返回的所有帧数据
    let _curFrameIdx       = 0;    // 当前帧下标
    let _globalAnnotations = {};   // 跨帧持久化标注 {posKey: {jersey,team,x,y}}
    let _frameDetections   = {};   // {frameIdx: detections[]}
    const API       = window.location.origin;

    // ── roundRect 兼容 polyfill ──
    if (!CanvasRenderingContext2D.prototype.roundRect) {
        CanvasRenderingContext2D.prototype.roundRect = function(x, y, w, h, r) {
            r = Math.min(r, w / 2, h / 2);
            this.moveTo(x + r, y);
            this.arcTo(x + w, y,     x + w, y + h, r);
            this.arcTo(x + w, y + h, x,     y + h, r);
            this.arcTo(x,     y + h, x,     y,     r);
            this.arcTo(x,     y,     x + w, y,     r);
            this.closePath();
        };
    }

    window.showJerseyModal = function (videoId) {
        _videoId    = videoId;
        _detections = [];
        _activeDet  = null;
        _frameUrl   = null;
        _img        = null;
        _teamColors = { team_a: [128,128,128], team_b: [128,128,128] };
        _homeTeam   = null;

        // ── 多帧状态 ──
        _allFrames      = [];   // 后端返回的全部帧数据
        _curFrameIdx    = 0;    // 当前显示的帧下标
        _globalAnnotations = {}; // jerseyNumber → {team} 跨帧持久化标注
        _frameDetections = {};  // frameIdx → detections[]（每帧独立检测框，含 jersey 状态）

        const modal = document.getElementById('jersey-modal');
        modal.style.display = 'block';
        document.body.style.overflow = 'hidden';

        // 确保第一步可见，第二步隐藏
        const step1Footer = document.getElementById('jersey-step1-footer');
        const step2Panel  = document.getElementById('jersey-team-select-panel');
        if (step1Footer) step1Footer.style.display = 'flex';
        if (step2Panel)  step2Panel.style.display  = 'none';

        document.getElementById('jersey-frame-loading').style.display = 'flex';
        document.getElementById('jersey-loading-text').textContent = '正在提取关键帧...';
        document.getElementById('jersey-modal-subtitle').textContent = 'AI 正在定位关键帧，请稍候...';

        _fetchKeyFrame(videoId);
    };

    window.closeJerseyModal = function () {
        document.getElementById('jersey-modal').style.display = 'none';
        document.body.style.overflow = '';
        _hideBubble();
    };

    // skipJerseyModal 已移除：标注是必须流程，不允许跳过

    // 第一步完成 → 进入第二步（选择主队颜色）
    window.showTeamSelectPanel = function () {
        const step1Footer = document.getElementById('jersey-step1-footer');
        const step2Panel  = document.getElementById('jersey-team-select-panel');
        if (step1Footer) step1Footer.style.display = 'none';
        if (step2Panel)  step2Panel.style.display  = 'block';
        _hideBubble();

        // 渲染两队颜色色块
        const swatchA = document.getElementById('team-color-swatch-a');
        const swatchB = document.getElementById('team-color-swatch-b');
        if (swatchA) {
            const [r, g, b] = _teamColors.team_a;
            swatchA.style.background = `rgb(${r},${g},${b})`;
        }
        if (swatchB) {
            const [r, g, b] = _teamColors.team_b;
            swatchB.style.background = `rgb(${r},${g},${b})`;
        }

        // 重置选中状态
        _homeTeam = null;
        _resetTeamSelectUI();
    };

    // 返回第一步
    window.backToStep1 = function () {
        const step1Footer = document.getElementById('jersey-step1-footer');
        const step2Panel  = document.getElementById('jersey-team-select-panel');
        if (step1Footer) step1Footer.style.display = 'flex';
        if (step2Panel)  step2Panel.style.display  = 'none';
    };

    // 用户点击选择主队
    window.selectHomeTeam = function (team) {
        _homeTeam = team;
        _resetTeamSelectUI();

        // 高亮选中的色块
        const selId = team === 'team_a' ? 'team-select-a' : 'team-select-b';
        const selEl = document.getElementById(selId);
        if (selEl) {
            selEl.style.borderColor = 'rgba(255,130,40,0.8)';
            selEl.style.background  = 'rgba(255,130,40,0.1)';
            selEl.classList.add('home-selected');
            const badge = selEl.querySelector('.home-badge');
            if (badge) badge.style.display = 'block';
        }

        // 解锁提交按钮
        const btn = document.getElementById('jersey-submit-btn');
        if (btn) {
            btn.disabled         = false;
            btn.style.background = 'linear-gradient(135deg,rgba(0,255,136,0.22),rgba(0,212,255,0.15))';
            btn.style.border     = '1px solid rgba(0,255,136,0.4)';
            btn.style.color      = '#00ff88';
            btn.style.cursor     = 'pointer';
        }
    };

    function _resetTeamSelectUI() {
        ['team-select-a', 'team-select-b'].forEach(id => {
            const el = document.getElementById(id);
            if (!el) return;
            el.classList.remove('home-selected');
            el.style.borderColor = 'rgba(255,255,255,0.12)';
            el.style.background  = 'rgba(255,255,255,0.04)';
            const badge = el.querySelector('.home-badge');
            if (badge) badge.style.display = 'none';
        });
    }

    // 第二步确认提交
    window.confirmJerseyModal = function () {
        if (!_homeTeam) return;   // 未选主队时不提交
        const teamSwap = (_homeTeam === 'team_b');

        // ── 汇总所有帧的标注 ──
        _saveCurrentFrameAnnotations();

        const jerseyMap = {};           // jerseyNumber → {jersey_number, team}  (旧方案兼容)
        const nonPlayerList = [];       // 旧方案兼容
        const trackJerseyMap = {};      // track_id → {jersey, team}  (v2 精确方案)
        const excludedTrackIds = [];    // 排除的 track_id 列表（视频渲染用：non_player + 未标注）
        const nonPlayerTrackIds = [];   // 仅明确标为非球员的 track_id（排除用）
        const seenExcluded = new Set();
        const seenNonPlayer = new Set();

        // 第一遍：先收集所有已标注号码的 track_id（以此为准，不能被 excluded 覆盖）
        const annotatedTrackIds = new Set();
        Object.values(_frameDetections).forEach(dets => {
            (dets || []).forEach(d => {
                if (d.team !== 'non_player' && d.jersey != null && d.track_id) {
                    annotatedTrackIds.add(d.track_id);
                    trackJerseyMap[d.track_id] = { jersey: d.jersey, team: d.team };
                    jerseyMap[d.jersey] = { det_id: d.id, jersey_number: d.jersey, team: d.team };
                }
            });
        });

        // 第二遍：收集非球员 & 未标注（只要不在已标注集合里才能进 excluded）
        // 关键修复：同一 track_id 在某帧已标注、在另一帧未标注是正常现象（用户只在关键帧标注），
        //           以"标注过"为准，不能因为在其他帧未标注就把它加入 excludedTrackIds。
        Object.values(_frameDetections).forEach(dets => {
            (dets || []).forEach(d => {
                if (d.team === 'non_player') {
                    nonPlayerList.push({ det_id: d.id, x: d.x, y: d.y, w: d.w, h: d.h });
                    if (d.track_id && !annotatedTrackIds.has(d.track_id) && !seenExcluded.has(d.track_id)) {
                        excludedTrackIds.push(d.track_id);
                        seenExcluded.add(d.track_id);
                    }
                    if (d.track_id && !seenNonPlayer.has(d.track_id)) {
                        nonPlayerTrackIds.push(d.track_id);
                        seenNonPlayer.add(d.track_id);
                    }
                } else if (d.jersey == null) {
                    // 未标注号码的球员：已标注过的 track 不再加入 excluded
                    nonPlayerList.push({ det_id: d.id, x: d.x, y: d.y, w: d.w, h: d.h });
                    if (d.track_id && !annotatedTrackIds.has(d.track_id) && !seenExcluded.has(d.track_id)) {
                        excludedTrackIds.push(d.track_id);
                        seenExcluded.add(d.track_id);
                    }
                }
            });
        });

        const annotations = Object.values(jerseyMap);

        // ── 人数合规校验（每队最多11人）──
        const countA = annotations.filter(a => a.team === 'team_a').length;
        const countB = annotations.filter(a => a.team === 'team_b').length;
        const MAX_TEAM = 11;
        if (countA > MAX_TEAM || countB > MAX_TEAM) {
            const overA = countA > MAX_TEAM ? `主队(橙色)已标 ${countA} 人` : '';
            const overB = countB > MAX_TEAM ? `客队(蓝色)已标 ${countB} 人` : '';
            const overMsg = [overA, overB].filter(Boolean).join('、');
            const confirmed = window.confirm(
                `⚠️ 标准足球每队上场 11 人，当前 ${overMsg}，超出部分将被系统自动截断，只保留出现最多的前11名球员。\n\n建议返回修正后再提交，是否继续？`
            );
            if (!confirmed) return;
        }

        window.closeJerseyModal();
        window._triggerAnalysis(_videoId, annotations, teamSwap, nonPlayerList, trackJerseyMap, excludedTrackIds, nonPlayerTrackIds);
    };

    window.resetJerseyLabels = function () {
        // 清除全部帧的标注状态
        Object.values(_frameDetections).forEach(dets => {
            (dets || []).forEach((d, i) => {
                d.jersey = null;
                if (d.team === 'non_player') {
                    d.team = d._prevTeam || (i % 2 === 0 ? 'team_a' : 'team_b');
                }
            });
        });
        // 清除跨帧全局标注缓存，防止回填复现
        _globalAnnotations = {};
        // 重新加载当前帧
        _detections = _frameDetections[_curFrameIdx] || [];
        _hideBubble();
        _activeDet = null;
        _render();
        _updateProgress();
    };

    window.cancelJerseyInput = function () { _hideBubble(); _activeDet = null; _render(); };

    // ── 辅助：将 _detections 中的修改同步回 _frameDetections[_curFrameIdx] ──
    function _syncDetToFrameStore(det) {
        const stored = (_frameDetections[_curFrameIdx] || []).find(d => d.id === det.id);
        if (stored) {
            stored.team      = det.team;
            stored.jersey    = det.jersey;
            stored._prevTeam = det._prevTeam;
        }
    }

    // ── 标记/撤销非球员（通过 track_id 跨帧同步） ──
    window.markAsNonPlayer = function () {
        if (_activeDet === null) return;
        const det = _detections.find(d => d.id === _activeDet);
        if (!det) return;

        const isUndoing = (det.team === 'non_player');
        const tid = det.track_id || null;

        if (isUndoing) {
            det.team      = det._prevTeam || 'team_a';
            det.jersey    = null;
            det._autoFilled = false; // 撤销时清除自动回填标志，确保 _saveCurrentFrameAnnotations 能保存
        } else {
            det._prevTeam   = det.team;
            det.team        = 'non_player';
            det.jersey      = null;
            det._autoFilled = false; // 用户主动标注，不应是自动回填
        }

        // 同步回当前帧的 _frameDetections 原始存储
        _syncDetToFrameStore(det);

        // 通过 track_id 同步到其他帧的同一追踪目标
        if (tid) {
            Object.keys(_frameDetections).forEach(fi => {
                if (parseInt(fi) === _curFrameIdx) return;
                (_frameDetections[fi] || []).forEach(fd => {
                    if (fd.track_id !== tid) return;
                    if (isUndoing) {
                        // 撤销：恢复所有帧中同 track_id 的非球员标记
                        if (fd.team === 'non_player') {
                            fd.team       = fd._prevTeam || 'team_a';
                            fd.jersey     = null;
                            fd._autoFilled = false;
                        }
                    } else {
                        // 标记：将所有帧中同 track_id 设为非球员
                        if (fd.team !== 'non_player') {
                            fd._prevTeam  = fd.team;
                            fd.team       = 'non_player';
                            fd.jersey     = null;
                            fd._autoFilled = false;
                        }
                    }
                });
            });

            // 同步清理/添加全局标注中同 track_id 的非球员记录
            if (isUndoing) {
                // 撤销：删除 _globalAnnotations 中所有属于该 track_id 的非球员记录
                Object.keys(_globalAnnotations).forEach(key => {
                    const ann = _globalAnnotations[key];
                    if (ann.track_id === tid && ann.isNonPlayer) {
                        delete _globalAnnotations[key];
                    }
                });
            } else {
                // 标记：在 _globalAnnotations 中写入/更新该 track_id 的非球员标注
                // 使用当前帧的 det 信息作为来源记录
                const npKey = `np_tid_${tid}`;
                _globalAnnotations[npKey] = {
                    jersey: null, team: 'non_player', isNonPlayer: true,
                    track_id: tid,
                    x: det.x, y: det.y, w: det.w, h: det.h,
                    fromFrame: _curFrameIdx,
                };
            }
        }

        _hideBubble();
        _activeDet = null;
        _render();
        _updateProgress();
    };

    window.confirmJerseyInput = function () {
        const val = parseInt(document.getElementById('jersey-number-input').value);
        if (_activeDet !== null && val >= 1 && val <= 99) {
            const det = _detections.find(d => d.id === _activeDet);
            if (det) {
                det.jersey = val;
                _syncDetToFrameStore(det);  // 同步回原始存储
            }
        }
        _hideBubble();
        _activeDet = null;
        _render();
        _updateProgress();
    };

    async function _fetchKeyFrame(videoId) {
        try {
            // ── Phase A：先触发全局追踪，然后轮询直到完成 ──
            document.getElementById('jersey-loading-text').textContent = '正在进行全局追踪分析，请稍候...';

            // 1. 启动 Phase A
            const preTrackRes = await fetch(`${API}/pre_track/${videoId}`, { method: 'POST' });
            const preTrackData = await preTrackRes.json();

            // 2. 轮询 Phase A 进度（除非已缓存）
            if (preTrackData.status !== 'cached') {
                let phaseADone = false;
                // 最长等待 95 分钟（1小时视频 Phase A 约需 50-80 分钟）
                const POLL_TIMEOUT_MS = 95 * 60 * 1000;
                const pollStart = Date.now();
                let consecutiveErrors = 0;

                while (!phaseADone) {
                    await new Promise(r => setTimeout(r, 3000));

                    // 超时保护
                    if (Date.now() - pollStart > POLL_TIMEOUT_MS) {
                        throw new Error('Phase A 超时（超过95分钟），请检查服务器日志');
                    }

                    let progData;
                    try {
                        const progRes = await fetch(`${API}/pre_track_progress/${videoId}`);
                        if (!progRes.ok) {
                            consecutiveErrors++;
                            if (consecutiveErrors >= 5) throw new Error('服务器连续5次无响应，可能已重启');
                            continue;
                        }
                        progData = await progRes.json();
                        consecutiveErrors = 0;
                    } catch (fetchErr) {
                        consecutiveErrors++;
                        if (consecutiveErrors >= 5) throw fetchErr;
                        document.getElementById('jersey-loading-text').textContent =
                            `连接中断，重试中 (${consecutiveErrors}/5)...`;
                        await new Promise(r => setTimeout(r, 5000));
                        continue;
                    }

                    const p = Number(progData.progress) || 0;
                    const elapsed = Math.round((Date.now() - pollStart) / 1000 / 60);
                    document.getElementById('jersey-loading-text').textContent =
                        `全局追踪中... ${Math.max(0, Math.round(p))}%（已用时 ${elapsed} 分钟）`;

                    // cached=true 是最可靠的完成信号（pkl 写入后才设置）
                    if (progData.cached) {
                        phaseADone = true;
                    } else if (p >= 95) {
                        // 再额外等 2 秒确认 pkl 已写完
                        await new Promise(r => setTimeout(r, 2000));
                        phaseADone = true;
                    } else if (p < 0) {
                        // 进度为 -1 可能是服务器重启导致内存丢失，再查一次 cached 状态
                        // 后端已修复：若 pkl 存在会将 progress 修正为 95，此处仍做保底检查
                        throw new Error('Phase A 追踪分析失败（progress=-1），请重新上传或检查服务器日志');
                    }
                }
            }

            document.getElementById('jersey-loading-text').textContent = '正在选取标注帧...';

            // 3. 调用 v2 选帧接口
            const res  = await fetch(`${API}/quick_detect_v2/${videoId}`);
            if (!res.ok) throw new Error('服务器响应异常 ' + res.status);
            const data = await res.json();

            // ── 多帧支持 ──
            _allFrames = data.frames && data.frames.length > 0
                ? data.frames
                : [{ frame_url: data.frame_url, frame_time: data.frame_time,
                     player_count: (data.detections||[]).length,
                     detections: data.detections || [], team_colors: data.team_colors }];

            _frameW     = data.frame_w  || 1280;
            _frameH     = data.frame_h  || 720;
            _teamColors = data.team_colors || { team_a: [128,128,128], team_b: [128,128,128] };

            // 初始化每帧检测框（加入 jersey/id/track_id 字段）
            // 当 Phase A 颜色聚类失败时 d.team 可能为 'unknown'，
            // 此时用 A/B 交替分配作为初始 team，确保提交时 trackJerseyMap 里有有效 team
            _allFrames.forEach((fd, fi) => {
                _frameDetections[fi] = (fd.detections || []).map((d, i) => ({
                    id    : fi * 1000 + i,
                    x: d.x, y: d.y, w: d.w, h: d.h,
                    team  : (d.team && d.team !== 'unknown') ? d.team : (i % 2 === 0 ? 'team_a' : 'team_b'),
                    jersey: null,
                    track_id: d.track_id || null,
                    _fromFrame: fi,
                }));
            });

            _curFrameIdx = 0;
            _switchToFrame(0);

        } catch (e) {
            console.warn('[JerseyModal] 关键帧提取失败:', e.message);
            _showLoadError('关键帧提取失败，请稍后重试或刷新页面');
        }
    }

    // ── 切换到指定帧 ─────────────────────────────────────────────────────
    function _switchToFrame(fi) {
        if (!_allFrames || fi < 0 || fi >= _allFrames.length) return;

        // 保存当前帧的标注到全局
        _saveCurrentFrameAnnotations();

        _curFrameIdx = fi;
        const fd = _allFrames[fi];
        _frameUrl   = fd.frame_url;
        _teamColors = fd.team_colors || _teamColors;

            // 恢复该帧检测框（深拷贝，防止回填结果污染 _frameDetections 的原始存储）
        _detections = (_frameDetections[fi] || []).map(d => Object.assign({}, d));
        _applyGlobalAnnotations(_detections, fi);

        // 更新导航 UI
        _updateFrameNav();

        // 加载帧图片
        _activeDet = null;
        _hideBubble();
        document.getElementById('jersey-frame-loading').style.display = 'flex';
        document.getElementById('jersey-loading-text').textContent = '正在加载关键帧...';

        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
            _img = img;
            document.getElementById('jersey-frame-loading').style.display = 'none';
            const totalAnnotated = _countGlobalAnnotations();
            const totalPlayers   = _allFrames.reduce((s, f) => s + (f.player_count || 0), 0);
            document.getElementById('jersey-modal-subtitle').textContent =
                `帧 ${fi + 1}/${_allFrames.length}｜第 ${fd.frame_time || '--'} 秒｜本帧 ${_detections.length} 人｜共已标注 ${totalAnnotated} 人`;
            // 更新图例颜色
            const dotA = document.getElementById('legend-dot-a');
            const dotB = document.getElementById('legend-dot-b');
            if (dotA) { const [r,g,b] = _teamColors.team_a; dotA.style.background = `rgba(${r},${g},${b},0.6)`; dotA.style.borderColor = `rgb(${r},${g},${b})`; }
            if (dotB) { const [r,g,b] = _teamColors.team_b; dotB.style.background = `rgba(${r},${g},${b},0.6)`; dotB.style.borderColor = `rgb(${r},${g},${b})`; }
            _render();
            _updateProgress();
        };
        img.onerror = () => _showLoadError('帧图加载失败');
        img.src = `${API}${_frameUrl}?_t=${Date.now()}`;
    }

    // 保存当前帧已标注号码到全局（含 track_id、isNonPlayer）
    function _saveCurrentFrameAnnotations() {
        if (!_detections || _detections.length === 0) return;
        _detections.forEach(d => {
            if (d._autoFilled) {
                // 自动回填的框不写入全局标注；
                // 但若该框是回填的非球员且 track_id 已被用户撤销，
                // 需清除 _globalAnnotations 中残留的同 track_id 非球员记录
                // （由之前帧保存的 fX_id key）
                if (d.track_id) {
                    const npKey = `np_tid_${d.track_id}`;
                    if (!_globalAnnotations[npKey]) {
                        // np_tid_ key 已不存在，说明已撤销，清理同 tid 的所有 fX_id 非球员记录
                        Object.keys(_globalAnnotations).forEach(k => {
                            const a = _globalAnnotations[k];
                            if (a && a.track_id === d.track_id && a.isNonPlayer) {
                                delete _globalAnnotations[k];
                            }
                        });
                    }
                }
                return;
            }

            const key = `f${_curFrameIdx}_${d.id}`;

            if (d.team === 'non_player') {
                // 非球员标记也存入全局（用于跨帧同步）
                _globalAnnotations[key] = {
                    jersey: null, team: 'non_player', isNonPlayer: true,
                    track_id: d.track_id || null,
                    x: d.x, y: d.y, w: d.w, h: d.h,
                    fromFrame: _curFrameIdx
                };
            } else {
                // 普通球员框（含撤销后恢复的）：删除该框曾经留下的非球员记录
                delete _globalAnnotations[key];
                if (d.jersey != null) {
                    _globalAnnotations[key] = {
                        jersey: d.jersey, team: d.team, isNonPlayer: false,
                        track_id: d.track_id || null,
                        x: d.x, y: d.y, w: d.w, h: d.h,
                        fromFrame: _curFrameIdx
                    };
                }
            }
        });
        // 将用户主动标注的框同步回 _frameDetections（仅保存主动标注，不保存自动回填结果）
        if (_curFrameIdx !== undefined && (_frameDetections[_curFrameIdx] || []).length > 0) {
            _frameDetections[_curFrameIdx] = _detections.map(d => {
                if (d._autoFilled) {
                    const orig = Object.assign({}, d);
                    orig.jersey = null;
                    if (orig.team === 'non_player') orig.team = orig._prevTeam || 'team_a';
                    orig._autoFilled = false;
                    return orig;
                }
                return Object.assign({}, d);
            });
        }
    }

    // 跨帧一对一回填：优先使用 track_id 精确匹配，降级到坐标距离匹配
    function _applyGlobalAnnotations(detections, excludeFrame) {
        // 只取来自其他帧的标注
        const anns = Object.values(_globalAnnotations).filter(a => a.fromFrame !== excludeFrame);
        if (anns.length === 0) return;

        const usedDetIds  = new Set(); // 已匹配的检测框 id
        const usedTidAnns = new Set(); // 已匹配的 ann.track_id（防止同一 ann 重复匹配多框）

        // ── 第一轮：track_id 精确匹配（球员号码回填） ──
        const unset = detections.filter(d => d.jersey == null && d.team !== 'non_player');
        for (const d of unset) {
            if (!d.track_id) continue;
            for (const ann of anns) {
                if (!ann.track_id || ann.isNonPlayer) continue;
                if (usedDetIds.has(d.id) || usedTidAnns.has(ann.track_id)) continue;
                if (d.track_id === ann.track_id) {
                    d.jersey      = ann.jersey;
                    d.team        = ann.team;
                    d._autoFilled = true;
                    usedDetIds.add(d.id);
                    usedTidAnns.add(ann.track_id);
                    break;
                }
            }
        }

        // ── 第二轮：track_id 同步非球员标记 ──
        for (const d of detections) {
            if (!d.track_id || usedDetIds.has(d.id)) continue;
            for (const ann of anns) {
                if (!ann.track_id || !ann.isNonPlayer) continue;
                if (ann.track_id === d.track_id) {
                    d._prevTeam   = d.team;
                    d.team        = 'non_player';
                    d.jersey      = null;
                    d._autoFilled = true;
                    usedDetIds.add(d.id);
                    break;
                }
            }
        }

        // ── 第三轮：坐标距离降级匹配（无 track_id 的球员框） ──
        const stillUnset = detections.filter(d =>
            d.jersey == null && d.team !== 'non_player' && !usedDetIds.has(d.id));
        // 只取未通过 track_id 匹配过的、且非非球员的 ann
        const coordAnns = anns.filter(a =>
            !a.isNonPlayer && (!a.track_id || !usedTidAnns.has(a.track_id)));
        if (stillUnset.length === 0 || coordAnns.length === 0) return;

        const CROSS_FRAME_TOL = 80;
        const pairs = [];
        for (const d of stillUnset) {
            const cx = d.x + d.w / 2;
            const cy = d.y + d.h / 2;
            for (const ann of coordAnns) {
                const annCx = ann.x + ann.w / 2;
                const annCy = ann.y + ann.h / 2;
                const dx = Math.abs(annCx - cx);
                const dy = Math.abs(annCy - cy);
                if (dx < CROSS_FRAME_TOL && dy < CROSS_FRAME_TOL) {
                    pairs.push({ d, ann, dist: dx * dx + dy * dy });
                }
            }
        }

        // 按距离排序，贪心一对一匹配
        const usedCoordAnns = new Set();
        pairs.sort((a, b) => a.dist - b.dist);
        for (const { d, ann } of pairs) {
            if (usedDetIds.has(d.id)) continue;
            const annId = `${ann.fromFrame}_${ann.jersey}_${ann.team}`;
            if (usedCoordAnns.has(annId)) continue;
            d.jersey      = ann.jersey;
            d.team        = ann.team;
            d._autoFilled = true;
            usedDetIds.add(d.id);
            usedCoordAnns.add(annId);
        }
    }

    // 计算全局已标注总人数（去重：按 jersey+team 唯一，仅统计球员，不含非球员）
    function _countGlobalAnnotations() {
        const seen = new Set();
        Object.values(_globalAnnotations).forEach(a => {
            if (!a.isNonPlayer && a.jersey != null) seen.add(`${a.jersey}_${a.team}`);
        });
        return seen.size;
    }

    // 更新帧导航栏 UI
    function _updateFrameNav() {
        const nav = document.getElementById('jersey-frame-nav');
        if (!nav || !_allFrames) return;
        if (_allFrames.length <= 1) { nav.style.display = 'none'; return; }
        nav.style.display = 'flex';
        document.getElementById('jersey-frame-prev').disabled = (_curFrameIdx === 0);
        document.getElementById('jersey-frame-next').disabled = (_curFrameIdx >= _allFrames.length - 1);
        document.getElementById('jersey-frame-indicator').textContent =
            `帧 ${_curFrameIdx + 1} / ${_allFrames.length}`;
        // 帧点指示器
        const dots = document.getElementById('jersey-frame-dots');
        if (dots) {
            dots.innerHTML = _allFrames.map((f, i) => {
                const annotated = (_frameDetections[i] || []).filter(d => d.jersey != null).length;
                const active = i === _curFrameIdx;
                return `<span onclick="jerseyGoToFrame(${i})" title="帧${i+1}｜${f.player_count||0}人${annotated?'｜已标'+annotated:''}"
                    style="display:inline-block;width:${active?'22px':'10px'};height:10px;border-radius:5px;cursor:pointer;
                    background:${active?'#00ff88':annotated?'rgba(0,255,136,0.4)':'rgba(255,255,255,0.2)'};
                    border:1.5px solid ${active?'#00ff88':annotated?'rgba(0,255,136,0.6)':'rgba(255,255,255,0.2)'};
                    transition:all 0.2s;margin:0 2px;"></span>`;
            }).join('');
        }
    }

    // 帧导航：上一帧 / 下一帧 / 跳转（_switchToFrame 内部已调用 _saveCurrentFrameAnnotations，不重复调用）
    window.jerseyPrevFrame = function () { _switchToFrame(_curFrameIdx - 1); };
    window.jerseyNextFrame = function () { _switchToFrame(_curFrameIdx + 1); };
    window.jerseyGoToFrame = function (fi) { _switchToFrame(fi); };

    function _showLoadError(msg) {
        const loadingEl = document.getElementById('jersey-frame-loading');
        const textEl    = document.getElementById('jersey-loading-text');
        if (textEl) textEl.textContent = msg;
        if (loadingEl) {
            const spin = loadingEl.querySelector('div');
            if (spin) spin.style.display = 'none';
        }
    }

    function _render() {
        const canvas = document.getElementById('jersey-canvas');
        if (!canvas || !_img) return;
        canvas.width  = _frameW;
        canvas.height = _frameH;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(_img, 0, 0, _frameW, _frameH);

        _detections.forEach(d => {
            const isActive    = d.id === _activeDet;
            const isNonPlayer = d.team === 'non_player';
            const isLabeled   = d.jersey != null && !isNonPlayer;
            const tc          = _teamColors[d.team] || [128, 128, 128];
            const [tr, tg, tb] = tc;

            let fillColor, strokeColor, labelColor;
            if (isNonPlayer) {
                fillColor   = 'rgba(255,180,0,0.18)';
                strokeColor = isActive ? '#fff' : '#ffb400';
                labelColor  = '#ffb400';
            } else if (isLabeled) {
                fillColor   = 'rgba(0,255,136,0.22)'; strokeColor = isActive ? '#fff' : '#00ff88';
                labelColor  = '#00ff88';
            } else {
                fillColor   = `rgba(${tr},${tg},${tb},0.22)`;
                strokeColor = isActive ? '#fff' : `rgb(${tr},${tg},${tb})`;
                labelColor  = `rgb(${tr},${tg},${tb})`;
            }

            ctx.save();
            if (isActive) { ctx.shadowColor = strokeColor; ctx.shadowBlur = 14; }
            ctx.strokeStyle = strokeColor;
            ctx.lineWidth   = isActive ? 3 : 2;
            ctx.fillStyle   = fillColor;
            ctx.beginPath();
            ctx.roundRect(d.x, d.y, d.w, d.h, 4);
            ctx.fill(); ctx.stroke();
            ctx.shadowBlur = 0;

            // 非球员画叉
            if (isNonPlayer) {
                ctx.strokeStyle = labelColor;
                ctx.lineWidth   = 2;
                ctx.globalAlpha = 0.7;
                ctx.beginPath(); ctx.moveTo(d.x + 4, d.y + 4); ctx.lineTo(d.x + d.w - 4, d.y + d.h - 4); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(d.x + d.w - 4, d.y + 4); ctx.lineTo(d.x + 4, d.y + d.h - 4); ctx.stroke();
                ctx.globalAlpha = 1.0;
            }

            const label    = isNonPlayer ? '×' : (isLabeled ? `#${d.jersey}` : '?');
            const fontSize = Math.max(13, d.h * 0.28);
            ctx.font         = `bold ${fontSize}px sans-serif`;
            ctx.fillStyle    = isNonPlayer ? 'rgba(80,40,0,0.6)' : (isLabeled ? 'rgba(0,0,0,0.55)' : 'rgba(0,0,0,0.45)');
            ctx.textAlign    = 'center';
            ctx.textBaseline = 'middle';
            const cx = d.x + d.w / 2, cy = d.y + d.h / 2;
            const tw = ctx.measureText(label).width + 10;
            const th = fontSize + 6;
            ctx.beginPath();
            ctx.roundRect(cx - tw / 2, cy - th / 2, tw, th, th / 2);
            ctx.fill();
            ctx.fillStyle = labelColor;
            ctx.fillText(label, cx, cy);
            ctx.restore();
        });
    }

    // Canvas 点击事件（需要在弹窗模块中绑定，因为 app.js DOMContentLoaded 时 canvas 已在 DOM 中）
    document.addEventListener('click', function (e) {
        const canvas = document.getElementById('jersey-canvas');
        if (!canvas || e.target !== canvas) return;
        const rect   = canvas.getBoundingClientRect();
        const scaleX = _frameW / rect.width;
        const scaleY = _frameH / rect.height;
        const mx     = (e.clientX - rect.left) * scaleX;
        const my     = (e.clientY - rect.top)  * scaleY;

        let hit = null;
        for (let i = _detections.length - 1; i >= 0; i--) {
            const d = _detections[i];
            if (mx >= d.x && mx <= d.x + d.w && my >= d.y && my <= d.y + d.h) { hit = d; break; }
        }

        if (!hit) { _hideBubble(); _activeDet = null; _render(); return; }
        _activeDet = hit.id;
        _render();

        const bubble = document.getElementById('jersey-input-bubble');
        const lbl    = document.getElementById('jersey-bubble-label');
        const input  = document.getElementById('jersey-number-input');
        const sb     = rect.width / _frameW;
        const bx     = (hit.x + hit.w / 2) * sb;
        const byBelow = (hit.y + hit.h) * sb + 6;   // 球员框下方
        const byAbove = hit.y * sb;                  // 球员框上方基准

        lbl.innerHTML = (function() {
            if (hit.team === 'non_player') {
                return `<i class="fas fa-user-slash"></i> <span style="color:#ffb400">非球员</span> 球衣号码`;
            }
            const tc  = _teamColors[hit.team] || [128,128,128];
            const css = `color:rgb(${tc[0]},${tc[1]},${tc[2]})`;
            const grp = hit.team === 'team_a' ? 'A 组' : 'B 组';
            return `<i class="fas fa-user"></i> <span style="${css}">${grp}</span> 球衣号码`;
        })();
        input.value   = (hit.jersey && hit.team !== 'non_player') ? hit.jersey : '';
        // 非球员按钮状态
        const npBtn = document.getElementById('jersey-nonplayer-btn');
        if (npBtn) {
            if (hit.team === 'non_player') {
                npBtn.style.background  = 'rgba(255,180,0,0.28)';
                npBtn.style.borderColor = 'rgba(255,180,0,0.65)';
                npBtn.innerHTML = '<i class="fas fa-user-check"></i> 已标为非球员（点击撤销）';
            } else {
                npBtn.style.background  = 'rgba(255,180,0,0.1)';
                npBtn.style.borderColor = 'rgba(255,180,0,0.35)';
                npBtn.innerHTML = '<i class="fas fa-user-slash"></i> 标为非球员（裁判/工作人员）';
            }
        }

        // 先显示（不可见位置）以获取真实高度，再计算方向
        bubble.style.visibility = 'hidden';
        bubble.style.display    = 'block';
        bubble.style.top        = '0px';
        const bubbleH = bubble.offsetHeight || 140;
        bubble.style.visibility = '';

        // 下方空间不够则向上弹出，并确保不超出顶部
        let finalTop;
        if (byBelow + bubbleH > rect.height - 4) {
            // 向上：气泡底边贴球员框顶部
            finalTop = Math.max(4, byAbove - bubbleH - 6);
        } else {
            finalTop = byBelow;
        }
        // 水平：确保不超出右边界，也不超出左边界
        const bubbleW = bubble.offsetWidth || 185;
        let finalLeft = bx - bubbleW / 2;
        finalLeft = Math.max(4, Math.min(finalLeft, rect.width - bubbleW - 4));

        bubble.style.left = `${finalLeft}px`;
        bubble.style.top  = `${finalTop}px`;
        setTimeout(() => input.focus(), 50);
    });

    function _hideBubble() {
        const b = document.getElementById('jersey-input-bubble');
        if (b) b.style.display = 'none';
    }

    function _updateProgress() {
        const dotsEl = document.getElementById('jersey-progress-dots');
        const textEl = document.getElementById('jersey-progress-text');
        if (!dotsEl) return;
        const total      = _detections.length;
        const labeled    = _detections.filter(d => d.jersey != null && d.team !== 'non_player').length;
        const nonPlayers = _detections.filter(d => d.team === 'non_player').length;
        dotsEl.innerHTML = _detections.map((d, idx) => {
            let bg, title;
            if (d.team === 'non_player') {
                bg    = '#ffb400';
                title = `非球员 #${idx + 1}（点击定位）`;
            } else if (d.jersey != null) {
                bg    = '#00ff88';
                title = `球员 #${idx + 1}  球衣 ${d.jersey}号（点击定位）`;
            } else {
                bg    = 'rgba(255,255,255,0.18)';
                title = `球员 #${idx + 1}  未标注（点击定位）`;
            }
            const isHighlighted = d.id === _activeDet;
            const ring = isHighlighted
                ? 'box-shadow:0 0 0 2px #fff,0 0 8px rgba(255,255,255,0.7);'
                : '';
            return `<span data-det-id="${d.id}" title="${title}" style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${bg};transition:background 0.2s,box-shadow 0.2s;cursor:pointer;${ring}"></span>`;
        }).join('');

        // 绑定点击：高亮对应框，并打开输入气泡
        dotsEl.querySelectorAll('span[data-det-id]').forEach(span => {
            span.addEventListener('click', function () {
                const id  = parseInt(this.getAttribute('data-det-id'));
                const det = _detections.find(d => d.id === id);
                if (!det) return;

                // 设为激活框并重绘
                _activeDet = id;
                _render();
                _updateProgress();   // 同步高亮选中点

                // 弹出输入气泡（复用画布点击逻辑）
                const canvas = document.getElementById('jersey-canvas');
                if (!canvas) return;
                const rect = canvas.getBoundingClientRect();
                const sb   = rect.width / _frameW;

                const bubble = document.getElementById('jersey-input-bubble');
                const lbl    = document.getElementById('jersey-bubble-label');
                const input  = document.getElementById('jersey-number-input');

                lbl.innerHTML = (function() {
                    if (det.team === 'non_player') {
                        return `<i class="fas fa-user-slash"></i> <span style="color:#ffb400">非球员</span> 球衣号码`;
                    }
                    const tc  = _teamColors[det.team] || [128,128,128];
                    const css = `color:rgb(${tc[0]},${tc[1]},${tc[2]})`;
                    const grp = det.team === 'team_a' ? 'A 组' : 'B 组';
                    return `<i class="fas fa-user"></i> <span style="${css}">${grp}</span> 球衣号码`;
                })();
                input.value = (det.jersey && det.team !== 'non_player') ? det.jersey : '';

                const npBtn = document.getElementById('jersey-nonplayer-btn');
                if (npBtn) {
                    if (det.team === 'non_player') {
                        npBtn.style.background  = 'rgba(255,180,0,0.28)';
                        npBtn.style.borderColor = 'rgba(255,180,0,0.65)';
                        npBtn.innerHTML = '<i class="fas fa-user-check"></i> 已标为非球员（点击撤销）';
                    } else {
                        npBtn.style.background  = 'rgba(255,180,0,0.1)';
                        npBtn.style.borderColor = 'rgba(255,180,0,0.35)';
                        npBtn.innerHTML = '<i class="fas fa-user-slash"></i> 标为非球员（裁判/工作人员）';
                    }
                }

                // 计算气泡位置（与画布点击逻辑一致）
                const bx     = (det.x + det.w / 2) * sb;
                const byBelow = (det.y + det.h) * sb + 6;
                const byAbove = det.y * sb;

                bubble.style.visibility = 'hidden';
                bubble.style.display    = 'block';
                bubble.style.top        = '0px';
                const bubbleH = bubble.offsetHeight || 140;
                bubble.style.visibility = '';

                let finalTop;
                if (byBelow + bubbleH > rect.height - 4) {
                    finalTop = Math.max(4, byAbove - bubbleH - 6);
                } else {
                    finalTop = byBelow;
                }
                const bubbleW = bubble.offsetWidth || 185;
                let finalLeft = bx - bubbleW / 2;
                finalLeft = Math.max(4, Math.min(finalLeft, rect.width - bubbleW - 4));

                bubble.style.left = `${finalLeft}px`;
                bubble.style.top  = `${finalTop}px`;
                setTimeout(() => input.focus(), 50);
            });
        });

        // 统计每队当前跨帧去重后的人数（用于实时警告）
        const teamCounts = { team_a: new Set(), team_b: new Set() };
        Object.values(_frameDetections).forEach(dets => {
            (dets || []).forEach(d => {
                if (d.jersey != null && d.team && d.team !== 'non_player') {
                    teamCounts[d.team] && teamCounts[d.team].add(d.jersey);
                }
            });
        });
        const cntA = teamCounts.team_a.size, cntB = teamCounts.team_b.size;
        const overLimit = cntA > 11 || cntB > 11;
        const parts = [`已标注 ${labeled}/${total}`];
        if (cntA > 0 || cntB > 0) {
            const aStr = `橙队${cntA}${cntA > 11 ? '⚠️' : ''}`;
            const bStr = `蓝队${cntB}${cntB > 11 ? '⚠️' : ''}`;
            parts.push(`${aStr} / ${bStr}`);
        }
        if (nonPlayers > 0) parts.push(`非球员 ${nonPlayers}`);
        if (overLimit) parts.push('超过11人将被截断');
        if (textEl) textEl.textContent = total > 0 ? parts.join('  ') : '';
        if (textEl) textEl.style.color = overLimit ? '#ffb400' : '';
        const btn = document.getElementById('jersey-confirm-btn');
        if (btn && (labeled > 0 || nonPlayers > 0)) {
            btn.style.borderColor = 'rgba(0,255,136,0.65)';
            btn.style.boxShadow   = '0 0 22px rgba(0,255,136,0.18)';
        }
    }


    document.addEventListener('keydown', e => {
        // 标注弹窗打开时禁用 ESC 关闭（标注是必须流程）
        // 其他弹窗仍保持正常行为
    });
})();

// ═══════════════════════════════════════════════════════════════
// 评测指标面板逻辑
// ═══════════════════════════════════════════════════════════════

/** 刷新估算评测指标 */
async function loadMetrics() {
    const icon = document.getElementById('metrics-refresh-icon');
    if (icon) icon.style.animation = 'spin 1s linear infinite';

    try {
        const res = await fetch('/api/metrics/summary');
        const data = await res.json();

        if (data.error) {
            console.error('[Metrics]', data.error);
            return;
        }

        const mot = data.mot || {};
        const hl  = data.highlight || {};

        // ── KPI 卡片 ──
        const fmt = (v, pct=true) => v == null ? '--' : pct ? (v*100).toFixed(1)+'%' : v;
        _setKpi('kpi-mota',      fmt(mot.mota));
        _setKpi('kpi-motp',      fmt(mot.motp));
        _setKpi('kpi-precision', fmt(mot.precision));
        _setKpi('kpi-recall',    fmt(mot.recall));
        _setKpi('kpi-f1',        fmt(mot.f1));
        _setKpi('kpi-idsw',      mot.id_switches ?? '--');
        _setKpi('kpi-mt',        mot.mt ?? '--');
        _setKpi('kpi-ml',        mot.ml ?? '--');

        _setKpi('kpi-hl-precision', fmt(hl.precision));
        _setKpi('kpi-hl-recall',    fmt(hl.recall));
        _setKpi('kpi-hl-f1',        fmt(hl.f1));
        _setKpi('kpi-hl-optical-w', hl.optical_flow_weight != null ? hl.optical_flow_weight.toFixed(2) : '--');
        _setKpi('kpi-hl-crowd-w',   hl.crowd_density_weight != null ? hl.crowd_density_weight.toFixed(2) : '--');
        _setKpi('kpi-hl-events',    hl.total_events ?? '--');

        // ── 特征贡献度条 ──
        const ofVal = hl.optical_flow_contrib;
        const cdVal = hl.crowd_density_contrib;
        const fuVal = hl.fused_score_mean;
        const maxV = Math.max(ofVal||0, cdVal||0, fuVal||0, 0.001);
        _setBar('bar-of',    'val-of',    ofVal, maxV);
        _setBar('bar-cd',    'val-cd',    cdVal, maxV);
        _setBar('bar-fused', 'val-fused', fuVal, maxV);

        // ── 逐视频 MOT 明细 ──
        const motBody = document.getElementById('mot-table-body');
        if (motBody) {
            if (!mot.per_video || mot.per_video.length === 0) {
                motBody.innerHTML = '<tr><td colspan="9" style="text-align:center;padding:24px;color:rgba(255,255,255,0.3);"><i class="fas fa-info-circle"></i> 暂无分析数据</td></tr>';
            } else {
                motBody.innerHTML = mot.per_video.map(r => `
                    <tr>
                        <td>${r.name || '--'}</td>
                        <td>${_scoreCell(r.mota)}</td>
                        <td>${_scoreCell(r.precision)}</td>
                        <td>${_scoreCell(r.recall)}</td>
                        <td>${_scoreCell(r.f1)}</td>
                        <td>${r.id_switches ?? '--'}</td>
                        <td>${r.fp ?? '--'}</td>
                        <td>${r.fn ?? '--'}</td>
                        <td>${r.speed_sec_per_min != null ? r.speed_sec_per_min+'s/min' : '--'}</td>
                    </tr>`).join('');
            }
        }

        // ── 逐视频精彩检测明细 ──
        const hlBody = document.getElementById('hl-table-body');
        if (hlBody) {
            if (!hl.per_video || hl.per_video.length === 0) {
                hlBody.innerHTML = '<tr><td colspan="8" style="text-align:center;padding:24px;color:rgba(255,255,255,0.3);"><i class="fas fa-info-circle"></i> 暂无精彩检测数据</td></tr>';
            } else {
                hlBody.innerHTML = hl.per_video.map(r => `
                    <tr>
                        <td>${r.name || '--'}</td>
                        <td>${r.segment_count ?? '--'}</td>
                        <td>${_scoreCell(r.precision)}</td>
                        <td>${_scoreCell(r.recall)}</td>
                        <td>${_scoreCell(r.f1)}</td>
                        <td>${r.optical_flow_mean != null ? r.optical_flow_mean.toFixed(3) : '--'}</td>
                        <td>${r.crowd_density_mean != null ? r.crowd_density_mean.toFixed(3) : '--'}</td>
                        <td>${r.threshold != null ? r.threshold.toFixed(2) : '--'}</td>
                    </tr>`).join('');
            }
        }

        // 同时加载 GT 列表
        loadGTList();

    } catch(e) {
        console.error('[Metrics] 请求失败', e);
    } finally {
        if (icon) icon.style.animation = '';
    }
}

function _setKpi(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}

function _setBar(barId, valId, val, maxV) {
    const bar = document.getElementById(barId);
    const lbl = document.getElementById(valId);
    if (!bar || !lbl) return;
    const pct = val != null ? Math.round(val / maxV * 100) : 0;
    bar.style.width = pct + '%';
    lbl.textContent = val != null ? val.toFixed(3) : '--';
}

function _scoreCell(v) {
    if (v == null) return '--';
    const pct = (v * 100).toFixed(1) + '%';
    const cls = v >= 0.8 ? 'tag-good' : v >= 0.6 ? 'tag-mid' : 'tag-low';
    return `<span class="${cls}">${pct}</span>`;
}

/** 导出当前评测数据为 CSV */
async function exportMetricsCSV() {
    try {
        const res = await fetch('/api/metrics/summary');
        const data = await res.json();
        if (data.error) { alert('获取数据失败：' + data.error); return; }

        const rows = [['视频', 'MOTA', 'MOTP', 'Precision', 'Recall', 'F1', 'IDSw', 'FP', 'FN', '处理速度(s/min)']];
        (data.mot?.per_video || []).forEach(r => {
            rows.push([r.name, r.mota, r.motp, r.precision, r.recall, r.f1,
                       r.id_switches, r.fp, r.fn, r.speed_sec_per_min]);
        });
        rows.push([]);
        rows.push(['视频', '检测片段', 'Precision', 'Recall', 'F1', '光流均值', '密度均值', '阈值']);
        (data.highlight?.per_video || []).forEach(r => {
            rows.push([r.name, r.segment_count, r.precision, r.recall, r.f1,
                       r.optical_flow_mean, r.crowd_density_mean, r.threshold]);
        });

        const csv = rows.map(r => r.join(',')).join('\r\n');
        const blob = new Blob(['\uFEFF' + csv], { type: 'text/csv;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = 'metrics_' + new Date().toISOString().slice(0,10) + '.csv';
        a.click(); URL.revokeObjectURL(url);
    } catch(e) {
        alert('导出失败：' + e.message);
    }
}

/** 上传 GT CSV 文件 */
async function uploadGT() {
    const videoId = document.getElementById('gt-video-select')?.value;
    const gtType  = document.getElementById('gt-type-select')?.value;
    const fileInp = document.getElementById('gt-file-input');
    const status  = document.getElementById('gt-status');

    if (!videoId) { if(status) status.textContent = '请先选择视频'; return; }
    if (!fileInp?.files?.length) { if(status) status.textContent = '请选择 CSV 文件'; return; }

    const fd = new FormData();
    fd.append('video_id', videoId);
    fd.append('gt_type',  gtType);
    fd.append('file',     fileInp.files[0]);

    if(status) { status.textContent = '上传中...'; status.style.color = 'rgba(255,206,86,0.8)'; }
    try {
        const res  = await fetch('/api/metrics/upload_gt', { method: 'POST', body: fd });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || '上传失败');
        if(status) {
            status.textContent = `上传成功：${data.gt_type === 'mot' ? 'MOT' : '精彩检测'} GT，共 ${data.rows} 行`;
            status.style.color = '#00ff88';
        }
        if(fileInp) fileInp.value = '';
        loadGTList();
        // 自动计算真实指标
        computeRealMetrics(videoId);
    } catch(e) {
        if(status) { status.textContent = '上传失败：' + e.message; status.style.color = '#ff6384'; }
    }
}

/** 加载 GT 列表并更新 gt-video-select */
async function loadGTList() {
    try {
        // 同时获取视频列表和 GT 列表
        const [vRes, gtRes] = await Promise.all([
            fetch('/api/videos'),
            fetch('/api/metrics/gt_list'),
        ]);
        const videos  = vRes.ok  ? await vRes.json()  : [];
        const gtItems = gtRes.ok ? await gtRes.json() : [];

        // 更新下拉框
        const sel = document.getElementById('gt-video-select');
        if (sel) {
            const cur = sel.value;
            sel.innerHTML = '<option value="">— 请选择视频 —</option>';
            (videos.videos || videos || []).forEach(v => {
                const opt = document.createElement('option');
                opt.value = v.id || v.video_id || v.video_uuid;
                opt.textContent = (v.name || v.filename || v.id || '').replace(/\.[^.]+$/, '');
                sel.appendChild(opt);
            });
            if (cur) sel.value = cur;
        }

        // 更新 GT 列表表格
        const body = document.getElementById('gt-list-body');
        const wrap = document.getElementById('gt-list-wrap');
        if (body) {
            if (gtItems.length === 0) {
                if(wrap) wrap.style.display = 'none';
            } else {
                if(wrap) wrap.style.display = 'block';
                body.innerHTML = gtItems.map(g => `
                    <tr>
                        <td>${g.name}</td>
                        <td>${g.has_mot ? `<span class="tag-good"><i class="fas fa-check"></i> ${g.mot_rows} 行</span>` : '<span style="color:rgba(255,255,255,0.25)">—</span>'}</td>
                        <td>${g.has_hl  ? `<span class="tag-good"><i class="fas fa-check"></i> ${g.hl_rows} 个事件</span>` : '<span style="color:rgba(255,255,255,0.25)">—</span>'}</td>
                        <td>
                            <button onclick="computeRealMetrics('${g.video_id}')" style="background:rgba(0,255,136,0.1);border:1px solid rgba(0,255,136,0.3);color:var(--primary);padding:4px 10px;border-radius:6px;cursor:pointer;font-size:0.78rem;margin-right:4px;">计算</button>
                            <button onclick="deleteGT('${g.video_id}')" style="background:rgba(255,99,132,0.08);border:1px solid rgba(255,99,132,0.3);color:#ff6384;padding:4px 10px;border-radius:6px;cursor:pointer;font-size:0.78rem;">删除</button>
                        </td>
                    </tr>`).join('');
            }
        }
    } catch(e) {
        console.error('[GT] loadGTList 失败', e);
    }
}

/** 删除 GT */
async function deleteGT(videoId) {
    if (!confirm('确认删除该视频的所有 GT 数据？')) return;
    await fetch(`/api/metrics/gt/${videoId}?gt_type=all`, { method: 'DELETE' });
    document.getElementById('real-metrics-card').style.display = 'none';
    loadGTList();
}

/** 计算并展示真实 GT 评测结果 */
async function computeRealMetrics(videoId) {
    const card = document.getElementById('real-metrics-card');
    const nameSpan = document.getElementById('real-metrics-video-name');
    if (!card) return;
    card.style.display = 'block';
    if(nameSpan) nameSpan.textContent = '计算中...';

    try {
        const res  = await fetch(`/api/metrics/real/${videoId}`);
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || '计算失败');

        if(nameSpan) nameSpan.textContent = data.video_name || videoId;

        // MOT 真实结果
        const motBlock = document.getElementById('real-mot-block');
        if (data.mot) {
            motBlock.style.display = 'block';
            document.getElementById('real-mot-gt-frames').textContent = data.mot.gt_total;
            document.getElementById('real-mot-tp').textContent = data.mot.tp;
            document.getElementById('real-mot-fp').textContent = data.mot.fp;
            document.getElementById('real-mot-fn').textContent = data.mot.fn;
            document.getElementById('real-mot-idsw').textContent = data.mot.id_switches;
            document.getElementById('real-mot-kpi').innerHTML = [
                ['MOTA', data.mot.mota],
                ['MOTP', data.mot.motp],
                ['Precision', data.mot.precision],
                ['Recall', data.mot.recall],
                ['F1', data.mot.f1],
                ['IDSw', data.mot.id_switches],
            ].map(([lbl, v]) => {
                const isNum = typeof v === 'number' && v <= 1;
                const disp  = isNum ? (v*100).toFixed(1)+'%' : (v ?? '--');
                const cls   = isNum ? (v>=0.8?'tag-good':v>=0.6?'tag-mid':'tag-low') : '';
                return `<div style="background:rgba(0,255,136,0.05);border:1px solid rgba(0,255,136,0.1);border-radius:8px;padding:10px 12px;text-align:center;">
                    <div class="kpi-val ${cls}" style="font-size:1.3rem;">${disp}</div>
                    <div class="kpi-lbl">${lbl}</div>
                </div>`;
            }).join('');
        } else {
            if(motBlock) motBlock.style.display = 'none';
        }

        // 精彩检测真实结果
        const hlBlock = document.getElementById('real-hl-block');
        if (data.highlight) {
            hlBlock.style.display = 'block';
            document.getElementById('real-hl-gt').textContent  = data.highlight.gt_total;
            document.getElementById('real-hl-sys').textContent = data.highlight.sys_total;
            document.getElementById('real-hl-tp').textContent  = data.highlight.tp;
            document.getElementById('real-hl-fp').textContent  = data.highlight.fp;
            document.getElementById('real-hl-fn').textContent  = data.highlight.fn;
            document.getElementById('real-hl-kpi').innerHTML = [
                ['Precision', data.highlight.precision],
                ['Recall',    data.highlight.recall],
                ['F1',        data.highlight.f1],
            ].map(([lbl, v]) => {
                const disp = v != null ? (v*100).toFixed(1)+'%' : '--';
                const cls  = v != null ? (v>=0.8?'tag-good':v>=0.6?'tag-mid':'tag-low') : '';
                return `<div style="background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.12);border-radius:8px;padding:10px 12px;text-align:center;">
                    <div class="kpi-val ${cls}" style="font-size:1.3rem;">${disp}</div>
                    <div class="kpi-lbl">${lbl}</div>
                </div>`;
            }).join('');
        } else {
            if(hlBlock) hlBlock.style.display = 'none';
        }

    } catch(e) {
        if(nameSpan) nameSpan.textContent = '计算失败：' + e.message;
        console.error('[RealMetrics]', e);
    }
}

// 切换到评测面板时自动加载
document.addEventListener('DOMContentLoaded', () => {
    // 监听 switchPage 调用时自动刷新
    const origSwitch = window.switchPage;
    window.switchPage = function(page) {
        if (typeof origSwitch === 'function') origSwitch(page);
        if (page === 'metrics') {
            loadMetrics();
        }
    };
});
