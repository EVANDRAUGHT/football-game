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
                    <li><strong>决策画像生成</strong>: 五维能力雷达图 + AI评估建议</li>
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
    const analysisPlaceholder = document.getElementById('analysis-placeholder');
    const analysisContent = document.getElementById('analysis-content');
    const videoContainer = document.getElementById('video-container');

    const API_BASE = "http://127.0.0.1:9999";

    let uploadedVideoId = null;
    
    // 暴露到全局，供 highlight.js 使用
    window.uploadedVideoId = null;
    let isAnalyzing = false;

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
            isAnalyzing = false;
            analysisContent.style.display = 'none';
            analysisPlaceholder.style.display = 'block';
            progressContainer.style.display = 'none';
            
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

        const xhr = new XMLHttpRequest();
        xhr.open('POST', `${API_BASE}/upload?owner=${ownerParam}`, true);
        
        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable) {
                const percent = (e.loaded / e.total * 100).toFixed(0);
                statusBadge.textContent = `上传中: ${percent}%`;
                uploadBtn.textContent = `上传中 (${percent}%)`;
            }
        };

        xhr.onload = function() {
            if (xhr.status === 200) {
                const data = JSON.parse(xhr.responseText);
                uploadedVideoId = data.video_id;
                window.uploadedVideoId = data.video_id;  // 同步到全局
                
                statusBadge.textContent = '上传成功！您可以开始分析视频';
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
                
                uploadBtn.disabled = false;
                uploadBtn.style.opacity = '1';
                uploadBtn.textContent = '开始分析';
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
        
        isAnalyzing = true;
        uploadBtn.disabled = true;
        uploadBtn.textContent = '正在深度分析中...';
        statusBadge.textContent = 'AI 引擎正在分析 3 分钟核心片段...';
        statusBadge.style.color = 'var(--accent)';
        
        progressContainer.style.display = 'block';
        progressBar.style.width = '0%';

        // 若未勾选"生成精彩视频"，点击开始分析后隐藏该选项区域
        const highlightCheckbox = document.getElementById('generate-highlight-checkbox');
        const highlightWrap = highlightCheckbox && highlightCheckbox.closest('div[style]');
        if (highlightCheckbox && !highlightCheckbox.checked) {
            // 找到包裹整个选框区域的父容器并隐藏
            const wrapEl = document.getElementById('generate-highlight-checkbox')
                ?.closest('div[style*="border-radius: 8px"]');
            if (wrapEl) wrapEl.style.display = 'none';
        }
        
        // 重置重试计数器
        retryCount = 0;
        
        showResults(uploadedVideoId);
    });

    let retryCount = 0;
    const maxRetries = 900; // 最多30分钟 (900次 × 2秒)
    
    async function showResults(videoId) {
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
                    if (prog < 100) {
                        statusBadge.textContent = `分析进度: ${prog}%`;
                    } else if (prog === 100) {
                        // 进度达到100%，直接跳过后续轮询，渲染结果
                        console.log('[INFO] 分析完成，进度100%，开始渲染结果...');
                    }
                }
            }

            // 检查是否有错误信息
            if (data.detailed_analysis && data.detailed_analysis.error) {
                statusBadge.textContent = '分析失败：' + (data.detailed_analysis.message || '未知错误');
                statusBadge.style.color = '#ff6384';
                progressContainer.style.display = 'none';
                isAnalyzing = false;
                uploadBtn.disabled = false;
                uploadBtn.textContent = '重试分析';
                
                // 显示详细错误信息
                analysisContent.innerHTML = `
                    <div style="padding: 30px; text-align: center; color: var(--text-dim);">
                        <i class="fas fa-exclamation-triangle" style="font-size: 3rem; color: #ff6384; margin-bottom: 20px;"></i>
                        <h3 style="color: #ff6384; margin-bottom: 15px;">视频分析失败</h3>
                        <p style="line-height: 1.6; margin-bottom: 20px;">${data.decision_summary || '未知错误'}</p>
                        <div style="background: rgba(255,99,132,0.1); border-left: 3px solid #ff6384; padding: 15px; border-radius: 8px; text-align: left; max-width: 600px; margin: 0 auto;">
                            <strong style="color: #ff6384;">可能的原因：</strong>
                            <ul style="margin-top: 10px; padding-left: 20px;">
                                <li>视频格式不支持（建议使用 MP4 格式）</li>
                                <li>视频文件损坏或无法读取</li>
                                <li>OCR/YOLO 依赖未正确安装</li>
                                <li>视频中未检测到足够的球员</li>
                            </ul>
                        </div>
                    </div>
                `;
                analysisPlaceholder.style.display = 'none';
                analysisContent.style.display = 'block';
                return;
            }

            // 🔑 修复3: 更严格的完成判断
            const isComplete = (
                data.progress === 100 && 
                data.detailed_analysis && 
                data.detailed_analysis.athletes && 
                Array.isArray(data.detailed_analysis.athletes) &&
                data.detailed_analysis.athletes.length > 0
            );

            if (!isComplete) {
                // 特殊处理：progress=100但数据未准备好（竞态条件）
                if (data.progress === 100 && !data.detailed_analysis) {
                    console.warn('[WARN] 进度100%但数据未就绪，延迟1秒重试...');
                    retryCount++;  // 增加重试计数
                    setTimeout(() => showResults(videoId), 1000);
                    return;
                }
                
                // 分析未完成，继续轮询
                if (data.progress === -1) {
                    statusBadge.textContent = '分析失败：后端处理异常';
                    statusBadge.style.color = '#ff6384';
                    isAnalyzing = false;
                    uploadBtn.disabled = false;
                    uploadBtn.textContent = '重试分析';
                    return;
                }
                
                // 检查重试次数
                retryCount++;
                if (retryCount >= maxRetries) {
                    statusBadge.textContent = '分析超时（超过30分钟未响应）';
                    statusBadge.style.color = '#ff6384';
                    progressContainer.style.display = 'none';
                    isAnalyzing = false;
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
                
                setTimeout(() => showResults(videoId), pollInterval);
                return;
            }
            
            // ✅ 分析完成，渲染结果
            console.log('[SUCCESS] 分析完成，开始渲染结果页面');
            retryCount = 0;

            // 完成
            statusBadge.textContent = '分析任务圆满完成';
            statusBadge.style.color = 'var(--primary)';
            progressContainer.style.display = 'none';
            uploadBtn.textContent = '分析完成';
            uploadBtn.disabled = true;
            
            // 🎬 触发精彩视频生成事件
            console.log('[APP] ===== 准备触发 videoAnalysisCompleted 事件 =====');
            console.log('[APP] uploadedVideoId:', uploadedVideoId);
            console.log('[APP] window.uploadedVideoId:', window.uploadedVideoId);
            
            const event = new CustomEvent('videoAnalysisCompleted', {
                detail: {
                    videoId: uploadedVideoId,
                    analysisData: data
                }
            });
            
            console.log('[APP] 触发事件:', event);
            document.dispatchEvent(event);
            console.log('[APP] ===== 事件已触发 =====');
            
            analysisPlaceholder.style.display = 'none';
            analysisContent.style.display = 'block';
            
            // 显示分析概览
            const summaryCard = document.getElementById('analysis-summary');
            if (summaryCard) {
                summaryCard.style.display = 'block';
                const avgConfidence = (data.detailed_analysis.athletes.reduce((sum, p) => sum + (p.path_accuracy || 0.95), 0) / data.detailed_analysis.athletes.length * 100).toFixed(0);
                document.getElementById('summary-confidence').textContent = avgConfidence + '%';
            }

            // ===== 控球率 / 热力图 / 战术板 =====
            const advStats = document.getElementById('advanced-stats');
            if (advStats) advStats.style.display = 'block';

            // 控球率
            const poss = data.detailed_analysis.possession;
            if (poss) {
                const pa = poss.team_a || 50;
                const pb = poss.team_b || 50;
                const barA = document.getElementById('poss-bar-a');
                const lblA = document.getElementById('poss-a-label');
                const lblB = document.getElementById('poss-b-label');
                if (barA) barA.style.width = pa + '%';
                if (lblA) lblA.textContent = `主队 ${pa}%`;
                if (lblB) lblB.textContent = `客队 ${pb}%`;
            }

            // 热力图
            const hmUrls = data.detailed_analysis.heatmap_urls || {};
            ['team_a', 'team_b'].forEach(team => {
                const suffix = team === 'team_a' ? 'a' : 'b';
                const img = document.getElementById(`heatmap-team-${suffix}`);
                const placeholder = document.getElementById(`heatmap-${suffix}-placeholder`);
                const url = hmUrls[team];
                if (img && url) {
                    img.src = `http://127.0.0.1:9999${url}?_t=${Date.now()}`;
                    img.style.display = 'block';
                    if (placeholder) placeholder.style.display = 'none';
                }
            });

            // 战术板
            const tactUrl = data.detailed_analysis.tactical_url;
            const tactImg = document.getElementById('tactical-view');
            const tactPh = document.getElementById('tactical-placeholder');
            if (tactImg && tactUrl) {
                tactImg.src = `http://127.0.0.1:9999${tactUrl}?_t=${Date.now()}`;
                tactImg.style.display = 'block';
                if (tactPh) tactPh.style.display = 'none';

                // 生成战术分析简报
                const tactDesc = document.getElementById('tactical-desc');
                const tactDescText = document.getElementById('tactical-desc-text');
                if (tactDesc && tactDescText) {
                    const poss = data.detailed_analysis.possession || { team_a: 50, team_b: 50 };
                    const athletes = data.detailed_analysis.athletes || [];
                    const ballCount = data.detailed_analysis.ball_trajectory_count || 0;

                    // 计算主客队平均能力
                    const midIdx = Math.floor(athletes.length / 2);
                    const teamA = athletes.slice(0, midIdx || 1);
                    const teamB = athletes.slice(midIdx || 1);
                    const avgAbil = (team) => team.length
                        ? Math.round(team.reduce((s, p) => s + Object.values(p.abilities || {}).reduce((a,b)=>a+b,0) / (Object.keys(p.abilities||{}).length||1), 0) / team.length)
                        : 0;
                    const aScore = avgAbil(teamA);
                    const bScore = avgAbil(teamB);
                    const domTeam = poss.team_a >= poss.team_b ? '主队' : '客队';
                    const domPoss = Math.max(poss.team_a, poss.team_b);

                    // 根据控球率推断战术风格
                    let styleA = '', styleB = '';
                    if (poss.team_a >= 60) styleA = '以控球为核心，倾向阵地进攻';
                    else if (poss.team_a <= 40) styleA = '以防守反击为主，快速转换';
                    else styleA = '控球与反击兼备，战术均衡';
                    if (poss.team_b >= 60) styleB = '以控球为核心，倾向阵地进攻';
                    else if (poss.team_b <= 40) styleB = '以防守反击为主，快速转换';
                    else styleB = '控球与反击兼备，战术均衡';

                    // 根据最强/最弱能力推断阵型倾向
                    const allSkills = {};
                    athletes.forEach(p => Object.entries(p.abilities||{}).forEach(([k,v])=>{ allSkills[k]=(allSkills[k]||0)+v; }));
                    const topSkill = Object.entries(allSkills).sort((a,b)=>b[1]-a[1])[0]?.[0] || '';
                    const formationHint = { '防守': '偏向防守阵型（如4-5-1或5-4-1）', '射门': '进攻型阵型（如4-3-3或3-4-3）', '传球': '控制型阵型（如4-3-3 Tiki-Taka）', '速度': '快节奏边路打法', '体能': '高强度压迫型战术' }[topSkill] || '均衡型阵型';

                    tactDescText.textContent =
                        `本场比赛双方跑动轨迹覆盖全场，共追踪 ${ballCount} 个球轨迹采样点。` +
                        `控球率方面，${domTeam}以 ${domPoss}% 占据主动，` +
                        `主队${styleA}，客队${styleB}。` +
                        `综合能力评估：主队均分 ${aScore}/100，客队均分 ${bScore}/100，` +
                        `${aScore >= bScore ? '主队' : '客队'}在整体实力上略占优势。` +
                        `从轨迹分布看，全队整体趋向${formationHint}，` +
                        `两侧边路和中路均有明显跑动覆盖，攻守转换较为积极。`;
                    tactDesc.style.display = 'block';
                }
            }
            
            // ===== 球员选择器（下拉框，支持全场所有球员） =====
            const selectorSection = document.getElementById('player-selector-section');
            const selectorA = document.getElementById('selector-team-a');
            const selectorB = document.getElementById('selector-team-b');
            if (selectorSection && selectorA && selectorB) {
                selectorSection.style.display = 'block';

                // 重置下拉框
                selectorA.innerHTML = '<option value="" style="color:#000;background:#fff;">— 请选择主队球员 —</option>';
                selectorB.innerHTML = '<option value="" style="color:#000;background:#fff;">— 请选择客队球员 —</option>';

                // 取全量球员（支持所有球员，不限数量）
                const allAthletes = data.detailed_analysis.athletes || [];
                const athletesA = data.detailed_analysis.athletes_team_a
                    || allAthletes.filter(a => a.team === 'team_a' || a.player_id?.startsWith('A'));
                const athletesB = data.detailed_analysis.athletes_team_b
                    || allAthletes.filter(a => a.team === 'team_b' || a.player_id?.startsWith('B'));

                // 存储所有球员数据，供选择时查询
                window._allAthletes = allAthletes;

                // 按球衣号排序
                const sortByNum = arr => [...arr].sort((a, b) => {
                    const na = a.jersey_number || parseInt((a.player_id||'').replace(/\D/g,'')) || 99;
                    const nb = b.jersey_number || parseInt((b.player_id||'').replace(/\D/g,'')) || 99;
                    return na - nb;
                });

                function buildOption(player) {
                    const opt = document.createElement('option');
                    const num = player.jersey_number || parseInt((player.player_id || '').replace(/\D/g,'')) || '?';
                    const avgScore = Object.keys(player.abilities || {}).length
                        ? Math.round(Object.values(player.abilities).reduce((a,b)=>a+b,0) / Object.keys(player.abilities).length)
                        : 0;
                    opt.value = player.player_id || num;
                    opt.textContent = `#${num} ${player.name}  （综合 ${avgScore}/100）`;
                    opt.style.cssText = 'color:#000;background:#fff;';
                    opt.dataset.playerId = player.player_id || num;
                    return opt;
                }

                sortByNum(athletesA).forEach(p => selectorA.appendChild(buildOption(p)));
                sortByNum(athletesB).forEach(p => selectorB.appendChild(buildOption(p)));

                // 下拉框 change 事件：选中球员后展示详情
                selectorA.addEventListener('change', () => {
                    if (!selectorA.value) return;
                    selectorB.value = '';  // 清除另一个下拉框
                    const player = allAthletes.find(p =>
                        (p.player_id || p.jersey_number?.toString()) === selectorA.value ||
                        p.player_id === selectorA.value
                    ) || athletesA.find(p => String(p.player_id || p.jersey_number) === selectorA.value);
                    if (player) showPlayerDetail(player, '255,130,40');
                });
                selectorB.addEventListener('change', () => {
                    if (!selectorB.value) return;
                    selectorA.value = '';  // 清除另一个下拉框
                    const player = allAthletes.find(p =>
                        (p.player_id || p.jersey_number?.toString()) === selectorB.value ||
                        p.player_id === selectorB.value
                    ) || athletesB.find(p => String(p.player_id || p.jersey_number) === selectorB.value);
                    if (player) showPlayerDetail(player, '80,120,255');
                });
            }

            // 关闭详情卡片
            const detailClose = document.getElementById('player-detail-close');
            if (detailClose) {
                detailClose.onclick = () => {
                    document.getElementById('player-detail-card').style.display = 'none';
                    const sa = document.getElementById('selector-team-a');
                    const sb = document.getElementById('selector-team-b');
                    if (sa) sa.value = '';
                    if (sb) sb.value = '';
                };
            }

            const container = document.getElementById('player-analysis-container');
            container.innerHTML = '';

            if (data.detailed_analysis.export_url) {
                const exportContainer = document.createElement('div');
                exportContainer.style.cssText = 'margin-top: 20px; text-align: center;';
                
                // 按钮容器
                const btnContainer = document.createElement('div');
                btnContainer.style.cssText = 'display: flex; gap: 15px; justify-content: center; align-items: center; flex-wrap: wrap; margin-bottom: 20px;';
                
                // 在线预览按钮 - 在页面内嵌入播放器
                const previewBtn = document.createElement('button');
                previewBtn.className = 'upload-btn';
                previewBtn.style.cssText = 'display: inline-flex; align-items: center; gap: 8px; padding: 12px 30px; background: var(--primary); margin: 0; border: none; cursor: pointer;';
                previewBtn.innerHTML = '<i class="fas fa-play-circle"></i> 在线预览标注视频';
                previewBtn.onclick = () => {
                    // 切换视频播放器显示状态
                    const existingPlayer = document.getElementById('inline-video-player');
                    if (existingPlayer) {
                        existingPlayer.style.display = existingPlayer.style.display === 'none' ? 'block' : 'none';
                    } else {
                        // 创建内嵌视频播放器
                        const playerContainer = document.createElement('div');
                        playerContainer.id = 'inline-video-player';
                        playerContainer.style.cssText = 'margin: 20px auto; max-width: 900px; background: var(--card-bg); border-radius: 12px; padding: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.3);';
                        
                        playerContainer.innerHTML = `
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                                <h4 style="margin: 0; color: var(--primary);"><i class="fas fa-video"></i> 标注视频播放器</h4>
                                <button onclick="document.getElementById('inline-video-player').style.display='none'" 
                                        style="background: transparent; border: none; color: var(--text-dim); cursor: pointer; font-size: 1.2rem;">&times;</button>
                            </div>
                            <video controls style="width: 100%; border-radius: 8px; background: #000;" preload="metadata">
                                <source src="${API_BASE}/preview/${videoId}" type="video/mp4">
                                您的浏览器不支持视频播放
                            </video>
                            <div style="margin-top: 10px; color: var(--text-dim); font-size: 0.85rem; text-align: center;">
                                <i class="fas fa-info-circle"></i> 点击播放按钮开始观看 | 支持全屏播放
                            </div>
                        `;
                        
                        exportContainer.appendChild(playerContainer);
                    }
                };
                
                // 下载按钮
                const downloadBtn = document.createElement('button');
                downloadBtn.className = 'upload-btn';
                downloadBtn.style.cssText = 'display: inline-flex; align-items: center; gap: 8px; padding: 12px 30px; background: var(--accent); margin: 0; border: none; cursor: pointer;';
                downloadBtn.innerHTML = '<i class="fas fa-download"></i> 下载分析结果';
                downloadBtn.onclick = () => {
                    // 创建隐藏的 a 标签触发下载
                    const link = document.createElement('a');
                    link.href = `${API_BASE}/download/${videoId}`;
                    link.download = `足球分析_${videoId.substring(0, 8)}.mp4`;
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                };
                
                // 新标签页预览按钮
                const newTabBtn = document.createElement('button');
                newTabBtn.className = 'upload-btn';
                newTabBtn.style.cssText = 'display: inline-flex; align-items: center; gap: 8px; padding: 12px 30px; background: #666; margin: 0; border: none; cursor: pointer;';
                newTabBtn.innerHTML = '<i class="fas fa-external-link-alt"></i> 新标签页打开';
                newTabBtn.onclick = () => {
                    window.open(`${API_BASE}/preview/${videoId}`, '_blank');
                };
                
                btnContainer.appendChild(previewBtn);
                btnContainer.appendChild(downloadBtn);
                btnContainer.appendChild(newTabBtn);
                exportContainer.appendChild(btnContainer);
                container.prepend(exportContainer);
            }
            drawTracking(data.detailed_analysis);
        } catch (error) {
            setTimeout(() => showResults(videoId), 5000);
        }
    }

    function showPlayerDetail(player, teamColor) {
        const card = document.getElementById('player-detail-card');
        if (!card) return;

        const num = player.jersey_number || parseInt((player.player_id || '').replace(/\D/g,'')) || '?';
        const abilities = player.abilities || {};
        const avgScore = Object.keys(abilities).length
            ? Math.round(Object.values(abilities).reduce((a,b) => a+b, 0) / Object.keys(abilities).length)
            : 0;
        const confidence = ((player.path_accuracy || 0.95) * 100).toFixed(0);

        document.getElementById('detail-player-name').innerHTML = `<i class="fas fa-user-check"></i> ${player.name}`;
        document.getElementById('detail-confidence-badge').innerHTML = `<i class="fas fa-crosshairs"></i> 置信度 ${confidence}%`;
        document.getElementById('detail-score-badge').textContent = `综合评分 ${avgScore}/100`;
        document.getElementById('detail-suggestion').textContent = player.suggestion || '暂无评估数据';

        // 能力详情列表
        const abList = document.getElementById('detail-abilities-list');
        abList.innerHTML = Object.entries(abilities).map(([key, val]) => `
            <div style="display:flex;justify-content:space-between;margin-bottom:5px;font-size:0.85rem;">
                <span style="color:rgba(255,255,255,0.7);">${key}</span>
                <span style="color:var(--primary);font-weight:bold;">${val}</span>
            </div>
        `).join('');

        // 颜色主题（由队伍颜色决定）
        const isTeamA = player.team === 'team_a' || (player.player_id || '').startsWith('A');
        const colorTheme = isTeamA
            ? { bg: 'rgba(255,130,40,0.2)', border: '#ff8228' }
            : { bg: 'rgba(80,120,255,0.2)', border: '#5078ff' };

        card.style.display = 'block';
        // 滚动到详情卡片
        card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        // 渲染雷达图（先销毁旧图）
        const existingChart = Chart.getChart('detail-radar-canvas');
        if (existingChart) existingChart.destroy();

        const canvas = document.getElementById('detail-radar-canvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const orderedLabels = ["防守", "射门", "传球", "速度", "体能"];
        const orderedValues = orderedLabels.map(label => abilities[label] || 0);

        new Chart(ctx, {
            type: 'radar',
            data: {
                labels: orderedLabels,
                datasets: [{
                    data: orderedValues,
                    backgroundColor: colorTheme.bg,
                    borderColor: colorTheme.border,
                    borderWidth: 2,
                    pointBackgroundColor: colorTheme.border,
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: colorTheme.border
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        min: 0, max: 100,
                        ticks: { display: true, stepSize: 20, color: 'rgba(255,255,255,0.3)', backdropColor: 'transparent' },
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        angleLines: { color: 'rgba(255,255,255,0.1)' },
                        pointLabels: { color: 'rgba(255,255,255,0.8)', font: { size: 12 } }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: { label: ctx => ctx.label + ': ' + ctx.parsed.r + '/100' }
                    }
                }
            }
        });
    }

    function renderRadarChart(canvasId, abilities, playerIndex) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) {
            console.error(`Canvas not found: ${canvasId}`);
            return;
        }
        const ctx = canvas.getContext('2d');
        if (Chart.getChart(canvasId)) Chart.getChart(canvasId).destroy();
        
        // 为每个球员使用不同的颜色主题，增强视觉差异
        const colorThemes = [
            { bg: 'rgba(255, 99, 132, 0.2)', border: '#ff6384' },   // 红色
            { bg: 'rgba(54, 162, 235, 0.2)', border: '#36a2eb' },   // 蓝色
            { bg: 'rgba(255, 206, 86, 0.2)', border: '#ffce56' },   // 黄色
            { bg: 'rgba(75, 192, 192, 0.2)', border: '#4bc0c0' }    // 青色
        ];
        
        // 使用传入的索引来选择颜色
        const theme = colorThemes[playerIndex % 4];
        
        // 确保能力值按固定顺序显示（防守、射门、传球、速度、体能）
        const orderedLabels = ["防守", "射门", "传球", "速度", "体能"];
        const orderedValues = orderedLabels.map(label => abilities[label] || 0);
        
        console.log(`渲染雷达图 ${canvasId} (球员${playerIndex}):`, orderedValues);  // 调试日志
        
        new Chart(ctx, {
            type: 'radar',
            data: {
                labels: orderedLabels,
                datasets: [{
                    data: orderedValues,
                    backgroundColor: theme.bg,
                    borderColor: theme.border,
                    borderWidth: 2,
                    pointBackgroundColor: theme.border,
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: theme.border
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { 
                    r: { 
                        min: 0, 
                        max: 100,
                        ticks: { 
                            display: true,
                            stepSize: 20,
                            color: 'rgba(255, 255, 255, 0.3)',
                            backdropColor: 'transparent'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        angleLines: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        pointLabels: {
                            color: 'rgba(255, 255, 255, 0.8)',
                            font: { size: 12 }
                        }
                    } 
                },
                plugins: { 
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.label + ': ' + context.parsed.r + '/100';
                            }
                        }
                    }
                }
            }
        });
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
});
