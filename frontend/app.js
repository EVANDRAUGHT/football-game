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
    const uploadBtn = document.getElementById('upload-btn');
    const fileInfo = document.getElementById('file-info');
    const filenameDisplay = document.getElementById('filename-display');
    const statusBadge = document.getElementById('status');
    const progressBar = document.getElementById('progress-bar');
    const progressContainer = document.getElementById('progress-container');
    const analysisPlaceholder = document.getElementById('analysis-placeholder');
    const analysisContent = document.getElementById('analysis-content');

    const API_BASE = "http://127.0.0.1:9999";

    let uploadedVideoId = null;
    
    // 暴露到全局，供 highlight.js 使用
    window.uploadedVideoId = null;
    let isAnalyzing = false;

    // 点击上传区域时，动态找当前的 file-input（因为 innerHTML 可能被替换）
    dropZone.addEventListener('click', () => {
        const fi = document.getElementById('file-input');
        if (fi) fi.click();
    });
    
    // 允许在上传后通过点击 filenameDisplay 重新上传
    filenameDisplay.style.cursor = 'pointer';
    filenameDisplay.addEventListener('click', () => {
        if (!isAnalyzing) {
            const fi = document.getElementById('file-input');
            if (fi) fi.click();
        }
    });

    // 初始的 file-input change 事件（后续通过 _updateDropZone 重新绑定）
    document.getElementById('file-input').addEventListener('change', (e) => {
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
            
            filenameDisplay.textContent = `准备上传: ${file.name}`;
            fileInfo.style.display = 'block';
            
            uploadBtn.disabled = true;
            uploadBtn.textContent = '正在极速上传...';
            uploadBtn.style.opacity = '0.5';
            
            autoUpload(file);
        }
    }

    // 更新 drop-zone 显示（已选文件时显示文件名，可重新点击）
    function _updateDropZone(filename) {
        dropZone.innerHTML = `
            <input type="file" id="file-input" hidden accept="video/*">
            <i class="fas fa-check-circle" style="color:var(--primary);"></i>
            <p style="color:var(--primary);font-weight:600;">${filename}</p>
            <small style="color:rgba(255,255,255,0.4);">点击可重新选择视频</small>
        `;
        // 重新绑定 file-input change 事件（因为 innerHTML 替换了旧 input）
        const newInput = document.getElementById('file-input');
        if (newInput) {
            newInput.addEventListener('change', (e) => {
                if (e.target.files[0]) handleFile(e.target.files[0]);
            });
        }
    }

    async function autoUpload(file) {
        const formData = new FormData();
        formData.append('file', file);

        // 更新 drop-zone 为"上传中"状态
        dropZone.innerHTML = `
            <input type="file" id="file-input" hidden accept="video/*">
            <i class="fas fa-spinner fa-spin" style="color:var(--primary);"></i>
            <p style="color:var(--primary);">正在上传...</p>
            <small>${file.name}</small>
        `;

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
                
                _updateDropZone(file.name);
                uploadBtn.disabled = false;
                uploadBtn.style.opacity = '1';
                uploadBtn.textContent = '开始分析';
            } else {
                statusBadge.textContent = '上传失败，请重试';
                uploadBtn.textContent = '重试上传';
                uploadBtn.disabled = false;
                // 恢复 drop-zone
                dropZone.innerHTML = `
                    <input type="file" id="file-input" hidden accept="video/*">
                    <i class="fas fa-cloud-upload-alt"></i>
                    <p>点击或拖拽视频文件至此</p>
                    <small>支持 MP4, MOV 格式</small>
                `;
                const newInput = document.getElementById('file-input');
                if (newInput) newInput.addEventListener('change', (e) => { if (e.target.files[0]) handleFile(e.target.files[0]); });
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
        
        // 重置重试计数器
        retryCount = 0;
        
        showResults(uploadedVideoId);
    });

    let retryCount = 0;
    const maxRetries = 900; // 最多30分钟 (900次 × 2秒)，兼容较长视频或低配机器
    
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
                    statusBadge.textContent = '等待超时（30分钟），后端可能仍在处理中，可点击"查询结果"继续等待';
                    statusBadge.style.color = '#ffce56';
                    progressContainer.style.display = 'none';
                    isAnalyzing = false;
                    uploadBtn.disabled = false;
                    uploadBtn.textContent = '查询结果';
                    // 超时后点按钮可重新轮询一次
                    uploadBtn.onclick = () => {
                        retryCount = 0;
                        isAnalyzing = true;
                        uploadBtn.disabled = true;
                        uploadBtn.textContent = '查询中...';
                        showResults(videoId);
                    };
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
            
            analysisPlaceholder.style.display = 'none';
            analysisContent.style.display = 'block';

            const event = new CustomEvent('videoAnalysisCompleted', {
                detail: {
                    videoId: uploadedVideoId,
                    analysisData: data
                }
            });
            console.log('[APP] 触发事件:', event);
            document.dispatchEvent(event);
            console.log('[APP] ===== 事件已触发 =====');
            
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
            }
            
            const container = document.getElementById('player-analysis-container');
            container.innerHTML = '';
            
            data.detailed_analysis.athletes.forEach((player, index) => {
                const playerDiv = document.createElement('div');
                playerDiv.className = 'player-card';
                
                // 计算综合评分（与后端保持一致的算法）
                const abilities = player.abilities;
                const avgScore = Math.round(Object.values(abilities).reduce((a,b) => a+b, 0) / Object.keys(abilities).length);
                const confidence = ((player.path_accuracy || 0.95) * 100).toFixed(0);
                
                // 使用索引而非 player_id 生成唯一的 canvas ID
                const radarCanvasId = `radar-player-${index}`;
                
                playerDiv.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
                        <h3 style="color: var(--primary); margin: 0;"><i class="fas fa-user-check"></i> ${player.name}</h3>
                        <div style="display: flex; gap: 10px; align-items: center;">
                            <span class="status-badge" style="border-color: var(--accent); color: var(--accent); font-size: 0.85rem;">
                                <i class="fas fa-crosshairs"></i> 置信度 ${confidence}%
                            </span>
                            <span class="status-badge" style="border-color: var(--primary); color: var(--primary); font-size: 0.85rem;">
                                综合评分 ${avgScore}/100
                            </span>
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1.2fr; gap: 2rem;">
                        <div>
                            <div style="height: 250px; position: relative;">
                                <canvas id="${radarCanvasId}"></canvas>
                            </div>
                            <div style="margin-top: 15px; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 8px;">
                                <div style="font-size: 0.75rem; color: var(--text-dim); margin-bottom: 8px;">能力详情:</div>
                                ${Object.entries(player.abilities).map(([key, val]) => `
                                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px; font-size: 0.85rem;">
                                        <span style="color: rgba(255,255,255,0.7);">${key}</span>
                                        <span style="color: var(--primary); font-weight: bold;">${val}</span>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                        <div>
                            <h4 style="color: var(--accent); margin: 0 0 10px 0;"><i class="fas fa-brain"></i> AI 深度评估</h4>
                            <p style="color: var(--text-dim); line-height: 1.6; font-size: 0.9rem;">${player.suggestion}</p>
                            <div style="margin-top: 15px; padding: 12px; background: rgba(255,206,86,0.05); border-left: 3px solid #ffce56; border-radius: 4px;">
                                <div style="font-size: 0.8rem; color: rgba(255,255,255,0.5); margin-bottom: 5px;">
                                    <i class="fas fa-flask"></i> 数据来源说明
                                </div>
                                <div style="font-size: 0.75rem; color: rgba(255,255,255,0.4); line-height: 1.5;">
                                    能力值基于预设模板生成（演示模式）。实际应用需结合多场比赛数据与专业标注。
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                container.appendChild(playerDiv);
                
                // 使用索引传递给 renderRadarChart
                renderRadarChart(radarCanvasId, player.abilities, index);
            });

            if (data.detailed_analysis.export_url) {
                const exportContainer = document.createElement('div');
                exportContainer.style.cssText = 'margin-top: 20px; text-align: center;';

                // 按钮行
                const btnContainer = document.createElement('div');
                btnContainer.style.cssText = 'display:flex;gap:15px;justify-content:center;align-items:center;flex-wrap:wrap;margin-bottom:20px;';

                // 下载按钮
                const downloadBtn = document.createElement('button');
                downloadBtn.className = 'upload-btn';
                downloadBtn.style.cssText = 'display:inline-flex;align-items:center;gap:8px;padding:12px 30px;background:var(--accent);margin:0;border:none;cursor:pointer;';
                downloadBtn.innerHTML = '<i class="fas fa-download"></i> 下载分析结果';
                downloadBtn.onclick = () => {
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
                newTabBtn.style.cssText = 'display:inline-flex;align-items:center;gap:8px;padding:12px 30px;background:#666;margin:0;border:none;cursor:pointer;';
                newTabBtn.innerHTML = '<i class="fas fa-external-link-alt"></i> 新标签页打开';
                newTabBtn.onclick = () => window.open(`${API_BASE}/preview/${videoId}`, '_blank');

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
        // tracking-overlay 已移除，轨迹信息改由标注视频和战术板展示
        return;
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

    // ===== 实时 2D 战术板 =====
    let _liveTacticTimer = null;   // 轮询定时器
    let _liveSrcWidth = 1920;      // 视频原始宽高（用于坐标归一化）
    let _liveSrcHeight = 1080;
    const CANVAS_W = 800;
    const CANVAS_H = 520;
    const MARGIN = 30;             // 球场边距（与后端 build_tactical_view 一致）

    function startLiveTactical(videoId) {
        const panel = document.getElementById('live-tactical-panel');
        const canvas = document.getElementById('live-tactical-canvas');
        const statusEl = document.getElementById('live-tactic-status');
        if (!panel || !canvas) return;

        // 设置 canvas 固有分辨率
        canvas.width = CANVAS_W;
        canvas.height = CANVAS_H;
        panel.style.display = 'block';

        // 启动之前先清一次旧定时器
        if (_liveTacticTimer) clearInterval(_liveTacticTimer);

        _liveTacticTimer = setInterval(async () => {
            try {
                const res = await fetch(`${API_BASE}/api/live-frame/${videoId}`);
                const d = await res.json();
                if (d.status !== 'ok') return;

                // 记录源分辨率
                if (d.width)  _liveSrcWidth  = d.width;
                if (d.height) _liveSrcHeight = d.height;

                // 更新状态文字
                const sec = (d.fno / (d.fps || 25)).toFixed(1);
                if (statusEl) statusEl.textContent = `帧 ${d.fno} | ${sec}s`;

                drawLiveTactical(canvas, d);
            } catch (_) { /* 网络偶尔失败忽略 */ }
        }, 200);  // 200ms 轮询 ≈ 5fps 刷新
    }

    function stopLiveTactical() {
        if (_liveTacticTimer) {
            clearInterval(_liveTacticTimer);
            _liveTacticTimer = null;
        }
    }

    /** 将视频像素坐标映射到 canvas 球场坐标 */
    function mapToField(x, y) {
        const fx = MARGIN + (x / _liveSrcWidth)  * (CANVAS_W - 2 * MARGIN);
        const fy = MARGIN + (y / _liveSrcHeight) * (CANVAS_H - 2 * MARGIN);
        return [fx, fy];
    }

    function drawLiveTactical(canvas, d) {
        const ctx = canvas.getContext('2d');

        // --- 绘制球场背景 ---
        ctx.fillStyle = '#1a6b2a';
        ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

        // 草坪斑马纹
        for (let i = 0; i < 8; i++) {
            const x0 = MARGIN + i * (CANVAS_W - 2 * MARGIN) / 8;
            const x1 = MARGIN + (i + 1) * (CANVAS_W - 2 * MARGIN) / 8;
            ctx.fillStyle = i % 2 === 0 ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.03)';
            ctx.fillRect(x0, MARGIN, x1 - x0, CANVAS_H - 2 * MARGIN);
        }

        // 场地线条
        ctx.strokeStyle = 'rgba(255,255,255,0.75)';
        ctx.lineWidth = 1.5;
        // 外框
        ctx.strokeRect(MARGIN, MARGIN, CANVAS_W - 2 * MARGIN, CANVAS_H - 2 * MARGIN);
        // 中线
        ctx.beginPath();
        ctx.moveTo(CANVAS_W / 2, MARGIN);
        ctx.lineTo(CANVAS_W / 2, CANVAS_H - MARGIN);
        ctx.stroke();
        // 中圈
        ctx.beginPath();
        ctx.arc(CANVAS_W / 2, CANVAS_H / 2, 60, 0, Math.PI * 2);
        ctx.stroke();
        // 中心点
        ctx.fillStyle = 'rgba(255,255,255,0.75)';
        ctx.beginPath();
        ctx.arc(CANVAS_W / 2, CANVAS_H / 2, 3, 0, Math.PI * 2);
        ctx.fill();
        // 左禁区
        ctx.strokeRect(MARGIN, CANVAS_H / 2 - 80, 120, 160);
        ctx.strokeRect(MARGIN, CANVAS_H / 2 - 40, 50, 80);
        // 右禁区
        ctx.strokeRect(CANVAS_W - MARGIN - 120, CANVAS_H / 2 - 80, 120, 160);
        ctx.strokeRect(CANVAS_W - MARGIN - 50, CANVAS_H / 2 - 40, 50, 80);

        // --- 绘制球轨迹（渐隐，越新越亮）---
        const trail = d.ball_trail || [];
        if (trail.length > 1) {
            for (let i = 1; i < trail.length; i++) {
                const alpha = i / trail.length;           // 0→1，越新越大
                const [x1, y1] = mapToField(trail[i - 1].x, trail[i - 1].y);
                const [x2, y2] = mapToField(trail[i].x, trail[i].y);
                ctx.beginPath();
                ctx.strokeStyle = `rgba(0, 230, 255, ${alpha * 0.9})`;
                ctx.lineWidth = 1.5 + alpha * 1.5;
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
                ctx.stroke();
            }
        }

        // --- 绘制球员圆点 ---
        const players = d.players || [];
        players.forEach(p => {
            const [cx, cy] = mapToField(p.x, p.y);
            const isTeamA = p.team === 'team_a';

            // 外光晕
            const grd = ctx.createRadialGradient(cx, cy, 2, cx, cy, 10);
            if (isTeamA) {
                grd.addColorStop(0, 'rgba(255,112,67,0.6)');
                grd.addColorStop(1, 'rgba(255,112,67,0)');
            } else {
                grd.addColorStop(0, 'rgba(66,165,245,0.6)');
                grd.addColorStop(1, 'rgba(66,165,245,0)');
            }
            ctx.beginPath();
            ctx.fillStyle = grd;
            ctx.arc(cx, cy, 10, 0, Math.PI * 2);
            ctx.fill();

            // 实心圆
            ctx.beginPath();
            ctx.fillStyle = isTeamA ? '#ff7043' : '#42a5f5';
            ctx.arc(cx, cy, 5, 0, Math.PI * 2);
            ctx.fill();

            // 白边
            ctx.beginPath();
            ctx.strokeStyle = 'rgba(255,255,255,0.8)';
            ctx.lineWidth = 1;
            ctx.arc(cx, cy, 5, 0, Math.PI * 2);
            ctx.stroke();
        });

        // --- 绘制球本体 ---
        if (d.ball) {
            const [bx, by] = mapToField(d.ball.x, d.ball.y);
            // 外发光
            const grd = ctx.createRadialGradient(bx, by, 1, bx, by, 9);
            grd.addColorStop(0, 'rgba(0,229,255,0.9)');
            grd.addColorStop(1, 'rgba(0,229,255,0)');
            ctx.beginPath();
            ctx.fillStyle = grd;
            ctx.arc(bx, by, 9, 0, Math.PI * 2);
            ctx.fill();
            // 实心圆
            ctx.beginPath();
            ctx.fillStyle = '#ffffff';
            ctx.arc(bx, by, 4, 0, Math.PI * 2);
            ctx.fill();
            ctx.beginPath();
            ctx.strokeStyle = '#00e5ff';
            ctx.lineWidth = 1.5;
            ctx.arc(bx, by, 4, 0, Math.PI * 2);
            ctx.stroke();
        }

        // --- 左下角：球员数量 + 时间戳信息 ---
        const pCount = (d.players || []).length;
        ctx.fillStyle = 'rgba(0,0,0,0.45)';
        ctx.fillRect(MARGIN, CANVAS_H - MARGIN - 20, 180, 18);
        ctx.fillStyle = pCount > 0 ? 'rgba(0,255,136,0.9)' : 'rgba(255,200,0,0.8)';
        ctx.font = '12px monospace';
        ctx.textAlign = 'left';
        ctx.fillText(`球员:${pCount}  ${d.t !== undefined ? d.t.toFixed(1)+'s' : ''}`, MARGIN + 6, CANVAS_H - MARGIN - 6);
    }

    // 点击"开始分析"后，延迟3秒启动实时战术板轮询（等后端第一批帧数据写入）
    uploadBtn.addEventListener('click', () => {
        setTimeout(() => {
            if (uploadedVideoId) startLiveTactical(uploadedVideoId);
        }, 3000);
    });

    // ===== 播放同步俯瞰图 =====
    let _syncTimeline = null;   // { fps, width, height, frames: [{t, players, ball, ball_trail}] }
    let _syncVideoId  = null;
    let _syncAnimFrame = null;  // requestAnimationFrame 句柄（视频播放时用）
    let _syncPreviewTimer = null; // setInterval 句柄（静态预览自动轮播用）
    let _syncPreviewIdx = 0;      // 当前预览帧索引

    /** 二分查找时序数组中最接近 currentTime 的帧索引 */
    function _findFrameIdx(frames, t) {
        let lo = 0, hi = frames.length - 1;
        while (lo < hi) {
            const mid = (lo + hi + 1) >> 1;
            if (frames[mid].t <= t) lo = mid;
            else hi = mid - 1;
        }
        return lo;
    }

    function stopSyncTactical() {
        if (_syncAnimFrame) {
            cancelAnimationFrame(_syncAnimFrame);
            _syncAnimFrame = null;
        }
    }

    function stopSyncPreview() {
        if (_syncPreviewTimer) {
            clearInterval(_syncPreviewTimer);
            _syncPreviewTimer = null;
        }
    }

    /** 启动自动轮播预览（视频未播放时展示数据动态效果） */
    function startSyncPreview(mapCanvas, statusEl) {
        stopSyncPreview();
        if (!_syncTimeline || !_syncTimeline.frames || _syncTimeline.frames.length === 0) return;
        // 不重置 _syncPreviewIdx，由调用方决定从哪帧开始
        const frames = _syncTimeline.frames;
        _syncPreviewTimer = setInterval(() => {
            if (!_syncTimeline) { stopSyncPreview(); return; }
            _liveSrcWidth  = _syncTimeline.width  || 1920;
            _liveSrcHeight = _syncTimeline.height || 1080;
            const f = frames[_syncPreviewIdx];
            drawLiveTactical(mapCanvas, { ...f, width: _syncTimeline.width, height: _syncTimeline.height });
            if (statusEl) statusEl.textContent = `预览 ${f.t.toFixed(1)}s | 共${frames.length}帧 · 点击播放同步 ▶`;
            _syncPreviewIdx = (_syncPreviewIdx + 1) % frames.length;
        }, 200); // 5fps 轮播
    }

    /** ---- sync-player 公共绘制函数（外层，避免闭包陷阱）---- */
    function _syncGetEls() {
        return {
            videoEl:   document.getElementById('sync-video'),
            mapCanvas: document.getElementById('sync-map-canvas'),
            statusEl:  document.getElementById('sync-map-status'),
        };
    }

    function _drawSyncFrame() {
        const { videoEl, mapCanvas, statusEl } = _syncGetEls();
        if (!videoEl || !mapCanvas) return;
        if (!_syncTimeline) { _syncAnimFrame = requestAnimationFrame(_drawSyncFrame); return; }
        const t = videoEl.currentTime;
        const frames = _syncTimeline.frames;
        if (!frames || frames.length === 0) { _syncAnimFrame = requestAnimationFrame(_drawSyncFrame); return; }
        const idx = _findFrameIdx(frames, t);
        _liveSrcWidth  = _syncTimeline.width  || 1920;
        _liveSrcHeight = _syncTimeline.height || 1080;
        drawLiveTactical(mapCanvas, { ...frames[idx], t, width: _syncTimeline.width, height: _syncTimeline.height });
        if (statusEl) statusEl.textContent = `同步中 ${t.toFixed(2)}s`;
        _syncAnimFrame = requestAnimationFrame(_drawSyncFrame);
    }

    function _drawSyncOneFrame() {
        const { videoEl, mapCanvas, statusEl } = _syncGetEls();
        if (!videoEl || !mapCanvas || !_syncTimeline) return;
        const t = videoEl.currentTime;
        const frames = _syncTimeline.frames;
        if (!frames || frames.length === 0) return;
        const idx = _findFrameIdx(frames, t);
        _liveSrcWidth  = _syncTimeline.width  || 1920;
        _liveSrcHeight = _syncTimeline.height || 1080;
        drawLiveTactical(mapCanvas, { ...frames[idx], t, width: _syncTimeline.width, height: _syncTimeline.height });
        if (statusEl) statusEl.textContent = `暂停 ${t.toFixed(2)}s`;
    }

    /** 初始化 sync-player-section 里的视频和canvas，并绑定同步事件（只绑一次） */
    let _syncPlayerInited = false;

    function initSyncPlayer(videoId) {
        const videoEl  = document.getElementById('sync-video');
        const mapCanvas = document.getElementById('sync-map-canvas');
        const statusEl  = document.getElementById('sync-map-status');
        if (!videoEl || !mapCanvas) return;

        // 重置状态
        stopSyncTactical();
        stopSyncPreview();
        _syncTimeline = null;

        // 设置视频源
        videoEl.src = `${API_BASE}/preview/${videoId}`;
        videoEl.load();

        // 设置 canvas 固有分辨率
        mapCanvas.width  = CANVAS_W;
        mapCanvas.height = CANVAS_H;

        if (statusEl) statusEl.textContent = '加载时序数据中...';

        // 只绑定一次事件监听器（用命名函数便于后续调试）
        if (!_syncPlayerInited) {
            _syncPlayerInited = true;

            videoEl.addEventListener('play', () => {
                console.log('[SYNC] 视频播放，启动RAF同步');
                stopSyncPreview();
                stopSyncTactical();
                _syncAnimFrame = requestAnimationFrame(_drawSyncFrame);
            });
            videoEl.addEventListener('pause', () => {
                console.log('[SYNC] 视频暂停');
                stopSyncTactical();
                _drawSyncOneFrame();
                const { mapCanvas: mc, statusEl: se } = _syncGetEls();
                if (_syncTimeline && _syncTimeline.frames) {
                    _syncPreviewIdx = _findFrameIdx(_syncTimeline.frames, videoEl.currentTime);
                }
                startSyncPreview(mc, se);
            });
            videoEl.addEventListener('ended', () => {
                console.log('[SYNC] 视频结束');
                stopSyncTactical();
                _syncPreviewIdx = 0;
                const { mapCanvas: mc, statusEl: se } = _syncGetEls();
                startSyncPreview(mc, se);
            });
            videoEl.addEventListener('seeked', () => {
                if (!videoEl.paused) return;
                stopSyncTactical();
                _drawSyncOneFrame();
            });
            videoEl.addEventListener('timeupdate', () => {
                if (videoEl.paused && _syncTimeline) _drawSyncOneFrame();
            });
        }
    }

    /** 拉取时序数据并启动轮播/自动播放 */
    function _loadSyncTimeline(vid) {
        const statusEl = document.getElementById('sync-map-status');
        const mapCanvas = document.getElementById('sync-map-canvas');

        // 初始化播放器
        initSyncPlayer(vid);

        if (statusEl) statusEl.textContent = '加载时序数据中...';
        console.log('[SYNC] 开始拉取时序数据 video_id=', vid);

        fetch(`${API_BASE}/api/frame-timeline/${vid}`)
            .then(r => r.json())
            .then(d => {
                console.log('[SYNC] 时序数据响应:', d.status,
                    'frames=', d.frames ? d.frames.length : 0,
                    '首帧players=', d.frames && d.frames[0] ? d.frames[0].players.length : 'N/A');
                if (d.status !== 'ok') {
                    if (statusEl) statusEl.textContent = '时序数据暂不可用（重新分析后可用）';
                    _drawEmptyField();
                    return;
                }
                _syncTimeline = d;
                _syncVideoId  = vid;
                _liveSrcWidth  = d.width  || 1920;
                _liveSrcHeight = d.height || 1080;

                const frameCount = d.frames ? d.frames.length : 0;
                let totalPlayers = 0;
                if (d.frames) d.frames.forEach(f => totalPlayers += (f.players ? f.players.length : 0));
                const avgPlayers = frameCount > 0 ? (totalPlayers / frameCount).toFixed(1) : 0;
                if (statusEl) statusEl.textContent = `${frameCount}帧 · 均${avgPlayers}球员/帧`;

                if (mapCanvas && d.frames && d.frames.length > 0) {
                    // 先从有球员的帧开始轮播
                    _syncPreviewIdx = 0;
                    for (let i = 0; i < d.frames.length; i++) {
                        if (d.frames[i].players && d.frames[i].players.length >= 2) {
                            _syncPreviewIdx = i;
                            break;
                        }
                    }
                    // 立即渲染一帧
                    drawLiveTactical(mapCanvas, { ...d.frames[_syncPreviewIdx], width: d.width, height: d.height });
                    // 启动自动轮播（视频播放时会自动切换到RAF同步）
                    startSyncPreview(mapCanvas, statusEl);
                } else {
                    _drawEmptyField();
                }

                // 尝试自动播放视频（若浏览器允许静音自动播放）
                const syncVideo = document.getElementById('sync-video');
                if (syncVideo) {
                    syncVideo.muted = true;
                    const tryPlay = () => syncVideo.play().catch(() => {});
                    if (syncVideo.readyState >= 2) tryPlay();
                    else syncVideo.addEventListener('canplay', tryPlay, { once: true });
                }
            })
            .catch((err) => {
                console.error('[SYNC] 时序数据加载失败:', err);
                if (statusEl) statusEl.textContent = '时序数据加载失败';
                _drawEmptyField();
            });
    }

    document.addEventListener('videoAnalysisCompleted', (e) => {
        stopLiveTactical();
        const liveStatusEl = document.getElementById('live-tactic-status');
        if (liveStatusEl) liveStatusEl.textContent = '分析完成';

        if (!e.detail || !e.detail.videoId) return;
        const vid = e.detail.videoId;
        _loadSyncTimeline(vid);
    });

    /** 在 sync-map-canvas 上画一个带提示文字的空球场 */
    function _drawEmptyField() {
        const mapCanvas = document.getElementById('sync-map-canvas');
        if (!mapCanvas) return;
        if (!mapCanvas.width) { mapCanvas.width = CANVAS_W; mapCanvas.height = CANVAS_H; }
        drawLiveTactical(mapCanvas, { players: [], ball: null, ball_trail: [], width: _liveSrcWidth, height: _liveSrcHeight });
        const ctx = mapCanvas.getContext('2d');
        ctx.fillStyle = 'rgba(255,255,255,0.5)';
        ctx.font = '16px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('等待数据...', CANVAS_W / 2, CANVAS_H / 2);
    }

});
