/**
 * 精彩视频生成功能模块
 */

console.log('[HIGHLIGHT] ===== 模块开始加载 =====');
console.log('[HIGHLIGHT] 当前时间:', new Date().toISOString());

// 全局变量
const API_BASE = 'http://127.0.0.1:9999';
let highlightGenerationInProgress = false;

// 页面加载完成后初始化
console.log('[HIGHLIGHT] 注册 DOMContentLoaded 事件监听器');
document.addEventListener('DOMContentLoaded', () => {
    console.log('[HIGHLIGHT] ===== DOMContentLoaded 触发 =====');
    console.log('[HIGHLIGHT] document.readyState:', document.readyState);
    
    const generateHighlightCheckbox = document.getElementById('generate-highlight-checkbox');
    const highlightSettings = document.getElementById('highlight-settings');
    const uploadBtn = document.getElementById('upload-btn');
    
    console.log('[HIGHLIGHT] DOM 元素检查:', {
        checkbox: !!generateHighlightCheckbox,
        settings: !!highlightSettings,
        uploadBtn: !!uploadBtn
    });
    
    // 复选框切换显示设置
    if (generateHighlightCheckbox) {
        console.log('[HIGHLIGHT] 绑定复选框事件');
        generateHighlightCheckbox.addEventListener('change', (e) => {
            console.log('[HIGHLIGHT] 复选框状态变化:', e.target.checked);
            if (highlightSettings) {
                highlightSettings.style.display = e.target.checked ? 'block' : 'none';
            }
        });
    } else {
        console.error('[HIGHLIGHT] ❌ 找不到复选框元素 #generate-highlight-checkbox');
    }
    
    // 监听自定义事件：视频分析完成
    console.log('[HIGHLIGHT] 绑定 videoAnalysisCompleted 事件监听器');
    document.addEventListener('videoAnalysisCompleted', (e) => {
        console.log('[HIGHLIGHT] ===== 收到视频分析完成事件 =====');
        console.log('[HIGHLIGHT] 事件详情:', e.detail);
        console.log('[HIGHLIGHT] 事件中的 videoId:', e.detail?.videoId);
        console.log('[HIGHLIGHT] window.uploadedVideoId:', window.uploadedVideoId);
        
        // 🔥 重要：从事件中更新 videoId（防止使用旧的ID）
        if (e.detail && e.detail.videoId) {
            window.uploadedVideoId = e.detail.videoId;
            console.log('[HIGHLIGHT] ✅ 已更新 window.uploadedVideoId:', window.uploadedVideoId);
        }
        
        const checkbox = document.getElementById('generate-highlight-checkbox');
        console.log('[HIGHLIGHT] 复选框状态:', checkbox ? checkbox.checked : '未找到');
        
        if (checkbox && checkbox.checked && window.uploadedVideoId) {
            console.log('[HIGHLIGHT] ✅ 条件满足，1秒后启动精彩视频生成');
            console.log('[HIGHLIGHT] 使用的 videoId:', window.uploadedVideoId);
            setTimeout(() => {
                console.log('[HIGHLIGHT] 调用 startHighlightGeneration()');
                startHighlightGeneration();
            }, 1000);
        } else {
            console.log('[HIGHLIGHT] ⚠️ 不满足生成条件，跳过');
            if (!window.uploadedVideoId) {
                console.error('[HIGHLIGHT] ❌ 错误：videoId 不存在！');
            }
        }
    });
    
    console.log('[HIGHLIGHT] ===== 初始化完成 =====');
});

/**
 * 开始生成精彩视频
 */
async function startHighlightGeneration() {
    if (!window.uploadedVideoId) {
        console.warn('[HIGHLIGHT] 没有已上传的视频');
        return;
    }
    
    if (highlightGenerationInProgress) {
        console.warn('[HIGHLIGHT] 精彩视频生成已在进行中');
        return;
    }
    
    highlightGenerationInProgress = true;
    
    // 获取设置
    const targetDuration = parseFloat(document.getElementById('target-duration')?.value || 180);
    const enableSlowmo = document.getElementById('enable-slowmo')?.checked !== false;
    const enableZoom = document.getElementById('enable-zoom')?.checked !== false;
    const enablePip = document.getElementById('enable-pip')?.checked !== false;
    
    console.log('[HIGHLIGHT] 开始生成精彩视频', {
        videoId: window.uploadedVideoId,
        targetDuration,
        enableSlowmo,
        enableZoom,
        enablePip
    });
    
    // 显示生成状态通知
    showHighlightNotification('开始生成精彩视频...', 'info');
    
    try {
        // 调用后端 API 启动生成任务
        const response = await fetch(`${API_BASE}/api/generate-highlight`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                video_id: window.uploadedVideoId,
                target_duration: targetDuration,
                enable_slowmo: enableSlowmo,
                enable_zoom: enableZoom,
                enable_pip: enablePip
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        if (result.status === 'success') {
            console.log('[HIGHLIGHT] 生成任务已启动:', result);
            
            // 开始轮询进度
            pollHighlightProgress(window.uploadedVideoId);
        } else {
            throw new Error(result.message || '启动生成任务失败');
        }
    } catch (error) {
        console.error('[HIGHLIGHT] 启动生成失败:', error);
        showHighlightNotification(`生成失败: ${error.message}`, 'error');
        highlightGenerationInProgress = false;
    }
}

/**
 * 轮询精彩视频生成进度
 */
async function pollHighlightProgress(videoId) {
    let retryCount = 0;
    const maxRetries = 1350; // 最多45分钟（1350次 × 2秒），兼容90分钟比赛视频
    // 心跳检测：超过 120 秒无进度更新则提示用户
    const HEARTBEAT_TIMEOUT = 120;

    const checkProgress = async () => {
        try {
            const response = await fetch(`${API_BASE}/api/highlight-progress/${videoId}`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const data = await response.json();
            
            console.log('[HIGHLIGHT] 进度:', data);
            
            if (data.status === 'completed') {
                console.log('[HIGHLIGHT] ✅ 精彩视频生成完成!');
                showHighlightNotification(
                    `🎉 精彩视频生成完成！文件大小: ${data.file_size_mb} MB`,
                    'success',
                    {
                        downloadUrl: `${API_BASE}${data.download_url}`,
                        filename: data.output_file
                    }
                );
                addHighlightVideoToList({
                    videoId: videoId,
                    filename: data.output_file,
                    downloadUrl: `${API_BASE}${data.download_url}`,
                    streamUrl: `${API_BASE}/api/stream-highlight/${videoId}`,
                    fileSize: data.file_size_mb
                });
                highlightGenerationInProgress = false;
                return;

            } else if (data.status === 'failed') {
                console.error('[HIGHLIGHT] ❌ 精彩视频生成失败:', data.message);
                hideHighlightNotification();
                // 显示带技术说明和操作建议的错误卡片
                showHighlightError(data.message, data.error_detail);
                highlightGenerationInProgress = false;
                return;

            } else if (data.status === 'processing') {
                const progress = data.progress || 0;

                let statusMessage = '';
                if (progress < 10) {
                    statusMessage = '初始化分析引擎...';
                } else if (progress < 50) {
                    statusMessage = '分析视频内容，识别精彩时刻...';
                } else if (progress < 60) {
                    statusMessage = '选择最佳片段...';
                } else if (progress < 80) {
                    statusMessage = '应用特效（慢动作、特写）...';
                } else if (progress < 100) {
                    statusMessage = '合并片段，生成最终视频...';
                } else {
                    statusMessage = '即将完成...';
                }

                // ── 心跳检测：超过 HEARTBEAT_TIMEOUT 秒无更新时附加警告 ──
                let heartbeatWarning = '';
                if (data.last_heartbeat) {
                    const staleSecs = Math.floor(Date.now() / 1000 - data.last_heartbeat);
                    if (staleSecs > HEARTBEAT_TIMEOUT) {
                        heartbeatWarning = `（已 ${staleSecs} 秒无响应，处理中请稍候）`;
                    }
                }

                updateHighlightProgress(progress, statusMessage + heartbeatWarning);
                console.log(`[HIGHLIGHT] 📊 进度: ${progress}% - ${statusMessage}`);

                retryCount++;
                if (retryCount < maxRetries) {
                    setTimeout(checkProgress, 2000);
                } else {
                    hideHighlightNotification();
                    showHighlightError(
                        '生成超时（超过 45 分钟）',
                        '可能原因：视频过长、服务器资源不足。建议：上传不超过 60 分钟的视频片段，或关闭慢动作/画中画特效后重试。'
                    );
                    highlightGenerationInProgress = false;
                }
            } else {
                console.warn('[HIGHLIGHT] 任务不存在，关闭通知');
                hideHighlightNotification();
                highlightGenerationInProgress = false;
            }
        } catch (error) {
            console.error('[HIGHLIGHT] 查询进度失败:', error);
            retryCount++;
            if (retryCount < maxRetries) {
                setTimeout(checkProgress, 3000); // 失败后稍微延长间隔
            } else {
                showHighlightError('网络连接异常', '无法连接到后端服务，请检查服务是否正常运行后刷新页面重试。');
                highlightGenerationInProgress = false;
            }
        }
    };
    
    checkProgress();
}

/**
 * 更新精彩视频生成进度
 */
function updateHighlightProgress(progress, message) {
    const notification = document.getElementById('highlight-notification');
    if (notification) {
        const progressBar = notification.querySelector('.highlight-progress-bar');
        const messageEl = notification.querySelector('.highlight-message');
        
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
        }
        
        if (messageEl) {
            messageEl.textContent = message || `生成中... ${progress}%`;
        }
    }
}

/**
 * 隐藏/移除精彩视频生成通知
 */
function hideHighlightNotification() {
    const notification = document.getElementById('highlight-notification');
    if (notification) {
        notification.remove();
    }
}

/**
 * 显示带技术说明和操作建议的错误卡片
 */
function showHighlightError(friendlyMsg, techDetail) {
    hideHighlightNotification();

    // 根据 techDetail 自动补充操作建议
    let suggestions = [];
    if (!techDetail) techDetail = '';
    const td = techDetail.toLowerCase();
    if (td.includes('winError 2') || td.includes('ffmpeg') || td.includes('找不到指定')) {
        suggestions = ['重启后端服务后重试', '确认系统已安装 imageio-ffmpeg（pip install imageio-ffmpeg）'];
    } else if (td.includes('超时') || td.includes('timeout')) {
        suggestions = ['上传时长不超过 30 分钟的视频', '关闭「慢动作」和「画中画」特效以加快速度', '在服务器资源较空闲时重试'];
    } else if (td.includes('codec') || td.includes('decoder') || td.includes('格式')) {
        suggestions = ['将视频转换为 H.264/MP4 格式（可使用 HandBrake 或 ffmpeg）', '重新上传转换后的视频'];
    } else if (td.includes('不存在') || td.includes('no such file')) {
        suggestions = ['视频文件已被删除，请重新上传'];
    } else {
        suggestions = ['重新上传视频后再试', '若问题持续，请联系管理员'];
    }

    const notification = document.createElement('div');
    notification.id = 'highlight-notification';
    notification.style.cssText = `
        position: fixed; top: 20px; right: 20px;
        background: rgba(255,99,132,0.08); border: 2px solid rgba(255,99,132,0.5);
        border-radius: 12px; padding: 20px; min-width: 320px; max-width: 440px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35); z-index: 10000; animation: slideIn 0.3s ease;
    `;
    notification.innerHTML = `
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
            <i class="fas fa-exclamation-circle" style="font-size:1.4rem;color:#ff6384;"></i>
            <strong style="color:#ff6384;font-size:0.95rem;">精彩视频生成失败</strong>
            <button onclick="hideHighlightNotification()"
                style="margin-left:auto;background:none;border:none;color:rgba(255,255,255,0.4);cursor:pointer;font-size:1rem;">&times;</button>
        </div>
        <p style="color:#fff;margin:0 0 8px;font-size:0.9rem;">${friendlyMsg}</p>
        ${techDetail ? `<p style="color:rgba(255,255,255,0.45);margin:0 0 10px;font-size:0.78rem;word-break:break-all;">技术详情：${techDetail.substring(0, 200)}</p>` : ''}
        <div style="border-top:1px solid rgba(255,99,132,0.2);padding-top:10px;">
            <p style="color:#ffce56;font-size:0.82rem;margin:0 0 6px;font-weight:600;">建议操作：</p>
            <ul style="margin:0;padding-left:18px;color:rgba(255,255,255,0.75);font-size:0.82rem;">
                ${suggestions.map(s => `<li style="margin-bottom:4px;">${s}</li>`).join('')}
            </ul>
        </div>
    `;
    document.body.appendChild(notification);
}

/**
 * 显示精彩视频生成通知
 */
function showHighlightNotification(message, type = 'info', data = null) {
    // 移除旧的通知
    const existingNotification = document.getElementById('highlight-notification');
    if (existingNotification) {
        existingNotification.remove();
    }
    
    // 颜色配置
    const colors = {
        info: { bg: 'rgba(0,212,255,0.1)', border: 'rgba(0,212,255,0.5)', text: '#00d4ff' },
        success: { bg: 'rgba(0,255,136,0.1)', border: 'rgba(0,255,136,0.5)', text: '#00ff88' },
        warning: { bg: 'rgba(255,206,86,0.1)', border: 'rgba(255,206,86,0.5)', text: '#ffce56' },
        error: { bg: 'rgba(255,99,132,0.1)', border: 'rgba(255,99,132,0.5)', text: '#ff6384' }
    };
    
    const style = colors[type] || colors.info;
    
    // 创建通知元素
    const notification = document.createElement('div');
    notification.id = 'highlight-notification';
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${style.bg};
        border: 2px solid ${style.border};
        border-radius: 12px;
        padding: 20px;
        min-width: 300px;
        max-width: 400px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        z-index: 10000;
        animation: slideIn 0.3s ease;
    `;
    
    let content = `
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 10px;">
            <i class="fas fa-video" style="font-size: 1.5rem; color: ${style.text};"></i>
            <strong style="color: ${style.text}; font-size: 1rem;">精彩视频生成</strong>
        </div>
        <p class="highlight-message" style="color: var(--text-primary); margin: 0 0 10px 0;">${message}</p>
    `;
    
    // 添加进度条（仅在处理中时显示）
    if (type === 'info') {
        content += `
            <div style="background: rgba(255,255,255,0.1); height: 6px; border-radius: 3px; overflow: hidden; margin-bottom: 10px;">
                <div class="highlight-progress-bar" style="background: ${style.text}; height: 100%; width: 0%; transition: width 0.3s;"></div>
            </div>
        `;
    }
    
    // 添加下载按钮（仅在成功时显示）
    if (type === 'success' && data && data.downloadUrl) {
        content += `
            <a href="${data.downloadUrl}" download="${data.filename}" style="
                display: inline-block;
                background: ${style.text};
                color: var(--bg-dark);
                padding: 8px 16px;
                border-radius: 6px;
                text-decoration: none;
                font-weight: bold;
                margin-top: 10px;
            ">
                <i class="fas fa-download"></i> 下载精彩视频
            </a>
        `;
    }
    
    // 添加关闭按钮
    content += `
        <button onclick="this.closest('#highlight-notification').remove()" style="
            position: absolute;
            top: 10px;
            right: 10px;
            background: transparent;
            border: none;
            color: var(--text-dim);
            font-size: 1.2rem;
            cursor: pointer;
            padding: 0;
            width: 24px;
            height: 24px;
        ">&times;</button>
    `;
    
    notification.innerHTML = content;
    document.body.appendChild(notification);
    
    // 自动关闭（错误和成功状态保留更长时间）
    if (type !== 'info') {
        setTimeout(() => {
            if (notification.parentNode) {
                notification.style.animation = 'slideOut 0.3s ease';
                setTimeout(() => notification.remove(), 300);
            }
        }, type === 'success' ? 10000 : 5000);
    }
}

// 添加动画样式
if (!document.getElementById('highlight-animations')) {
    const style = document.createElement('style');
    style.id = 'highlight-animations';
    style.textContent = `
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        @keyframes slideOut {
            from {
                transform: translateX(0);
                opacity: 1;
            }
            to {
                transform: translateX(100%);
                opacity: 0;
            }
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
        }
    `;
    document.head.appendChild(style);
}

/**
 * 在主界面添加精彩视频到列表
 */
function addHighlightVideoToList(videoData) {
    const section = document.getElementById('highlight-videos-section');
    const list = document.getElementById('highlight-videos-list');
    
    if (!section || !list) {
        console.warn('[HIGHLIGHT] 找不到精彩视频区域');
        return;
    }
    
    // 显示区域
    section.style.display = 'block';
    
    // 预览使用流式 URL，下载使用下载 URL
    const previewUrl = videoData.streamUrl || videoData.downloadUrl;
    
    // 创建视频卡片
    const videoCard = document.createElement('div');
    videoCard.style.cssText = `
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
    `;
    
    videoCard.innerHTML = `
        <div style="flex: 1;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                <i class="fas fa-video" style="color: #00d4ff;"></i>
                <strong style="color: var(--text-primary); font-size: 0.95rem;">${videoData.filename}</strong>
            </div>
            <div style="color: var(--text-dim); font-size: 0.8rem;">
                <span><i class="fas fa-hdd"></i> ${videoData.fileSize} MB</span>
                <span style="margin-left: 15px;"><i class="fas fa-check-circle" style="color: #00ff88;"></i> 生成完成</span>
            </div>
        </div>
        <div style="display: flex; gap: 8px;">
            <a href="${videoData.downloadUrl}" download="${videoData.filename}" style="
                background: linear-gradient(135deg, #00d4ff, #0099cc);
                color: white;
                padding: 8px 16px;
                border-radius: 6px;
                text-decoration: none;
                font-weight: bold;
                font-size: 0.85rem;
                display: flex;
                align-items: center;
                gap: 6px;
                white-space: nowrap;
                transition: transform 0.2s;
            " onmouseover="this.style.transform='translateY(-2px)'" onmouseout="this.style.transform='translateY(0)'">
                <i class="fas fa-download"></i> 下载
            </a>
            <button onclick="previewHighlightVideo('${previewUrl}')" style="
                background: rgba(255,255,255,0.1);
                border: 1px solid rgba(255,255,255,0.2);
                color: var(--text-primary);
                padding: 8px 12px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 0.85rem;
                display: flex;
                align-items: center;
                gap: 6px;
                transition: background 0.2s;
            " onmouseover="this.style.background='rgba(255,255,255,0.15)'" onmouseout="this.style.background='rgba(255,255,255,0.1)'">
                <i class="fas fa-play"></i> 预览
            </button>
        </div>
    `;
    
    list.appendChild(videoCard);
}

/**
 * 预览精彩视频
 */
function previewHighlightVideo(videoUrl) {
    // 创建模态框播放视频
    const modal = document.createElement('div');
    modal.id = 'highlight-preview-modal';
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0,0,0,0.9);
        z-index: 20000;
        display: flex;
        align-items: center;
        justify-content: center;
        animation: fadeIn 0.3s ease;
    `;
    
    modal.innerHTML = `
        <div style="position: relative; width: 90%; max-width: 1200px;">
            <button onclick="document.getElementById('highlight-preview-modal').remove()" style="
                position: absolute;
                top: -40px;
                right: 0;
                background: rgba(255,255,255,0.2);
                border: none;
                color: white;
                font-size: 2rem;
                cursor: pointer;
                width: 40px;
                height: 40px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
            ">&times;</button>
            <video controls autoplay style="width: 100%; border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.5);">
                <source src="${videoUrl}" type="video/mp4">
                您的浏览器不支持视频播放。
            </video>
        </div>
    `;
    
    // 点击背景关闭
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.remove();
        }
    });
    
    document.body.appendChild(modal);
}

console.log('[HIGHLIGHT] 精彩视频生成模块已加载');
