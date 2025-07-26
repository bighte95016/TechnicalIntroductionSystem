#!/usr/bin/env python3
"""
RAG聊天機器人 Web UI - 虛擬人物版
包含完整的虛擬人物動態變化功能
"""

import os
import sys
import threading
import time
import tempfile
import wave
import numpy as np
from flask import Flask, render_template_string, request, jsonify, send_file
from flask_socketio import SocketIO, emit

# 導入RAG系統模組
from roboticBar_printChunk_QAsplit_indexTTSclone_JP_UI import (
    load_documents, split_documents, create_vector_store, initialize_llm,
    create_qa_chain, PDF_DIRECTORY, VECTORSTORE_DIR, BM25_CACHE_DIR,
    speech_to_text, text_to_speech, map_whisper_language_to_supported,
    load_whisper_model, ENABLE_TTS_OUTPUT, ENABLE_VOICE_INPUT, clean_text_for_tts,
    TTS_MODEL_DIR, TTS_CONFIG_PATH, TTS_VOICE_PATH, USE_PYTTSX3_FOR_JAPANESE,
    ENABLE_PROMPT_AUDIO
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'rag_avatar_chatbot_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# 全局變量
qa_chain = None
system_ready = False
initialization_status = "正在初始化..."

# TTS播放鎖
tts_playback_lock = threading.Lock()
prompt_playback_lock = threading.Lock()

# 全局IndexTTS系統實例
global_tts_system = None
tts_system_lock = threading.Lock()

# 虛擬人物狀態管理
avatar_state = 'idle'  # idle, thinking, speaking
avatar_lock = threading.Lock()

# HTML模板 - 包含虛擬人物功能
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG聊天機器人 - 虛擬人物版</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        body {
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            max-width: 2000px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .main-container {
            display: flex;
            gap: 20px;
            height: calc(100vh - 40px);
        }
        
        .left-panel {
            flex: 1;
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow-y: auto;
        }
        
        .right-panel {
            width: 900px;
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        
        .avatar-container {
            width: 100%;
            height: 900px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 15px;
            overflow: hidden;
            position: relative;
            margin-bottom: 20px;
        }
        
        .avatar-image {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        
        .avatar-video {
            width: 100%;
            height: 100%;
            object-fit: contain;
            border-radius: 10px;
        }
        
        .avatar-status {
            text-align: center;
            padding: 15px;
            border-radius: 10px;
            font-weight: bold;
            background: #e3f2fd;
            color: #1976d2;
            border: 2px solid #bbdefb;
            margin-bottom: 15px;
            transition: all 0.3s ease;
        }
        
        .avatar-status.thinking {
            background: #fff3e0;
            color: #f57c00;
            border-color: #ffcc02;
            animation: pulse-orange 1.5s infinite;
        }
        
        .avatar-status.speaking {
            background: #e8f5e8;
            color: #2e7d32;
            border-color: #c8e6c9;
            animation: pulse-green 1.5s infinite;
        }
        
        @keyframes pulse-orange {
            0% { box-shadow: 0 0 0 0 rgba(245, 124, 0, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(245, 124, 0, 0); }
            100% { box-shadow: 0 0 0 0 rgba(245, 124, 0, 0); }
        }
        
        @keyframes pulse-green {
            0% { box-shadow: 0 0 0 0 rgba(46, 125, 50, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(46, 125, 50, 0); }
            100% { box-shadow: 0 0 0 0 rgba(46, 125, 50, 0); }
        }
        
        .avatar-info {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 14px;
            color: #6c757d;
        }
        
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }
        
        .status {
            text-align: center;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 10px;
            font-weight: bold;
        }
        
        .status.ready {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .status.loading {
            background: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
        }
        
        .status.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .input-section {
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .text-input-panel, .voice-input-panel {
            flex: 1;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 15px;
            border: 2px solid #e9ecef;
        }
        
        .panel-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #495057;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        #question {
            width: 100%;
            height: 100px;
            padding: 15px;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            font-size: 16px;
            resize: vertical;
            box-sizing: border-box;
        }
        
        #question:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 50px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 15px;
            width: 100%;
            transition: all 0.3s ease;
        }
        
        .button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        
        .button:disabled {
            background: #6c757d;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        .voice-button {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            font-size: 18px;
            height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        
        .voice-button.recording {
            background: linear-gradient(135deg, #dc3545 0%, #e74c3c 100%);
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(220, 53, 69, 0); }
            100% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0); }
        }
        
        .recording-indicator {
            text-align: center;
            margin-top: 10px;
            font-size: 14px;
            color: #666;
        }
        
        .chat-history {
            max-height: 400px;
            overflow-y: auto;
            margin-top: 20px;
            border-top: 2px solid #e9ecef;
            padding-top: 20px;
        }
        
        .message {
            margin-bottom: 20px;
            padding: 15px;
            border-radius: 10px;
        }
        
        .question-msg {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
        }
        
        .answer-msg {
            background: #f3e5f5;
            border-left: 4px solid #9c27b0;
            white-space: pre-wrap;
        }
        
        .timestamp {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
        
        @media (max-width: 1200px) {
            .main-container {
                flex-direction: column;
                height: auto;
            }
            .right-panel {
                width: 100%;
            }
        }
        
        @media (max-width: 768px) {
            .input-section {
                flex-direction: column;
            }
            .main-container {
                padding: 10px;
            }
            .left-panel, .right-panel {
                padding: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="main-container">
        <!-- 左側面板 - 聊天界面 -->
        <div class="left-panel">
            <h1>🤖 RAG聊天機器人 - 虛擬人物版</h1>
            
            <div id="status" class="status loading">
                <span id="statusText">正在初始化系統...</span>
            </div>
            
            <div class="input-section">
                <div class="text-input-panel">
                    <div class="panel-title">📝 文字輸入</div>
                    <textarea id="question" placeholder="請輸入您的問題..."></textarea>
                    <button class="button" id="textBtn" onclick="askTextQuestion()">提交問題</button>
                </div>
                
                <div class="voice-input-panel">
                    <div class="panel-title">🎤 語音輸入</div>
                    <button class="button voice-button" id="voiceBtn" onclick="toggleVoiceRecording()">
                        <span id="voiceIcon">🎤</span>
                        <span id="voiceText">開始錄音</span>
                    </button>
                    <div class="recording-indicator" id="recordingIndicator"></div>
                </div>
            </div>
            
            <div id="chatHistory" class="chat-history"></div>
        </div>
        
        <!-- 右側面板 - 虛擬人物 -->
        <div class="right-panel">
            <div class="avatar-container" id="avatarContainer">
                <!-- 靜態圖片 -->
                <img id="avatarImage" 
                     src="/static/avatarImage_files/StaticStatus.png" 
                     alt="AI助手虛擬人物" 
                     class="avatar-image"
                     style="display: block;">
                
                <!-- 說話視頻 -->
                <video id="avatarSpeakingVideo" 
                       class="avatar-video" 
                       style="display: none;" 
                       muted
                       loop
                       preload="auto"
                       playsinline>
                    <source src="/static/avatarImage_files/TalkStatus.mp4" type="video/mp4">
                    您的瀏覽器不支持視頻播放。
                </video>
            </div>
            
            <div id="avatarStatus" class="avatar-status">
                🤖 AI助手待命中
            </div>
            
            <div class="avatar-info">
                <div>💭 狀態: <span id="avatarCurrentState">待命</span></div>
                <div>🎵 語音: <span id="avatarVoiceState">靜音</span></div>
                <div>🧠 思考: <span id="avatarThinkState">空閒</span></div>
            </div>
        </div>
    </div>
    
    <script>
        const socket = io();
        let systemReady = false;
        let isRecording = false;
        let mediaRecorder = null;
        let audioChunks = [];
        
        // 虛擬人物狀態管理
        let avatarState = 'idle'; // idle, thinking, speaking
        let isAvatarAnimating = false;
        
        // 虛擬人物控制函數
        function setAvatarState(newState, statusText = '', voiceState = '', thinkState = '') {
            const avatarImage = document.getElementById('avatarImage');
            const avatarSpeakingVideo = document.getElementById('avatarSpeakingVideo');
            const avatarStatus = document.getElementById('avatarStatus');
            const avatarCurrentState = document.getElementById('avatarCurrentState');
            const avatarVoiceState = document.getElementById('avatarVoiceState');
            const avatarThinkState = document.getElementById('avatarThinkState');
            
            console.log(`🎭 虛擬人物狀態切換: ${avatarState} → ${newState}`);
            
            // 停止所有視頻
            avatarSpeakingVideo.pause();
            
            // 隱藏所有元素
            avatarImage.style.display = 'none';
            avatarSpeakingVideo.style.display = 'none';
            
            // 清除所有狀態類
            avatarStatus.classList.remove('thinking', 'speaking');
            
            avatarState = newState;
            isAvatarAnimating = true;
            
            switch (newState) {
                case 'idle':
                    avatarImage.style.display = 'block';
                    avatarStatus.textContent = statusText || '🤖 AI助手待命中';
                    avatarCurrentState.textContent = '待命';
                    avatarVoiceState.textContent = voiceState || '靜音';
                    avatarThinkState.textContent = thinkState || '空閒';
                    isAvatarAnimating = false;
                    console.log('🖼️ 虛擬人物切換到靜態狀態');
                    break;
                    
                case 'thinking': // 有聲思考 (播放提示音)
                    avatarSpeakingVideo.style.display = 'block';
                    avatarSpeakingVideo.currentTime = 0;
                    avatarSpeakingVideo.play().catch(error => {
                        console.log('❌ 思考時影片播放失敗:', error);
                        avatarSpeakingVideo.style.display = 'none';
                        avatarImage.style.display = 'block';
                    });
                    
                    avatarStatus.textContent = statusText || '🤔 AI助手正在思考...';
                    avatarStatus.classList.add('thinking');
                    avatarCurrentState.textContent = '思考中';
                    avatarVoiceState.textContent = voiceState || '提示音播放中';
                    avatarThinkState.textContent = thinkState || '分析問題';
                    console.log('🤔 虛擬人物切換到有聲思考狀態 (播放影片)');
                    break;

                case 'processing': // 無聲思考 (LLM處理)
                    avatarImage.style.display = 'block';
                    
                    avatarStatus.textContent = statusText || '🧠 AI助手處理中...';
                    avatarStatus.classList.add('thinking'); // 沿用思考的樣式
                    avatarCurrentState.textContent = '處理中';
                    avatarVoiceState.textContent = voiceState || '靜音';
                    avatarThinkState.textContent = thinkState || '生成回答';
                    console.log('🧠 虛擬人物切換到無聲處理狀態 (靜態圖片)');
                    break;
                    
                case 'speaking':
                    avatarSpeakingVideo.style.display = 'block';
                    avatarSpeakingVideo.currentTime = 0;
                    avatarSpeakingVideo.play().catch(error => {
                        console.log('❌ 說話動畫播放失敗:', error);
                        setAvatarState('idle', '🤖 AI助手待命中');
                    });
                    
                    avatarStatus.textContent = statusText || '🗣️ AI助手正在回答';
                    avatarStatus.classList.add('speaking');
                    avatarCurrentState.textContent = '說話中';
                    avatarVoiceState.textContent = voiceState || '播放中';
                    avatarThinkState.textContent = thinkState || '表達想法';
                    console.log('🗣️ 虛擬人物切換到說話狀態');
                    break;
                    
                default:
                    console.log('⚠️ 未知的虛擬人物狀態:', newState);
                    setAvatarState('idle');
                    break;
            }
        }
        
        // 移除輔助函數 setAvatarThinking, setAvatarSpeaking, setAvatarIdle
        
        // WebSocket事件處理
        socket.on('system_status', function(data) {
            updateSystemStatus(data.status, data.message);
        });
        
        // 虛擬人物狀態變化事件處理
        socket.on('avatar_state_change', function(data) {
            console.log('🎭 收到虛擬人物狀態變化:', data);
            // 直接使用後端傳來的狀態進行設置
            setAvatarState(
                data.state,
                data.message,      // statusText
                data.voice_state,  // voiceState
                data.think_state   // thinkState
            );
        });
        
        function updateSystemStatus(status, message) {
            const statusDiv = document.getElementById('status');
            const statusText = document.getElementById('statusText');
            const textBtn = document.getElementById('textBtn');
            const voiceBtn = document.getElementById('voiceBtn');
            
            statusText.textContent = message;
            
            if (status === 'ready') {
                statusDiv.className = 'status ready';
                systemReady = true;
                textBtn.disabled = false;
                voiceBtn.disabled = false;
            } else if (status === 'error') {
                statusDiv.className = 'status error';
                systemReady = false;
                textBtn.disabled = true;
                voiceBtn.disabled = true;
            } else {
                statusDiv.className = 'status loading';
                systemReady = false;
                textBtn.disabled = true;
                voiceBtn.disabled = true;
            }
        }
        
        function addMessage(question, answer, audioUrl = null, language = 'zh') {
            const chatHistory = document.getElementById('chatHistory');
            const timestamp = new Date().toLocaleTimeString();
            
            console.log(`💬 添加新消息 (語言: ${language})`);
            
            const messageHTML = `
                <div class="message question-msg">
                    <strong>問題：</strong>${question}
                    <div class="timestamp">${timestamp}</div>
                </div>
                <div class="message answer-msg">
                    <strong>回答：</strong>${answer}
                    <div class="timestamp">${timestamp}</div>
                </div>
            `;
            
            chatHistory.innerHTML += messageHTML;
            chatHistory.scrollTop = chatHistory.scrollHeight;
            
            console.log(`✅ 消息已添加到聊天記錄`);
        }
        
        // 文字問答
        async function askTextQuestion() {
            const question = document.getElementById('question').value.trim();
            const textBtn = document.getElementById('textBtn');
            
            if (!question) {
                alert('請輸入問題');
                return;
            }
            
            if (!systemReady) {
                alert('系統尚未就緒，請稍候');
                return;
            }
            
            textBtn.disabled = true;
            textBtn.textContent = '處理中...';
            
            console.log('💬 開始處理文字問題');
            await processTextQuestion(question);
        }
        
        async function processTextQuestion(question) {
            const textBtn = document.getElementById('textBtn');
            
            try {
                const response = await fetch('/api/text_question', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question: question})
                });
                
                const data = await response.json();
                if (data.success) {
                    const validAudioUrl = (data.audio_url && data.audio_url !== 'null') ? data.audio_url : null;
                    const language = data.language || 'zh';
                    addMessage(data.question, data.answer, validAudioUrl, language);
                    document.getElementById('question').value = '';
                } else {
                    alert('錯誤: ' + data.message);
                }
            } catch (error) {
                alert('處理問題時發生錯誤: ' + error.message);
            } finally {
                textBtn.disabled = false;
                textBtn.textContent = '提交問題';
            }
        }
        
        // 語音錄音功能
        async function toggleVoiceRecording() {
            if (!systemReady) {
                alert('系統尚未就緒，請稍候');
                return;
            }
            
            if (!isRecording) {
                await startRecording();
            } else {
                await stopRecording();
            }
        }
        
        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                
                mediaRecorder.ondataavailable = function(event) {
                    audioChunks.push(event.data);
                };
                
                mediaRecorder.onstop = function() {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    processAudioBlob(audioBlob);
                };
                
                mediaRecorder.start();
                isRecording = true;
                
                // 更新UI
                const voiceBtn = document.getElementById('voiceBtn');
                const voiceIcon = document.getElementById('voiceIcon');
                const voiceText = document.getElementById('voiceText');
                const indicator = document.getElementById('recordingIndicator');
                
                voiceBtn.classList.add('recording');
                voiceIcon.textContent = '⏹️';
                voiceText.textContent = '停止錄音';
                indicator.textContent = '🔴 正在錄音中...';
                
            } catch (error) {
                alert('無法訪問麥克風: ' + error.message);
            }
        }
        
        async function stopRecording() {
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                isRecording = false;
                
                // 更新UI
                const voiceBtn = document.getElementById('voiceBtn');
                const voiceIcon = document.getElementById('voiceIcon');
                const voiceText = document.getElementById('voiceText');
                const indicator = document.getElementById('recordingIndicator');
                
                voiceBtn.classList.remove('recording');
                voiceBtn.disabled = true;
                voiceIcon.textContent = '⏳';
                voiceText.textContent = '處理中...';
                indicator.textContent = '正在處理音頻...';
            }
        }
        
        function processAudioBlob(audioBlob) {
            console.log('🎤 語音錄製完成，準備處理...');
            console.log('💬 開始處理語音問題');
            processVoiceRequest(audioBlob);
        }
        
        async function processVoiceRequest(audioBlob) {
            try {
                const formData = new FormData();
                formData.append('audio', audioBlob, 'recording.wav');
                
                const response = await fetch('/api/voice_question', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                if (data.success) {
                    const validAudioUrl = (data.audio_url && data.audio_url !== 'null') ? data.audio_url : null;
                    const language = data.language || 'zh';
                    addMessage(data.question, data.answer, validAudioUrl, language);
                } else {
                    alert('錯誤: ' + data.message);
                }
            } catch (error) {
                alert('處理語音時發生錯誤: ' + error.message);
            } finally {
                // 恢復UI
                const voiceBtn = document.getElementById('voiceBtn');
                const voiceIcon = document.getElementById('voiceIcon');
                const voiceText = document.getElementById('voiceText');
                const indicator = document.getElementById('recordingIndicator');
                
                voiceBtn.disabled = false;
                voiceIcon.textContent = '🎤';
                voiceText.textContent = '開始錄音';
                indicator.textContent = '';
            }
        }
        
        // 支援Enter鍵提交（Shift+Enter換行）
        document.getElementById('question').addEventListener('keydown', function(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                askTextQuestion();
            }
        });

        // 頁面載入完成後的初始化
        document.addEventListener('DOMContentLoaded', function() {
            console.log('RAG聊天機器人 Web UI 已載入（虛擬人物版）');
            
            // 預載入視頻以減少切換延遲
            const avatarSpeakingVideo = document.getElementById('avatarSpeakingVideo');
            
            avatarSpeakingVideo.load();
            
            avatarSpeakingVideo.addEventListener('loadeddata', function() {
                console.log('✅ 說話視頻預載入完成');
            });
            
            // 用戶交互啟用自動播放
            document.body.addEventListener('click', function enableAutoplay() {
                console.log('用戶已交互，音頻和視頻自動播放已啟用');
                
                // 預熱視頻播放
                [avatarSpeakingVideo].forEach((video, index) => {
                    video.currentTime = 0;
                    video.play().then(() => {
                        video.pause();
                        console.log(`✅ 視頻${index + 1}預熱完成`);
                    }).catch(error => {
                        console.log(`⚠️ 視頻${index + 1}預熱失敗:`, error);
                    });
                });
                
                document.body.removeEventListener('click', enableAutoplay);
            }, { once: true });
            
            // 確保虛擬人物初始狀態正確
            setAvatarState('idle', '🤖 AI助手待命中', '靜音', '空閒');
        });
    </script>
</body>
</html>
"""

# 這裡需要包含所有的後端函數，包括修改過的generate_tts_audio等
# 為了簡潔，我將只展示關鍵的修改部分

def get_global_tts_system():
    """獲取全局IndexTTS系統實例"""
    global global_tts_system, tts_system_lock
    
    if global_tts_system is not None:
        return global_tts_system
    
    with tts_system_lock:
        if global_tts_system is not None:
            return global_tts_system
        
        try:
            from index_tts.indextts.infer import IndexTTS
            
            if not os.path.exists(TTS_MODEL_DIR):
                print(f"❌ TTS模型目錄不存在: {TTS_MODEL_DIR}")
                return None
            if not os.path.exists(TTS_CONFIG_PATH):
                print(f"❌ TTS配置文件不存在: {TTS_CONFIG_PATH}")
                return None
            if not os.path.exists(TTS_VOICE_PATH):
                print(f"❌ 語音參考文件不存在: {TTS_VOICE_PATH}")
                return None
            
            print("🔧 正在初始化全局IndexTTS系統...")
            global_tts_system = IndexTTS(model_dir=TTS_MODEL_DIR, cfg_path=TTS_CONFIG_PATH)
            print(f"✅ 全局IndexTTS系統初始化成功")
            return global_tts_system
            
        except Exception as e:
            print(f"❌ 全局IndexTTS系統初始化失敗: {e}")
            global_tts_system = None
            return None

def play_chinese_english_tts_system(text, language, set_idle_on_finish=True, is_prompt=False):
    """使用IndexTTS直接透過系統播放中文或英文語音，並可選擇在播放完成後切換虛擬人物狀態"""
    global tts_playback_lock
    
    if not tts_playback_lock.acquire(blocking=False):
        print("⚠️ IndexTTS或日文TTS正在播放中，跳過此次播放")
        return False
    
    try:
        def play_speech_thread():
            try:
                tts_system = get_global_tts_system()
                if tts_system is None:
                    raise Exception("無法獲取IndexTTS系統")

                cleaned_text = clean_text_for_tts(text)
                if not cleaned_text:
                    raise Exception("文本清理後為空")

                print(f"🔊 正在生成 {language} 語音...")
                sampling_rate, wav_data = tts_system.infer(TTS_VOICE_PATH, cleaned_text, output_path=None)
                
                if wav_data is None or len(wav_data) == 0:
                    raise Exception("IndexTTS返回空音頻數據")

                print(f"✅ 音頻生成成功，準備播放...")
                
                # 在播放前一刻才發送狀態
                state_to_emit = 'thinking' if is_prompt else 'speaking'
                state_message = f'正在播放提示音 ({language})' if is_prompt else f'正在回答您的問題 ({language})'
                voice_state_message = '提示音播放中' if is_prompt else '回答播放中'
                think_state_message = '初步分析' if is_prompt else '表達答案'

                print(f"🎬 通知前端切換 '{state_to_emit}' 狀態")
                socketio.emit('avatar_state_change', {
                    'state': state_to_emit, 
                    'language': language,
                    'message': state_message,
                    'voice_state': voice_state_message,
                    'think_state': think_state_message
                })

                import sounddevice as sd
                sd.play(wav_data, sampling_rate)
                sd.wait()
                print(f"✅ {language} 語音播放完成")
                
                if set_idle_on_finish:
                    socketio.emit('avatar_state_change', {
                        'state': 'idle', 
                        'language': language, 
                        'message': '🤖 AI助手待命中',
                        'voice_state': '靜音',
                        'think_state': '空閒'
                    })
                
            except Exception as play_error:
                print(f"❌ IndexTTS播放過程中發生錯誤: {play_error}")
                if set_idle_on_finish:
                    socketio.emit('avatar_state_change', {'state': 'idle', 'language': language, 'message': 'TTS播放發生錯誤'})
            finally:
                tts_playback_lock.release()
                print("🔓 IndexTTS語音鎖已釋放")
        
        play_thread = threading.Thread(target=play_speech_thread, daemon=True)
        play_thread.start()
        
        return True
        
    except Exception as e:
        print(f"❌ IndexTTS語音播放時發生錯誤: {e}")
        tts_playback_lock.release()
        if set_idle_on_finish:
            socketio.emit('avatar_state_change', {'state': 'idle', 'language': language, 'message': 'TTS啟動錯誤'})
        return False

def play_japanese_tts_system(text, set_idle_on_finish=True, is_prompt=False):
    """使用pyttsx3直接透過系統播放日文語音，並可選擇在播放完成後切換虛擬人物狀態"""
    global tts_playback_lock
    
    if not tts_playback_lock.acquire(blocking=False):
        print("⚠️ IndexTTS或日文TTS正在播放中，跳過此次播放")
        return False
    
    try:
        import pyttsx3
        import threading
        
        def play_speech_thread():
            engine = None
            try:
                engine = pyttsx3.init()
                if engine is None:
                    raise Exception("無法初始化pyttsx3引擎")

                engine.setProperty('rate', 150)
                engine.setProperty('volume', 0.9)
                
                voices = engine.getProperty('voices')
                japanese_voice_found = False
                
                if voices:
                    for voice in voices:
                        voice_name = voice.name.lower()
                        if any(keyword in voice_name for keyword in ['japan', 'japanese', 'ja-jp', 'haruka', 'ayumi']):
                            try:
                                engine.setProperty('voice', voice.id)
                                japanese_voice_found = True
                                print(f"✅ 找到並設置日文語音: {voice.name}")
                                break
                            except Exception as voice_error:
                                print(f"⚠️ 設置語音失敗: {voice_error}")
                                continue
                
                if not japanese_voice_found:
                    print("⚠️ 未找到專用日文語音，使用默認語音")
                
                cleaned_text = clean_text_for_tts(text)
                if not cleaned_text:
                    raise Exception("日文文本清理後為空")
                
                # 在播放前一刻才發送狀態
                state_to_emit = 'thinking' if is_prompt else 'speaking'
                state_message = '正在播放提示音 (ja)' if is_prompt else '正在回答您的問題 (ja)'
                voice_state_message = '提示音播放中' if is_prompt else '回答播放中'
                think_state_message = '初步分析' if is_prompt else '表達答案'

                print(f"🎬 通知前端切換 '{state_to_emit}' 狀態")
                socketio.emit('avatar_state_change', {
                    'state': state_to_emit, 
                    'language': 'ja',
                    'message': state_message,
                    'voice_state': voice_state_message,
                    'think_state': think_state_message
                })

                print(f"🔊 開始播放日文語音...")
                engine.say(cleaned_text)
                engine.runAndWait()
                print(f"✅ 日文語音播放完成")

                if set_idle_on_finish:
                    socketio.emit('avatar_state_change', {
                        'state': 'idle', 
                        'language': 'ja', 
                        'message': '🤖 AI助手待命中',
                        'voice_state': '靜音',
                        'think_state': '空閒'
                    })
                
            except Exception as play_error:
                print(f"❌ 播放過程中發生錯誤: {play_error}")
                if set_idle_on_finish:
                    socketio.emit('avatar_state_change', {'state': 'idle', 'language': 'ja', 'message': 'TTS播放發生错误'})
            finally:
                if engine is not None:
                    try:
                        engine.stop()
                        del engine
                    except:
                        pass
                tts_playback_lock.release()
                print("🔓 日文TTS鎖已釋放")
        
        play_thread = threading.Thread(target=play_speech_thread, daemon=True)
        play_thread.start()
        
        return True
        
    except ImportError:
        print("❌ pyttsx3未安裝，無法播放日文語音")
        tts_playback_lock.release()
        if set_idle_on_finish:
            socketio.emit('avatar_state_change', {'state': 'idle', 'language': 'ja'})
        return False
    except Exception as e:
        print(f"❌ 日文語音播放時發生錯誤: {e}")
        tts_playback_lock.release()
        if set_idle_on_finish:
            socketio.emit('avatar_state_change', {'state': 'idle', 'language': 'ja', 'message': 'TTS啟動錯誤'})
        return False

def generate_tts_audio_with_avatar(text, language):
    """
    生成TTS音頻並直接透過系統播放。
    虛擬人物的狀態變化（speaking, idle）將在播放函數內部處理，以確保同步。
    """
    if not ENABLE_TTS_OUTPUT:
        print("⚠️ TTS功能已禁用")
        return False
    
    try:
        # 根據語言選擇TTS系統，播放函數將負責處理 avatar 狀態
        if language == "ja" and USE_PYTTSX3_FOR_JAPANESE:
            success = play_japanese_tts_system(text, set_idle_on_finish=True, is_prompt=False)
        else:
            success = play_chinese_english_tts_system(text, language, set_idle_on_finish=True, is_prompt=False)
        
        return success
            
    except Exception as e:
        print(f"❌ 生成TTS音頻時發生未預期錯誤: {e}")
        # 確保在啟動失敗時也能切換回idle
        socketio.emit('avatar_state_change', {
            'state': 'idle',
            'language': language,
            'message': 'TTS啟動時發生錯誤'
        })
        return False

def generate_prompt_audio_with_avatar_async(language="zh"):
    """異步播放提示音並控制虛擬人物思考狀態"""
    if not ENABLE_PROMPT_AUDIO or not ENABLE_TTS_OUTPUT:
        return None
    
    def play_prompt_in_background():
        # 使用專用的提示音鎖
        with prompt_playback_lock:
            try:
                print(f"🎵 準備播放 {language} 提示音（後台播放）...")
                
                prompt_texts = {
                    'zh': "感謝您的提問，我思考一下，請稍後。",
                    'en': "Thank you for your question. Let me think about it and get back to you shortly.",
                    'ja': "ご質問ありがとうございます。少しお考えください。"
                }
                
                prompt_text = prompt_texts.get(language, prompt_texts['zh'])
                
                # 使用TTS播放函數，讓它發送 thinking 狀態，且在結束時不切換回idle
                if language == "ja" and USE_PYTTSX3_FOR_JAPANESE:
                    play_japanese_tts_system(prompt_text, set_idle_on_finish=False, is_prompt=True)
                else:
                    play_chinese_english_tts_system(prompt_text, language, set_idle_on_finish=False, is_prompt=True)
                
                # 等待鎖被釋放，表示播放完成
                tts_playback_lock.acquire()
                tts_playback_lock.release()
                
                print(f"✅ {language} 提示音播放完成，進入靜默思考階段...")
                # 提示音結束後，切換到無聲的處理狀態
                socketio.emit('avatar_state_change', {
                    'state': 'processing',
                    'language': language,
                    'message': '正在深入思考您的問題...',
                    'voice_state': '靜音',
                    'think_state': '生成回答'
                })
                
            except Exception as e:
                print(f"❌ 後台播放 {language} 提示音時發生錯誤: {e}")
                socketio.emit('avatar_state_change', {
                    'state': 'idle',
                    'language': language,
                    'message': '提示音播放發生錯誤',
                    'voice_state': '靜音',
                    'think_state': '錯誤'
                })
    
    import threading
    prompt_thread = threading.Thread(target=play_prompt_in_background, daemon=True)
    prompt_thread.start()
    
    return prompt_thread

def initialize_rag_system():
    """初始化RAG系統"""
    global qa_chain, system_ready, initialization_status
    
    try:
        print("🚀 開始初始化RAG系統...")
        socketio.emit('system_status', {'status': 'initializing', 'message': '正在初始化RAG系統...'})
        
        initialization_status = "正在載入文檔..."
        socketio.emit('system_status', {'status': 'initializing', 'message': initialization_status})
        
        documents = load_documents(PDF_DIRECTORY)
        if not documents:
            raise Exception("無法載入文檔")
        
        initialization_status = "正在分割文檔..."
        socketio.emit('system_status', {'status': 'initializing', 'message': initialization_status})
        
        texts = split_documents(documents)
        if not texts:
            raise Exception("文檔分割失敗")
        
        initialization_status = "正在建立向量儲存..."
        socketio.emit('system_status', {'status': 'initializing', 'message': initialization_status})
        
        vectorstore = create_vector_store(texts, VECTORSTORE_DIR)
        if not vectorstore:
            raise Exception("向量儲存建立失敗")
        
        initialization_status = "正在初始化LLM..."
        socketio.emit('system_status', {'status': 'initializing', 'message': initialization_status})
        
        llm = initialize_llm()
        if not llm:
            raise Exception("LLM初始化失敗")
        
        initialization_status = "正在建立QA鏈..."
        socketio.emit('system_status', {'status': 'initializing', 'message': initialization_status})
        
        qa_chain = create_qa_chain(llm, vectorstore, texts)
        if not qa_chain:
            raise Exception("QA鏈建立失敗")
        
        if ENABLE_TTS_OUTPUT:
            initialization_status = "正在初始化TTS語音系統..."
            socketio.emit('system_status', {'status': 'initializing', 'message': initialization_status})
            
            print("🔧 預初始化IndexTTS系統...")
            tts_system = get_global_tts_system()
            if tts_system is not None:
                print("✅ IndexTTS系統預初始化成功")
            else:
                print("⚠️ IndexTTS系統預初始化失敗，但不影響文字功能")
        
        if ENABLE_VOICE_INPUT:
            initialization_status = "正在預載入Whisper語音識別模型..."
            socketio.emit('system_status', {'status': 'initializing', 'message': initialization_status})
            
            print("🎤 開始預載入Whisper模型...")
            try:
                whisper_loaded = load_whisper_model()
                if whisper_loaded:
                    print("✅ Whisper模型預載入成功！")
                else:
                    print("❌ Whisper模型預載入失敗")
            except Exception as whisper_error:
                print(f"❌ Whisper模型預載入時發生錯誤: {whisper_error}")
        
        system_ready = True
        initialization_status = "系統就緒"
        
        print("✅ RAG系統初始化完成！")
        socketio.emit('system_status', {'status': 'ready', 'message': 'RAG系統已就緒！可以開始提問了。'})
        
    except Exception as e:
        error_msg = f"RAG系統初始化失敗: {str(e)}"
        print(f"❌ {error_msg}")
        initialization_status = error_msg
        socketio.emit('system_status', {'status': 'error', 'message': error_msg})

@app.route('/')
def index():
    """主頁"""
    return HTML_TEMPLATE

@app.route('/api/text_question', methods=['POST'])
def text_question():
    """處理文字問題API"""
    global qa_chain, system_ready
    
    if not system_ready or not qa_chain:
        return jsonify({'success': False, 'message': '系統尚未就緒'})
    
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'success': False, 'message': '問題不能為空'})
            
        print(f"🔍 收到文字問題: {question}")
        
        # 異步播放中文提示音（不阻塞LLM處理）
        prompt_thread = None
        if ENABLE_PROMPT_AUDIO:
            try:
                prompt_thread = generate_prompt_audio_with_avatar_async("zh")
                print("🎵 提示音開始後台播放，同時開始LLM處理")
            except Exception as prompt_error:
                print(f"⚠️ 提示音播放失敗: {prompt_error}")
        
        print("🤖 開始LLM處理...")
        result = qa_chain.invoke({"query": question, "language": "zh"})
        answer = result.get('result', '抱歉，無法生成答案。').strip()
        
        print(f"✅ LLM處理完成: {answer[:100]}...")
        
        if prompt_thread is not None:
            try:
                print("⏳ 等待提示音播放完成...")
                prompt_thread.join(timeout=10)
                if prompt_thread.is_alive():
                    print("⚠️ 提示音播放超時，繼續播放回答語音")
                else:
                    print("✅ 提示音播放已完成，開始播放回答語音")
            except Exception as e:
                print(f"⚠️ 等待提示音完成時發生錯誤: {e}")
        
        tts_success = False
        if ENABLE_TTS_OUTPUT:
            try:
                print(f"🔊 開始生成 zh 語音回答...")
                tts_success = generate_tts_audio_with_avatar(answer, "zh")
                if tts_success:
                    print("✅ 語音回答播放成功")
                else:
                    print("❌ 語音回答播放失敗")
            except Exception as tts_error:
                print(f"❌ TTS處理時發生異常: {tts_error}")
        else:
            print("⚠️ TTS功能已禁用，跳過語音生成")
        
        return jsonify({
            'success': True,
            'question': question,
            'answer': answer,
            'language': 'zh',
            'audio_url': None,
            'tts_success': tts_success
        })
        
    except Exception as e:
        error_msg = f'處理問題時發生錯誤: {str(e)}'
        print(f"❌ {error_msg}")
        return jsonify({'success': False, 'message': error_msg})

@app.route('/api/voice_question', methods=['POST'])
def voice_question():
    """處理語音問題API"""
    global qa_chain, system_ready
    
    if not system_ready or not qa_chain:
        return jsonify({'success': False, 'message': '系統尚未就緒'})
    
    temp_file_path = None
    try:
        if 'audio' not in request.files:
            return jsonify({'success': False, 'message': '未找到音頻文件'})
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'success': False, 'message': '未選擇文件'})
        
        import tempfile
        temp_fd, temp_file_path = tempfile.mkstemp(suffix='.wav', prefix='voice_')
        
        try:
            with os.fdopen(temp_fd, 'wb') as temp_file:
                audio_file.save(temp_file)
            
            print(f"🎤 收到語音文件: {temp_file_path}")
            
            import roboticBar_printChunk_QAsplit_indexTTSclone_JP_UI as main_module
            original_filename = main_module.WAVE_OUTPUT_FILENAME
            main_module.WAVE_OUTPUT_FILENAME = temp_file_path
            
            try:
                if not load_whisper_model():
                    raise Exception("Whisper模型載入失敗")
                
                question, detected_lang = speech_to_text()
                print(f"語音識別完成: question='{question}', language='{detected_lang}'")
                
            finally:
                main_module.WAVE_OUTPUT_FILENAME = original_filename
            
            if not question or len(question.strip()) == 0:
                return jsonify({'success': False, 'message': '無法識別語音內容，請重新錄音'})
            
            print(f"🔍 語音識別結果: {question}")
            
            tts_language = map_whisper_language_to_supported(detected_lang)
            print(f"映射後的TTS語言: {tts_language}")
            
            prompt_thread = None
            if ENABLE_PROMPT_AUDIO:
                try:
                    prompt_thread = generate_prompt_audio_with_avatar_async(tts_language)
                    print(f"🎵 {tts_language} 提示音開始後台播放，同時開始LLM處理")
                except Exception as prompt_error:
                    print(f"⚠️ 提示音播放失敗: {prompt_error}")
            
            print("🤖 開始LLM處理...")
            result = qa_chain.invoke({"query": question, "language": tts_language})
            answer = result.get('result', '抱歉，無法生成答案。').strip()
            
            print(f"✅ LLM處理完成: {answer[:100]}...")
            
            if prompt_thread is not None:
                try:
                    print("⏳ 等待提示音播放完成...")
                    prompt_thread.join(timeout=10)
                    if prompt_thread.is_alive():
                        print("⚠️ 提示音播放超時，繼續播放回答語音")
                    else:
                        print("✅ 提示音播放已完成，開始播放回答語音")
                except Exception as e:
                    print(f"⚠️ 等待提示音完成時發生錯誤: {e}")
            
            tts_success = False
            if ENABLE_TTS_OUTPUT:
                try:
                    print(f"🔊 開始生成 {tts_language} 語音回答...")
                    tts_success = generate_tts_audio_with_avatar(answer, tts_language)
                    if tts_success:
                        print(f"✅ {tts_language} 語音回答播放成功")
                    else:
                        print(f"❌ {tts_language} 語音回答播放失敗")
                except Exception as tts_error:
                    print(f"❌ TTS處理時發生異常: {tts_error}")
            else:
                print("⚠️ TTS功能已禁用，跳過語音生成")
            
            return jsonify({
                'success': True,
                'question': question,
                'answer': answer,
                'language': tts_language,
                'audio_url': None,
                'tts_success': tts_success
            })
            
        except Exception as inner_error:
            print(f"處理音頻文件時發生內部錯誤: {inner_error}")
            raise inner_error
            
    except Exception as e:
        error_msg = f'處理語音時發生錯誤: {str(e)}'
        print(f"❌ {error_msg}")
        return jsonify({'success': False, 'message': error_msg})
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                print(f"✅ 臨時文件已清理: {temp_file_path}")
            except Exception as cleanup_error:
                print(f"⚠️ 清理臨時文件時發生錯誤: {cleanup_error}")

@app.route('/static/<filename>')
def serve_static_file(filename):
    """提供靜態文件服務"""
    return send_file(os.path.join('static', filename))

# 添加虛擬人物文件服務路由
@app.route('/static/avatarImage_files/<filename>')
def serve_avatar_file(filename):
    """提供虛擬人物文件服務"""
    return send_file(os.path.join('avatarImage_files', filename))

@socketio.on('connect')
def handle_connect():
    """客戶端連接事件"""
    print('客戶端已連接')
    if system_ready:
        emit('system_status', {'status': 'ready', 'message': 'RAG系統已就緒！可以開始提問了。'})
    else:
        emit('system_status', {'status': 'initializing', 'message': initialization_status})

@socketio.on('disconnect')
def handle_disconnect():
    """客戶端斷開連接事件"""
    print('客戶端已斷開連接')

if __name__ == '__main__':
    print("="*60)
    print("🎭 RAG聊天機器人 Web UI - 虛擬人物版")
    print("="*60)
    print("正在啟動服務器...")
    print("請在瀏覽器中打開: http://localhost:5003")
    print()
    print("功能特色:")
    print("📝 文字問答 - 直接輸入文字問題")
    print("🎤 語音問答 - 點擊按鈕錄音提問")
    print("🔊 語音回答 - 自動播放回答音頻")
    print("🎭 虛擬人物 - 動態表情和狀態變化")
    print("💭 思考動畫 - 播放提示音時顯示思考狀態")
    print("🗣️ 說話動畫 - 播放回答時顯示說話狀態")
    print("💬 聊天記錄 - 保存完整對話歷史")
    print()
    print("系統初始化包含:")
    print("🤖 RAG問答系統載入")
    print("🔊 IndexTTS語音合成系統預載入")
    print("🎤 Whisper語音識別模型預載入")
    print("📚 文檔向量化和索引建立")
    print("⏳ 首次啟動可能需要較長時間，請耐心等待...")
    print()
    print("虛擬人物文件需求:")
    print("📁 ./avatarImage_files/StaticStatus.png - 靜態待命圖片")
    print("📁 ./avatarImage_files/ThinkingStatus.mp4 - 思考動畫視頻")
    print("📁 ./avatarImage_files/TalkStatus.mp4 - 說話動畫視頻")
    print("="*60)
    
    # 創建必要目錄
    os.makedirs('static', exist_ok=True)
    os.makedirs('avatarImage_files', exist_ok=True)
    
    # 在後台線程中初始化RAG系統
    init_thread = threading.Thread(target=initialize_rag_system)
    init_thread.daemon = True
    init_thread.start()
    
    try:
        # 啟動Flask應用
        socketio.run(app, host='0.0.0.0', port=5003, debug=False, log_output=True)
    except KeyboardInterrupt:
        print("\n🛑 伺服器已停止")
    except Exception as e:
        print(f"\n❌ 伺服器啟動失敗: {e}") 