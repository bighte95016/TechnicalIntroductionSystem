#!/usr/bin/env python3
"""
真正的RAG聊天機器人 Web UI
能夠使用完整的RAG系統回答問題，支持語音功能
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
app.config['SECRET_KEY'] = 'real_rag_chatbot_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# 全局變量
qa_chain = None
system_ready = False
initialization_status = "正在初始化..."
current_recording = None
recording_lock = threading.Lock()

# 預生成的提示音文件路徑
PROMPT_AUDIO_FILES = {
    'zh': None,
    'en': None,
    'ja': None
}

# TTS播放鎖，確保同一時間只有一個語音播放
tts_playback_lock = threading.Lock()
# 提示音播放鎖，與回答語音分開管理
prompt_playback_lock = threading.Lock()

# 全局IndexTTS系統實例，避免重複初始化
global_tts_system = None
tts_system_lock = threading.Lock()  # 保護TTS系統初始化的鎖

class WebUIManager:
    """Web UI管理器，處理語音錄音"""
    
    def __init__(self):
        self.is_recording = False
        self.audio_data = []
        self.sample_rate = 16000
        self.temp_audio_file = None
        
    def start_recording(self):
        """開始錄音"""
        with recording_lock:
            if self.is_recording:
                return False
            
            self.is_recording = True
            self.audio_data = []
            
            # 創建臨時音頻文件
            self.temp_audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            self.temp_audio_file.close()
            
            return True
    
    def add_audio_chunk(self, audio_chunk):
        """添加音頻數據塊"""
        if self.is_recording:
            # 將base64音頻數據轉換為numpy數組
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            self.audio_data.extend(audio_array)
    
    def stop_recording(self):
        """停止錄音並保存文件"""
        with recording_lock:
            if not self.is_recording:
                return None
            
            self.is_recording = False
            
            if self.audio_data and self.temp_audio_file:
                # 保存音頻文件
                try:
                    with wave.open(self.temp_audio_file.name, 'wb') as wf:
                        wf.setnchannels(1)  # 單聲道
                        wf.setsampwidth(2)  # 16位
                        wf.setframerate(self.sample_rate)
                        wf.writeframes(np.array(self.audio_data, dtype=np.int16).tobytes())
                    
                    return self.temp_audio_file.name
                except Exception as e:
                    print(f"保存音頻文件時發生錯誤: {e}")
                    return None
            
            return None
    
    def cleanup_temp_file(self):
        """清理臨時文件"""
        if self.temp_audio_file and os.path.exists(self.temp_audio_file.name):
            try:
                os.unlink(self.temp_audio_file.name)
            except Exception as e:
                print(f"清理臨時文件時發生錯誤: {e}")

# 創建Web UI管理器實例
web_ui_manager = WebUIManager()

# 增強的HTML模板，包含語音功能
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG聊天機器人 - 語音版</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        body {
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
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
        
        .audio-player {
            margin-top: 10px;
            width: 100%;
        }
        
        @media (max-width: 768px) {
            .input-section {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 RAG聊天機器人 - 語音版</h1>
        
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
    
    <script>
        const socket = io();
        let systemReady = false;
        let isRecording = false;
        let mediaRecorder = null;
        let audioChunks = [];
        
        // WebSocket事件處理
        socket.on('system_status', function(data) {
            updateSystemStatus(data.status, data.message);
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
            
            // 所有語音都在服務器端播放，前端不需要顯示語音狀態
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
            
            // 直接處理問題，提示音在服務器端播放
            console.log('💬 開始處理文字問題，提示音將在服務器端播放');
            await processTextQuestion(question);
        }
        
        // 處理文字問題的實際邏輯
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
                    // 確保audio_url有效才傳遞，否則傳遞null
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
            
            // 直接處理語音請求，提示音在服務器端播放
            console.log('💬 開始處理語音問題，提示音將在服務器端播放');
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
                    // 確保audio_url有效才傳遞，否則傳遞null
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
            console.log('RAG聊天機器人 Web UI 已載入');
            
            // 添加用戶交互來啟用自動播放
            document.body.addEventListener('click', function enableAutoplay() {
                console.log('用戶已交互，音頻自動播放已啟用');
                // 移除監聽器，只需要一次交互
                document.body.removeEventListener('click', enableAutoplay);
            }, { once: true });
        });
        
        // 添加音頻播放狀態管理
        let isAudioPlaying = false;
        let currentAudio = null;  // 追蹤當前播放的音頻元素
        
        function playAudioSafely(audioElement) {
            console.log('🎵 playAudioSafely: 準備播放新音頻');
            
            // 強制停止當前播放的音頻
            if (currentAudio && !currentAudio.paused) {
                console.log('⏸️ playAudioSafely: 強制停止當前播放的音頻');
                currentAudio.pause();
                currentAudio.currentTime = 0;
                currentAudio = null;
            }
            
            // 停止所有其他可能正在播放的音頻（更徹底的清理）
            const allAudios = document.querySelectorAll('audio');
            console.log(`🔍 playAudioSafely: 找到 ${allAudios.length} 個音頻元素`);
            allAudios.forEach((audio, index) => {
                if (!audio.paused) {
                    console.log(`⏸️ playAudioSafely: 停止音頻元素 ${index}`);
                    audio.pause();
                    audio.currentTime = 0;
                }
            });
            
            // 重置全局狀態
            isAudioPlaying = false;
            currentAudio = null;
            
            // 等待一小段時間確保所有音頻都已停止
            setTimeout(() => {
                // 設置新的當前音頻
                currentAudio = audioElement;
                isAudioPlaying = true;
                
                // 設置音頻結束事件
                audioElement.onended = function() {
                    isAudioPlaying = false;
                    currentAudio = null;
                    console.log('🔊 playAudioSafely: 音頻播放完成');
                };
                
                // 設置音頻暫停事件
                audioElement.onpause = function() {
                    if (currentAudio === audioElement) {
                        isAudioPlaying = false;
                        currentAudio = null;
                        console.log('⏸️ playAudioSafely: 音頻被暫停');
                    }
                };
                
                // 開始播放
                const playPromise = audioElement.play();
                if (playPromise !== undefined) {
                    playPromise.then(() => {
                        console.log('🔊 playAudioSafely: 音頻開始播放');
                    }).catch(error => {
                        isAudioPlaying = false;
                        currentAudio = null;
                        console.log('⚠️ playAudioSafely: 音頻播放失敗:', error);
                        // 顯示控制器讓用戶手動播放
                        audioElement.controls = true;
                        audioElement.style.display = 'block';
                    });
                }
            }, 100);  // 100ms延遲確保清理完成
        }
    </script>
</body>
</html>
"""

def get_global_tts_system():
    """獲取全局IndexTTS系統實例，如果未初始化則進行初始化（單例模式）"""
    global global_tts_system, tts_system_lock
    
    # 如果已經初始化，直接返回
    if global_tts_system is not None:
        return global_tts_system
    
    # 使用鎖確保只有一個線程進行初始化
    with tts_system_lock:
        # 雙重檢查，防止在等待鎖的過程中其他線程已經初始化了
        if global_tts_system is not None:
            return global_tts_system
        
        try:
            # 導入所需模組
            from roboticBar_printChunk_QAsplit_indexTTSclone_JP_UI import (
                TTS_MODEL_DIR, TTS_CONFIG_PATH, TTS_VOICE_PATH
            )
            from index_tts.indextts.infer import IndexTTS
            
            # 檢查必要文件
            if not os.path.exists(TTS_MODEL_DIR):
                print(f"❌ TTS模型目錄不存在: {TTS_MODEL_DIR}")
                return None
            if not os.path.exists(TTS_CONFIG_PATH):
                print(f"❌ TTS配置文件不存在: {TTS_CONFIG_PATH}")
                return None
            if not os.path.exists(TTS_VOICE_PATH):
                print(f"❌ 語音參考文件不存在: {TTS_VOICE_PATH}")
                return None
            
            # 初始化IndexTTS系統（只初始化一次）
            print("🔧 正在初始化全局IndexTTS系統...")
            global_tts_system = IndexTTS(model_dir=TTS_MODEL_DIR, cfg_path=TTS_CONFIG_PATH)
            print(f"✅ 全局IndexTTS系統初始化成功，可重複使用")
            return global_tts_system
            
        except Exception as e:
            print(f"❌ 全局IndexTTS系統初始化失敗: {e}")
            import traceback
            print(f"詳細錯誤: {traceback.format_exc()}")
            global_tts_system = None
            return None

def cleanup_global_tts_system():
    """清理全局IndexTTS系統（在程序退出時調用）"""
    global global_tts_system
    if global_tts_system is not None:
        try:
            # 如果IndexTTS有清理方法，在這裡調用
            print("🔧 清理全局IndexTTS系統...")
            global_tts_system = None
            print("✅ 全局IndexTTS系統已清理")
        except Exception as e:
            print(f"⚠️ 清理IndexTTS系統時發生錯誤: {e}")

def play_japanese_tts_system(text):
    """使用pyttsx3直接透過系統播放日文語音"""
    global tts_playback_lock
    
    # 檢查是否已有語音在播放
    if not tts_playback_lock.acquire(blocking=False):
        print("⚠️ 另一個語音正在播放中，跳過此次播放")
        return False
    
    try:
        import pyttsx3
        import threading
        import time
        
        print(f"🎌 準備播放日文語音: {text[:30]}...")
        
        # 使用線程安全的方式播放語音
        def play_speech_thread():
            engine = None
            try:
                # 每次都創建新的引擎實例，避免run loop衝突
                engine = pyttsx3.init()
                if engine is None:
                    print("❌ 無法初始化pyttsx3引擎")
                    return False
                
                # 設置語音屬性
                try:
                    # 設置語速
                    engine.setProperty('rate', 150)  # 稍慢一點的語速
                    engine.setProperty('volume', 0.9)  # 音量
                    
                    # 嘗試設置日文語音
                    voices = engine.getProperty('voices')
                    japanese_voice_found = False
                    
                    if voices:
                        for voice in voices:
                            # 檢查語音名稱中是否包含日文相關關鍵詞
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
                    
                except Exception as setup_error:
                    print(f"⚠️ 語音設置過程中發生錯誤: {setup_error}")
                
                # 清理文本
                from roboticBar_printChunk_QAsplit_indexTTSclone_JP_UI import clean_text_for_tts
                cleaned_text = clean_text_for_tts(text)
                if not cleaned_text:
                    print("❌ 日文文本清理後為空")
                    return False
                
                # 播放語音
                print(f"🔊 開始播放日文語音...")
                engine.say(cleaned_text)
                engine.runAndWait()
                print(f"✅ 日文語音播放完成")
                return True
                
            except Exception as play_error:
                print(f"❌ 播放過程中發生錯誤: {play_error}")
                return False
            finally:
                # 確保引擎資源被正確清理
                if engine is not None:
                    try:
                        # 停止引擎
                        engine.stop()
                        print("🔧 pyttsx3引擎已停止")
                    except Exception as stop_error:
                        print(f"⚠️ 停止引擎時發生錯誤: {stop_error}")
                    
                    try:
                        # 刪除引擎實例
                        del engine
                        print("🔧 pyttsx3引擎實例已清理")
                    except Exception as del_error:
                        print(f"⚠️ 清理引擎實例時發生錯誤: {del_error}")
                
                # 釋放鎖
                tts_playback_lock.release()
                print("🔓 日文TTS鎖已釋放")
        
        # 在新線程中播放，避免阻塞主線程
        play_thread = threading.Thread(target=play_speech_thread, daemon=True)
        play_thread.start()
        
        # 等待一小段時間確保播放開始
        time.sleep(0.5)
        
        return True
        
    except ImportError:
        print("❌ pyttsx3未安裝，無法播放日文語音")
        tts_playback_lock.release()
        return False
    except Exception as e:
        print(f"❌ 日文語音播放時發生錯誤: {e}")
        import traceback
        print(f"詳細錯誤: {traceback.format_exc()}")
        tts_playback_lock.release()
        return False

def play_chinese_english_tts_system(text, language):
    """使用IndexTTS直接透過系統播放中文或英文語音"""
    global tts_playback_lock # 使用同一個鎖來管理所有TTS播放
    
    # 檢查是否已有語音在播放
    if not tts_playback_lock.acquire(blocking=False):
        print("⚠️ 另一個語音正在播放中，跳過此次播放")
        return False
    
    try:
        import threading
        import time
        
        print(f"🔊 準備使用IndexTTS播放 {language} 語音: {text[:30]}...")
        
        # 使用線程安全的方式播放語音
        def play_speech_thread():
            try:
                # 獲取全局IndexTTS系統（只初始化一次）
                tts_system = get_global_tts_system()
                if tts_system is None:
                    print("❌ 無法獲取IndexTTS系統")
                    return False
                
                # 導入所需模組
                from roboticBar_printChunk_QAsplit_indexTTSclone_JP_UI import (
                    clean_text_for_tts, TTS_VOICE_PATH
                )
                import sounddevice as sd
                
                # 清理文本
                cleaned_text = clean_text_for_tts(text)
                if not cleaned_text:
                    print("❌ 文本清理後為空")
                    return False
                
                print(f"🔊 正在生成 {language} 語音...")
                
                # 使用全局IndexTTS系統生成音頻數據
                sampling_rate, wav_data = tts_system.infer(
                    TTS_VOICE_PATH, 
                    cleaned_text, 
                    output_path=None  # 不保存到文件，直接返回音頻數據
                )
                
                if wav_data is None or len(wav_data) == 0:
                    print("❌ IndexTTS返回空音頻數據")
                    return False
                
                print(f"✅ 音頻生成成功: sampling_rate={sampling_rate}, data_length={len(wav_data)}")
                
                # 播放音頻
                print(f"🔊 開始播放 {language} 語音...")
                sd.play(wav_data, sampling_rate)
                sd.wait()  # 等待播放完成
                print(f"✅ {language} 語音播放完成")
                return True
                
            except Exception as play_error:
                print(f"❌ IndexTTS播放過程中發生錯誤: {play_error}")
                import traceback
                print(f"詳細錯誤: {traceback.format_exc()}")
                return False
            finally:
                # 釋放鎖
                tts_playback_lock.release()
                print("🔓 IndexTTS語音鎖已釋放")
        
        # 在新線程中播放，避免阻塞主線程
        play_thread = threading.Thread(target=play_speech_thread, daemon=True)
        play_thread.start()
        
        # 等待一小段時間確保播放開始
        time.sleep(0.5)
        
        return True
        
    except ImportError as import_error:
        print(f"❌ IndexTTS相關模組導入失敗: {import_error}")
        tts_playback_lock.release()
        return False
    except Exception as e:
        print(f"❌ IndexTTS語音播放時發生錯誤: {e}")
        import traceback
        print(f"詳細錯誤: {traceback.format_exc()}")
        tts_playback_lock.release()
        return False

def generate_tts_audio(text, language):
    """生成TTS音頻並直接透過系統播放，返回成功/失敗狀態"""
    if not ENABLE_TTS_OUTPUT:
        print("⚠️ TTS功能已禁用")
        return False
    
    try:
        # 所有語言都直接透過系統播放，不生成音檔
        if language == "ja" and USE_PYTTSX3_FOR_JAPANESE:
            # 日文使用pyttsx3
            print("日文將使用pyttsx3直接透過系統播放語音")
            success = play_japanese_tts_system(text)
            if success:
                print("✅ 日文語音播放成功")
                return True
            else:
                print("❌ 日文語音播放失敗")
                return False
        else:
            # 中文和英文使用IndexTTS系統播放
            print(f"正在使用IndexTTS直接播放 {language} 語音")
            success = play_chinese_english_tts_system(text, language)
            if success:
                print(f"✅ {language} 語音播放成功")
                return True
            else:
                print(f"❌ {language} 語音播放失敗")
                return False
            
    except Exception as e:
        print(f"❌ 生成TTS音頻時發生未預期錯誤: {e}")
        import traceback
        print(f"TTS錯誤詳情: {traceback.format_exc()}")
        return False

def pregenerate_prompt_audio():
    """所有語音都改為系統播放，不需要預生成提示音文件"""
    global PROMPT_AUDIO_FILES
    
    if not ENABLE_PROMPT_AUDIO or not ENABLE_TTS_OUTPUT:
        print("提示音功能已禁用，跳過預生成")
        return
    
    print("🎵 所有語音已改為系統直接播放，不需要預生成提示音文件")
    
    # 清空提示音文件字典，因為不再需要
    PROMPT_AUDIO_FILES = {
        'zh': None,
        'en': None,
        'ja': None
    }
    
    print("✅ 提示音系統已配置為直接播放模式")

def generate_prompt_audio(language="zh"):
    """直接透過系統播放提示音，並等待播放完成"""
    global prompt_playback_lock
    
    if not ENABLE_PROMPT_AUDIO or not ENABLE_TTS_OUTPUT:
        return None
    
    # 使用提示音專用鎖
    with prompt_playback_lock:
        try:
            print(f"🎵 準備播放 {language} 提示音...")
            
            # 提示音文本
            prompt_texts = {
                'zh': "感謝您的提問，我思考一下，請稍後。",
                'en': "Thank you for your question. Let me think about it and get back to you shortly.",
                'ja': "ご質問ありがとうございます。少しお考えください。"
            }
            
            prompt_text = prompt_texts.get(language, prompt_texts['zh'])
            
            # 根據語言選擇播放方式
            if language == "ja" and USE_PYTTSX3_FOR_JAPANESE:
                # 日文使用pyttsx3 - 同步播放
                play_japanese_prompt_sync(prompt_text)
            else:
                # 中文和英文使用IndexTTS - 同步播放
                play_indexTTS_prompt_sync(prompt_text, language)
            
            print(f"✅ {language} 提示音播放完成")
            return None  # 不返回音頻URL，因為是系統播放
            
        except Exception as e:
            print(f"❌ 播放 {language} 提示音時發生錯誤: {e}")
            import traceback
            print(f"詳細錯誤: {traceback.format_exc()}")
            return None

def play_japanese_prompt_sync(text):
    """同步播放日文提示音"""
    try:
        import pyttsx3
        
        engine = pyttsx3.init()
        if engine is None:
            print("❌ 無法初始化pyttsx3引擎")
            return False
        
        try:
            # 設置語音屬性
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            
            # 嘗試設置日文語音
            voices = engine.getProperty('voices')
            for voice in voices:
                voice_name = voice.name.lower()
                if any(keyword in voice_name for keyword in ['japan', 'japanese', 'ja-jp', 'haruka', 'ayumi']):
                    try:
                        engine.setProperty('voice', voice.id)
                        print(f"✅ 設置日文提示音語音: {voice.name}")
                        break
                    except:
                        continue
            
            # 清理文本
            from roboticBar_printChunk_QAsplit_indexTTSclone_JP_UI import clean_text_for_tts
            cleaned_text = clean_text_for_tts(text)
            if not cleaned_text:
                return False
            
            # 同步播放
            print(f"🔊 播放日文提示音...")
            engine.say(cleaned_text)
            engine.runAndWait()
            print(f"✅ 日文提示音播放完成")
            return True
            
        finally:
            try:
                engine.stop()
                del engine
            except:
                pass
                
    except Exception as e:
        print(f"❌ 日文提示音播放錯誤: {e}")
        return False

def play_indexTTS_prompt_sync(text, language):
    """同步播放IndexTTS提示音"""
    try:
        # 獲取全局IndexTTS系統（只初始化一次）
        tts_system = get_global_tts_system()
        if tts_system is None:
            print("❌ 無法獲取IndexTTS系統")
            return False
        
        # 導入所需模組
        from roboticBar_printChunk_QAsplit_indexTTSclone_JP_UI import (
            clean_text_for_tts, TTS_VOICE_PATH
        )
        import sounddevice as sd
        
        # 清理文本
        cleaned_text = clean_text_for_tts(text)
        if not cleaned_text:
            return False
        
        print(f"🔊 正在生成 {language} 提示音...")
        
        # 使用全局IndexTTS系統生成音頻數據
        sampling_rate, wav_data = tts_system.infer(
            TTS_VOICE_PATH, 
            cleaned_text, 
            output_path=None
        )
        
        if wav_data is None or len(wav_data) == 0:
            print("❌ IndexTTS返回空音頻數據")
            return False
        
        print(f"✅ 提示音生成成功，開始播放...")
        
        # 同步播放音頻
        sd.play(wav_data, sampling_rate)
        sd.wait()  # 等待播放完成
        print(f"✅ {language} 提示音播放完成")
        return True
        
    except Exception as e:
        print(f"❌ IndexTTS提示音播放錯誤: {e}")
        import traceback
        print(f"詳細錯誤: {traceback.format_exc()}")
        return False

def initialize_rag_system():
    """初始化RAG系統"""
    global qa_chain, system_ready, initialization_status
    
    try:
        print("🚀 開始初始化RAG系統...")
        socketio.emit('system_status', {'status': 'initializing', 'message': '正在初始化RAG系統...'})
        
        # 導入所需模組
        from roboticBar_printChunk_QAsplit_indexTTSclone_JP_UI import (
            load_documents, split_documents, create_vector_store, 
            initialize_llm, create_qa_chain, map_whisper_language_to_supported,
            PDF_DIRECTORY, VECTORSTORE_DIR, ENABLE_TTS_OUTPUT
        )
        
        initialization_status = "正在載入文檔..."
        socketio.emit('system_status', {'status': 'initializing', 'message': initialization_status})
        
        # 載入文檔
        documents = load_documents(PDF_DIRECTORY)
        if not documents:
            raise Exception("無法載入文檔")
        
        initialization_status = "正在分割文檔..."
        socketio.emit('system_status', {'status': 'initializing', 'message': initialization_status})
        
        # 分割文檔
        texts = split_documents(documents)
        if not texts:
            raise Exception("文檔分割失敗")
        
        initialization_status = "正在建立向量儲存..."
        socketio.emit('system_status', {'status': 'initializing', 'message': initialization_status})
        
        # 建立向量儲存
        vectorstore = create_vector_store(texts, VECTORSTORE_DIR)
        if not vectorstore:
            raise Exception("向量儲存建立失敗")
        
        initialization_status = "正在初始化LLM..."
        socketio.emit('system_status', {'status': 'initializing', 'message': initialization_status})
        
        # 初始化LLM
        llm = initialize_llm()
        if not llm:
            raise Exception("LLM初始化失敗")
        
        initialization_status = "正在建立QA鏈..."
        socketio.emit('system_status', {'status': 'initializing', 'message': initialization_status})
        
        # 建立QA鏈
        qa_chain = create_qa_chain(llm, vectorstore, texts)
        if not qa_chain:
            raise Exception("QA鏈建立失敗")
        
        # 預初始化IndexTTS系統（如果啟用TTS功能）
        if ENABLE_TTS_OUTPUT:
            initialization_status = "正在初始化TTS語音系統..."
            socketio.emit('system_status', {'status': 'initializing', 'message': initialization_status})
            
            print("🔧 預初始化IndexTTS系統...")
            tts_system = get_global_tts_system()
            if tts_system is not None:
                print("✅ IndexTTS系統預初始化成功")
            else:
                print("⚠️ IndexTTS系統預初始化失敗，但不影響文字功能")
        
        # 預生成提示音（現在不需要了，因為改為系統播放）
        pregenerate_prompt_audio()
        
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
        
        # 播放中文提示音
        if ENABLE_PROMPT_AUDIO:
            try:
                generate_prompt_audio("zh")
            except Exception as prompt_error:
                print(f"⚠️ 提示音播放失敗: {prompt_error}")
        
        # 處理問題
        result = qa_chain.invoke({"query": question, "language": "zh"})
        answer = result.get('result', '抱歉，無法生成答案。').strip()
        
        print(f"✅ 生成回答: {answer[:100]}...")
        
        # 生成語音回答（系統播放）
        tts_success = False
        if ENABLE_TTS_OUTPUT:
            try:
                print(f"🔊 開始生成 zh 語音回答...")
                tts_success = generate_tts_audio(answer, "zh")
                if tts_success:
                    print("✅ 語音回答播放成功")
                else:
                    print("❌ 語音回答播放失敗")
            except Exception as tts_error:
                print(f"❌ TTS處理時發生異常: {tts_error}")
                import traceback
                print(f"TTS異常詳情: {traceback.format_exc()}")
        else:
            print("⚠️ TTS功能已禁用，跳過語音生成")
        
        return jsonify({
            'success': True,
            'question': question,
            'answer': answer,
            'language': 'zh',  # 文字問題默認為中文
            'audio_url': None,  # 系統播放，不返回音頻URL
            'tts_success': tts_success  # 添加TTS成功狀態
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
        
        # 創建更安全的臨時文件
        import tempfile
        temp_fd, temp_file_path = tempfile.mkstemp(suffix='.wav', prefix='voice_')
        
        try:
            # 保存音頻文件
            with os.fdopen(temp_fd, 'wb') as temp_file:
                audio_file.save(temp_file)
            
            print(f"🎤 收到語音文件: {temp_file_path}")
            
            # 驗證文件是否存在且有內容
            if not os.path.exists(temp_file_path):
                raise Exception(f"臨時文件創建失敗: {temp_file_path}")
            
            file_size = os.path.getsize(temp_file_path)
            if file_size == 0:
                raise Exception("音頻文件為空")
            
            print(f"音頻文件大小: {file_size} bytes")
            
            # 語音轉文字
            import roboticBar_printChunk_QAsplit_indexTTSclone_JP_UI as main_module
            original_filename = main_module.WAVE_OUTPUT_FILENAME
            main_module.WAVE_OUTPUT_FILENAME = temp_file_path
            
            try:
                # 確保Whisper模型已載入
                if not load_whisper_model():
                    raise Exception("Whisper模型載入失敗")
                
                question, detected_lang = speech_to_text()
                print(f"語音識別完成: question='{question}', language='{detected_lang}'")
                
            finally:
                main_module.WAVE_OUTPUT_FILENAME = original_filename
            
            if not question or len(question.strip()) == 0:
                return jsonify({'success': False, 'message': '無法識別語音內容，請重新錄音'})
            
            print(f"🔍 語音識別結果: {question}")
            
            # 映射語言
            tts_language = map_whisper_language_to_supported(detected_lang)
            print(f"映射後的TTS語言: {tts_language}")
            
            # 播放對應語言的提示音
            if ENABLE_PROMPT_AUDIO:
                try:
                    generate_prompt_audio(tts_language)
                except Exception as prompt_error:
                    print(f"⚠️ 提示音播放失敗: {prompt_error}")
            
            # 處理問題
            result = qa_chain.invoke({"query": question, "language": tts_language})
            answer = result.get('result', '抱歉，無法生成答案。').strip()
            
            print(f"✅ 生成回答: {answer[:100]}...")
            
            # 生成語音回答
            tts_success = False
            if ENABLE_TTS_OUTPUT:
                try:
                    print(f"🔊 開始生成 {tts_language} 語音回答...")
                    tts_success = generate_tts_audio(answer, tts_language)
                    if tts_success:
                        print(f"✅ {tts_language} 語音回答播放成功")
                    else:
                        print(f"❌ {tts_language} 語音回答播放失敗")
                except Exception as tts_error:
                    print(f"❌ TTS處理時發生異常: {tts_error}")
                    import traceback
                    print(f"TTS異常詳情: {traceback.format_exc()}")
            else:
                print("⚠️ TTS功能已禁用，跳過語音生成")
            
            return jsonify({
                'success': True,
                'question': question,
                'answer': answer,
                'language': tts_language,
                'audio_url': None,  # 系統播放，不返回音頻URL
                'tts_success': tts_success  # 添加TTS成功狀態
            })
            
        except Exception as inner_error:
            print(f"處理音頻文件時發生內部錯誤: {inner_error}")
            raise inner_error
            
    except Exception as e:
        error_msg = f'處理語音時發生錯誤: {str(e)}'
        print(f"❌ {error_msg}")
        import traceback
        print(f"詳細錯誤信息: {traceback.format_exc()}")
        return jsonify({'success': False, 'message': error_msg})
    
    finally:
        # 清理臨時文件
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
    print("🌐 RAG聊天機器人 Web UI - 語音版")
    print("="*60)
    print("正在啟動服務器...")
    print("請在瀏覽器中打開: http://localhost:5002")
    print()
    print("功能特色:")
    print("📝 文字問答 - 直接輸入文字問題")
    print("🎤 語音問答 - 點擊按鈕錄音提問")
    print("🔊 語音回答 - 自動播放回答音頻")
    print("💬 聊天記錄 - 保存完整對話歷史")
    print("="*60)
    
    # 創建static目錄（用於存儲臨時音頻文件）
    os.makedirs('static', exist_ok=True)
    
    # 在後台線程中初始化RAG系統
    init_thread = threading.Thread(target=initialize_rag_system)
    init_thread.daemon = True
    init_thread.start()
    
    # 啟動Flask應用
    socketio.run(app, host='0.0.0.0', port=5002, debug=False) 