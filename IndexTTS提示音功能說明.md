# IndexTTS 提示音功能說明

## 功能概述

為 `qa_RAGsystem_printChunk_QAsplit_indexTTSclone.py` 添加了提示音功能，在用戶提問後、系統開始處理問題時播放友好的提示音，提升用戶體驗。

## 新增功能

### 1. 配置選項
```python
# --- 提示音功能設定 ---
ENABLE_PROMPT_AUDIO = True  # 是否啟用提示音功能（在用戶提問後播放友好提示）
```

### 2. 提示音播放函數
- `play_prompt_audio(detected_language="zh")`: 根據檢測到的語言播放相應的提示音
  - 中文提示音：「感謝您的提問，我思考一下，請稍後。」
  - 英文提示音：「Thank you for your question. Let me think about it and get back to you shortly.」
  - 使用後台線程播放，不阻塞主程序

- `play_recording_prompt(detected_language="zh")`: 播放錄音開始提示音
  - 中文錄音提示音：「請說。」
  - 英文錄音提示音：「Please speak now.」
  - 在用戶按下Enter鍵開始錄音時播放

### 3. 智能語言檢測
- 語音模式：使用 Whisper 檢測到的語言
- 文字模式：默認使用中文提示音
- 支援中英文雙語提示音

## 工作流程

1. **用戶按Enter** → 播放錄音提示音「請說」
2. **用戶提問** → 語音識別或文字輸入
3. **語言檢測** → 確定使用中文或英文提示音
4. **播放提示音** → 後台線程播放「感謝您的提問，我思考一下，請稍後。」或「Thank you for your question. Let me think about it and get back to you shortly.」
5. **問題處理** → 系統開始檢索和生成答案
6. **答案播放** → TTS 播放完整答案

## 觸發條件

提示音會在以下條件同時滿足時播放：
- `ENABLE_TTS_OUTPUT = True` （TTS功能已啟用）
- `ENABLE_PROMPT_AUDIO = True` （提示音功能已啟用）
- `debug_mode = False` （非調試模式）

## 技術實現

### 多線程播放
```python
def play_prompt_in_background():
    # 載入TTS系統
    # 生成提示音
    # 播放音頻
    
# 啟動後台線程
prompt_thread = threading.Thread(target=play_prompt_in_background)
prompt_thread.daemon = True
prompt_thread.start()
```

### 語言適配
```python
if detected_language == "en":
    prompt_text = "Let me think about that..."
    prompt_lang = "English"
else:
    prompt_text = "讓我想想..."
    prompt_lang = "中文"
```

## 系統集成

### 啟動信息顯示
```
TTS語音輸出功能: 已啟用 (語音: Dr_Lee.wav)
    🎵 提示音功能已啟用（問題處理時播放友好提示）
    🎤 錄音提示音已啟用（按Enter開始錄音時播放「請說」）
```

### 運行時流程
1. 用戶按Enter鍵，系統播放「請說」提示音
2. 用戶開始語音輸入，系統顯示「正在錄音...」
3. 用戶再次按Enter停止錄音，進行語音識別
4. 系統顯示「正在處理您的問題...」
5. 同時在後台播放思考提示音
6. 繼續進行問題檢索和答案生成
7. 最後播放完整答案

## 測試腳本

提供了 `test_indextts_prompt_audio.py` 測試腳本：
- 檢查必要文件是否存在
- 測試中文和英文提示音生成
- 測試多線程播放功能
- 驗證音頻品質和播放效果

## 使用方法

### 啟用/禁用提示音
```python
# 在文件頂部修改配置
ENABLE_PROMPT_AUDIO = True   # 啟用提示音
ENABLE_PROMPT_AUDIO = False  # 禁用提示音
```

### 運行測試
```bash
python test_indextts_prompt_audio.py
```

## 優勢特點

1. **非阻塞播放**：使用後台線程，不影響主程序響應速度
2. **智能語言檢測**：根據輸入語言自動選擇提示音語言
3. **友好用戶體驗**：在系統思考時提供音頻反饋
4. **可配置控制**：可通過配置開關輕鬆啟用/禁用
5. **調試模式友好**：調試模式下自動禁用，避免干擾

## 注意事項

1. **依賴要求**：需要 IndexTTS 系統和相關音頻文件
2. **音響設備**：需要正常工作的音響輸出設備
3. **性能影響**：提示音生成會消耗少量計算資源
4. **線程安全**：使用守護線程，程序退出時會自動清理

## 故障排除

### 常見問題
1. **提示音不播放**：檢查 `ENABLE_PROMPT_AUDIO` 和 `ENABLE_TTS_OUTPUT` 設定
2. **音頻文件錯誤**：確認 `TTS_VOICE_PATH` 指向的語音參考文件存在
3. **TTS初始化失敗**：檢查 IndexTTS 模型和配置文件
4. **音響無聲音**：檢查系統音量和音響設備

### 調試方法
```bash
# 運行測試腳本
python test_indextts_prompt_audio.py

# 檢查配置
print(f"提示音功能: {ENABLE_PROMPT_AUDIO}")
print(f"TTS功能: {ENABLE_TTS_OUTPUT}")
print(f"語音文件: {os.path.exists(TTS_VOICE_PATH)}")
```

## 未來改進

1. **自定義提示音文本**：允許用戶自定義提示音內容
2. **音效選擇**：提供多種提示音風格選擇
3. **音量控制**：獨立的提示音音量設定
4. **更多語言支持**：支援更多語言的提示音
5. **情境感知**：根據問題類型選擇不同提示音 