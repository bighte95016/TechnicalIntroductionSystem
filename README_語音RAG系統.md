# 語音 RAG 系統 - 完整語音交互版本

## 🎯 項目概述

這是一個集成了完整語音交互功能的檢索增強生成（RAG）系統，支持語音輸入問題和語音輸出答案，實現了真正的語音對話體驗。

### 🌟 主要特性

- **🎙️ 語音輸入**：使用 OpenAI Whisper 進行高精度語音識別
- **🔊 語音輸出**：使用 CoquiTTS 進行高品質語音合成
- **🎵 智能提示音**：在用戶提問後播放友好提示音，提升交互體驗
- **🌐 多語言支持**：自動檢測語言並相應回答（中文/英文）
- **📚 智能檢索**：支持混合檢索（Dense + BM25）
- **🧠 問答分離**：優化的文檔處理和檢索策略
- **⚡ 性能優化**：模型預熱、音頻緩存、GPU 加速

## 📁 項目結構

```
TechnicalIntroductionSystem/
├── qa_RAGsystem_printChunk_QAsplit_coquiTTS.py  # 主程序文件
├── requirements_voice.txt                       # 語音功能依賴
├── test_voice_features.py                      # 語音功能測試腳本
├── start_voice_rag.bat                         # Windows 啟動腳本
├── start_voice_rag.sh                          # Linux/macOS 啟動腳本
├── 語音功能使用說明.md                           # 詳細使用說明
├── README_語音RAG系統.md                        # 本文檔
├── technical_file/PDF/                         # PDF 文檔目錄
├── chroma_db/                                  # 向量數據庫（自動生成）
└── bm25_cache/                                 # BM25 緩存（自動生成）
```

## 🚀 快速開始

### 1. 環境要求

- **Python**: 3.8 或更高版本
- **內存**: 最少 4GB，推薦 8GB+
- **磁盤空間**: 至少 3GB（用於模型文件）
- **音頻設備**: 麥克風和音響/耳機
- **可選**: NVIDIA GPU（用於加速）

### 2. 安裝依賴

```bash
# 安裝所有語音功能依賴
pip install -r requirements_voice.txt
```

### 3. 準備文檔

將您的 PDF 文檔放在 `technical_file/PDF/` 目錄中。

### 4. 運行測試（推薦）

```bash
# 測試語音功能是否正常
python test_voice_features.py
```

### 5. 啟動系統

#### 方法一：使用啟動腳本（推薦）

**Windows:**
```cmd
start_voice_rag.bat
```

**Linux/macOS:**
```bash
chmod +x start_voice_rag.sh
./start_voice_rag.sh
```

#### 方法二：直接運行

```bash
python qa_RAGsystem_printChunk_QAsplit_coquiTTS.py
```

## 🎮 使用方法

### 語音交互模式（默認）

1. **開始對話**
   - 🎤 按 Enter 開始錄音
   - 🗣️ 清晰地說出您的問題
   - ⏹️ 再按 Enter 停止錄音

2. **系統響應**
   - 🎵 播放友好提示音（表示正在思考）
   - 🤖 自動識別語音內容
   - 📝 顯示識別到的問題
   - 💭 檢索相關文檔並生成答案
   - 🔊 自動播放語音答案

### 特殊指令

- `text` 或 `t`：切換到文字輸入模式
- `voice` 或 `v`：切換回語音輸入模式
- `debug`：切換調試模式（顯示檢索詳情）
- `help`：查看系統優化建議
- `quit` 或 `exit`：退出程序

### 語音退出命令

在語音模式下，您也可以說出以下命令來退出：
- 中文：「退出」、「結束」
- 英文：「exit」、「quit」

## ⚙️ 配置選項

### 語音輸入配置

```python
ENABLE_VOICE_INPUT = True          # 啟用語音輸入
WHISPER_MODEL_SIZE = "medium"      # 模型大小: tiny/base/small/medium/large
SPEECH_LANGUAGE = "auto"           # 語言檢測: auto/zh/en
```

### 語音輸出配置

```python
ENABLE_VOICE_OUTPUT = True         # 啟用語音輸出
ENABLE_PROMPT_AUDIO = True         # 啟用提示音功能
COQUI_TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
TTS_ENABLE_GPU = True              # GPU 加速
TTS_ENABLE_CACHE = True            # 音頻緩存
FIXED_ZH_SPEAKER = "Tammie Ema"    # 中文說話人
FIXED_EN_SPEAKER = "Tammie Ema"    # 英文說話人
```

### 檢索配置

```python
USE_HYBRID_RETRIEVAL = True        # 混合檢索模式
DENSE_WEIGHT = 0.5                 # Dense 檢索權重
SPARSE_WEIGHT = 0.5                # BM25 檢索權重
RETRIEVAL_K = 5                    # 檢索數量
SIMILARITY_THRESHOLD = 0.05        # 相似度閾值
```

## 🔧 故障排除

### 常見問題

1. **語音識別失敗**
   - 檢查麥克風權限和設備
   - 確保環境相對安靜
   - 嘗試更大的 Whisper 模型

2. **語音合成失敗**
   - 檢查網絡連接（首次下載模型）
   - 確保磁盤空間充足（約 2GB）
   - 檢查音頻播放設備

3. **依賴安裝問題**
   - Windows: 可能需要 Visual C++ Redistributable
   - Linux: 可能需要 `sudo apt-get install portaudio19-dev`
   - macOS: 可能需要 `brew install portaudio`

### 性能優化建議

1. **硬件優化**
   - 使用 NVIDIA GPU 加速
   - 使用 SSD 存儲
   - 確保充足內存

2. **模型選擇**
   - 根據精度需求選擇 Whisper 模型大小
   - 考慮使用較小的 TTS 模型

3. **系統調優**
   - 調整相似度閾值和檢索數量
   - 啟用音頻緩存
   - 優化文檔分割策略

## 📊 技術細節

### 語音識別技術棧

- **模型**: OpenAI Whisper
- **支持語言**: 多語言自動檢測
- **音頻格式**: 16kHz, 16-bit, 單聲道
- **加速**: CUDA GPU 支持

### 語音合成技術棧

- **模型**: CoquiTTS XTTS v2
- **音頻品質**: 22kHz 高品質輸出
- **說話人**: 固定說話人配置
- **緩存**: 智能音頻緩存機制

### RAG 系統架構

- **文檔處理**: 問答分離策略
- **向量存儲**: ChromaDB
- **檢索方式**: 混合檢索（Dense + BM25）
- **LLM**: 支持 Ollama 和 OpenAI

## 📈 更新日誌

### v2.0 - 語音交互版本
- ✅ 完整語音輸入輸出功能
- ✅ 多語言自動檢測
- ✅ CoquiTTS 高品質語音合成
- ✅ 智能緩存和性能優化
- ✅ 友好的用戶界面

### v1.0 - 基礎 RAG 系統
- ✅ PDF 文檔處理
- ✅ 向量檢索
- ✅ 問答生成

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request 來改進這個項目！

## 📄 許可證

本項目採用 MIT 許可證。

## 💡 提示和技巧

1. **首次使用**：模型下載需要時間，請耐心等待
2. **語音品質**：在安靜環境中使用以獲得最佳識別效果
3. **模型選擇**：可以根據需要調整 Whisper 和 TTS 模型
4. **性能調優**：查看調試模式了解系統檢索情況
5. **文檔準備**：確保 PDF 文檔格式清晰，便於處理

---

🎉 **享受您的語音 RAG 系統體驗！**

如有問題，請參考 `語音功能使用說明.md` 獲取更詳細的說明。 