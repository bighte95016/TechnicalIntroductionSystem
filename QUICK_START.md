# RAG系統打包 - 快速開始指南

## 🚀 三步驟快速打包

### 步驟1：準備環境
```bash
# 確保Python版本 >= 3.8
python --version

# 確保有足夠磁碟空間 (至少5GB)
```

### 步驟2：選擇打包方式

#### 方式A：一鍵打包（推薦）

**Windows用戶：**
```
雙擊運行：build.bat
```

**Linux/macOS用戶：**
```bash
./build.sh
```

#### 方式B：手動運行
```bash
python build_executable.py
```

### 步驟3：獲取結果
打包完成後，在 `dist/` 目錄中找到可執行文件：
- Windows: `RAG_System.exe`
- Linux/macOS: `RAG_System`

## 📋 檢查清單

### 打包前確認：
- [ ] Python 3.8+ 已安裝
- [ ] 有 5GB+ 可用空間
- [ ] 以下文件/目錄存在：
  - [ ] `qa_RAGsystem_printChunk_QAsplit_coquiTTSclone.py`
  - [ ] `technical_file/PDF/` (包含PDF文件)
  - [ ] `voice/speaker_oreo1min.wav`
  - [ ] `requirements.txt`
  - [ ] `build_config.spec`
  - [ ] `build_executable.py`

### 運行前確認：
- [ ] Ollama 服務正在運行
- [ ] 相應的 LLM 模型已下載
- [ ] 音頻設備工作正常（如使用語音功能）

## ⚠️ 常見問題

### 1. 依賴安裝失敗
```bash
# 如果遇到PyAudio安裝問題（Windows）
pip install pipwin
pipwin install pyaudio
```

### 2. 文件太大
- 預期大小：500MB - 2GB
- 如果超過2GB，檢查是否包含了不必要的文件

### 3. 運行時錯誤
- 確保 Ollama 服務正在運行
- 檢查防火牆設置
- 確認模型已下載

## 💡 優化建議

### 減少文件大小
1. 移除不需要的PDF文件
2. 使用較小的語音模型
3. 在 `build_config.spec` 中排除更多模組

### 提高性能
1. 使用SSD存儲
2. 確保至少8GB RAM
3. 預先下載所有模型

## 📞 需要幫助？

1. 查看 `BUILD_README.md` 了解詳細說明
2. 檢查控制台輸出的錯誤信息
3. 確認所有依賴服務正常運行

## 🎯 預期時間

- **準備階段**：5-10分鐘
- **依賴安裝**：10-20分鐘
- **打包過程**：10-30分鐘
- **總時間**：25-60分鐘

---

**成功率：** 大多數用戶可以在第一次嘗試時成功打包！ 