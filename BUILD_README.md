# RAG系統打包指南

這個指南將説明如何將RAG系統打包成可執行文件，方便分發和部署。

## 📋 目錄結構

確保您的項目包含以下文件和目錄：

```
RAG_Project/
├── qa_RAGsystem_printChunk_QAsplit_coquiTTSclone.py  # 主程序
├── requirements.txt              # 依賴列表
├── build_config.spec            # PyInstaller配置
├── build_executable.py          # 自動打包腳本
├── BUILD_README.md              # 本說明文檔
├── .env                         # 環境變量（可選）
├── technical_file/              # PDF文件目錄
│   └── PDF/
│       └── *.pdf
├── voice/                       # 語音文件目錄
│   └── speaker_oreo1min.wav
└── 其他必要文件...
```

## 🔧 系統要求

### 基本需求
- **Python**: 3.8 或更高版本
- **操作系統**: Windows 10/11, macOS 10.14+, Linux (Ubuntu 18.04+)
- **RAM**: 至少 8GB (推薦 16GB)
- **磁碟空間**: 至少 5GB 可用空間
- **網路**: 穩定的網路連接（用於下載模型）

### 依賴服務
- **Ollama**: 必須正在運行，且已下載相應的LLM模型
- **音頻設備**: 如果使用語音功能，需要麥克風和揚聲器

## 🚀 快速開始

### 方法1：使用自動打包腳本（推薦）

1. 打開命令提示符或終端
2. 切換到項目目錄
3. 運行自動打包腳本：

```bash
python build_executable.py
```

腳本將自動完成以下步驟：
- 檢查Python版本和系統需求
- 安裝所有依賴
- 檢查必要文件
- 清理舊的建置文件
- 測試主程序
- 執行打包
- 完成打包並提供使用說明

### 方法2：手動打包

如果自動腳本遇到問題，可以手動執行以下步驟：

1. **安裝依賴**：
```bash
pip install -r requirements.txt
```

2. **清理舊文件**：
```bash
rmdir /s build dist __pycache__ 2>nul
```

3. **執行打包**：
```bash
python -m PyInstaller build_config.spec
```

## 📦 打包配置說明

### PyInstaller配置 (build_config.spec)

- **單文件打包**: 生成單個可執行文件
- **包含數據**: 自動包含PDF文件、語音文件等
- **隱藏導入**: 包含所有必要的模組
- **UPX壓縮**: 減少文件大小
- **控制台模式**: 保持命令行介面

### 主要特性

- ✅ 包含所有語音功能 (Whisper + TTS)
- ✅ 包含向量數據庫功能
- ✅ 包含混合檢索功能
- ✅ 自動處理依賴
- ✅ 跨平台相容

## 🛠️ 常見問題與解決方案

### 1. PyAudio 安裝失敗 (Windows)

**問題**: `pip install pyaudio` 失敗

**解決方案**:
```bash
# 方法1：使用預編譯的wheel文件
pip install pipwin
pipwin install pyaudio

# 方法2：從https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
# 下載對應的.whl文件然後安裝
pip install PyAudio‑0.2.11‑cp39‑cp39‑win_amd64.whl
```

### 2. PyTorch 安裝問題

**問題**: PyTorch 安裝失敗或版本不匹配

**解決方案**:
```bash
# 卸載舊版本
pip uninstall torch torchaudio

# 重新安裝（CPU版本）
pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# 如果有GPU
pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

### 3. 打包文件過大

**問題**: 生成的可執行文件超過 1GB

**解決方案**:
- 在 `build_config.spec` 中添加更多排除項
- 使用 `--exclude-module` 排除不需要的模組
- 考慮使用目錄分發而非單文件打包

### 4. 運行時模型下載失敗

**問題**: 首次運行時 Whisper 或 TTS 模型下載失敗

**解決方案**:
```bash
# 預先下載 Whisper 模型
python -c "import whisper; whisper.load_model('medium')"

# 預先下載 TTS 模型
python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"
```

### 5. 語音功能不工作

**問題**: 麥克風或揚聲器不工作

**解決方案**:
- 檢查音頻設備驅動程式
- 確保 Windows 音頻服務正在運行
- 嘗試以管理員身份運行
- 檢查防火牆設置

### 6. Ollama 連接失敗

**問題**: 無法連接到 Ollama 服務

**解決方案**:
- 確保 Ollama 服務正在運行 (`ollama serve`)
- 檢查端口 11434 是否開放
- 確認模型已下載 (`ollama list`)
- 檢查防火牆設置

## 📊 性能優化建議

### 減少文件大小

1. **排除不需要的模組**:
```python
# 在 build_config.spec 中的 excludes 列表添加：
excludes = [
    'matplotlib', 'tkinter', 'PIL', 'cv2', 'sklearn', 
    'scipy', 'pandas', 'jupyter', 'notebook', 'IPython',
    'debugpy', 'IPython', 'jedi'
]
```

2. **使用目錄分發**:
```python
# 在 build_config.spec 最後取消註釋：
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='RAG_System_Dist'
)
```

### 加速啟動時間

1. **預編譯模組**:
```bash
python -m compileall .
```

2. **使用 SSD 存儲**:
- 將可執行文件存放在 SSD 上
- 確保有足夠的 RAM

## 🔄 更新和維護

### 更新依賴

```bash
# 更新所有依賴到最新版本
pip install -U -r requirements.txt

# 重新打包
python build_executable.py
```

### 版本控制

建議在每次打包時：
1. 更新版本號
2. 記錄更改內容
3. 測試所有功能
4. 建立版本標籤

## 📝 部署檢查清單

在分發可執行文件前，請確保：

- [ ] 所有功能正常工作
- [ ] 語音輸入/輸出正常
- [ ] 向量數據庫建立成功
- [ ] 問答功能正常
- [ ] 文件大小合理
- [ ] 包含所有必要的數據文件
- [ ] 提供使用說明文檔

## 📞 技術支援

如果遇到問題，請檢查：

1. **日誌文件**: 查看控制台輸出
2. **系統需求**: 確認滿足所有要求
3. **依賴服務**: 確認 Ollama 等服務正常
4. **防火牆**: 檢查網路連接
5. **權限**: 確保有足夠的讀寫權限

## 💡 進階技巧

### 自定義圖標

```python
# 在 build_config.spec 中修改：
exe = EXE(
    # ...
    icon='path/to/your/icon.ico',  # 添加自定義圖標
    # ...
)
```

### 添加版本信息

```python
# 在 build_config.spec 中添加：
version_info = {
    'version': '1.0.0',
    'company_name': 'Your Company',
    'file_description': 'RAG System',
    'product_name': 'RAG System'
}
```

### 多語言支持

確保在打包時包含所有語言資源：

```python
# 在 datas 中添加：
('locale/', 'locale/'),  # 語言文件
('i18n/', 'i18n/'),      # 國際化文件
```

---

**祝您打包成功！** 🎉

如有任何問題，請參考本文檔或聯繫技術支援。 