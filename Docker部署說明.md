# RAG 系統 Docker 部署說明

## 概述

本文檔說明如何使用 Docker 在另一台 Windows 電腦上部署 RAG (Retrieval-Augmented Generation) 語音問答系統。

## 系統需求

### 硬體需求
- **CPU**: 最少 4 核心，建議 8 核心或以上
- **記憶體**: 最少 8GB RAM，建議 16GB 或以上
- **存儲空間**: 最少 20GB 可用空間，建議 50GB 或以上
- **網路**: 穩定的網路連接（首次下載模型需要）
- **音頻設備**: 支援語音輸入/輸出（可選）

### 軟體需求
- **作業系統**: Windows 10/11 (64位)
- **Docker Desktop**: 最新版本
- **Git**: 用於克隆專案（可選）

## 部署步驟

### 1. 安裝必要軟體

#### 1.1 安裝 Docker Desktop
1. 前往 [Docker Desktop 官網](https://www.docker.com/products/docker-desktop/)
2. 下載並安裝 Docker Desktop for Windows
3. 安裝完成後重啟電腦
4. 啟動 Docker Desktop 並等待啟動完成

#### 1.2 驗證安裝
打開命令提示字元或 PowerShell，執行：
```bash
docker --version
docker compose version
```

### 2. 準備專案文件

#### 2.1 複製專案文件
將以下檔案複製到目標電腦的一個資料夾中：
- `roboticBar_printChunk_QAsplit_indexTTSclone_JP.py` (主程式)
- `Dockerfile`
- `docker-compose.yml`
- `requirements.txt`
- `.dockerignore`
- `start.bat` (Windows 啟動腳本)
- `start.sh` (Linux/Mac 啟動腳本)

#### 2.2 準備數據目錄
確保以下目錄結構存在（會在首次執行時自動創建）：
```
project-folder/
├── bar_file/
│   └── PDF/          # 放置 PDF 文件
├── voice/            # 語音參考文件
├── index_tts/        # TTS 模型文件
├── chroma_db/        # 向量資料庫（自動生成）
├── bm25_cache/       # BM25 快取（自動生成）
└── .env              # 環境變數文件（自動生成）
```

### 3. 啟動系統

#### 3.1 使用自動化腳本（推薦）
1. 雙擊執行 `start.bat`
2. 腳本會自動：
   - 檢查 Docker 安裝
   - 創建必要目錄
   - 啟動 Docker 容器
   - 下載必要的 AI 模型
   - 顯示系統狀態

#### 3.2 手動啟動
如果自動化腳本失敗，可以手動執行：

1. 打開命令提示字元，切換到專案目錄
2. 啟動服務：
   ```bash
   docker compose up -d
   ```
3. 下載模型：
   ```bash
   docker exec ollama ollama pull mxbai-embed-large
   docker exec ollama ollama pull gemma2:9b
   ```

### 4. 配置系統

#### 4.1 添加 PDF 文件
將您的 PDF 知識庫文件放置在 `bar_file/PDF/` 目錄中。

#### 4.2 配置 TTS（語音合成）
如果需要語音輸出功能：
1. 將 TTS 模型文件放置在 `index_tts/` 目錄中
2. 將語音參考文件放置在 `voice/` 目錄中

#### 4.3 環境變數配置
編輯 `.env` 文件（首次執行時自動創建）：
```env
# OpenAI API Key (如果使用 OpenAI)
OPENAI_API_KEY=your_openai_api_key_here

# 其他環境變數
ANONYMIZED_TELEMETRY=False
```

### 5. 使用系統

#### 5.1 啟動應用程序
```bash
docker compose exec rag-app python roboticBar_printChunk_QAsplit_indexTTSclone_JP.py
```

#### 5.2 系統功能
- **語音輸入**: 按 Enter 開始錄音
- **文字輸入**: 輸入 'text' 或 't' 切換到文字模式
- **調試模式**: 輸入 'debug' 查看檢索詳情
- **退出系統**: 輸入 'quit' 或 'exit'

## 常見問題與解決方案

### 問題1: Docker 啟動失敗
**症狀**: Docker Desktop 無法啟動或顯示錯誤
**解決方案**:
1. 確保已啟用 Windows 虛擬化功能
2. 重啟 Docker Desktop
3. 檢查 Windows 更新

### 問題2: 音頻功能不可用
**症狀**: 語音輸入/輸出功能無法使用
**解決方案**:
1. 在 Windows 上，音頻功能在 Docker 容器中有限制
2. 可以關閉語音功能，僅使用文字模式
3. 考慮在代碼中設置 `ENABLE_VOICE_INPUT = False`

### 問題3: 模型下載失敗
**症狀**: Ollama 模型下載中斷或失敗
**解決方案**:
1. 檢查網路連接
2. 手動重新下載：
   ```bash
   docker exec ollama ollama pull mxbai-embed-large
   docker exec ollama ollama pull gemma2:9b
   ```

### 問題4: 記憶體不足
**症狀**: 系統運行緩慢或容器崩潰
**解決方案**:
1. 增加 Docker Desktop 的記憶體限制
2. 使用較小的模型（如 qwen2.5:3b）
3. 關閉其他不必要的應用程序

### 問題5: 埠號衝突
**症狀**: 11434 埠號已被占用
**解決方案**:
1. 修改 `docker-compose.yml` 中的埠號對應
2. 或停止占用該埠號的其他服務

## 系統管理命令

### 查看系統狀態
```bash
docker compose ps
```

### 查看日誌
```bash
docker compose logs -f rag-app
docker compose logs -f ollama
```

### 停止系統
```bash
docker compose down
```

### 更新系統
```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```

### 清理系統
```bash
docker compose down -v
docker system prune -f
```

## 效能調優

### 1. GPU 加速（如果可用）
如果您的電腦有 NVIDIA GPU：
1. 安裝 NVIDIA Docker 支援
2. 取消 `docker-compose.yml` 中 GPU 配置的註釋

### 2. 模型選擇
根據您的硬體選擇合適的模型：
- **高配置**: gemma2:9b 或更大
- **中配置**: qwen2.5:7b
- **低配置**: qwen2.5:3b

### 3. 記憶體調優
在 Docker Desktop 設定中調整記憶體分配：
- 最少分配 6GB
- 建議分配 8GB 或以上

## 故障排除

### 日誌檢查
查看詳細錯誤日誌：
```bash
docker compose logs rag-app
```

### 容器狀態檢查
```bash
docker compose ps -a
```

### 重啟特定服務
```bash
docker compose restart rag-app
docker compose restart ollama
```

## 安全考慮

1. **網路安全**: 系統默認只綁定到本地主機
2. **數據隱私**: 所有數據處理都在本地進行
3. **文件權限**: 確保敏感文件有適當的權限設置

## 支援與維護

- 定期更新 Docker 映像檔
- 監控系統資源使用情況
- 備份重要的配置文件和數據
- 保持 Docker Desktop 最新版本

## 總結

通過以上步驟，您應該能夠在另一台 Windows 電腦上成功部署 RAG 系統。如果遇到問題，請參考常見問題部分或檢查系統日誌以獲取更多資訊。 