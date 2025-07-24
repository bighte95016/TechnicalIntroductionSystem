#!/bin/bash

echo "正在啟動 RAG 系統..."

# 檢查 Docker 和 Docker Compose 是否安裝
if ! command -v docker &> /dev/null; then
    echo "錯誤：Docker 未安裝。請先安裝 Docker。"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "錯誤：Docker Compose 未安裝。請先安裝 Docker Compose。"
    exit 1
fi

# 檢查必要的目錄和文件
echo "檢查必要的目錄和文件..."
mkdir -p bar_file/PDF
mkdir -p voice
mkdir -p index_tts
mkdir -p chroma_db
mkdir -p bm25_cache

# 檢查 .env 文件
if [ ! -f .env ]; then
    echo "創建預設 .env 文件..."
    cat > .env << EOL
# OpenAI API Key (如果使用 OpenAI)
OPENAI_API_KEY=your_openai_api_key_here

# 其他環境變數
ANONYMIZED_TELEMETRY=False
EOL
    echo "已創建 .env 文件，請根據需要修改其中的設定。"
fi

# 啟動服務
echo "啟動 Docker Compose 服務..."
if command -v docker-compose &> /dev/null; then
    docker-compose up -d
else
    docker compose up -d
fi

# 等待 Ollama 服務啟動
echo "等待 Ollama 服務啟動..."
sleep 10

# 檢查 Ollama 是否運行
if ! curl -s http://localhost:11434/api/version > /dev/null; then
    echo "錯誤：Ollama 服務未正常啟動。"
    exit 1
fi

echo "Ollama 服務已啟動。"

# 下載必要的模型
echo "檢查並下載必要的 Ollama 模型..."

# 下載嵌入模型
echo "下載嵌入模型 mxbai-embed-large..."
docker exec ollama ollama pull mxbai-embed-large

# 下載語言模型 (根據代碼中的設定)
echo "下載語言模型 gemma2:9b..."
docker exec ollama ollama pull gemma2:9b

echo "可選：下載其他語言模型..."
echo "  - qwen2.5:7b: docker exec ollama ollama pull qwen2.5:7b"
echo "  - qwen2.5:3b: docker exec ollama ollama pull qwen2.5:3b"

# 顯示服務狀態
echo ""
echo "服務狀態："
if command -v docker-compose &> /dev/null; then
    docker-compose ps
else
    docker compose ps
fi

echo ""
echo "RAG 系統啟動完成！"
echo ""
echo "如何使用："
echo "1. 將 PDF 文件放置在 bar_file/PDF/ 目錄中"
echo "2. 將 TTS 相關文件放置在 index_tts/ 目錄中"
echo "3. 將語音參考文件放置在 voice/ 目錄中"
echo "4. 運行應用程序："
if command -v docker-compose &> /dev/null; then
    echo "   docker-compose exec rag-app python roboticBar_printChunk_QAsplit_indexTTSclone_JP.py"
else
    echo "   docker compose exec rag-app python roboticBar_printChunk_QAsplit_indexTTSclone_JP.py"
fi
echo ""
echo "查看日誌："
if command -v docker-compose &> /dev/null; then
    echo "   docker-compose logs -f rag-app"
else
    echo "   docker compose logs -f rag-app"
fi 