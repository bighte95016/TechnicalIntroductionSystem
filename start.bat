@echo off
chcp 65001 >nul
echo 正在啟動 RAG 系統...

:: 檢查 Docker 是否安裝
docker --version >nul 2>&1
if errorlevel 1 (
    echo 錯誤：Docker 未安裝。請先安裝 Docker Desktop。
    pause
    exit /b 1
)

:: 檢查 Docker Compose 是否可用
docker compose version >nul 2>&1
if errorlevel 1 (
    docker-compose --version >nul 2>&1
    if errorlevel 1 (
        echo 錯誤：Docker Compose 未安裝。請先安裝 Docker Compose。
        pause
        exit /b 1
    )
    set COMPOSE_CMD=docker-compose
) else (
    set COMPOSE_CMD=docker compose
)

:: 檢查必要的目錄和文件
echo 檢查必要的目錄和文件...
if not exist "bar_file\PDF" mkdir "bar_file\PDF"
if not exist "voice" mkdir "voice"
if not exist "index_tts" mkdir "index_tts"
if not exist "chroma_db" mkdir "chroma_db"
if not exist "bm25_cache" mkdir "bm25_cache"

:: 檢查 .env 文件
if not exist ".env" (
    echo 創建預設 .env 文件...
    (
        echo # OpenAI API Key ^(如果使用 OpenAI^)
        echo OPENAI_API_KEY=your_openai_api_key_here
        echo.
        echo # 其他環境變數
        echo ANONYMIZED_TELEMETRY=False
    ) > .env
    echo 已創建 .env 文件，請根據需要修改其中的設定。
)

:: 啟動服務
echo 啟動 Docker Compose 服務...
%COMPOSE_CMD% up -d

:: 等待 Ollama 服務啟動
echo 等待 Ollama 服務啟動...
timeout /t 15 /nobreak >nul

:: 檢查 Ollama 是否運行
curl -s http://localhost:11434/api/version >nul 2>&1
if errorlevel 1 (
    echo 錯誤：Ollama 服務未正常啟動。請檢查 Docker 容器狀態。
    pause
    exit /b 1
)

echo Ollama 服務已啟動。

:: 下載必要的模型
echo 檢查並下載必要的 Ollama 模型...

echo 下載嵌入模型 mxbai-embed-large...
docker exec ollama ollama pull mxbai-embed-large

echo.
echo 選擇語言模型下載：
echo 代碼中配置的模型是 gemma3:12b (約12GB，高品質但需要較多資源)
echo.
echo 選項：
echo [1] gemma3:12b - 原始配置，高品質但需要至少16GB RAM
echo [2] qwen2.5:7b - 中等品質，需要約8GB RAM (推薦)
echo [3] qwen2.5:3b - 輕量版本，需要約4GB RAM
echo [4] 跳過下載，稍後手動下載
echo.
set /p model_choice="請選擇 (1-4，默認2): "

if "%model_choice%"=="" set model_choice=2
if "%model_choice%"=="1" (
    echo 下載 gemma3:12b ^(這可能需要較長時間，約12GB^)...
    docker exec ollama ollama pull gemma3:12b
    echo 注意：您可能需要修改代碼中的 OLLAMA_MODEL 設定才能使用其他模型。
) else if "%model_choice%"=="2" (
    echo 下載 qwen2.5:7b ^(推薦，約7GB^)...
    docker exec ollama ollama pull qwen2.5:7b
    echo 重要：請修改代碼中第59行的 OLLAMA_MODEL = "gemma3:12b"
    echo       為 OLLAMA_MODEL = "qwen2.5:7b"
) else if "%model_choice%"=="3" (
    echo 下載 qwen2.5:3b ^(輕量版，約3GB^)...
    docker exec ollama ollama pull qwen2.5:3b
    echo 重要：請修改代碼中第59行的 OLLAMA_MODEL = "gemma3:12b"
    echo       為 OLLAMA_MODEL = "qwen2.5:3b"
) else (
    echo 跳過模型下載。請稍後手動下載所需模型：
    echo   docker exec ollama ollama pull ^<model-name^>
)

:: 顯示服務狀態
echo.
echo 服務狀態：
%COMPOSE_CMD% ps

echo.
echo RAG 系統啟動完成！
echo.
echo 如何使用：
echo 1. 將 PDF 文件放置在 bar_file\PDF\ 目錄中
echo 2. 將 TTS 相關文件放置在 index_tts\ 目錄中
echo 3. 將語音參考文件放置在 voice\ 目錄中
if not "%model_choice%"=="1" (
    echo 4. 重要：如果選擇了非 gemma3:12b 模型，請修改代碼中的模型配置
)
echo 5. 運行應用程序：
echo    %COMPOSE_CMD% exec rag-app python roboticBar_printChunk_QAsplit_indexTTSclone_JP.py
echo.
echo 查看日誌：
echo    %COMPOSE_CMD% logs -f rag-app
echo.
echo 停止服務：
echo    %COMPOSE_CMD% down
echo.
pause 