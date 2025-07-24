@echo off
chcp 65001 >nul
echo 正在準備 RAG 系統部署包...

set DEPLOY_DIR=rag-system-deploy
set TIMESTAMP=%date:~0,4%%date:~5,2%%date:~8,2%-%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%

echo 創建部署目錄...
if exist "%DEPLOY_DIR%" rmdir /s /q "%DEPLOY_DIR%"
mkdir "%DEPLOY_DIR%"

echo 複製核心文件...
copy "roboticBar_printChunk_QAsplit_indexTTSclone_JP.py" "%DEPLOY_DIR%\" >nul
copy "Dockerfile" "%DEPLOY_DIR%\" >nul
copy "docker-compose.yml" "%DEPLOY_DIR%\" >nul
copy "docker-compose-windows.yml" "%DEPLOY_DIR%\" >nul
copy "requirements.txt" "%DEPLOY_DIR%\" >nul
copy ".dockerignore" "%DEPLOY_DIR%\" >nul
copy "start.bat" "%DEPLOY_DIR%\" >nul
copy "start.sh" "%DEPLOY_DIR%\" >nul
copy "package-for-deployment.bat" "%DEPLOY_DIR%\" >nul
copy "Docker部署說明.md" "%DEPLOY_DIR%\" >nul

echo 創建必要的目錄結構...
mkdir "%DEPLOY_DIR%\bar_file"
mkdir "%DEPLOY_DIR%\bar_file\PDF"
mkdir "%DEPLOY_DIR%\voice"
mkdir "%DEPLOY_DIR%\index_tts"
mkdir "%DEPLOY_DIR%\chroma_db"
mkdir "%DEPLOY_DIR%\bm25_cache"

echo 複製數據文件（如果存在）...
if exist "bar_file\PDF\*" (
    xcopy "bar_file\PDF\*" "%DEPLOY_DIR%\bar_file\PDF\" /E /I /Q >nul
    echo   - PDF 文件已複製
)

if exist "voice\*" (
    xcopy "voice\*" "%DEPLOY_DIR%\voice\" /E /I /Q >nul
    echo   - 語音文件已複製
)

if exist "index_tts\*" (
    xcopy "index_tts\*" "%DEPLOY_DIR%\index_tts\" /E /I /Q >nul
    echo   - TTS 文件已複製
)

if exist ".env" (
    copy ".env" "%DEPLOY_DIR%\" >nul
    echo   - 環境配置文件已複製
) else (
    echo 創建範例 .env 文件...
    (
        echo # OpenAI API Key ^(如果使用 OpenAI^)
        echo OPENAI_API_KEY=your_openai_api_key_here
        echo.
        echo # 其他環境變數
        echo ANONYMIZED_TELEMETRY=False
    ) > "%DEPLOY_DIR%\.env"
)

echo 創建部署指南...
(
    echo # RAG 系統快速部署指南
    echo.
    echo ## 部署步驟
    echo 1. 確保已安裝 Docker Desktop
    echo 2. 雙擊執行 start.bat
    echo 3. 等待系統啟動完成
    echo 4. 使用以下命令啟動應用：
    echo    docker compose exec rag-app python roboticBar_printChunk_QAsplit_indexTTSclone_JP.py
    echo.
    echo ## Windows 專用配置
    echo 如果遇到問題，請使用 docker-compose-windows.yml：
    echo    docker compose -f docker-compose-windows.yml up -d
    echo.
    echo ## 重要提醒
    echo - 首次啟動需要下載大約 2-4GB 的 AI 模型
    echo - 確保有穩定的網路連接
    echo - 建議至少 8GB RAM
    echo.
    echo 詳細說明請參考 "Docker部署說明.md"
) > "%DEPLOY_DIR%\README.txt"

echo 創建 Windows 優化版啟動腳本...
(
    echo @echo off
    echo chcp 65001 ^>nul
    echo echo 正在啟動 RAG 系統 ^(Windows 優化版^)...
    echo.
    echo :: 使用 Windows 優化的 Docker Compose 配置
    echo set COMPOSE_FILE=docker-compose-windows.yml
    echo.
    echo :: 檢查 Docker 是否安裝
    echo docker --version ^>nul 2^>^&1
    echo if errorlevel 1 ^(
    echo     echo 錯誤：Docker 未安裝。請先安裝 Docker Desktop。
    echo     pause
    echo     exit /b 1
    echo ^)
    echo.
    echo :: 啟動服務
    echo echo 啟動 Docker Compose 服務 ^(Windows 配置^)...
    echo docker compose -f %%COMPOSE_FILE%% up -d
    echo.
    echo :: 等待服務啟動
    echo echo 等待服務啟動...
    echo timeout /t 20 /nobreak ^>nul
    echo.
    echo :: 下載模型
    echo echo 下載必要的 AI 模型...
    echo docker exec ollama ollama pull mxbai-embed-large
    echo docker exec ollama ollama pull gemma2:9b
    echo.
    echo echo RAG 系統啟動完成！
    echo echo 使用以下命令進入應用：
    echo echo docker compose -f %%COMPOSE_FILE%% exec rag-app python roboticBar_printChunk_QAsplit_indexTTSclone_JP.py
    echo pause
) > "%DEPLOY_DIR%\start-windows.bat"

echo 打包完成！
echo.
echo 部署包已創建在目錄: %DEPLOY_DIR%
echo.
echo 部署包內容：
echo   ✓ 所有必要的配置文件
echo   ✓ Docker 配置（通用版和 Windows 優化版）
echo   ✓ 啟動腳本（通用版和 Windows 專用版）
echo   ✓ 完整的部署說明文檔
echo   ✓ 示例環境配置文件
if exist "bar_file\PDF\*" echo   ✓ PDF 知識庫文件
if exist "voice\*" echo   ✓ 語音參考文件
if exist "index_tts\*" echo   ✓ TTS 模型文件
echo.
echo 下一步：
echo 1. 將整個 %DEPLOY_DIR% 目錄複製到目標電腦
echo 2. 在目標電腦上雙擊執行 start-windows.bat （推薦）
echo    或執行 start.bat （通用版）
echo 3. 按照螢幕提示完成部署
echo.
echo 部署包已準備就緒！
pause 