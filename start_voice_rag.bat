@echo off
chcp 65001 > nul
echo ========================================
echo    語音 RAG 系統啟動腳本
echo ========================================
echo.

echo 🔍 檢查 Python 環境...
python --version > nul 2>&1
if errorlevel 1 (
    echo ❌ 未找到 Python，請先安裝 Python 3.8+
    pause
    exit /b 1
)

echo ✅ Python 環境正常

echo.
echo 🔍 檢查必要文件...
if not exist "qa_RAGsystem_printChunk_QAsplit_coquiTTS.py" (
    echo ❌ 未找到主程序文件
    pause
    exit /b 1
)

if not exist "requirements_voice.txt" (
    echo ❌ 未找到依賴文件
    pause
    exit /b 1
)

echo ✅ 程序文件完整

echo.
echo 📦 是否要安裝/更新依賴？(y/n)
set /p install_deps="> "
if /i "%install_deps%"=="y" (
    echo 正在安裝依賴...
    pip install -r requirements_voice.txt
    if errorlevel 1 (
        echo ❌ 依賴安裝失敗
        pause
        exit /b 1
    )
    echo ✅ 依賴安裝完成
)

echo.
echo 🧪 是否要運行語音功能測試？(y/n)
set /p run_test="> "
if /i "%run_test%"=="y" (
    echo 運行語音功能測試...
    python test_voice_features.py
    echo.
    echo 按任意鍵繼續啟動主程序...
    pause > nul
)

echo.
echo 🚀 啟動語音 RAG 系統...
echo ========================================
echo 提示：
echo - 🎤 語音輸入：按 Enter 開始錄音
echo - 🔊 語音輸出：自動播放答案
echo - 📝 文字模式：輸入 'text' 切換
echo - 🔚 退出程序：輸入 'quit' 或 'exit'
echo ========================================
echo.

python qa_RAGsystem_printChunk_QAsplit_coquiTTS.py

echo.
echo 程序已結束，按任意鍵退出...
pause > nul 