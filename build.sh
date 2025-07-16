#!/bin/bash

# 設置錯誤處理
set -e

echo "=============================================="
echo "RAG系統自動打包腳本"
echo "=============================================="
echo
echo "此腳本將自動打包您的RAG系統為可執行文件"
echo "請確保您已經："
echo "1. 安裝了Python 3.8或更高版本"
echo "2. 準備好了所有必要的文件和目錄"
echo "3. 確保有足夠的磁碟空間 (至少5GB)"
echo
echo "按Enter鍵開始打包，或按Ctrl+C取消..."
read -p ""

echo
echo "正在檢查Python環境..."
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "❌ 錯誤：未找到Python，請確保Python已安裝"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo "使用Python命令: $PYTHON_CMD"
$PYTHON_CMD --version

echo
echo "開始執行打包腳本..."
echo
$PYTHON_CMD build_executable.py

echo
echo "打包完成！"
echo "請檢查 dist 目錄中的可執行文件"
echo

# 檢查是否在macOS上，如果是則打開Finder
if [[ "$OSTYPE" == "darwin"* ]]; then
    if [ -d "dist" ]; then
        echo "正在打開 Finder 顯示結果..."
        open dist
    fi
fi

echo "按Enter鍵退出..."
read -p "" 