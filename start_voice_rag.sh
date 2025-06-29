#!/bin/bash
# 語音 RAG 系統啟動腳本 (Linux/macOS)

# 設置顏色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================"
echo "    語音 RAG 系統啟動腳本"
echo "========================================"
echo

# 檢查 Python 環境
echo -e "${BLUE}🔍 檢查 Python 環境...${NC}"
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo -e "${RED}❌ 未找到 Python，請先安裝 Python 3.8+${NC}"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
echo -e "${GREEN}✅ Python 環境正常 (版本: $PYTHON_VERSION)${NC}"

# 檢查必要文件
echo
echo -e "${BLUE}🔍 檢查必要文件...${NC}"
if [ ! -f "qa_RAGsystem_printChunk_QAsplit_coquiTTS.py" ]; then
    echo -e "${RED}❌ 未找到主程序文件${NC}"
    exit 1
fi

if [ ! -f "requirements_voice.txt" ]; then
    echo -e "${RED}❌ 未找到依賴文件${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 程序文件完整${NC}"

# 詢問是否安裝依賴
echo
echo -e "${YELLOW}📦 是否要安裝/更新依賴？(y/n)${NC}"
read -r install_deps
if [[ $install_deps =~ ^[Yy]$ ]]; then
    echo "正在安裝依賴..."
    $PYTHON_CMD -m pip install -r requirements_voice.txt
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ 依賴安裝失敗${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ 依賴安裝完成${NC}"
fi

# 詢問是否運行測試
echo
echo -e "${YELLOW}🧪 是否要運行語音功能測試？(y/n)${NC}"
read -r run_test
if [[ $run_test =~ ^[Yy]$ ]]; then
    echo "運行語音功能測試..."
    $PYTHON_CMD test_voice_features.py
    echo
    echo "按 Enter 鍵繼續啟動主程序..."
    read -r
fi

# 啟動主程序
echo
echo -e "${BLUE}🚀 啟動語音 RAG 系統...${NC}"
echo "========================================"
echo "提示："
echo "- 🎤 語音輸入：按 Enter 開始錄音"
echo "- 🔊 語音輸出：自動播放答案"
echo "- 📝 文字模式：輸入 'text' 切換"
echo "- 🔚 退出程序：輸入 'quit' 或 'exit'"
echo "========================================"
echo

$PYTHON_CMD qa_RAGsystem_printChunk_QAsplit_coquiTTS.py

echo
echo "程序已結束。" 