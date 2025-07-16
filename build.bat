@echo off
chcp 65001 >nul
echo ===============================================
echo RAG系統一鍵打包工具
echo ===============================================
echo.
echo 此工具將自動打包您的RAG系統為可執行文件
echo.
echo 打包選項：
echo 1. 完整打包（包含依賴安裝）
echo 2. 快速打包（跳過依賴安裝）
echo.
echo 開始執行打包腳本...
echo.

python build_executable.py

echo.
echo 打包完成！
echo 可執行文件位於 dist 目錄中
echo.
pause 