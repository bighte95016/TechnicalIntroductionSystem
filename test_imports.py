#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速測試腳本 - 驗證關鍵模組導入
用於驗證 PyInstaller 打包前的環境準備
"""

import sys

def test_import(module_name, description=""):
    """測試單個模組導入"""
    try:
        __import__(module_name)
        print(f"✅ {module_name} - {description}")
        return True
    except ImportError as e:
        print(f"❌ {module_name} - {description}")
        print(f"   錯誤: {e}")
        return False

def main():
    """主測試函數"""
    print("🧪 關鍵模組導入測試")
    print("=" * 50)
    
    # 定義要測試的模組
    test_modules = [
        ("langchain_ollama", "LangChain Ollama 主模組"),
        ("langchain_ollama.llms", "Ollama LLM 模組"),
        ("langchain_ollama.embeddings", "Ollama 嵌入模組"),
        ("langchain_ollama.chat_models", "Ollama 聊天模組"),
        ("langchain_core", "LangChain 核心模組"),
        ("langchain_community", "LangChain 社區模組"),
        ("ollama", "Ollama 客戶端"),
        ("chromadb", "ChromaDB 向量數據庫"),
        ("TTS", "Coqui TTS"),
        ("whisper", "OpenAI Whisper"),
        ("PyPDF2", "PDF 處理"),
        ("jieba", "中文分詞"),
        ("inflect", "英文語法變化"),
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("requests", "HTTP 請求"),
        ("dotenv", "環境變數加載"),
    ]
    
    success_count = 0
    total_count = len(test_modules)
    
    for module_name, description in test_modules:
        if test_import(module_name, description):
            success_count += 1
    
    print("\n" + "=" * 50)
    print(f"📊 測試結果: {success_count}/{total_count} 模組導入成功")
    
    if success_count == total_count:
        print("🎉 所有模組導入成功！可以進行 PyInstaller 打包。")
        return True
    else:
        print("⚠️ 部分模組導入失敗，建議先解決依賴問題。")
        failed_count = total_count - success_count
        print(f"❌ 失敗的模組數量: {failed_count}")
        
        print("\n💡 建議解決方案:")
        print("1. 運行 fix_langchain_ollama.py 診斷腳本")
        print("2. 手動安裝缺失的套件:")
        print("   pip install langchain-ollama langchain-core langchain-community")
        print("   pip install ollama chromadb TTS openai-whisper")
        print("   pip install PyPDF2 jieba inflect torch numpy requests python-dotenv")
        print("3. 檢查虛擬環境是否正確激活")
        
        return False

if __name__ == "__main__":
    success = main()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ 準備就緒，可以運行 fix_and_rebuild.bat 進行打包")
    else:
        print("❌ 請先解決上述問題，然後重新運行此測試")
    
    input("\n按 Enter 鍵退出...") 