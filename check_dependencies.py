#!/usr/bin/env python3
"""
依賴檢查腳本 - 檢查RAG TTS系統所需的所有依賴
"""

import sys
import importlib

def check_module(module_name, description=""):
    """檢查模組是否可以導入"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} - OK {description}")
        return True
    except ImportError as e:
        print(f"❌ {module_name} - MISSING {description}")
        print(f"   錯誤: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {module_name} - ERROR {description}")
        print(f"   錯誤: {e}")
        return False

def main():
    print("="*60)
    print("RAG TTS 系統依賴檢查")
    print("="*60)
    
    # 核心依賴
    print("\n核心依賴:")
    core_deps = [
        ("langchain_ollama", "- Ollama LLM 支援"),
        ("langchain_community", "- LangChain 社群套件"),
        ("langchain_openai", "- OpenAI 支援"),
        ("langchain.chains", "- LangChain 鏈"),
        ("langchain.prompts", "- LangChain 提示"),
        ("langchain.schema", "- LangChain 架構"),
    ]
    
    core_ok = 0
    for module, desc in core_deps:
        if check_module(module, desc):
            core_ok += 1
    
    # TTS 相關
    print("\nTTS 語音合成:")
    tts_deps = [
        ("TTS", "- Coqui TTS"),
        ("TTS.api", "- TTS API"),
        ("inflect", "- 英文語法處理"),
        ("torch", "- PyTorch"),
        ("simpleaudio", "- 音頻播放"),
    ]
    
    tts_ok = 0
    for module, desc in tts_deps:
        if check_module(module, desc):
            tts_ok += 1
    
    # 語音識別
    print("\n語音識別:")
    speech_deps = [
        ("whisper", "- Whisper 語音識別"),
        ("pyaudio", "- 音頻錄製"),
        ("numpy", "- 數值計算"),
    ]
    
    speech_ok = 0
    for module, desc in speech_deps:
        if check_module(module, desc):
            speech_ok += 1
    
    # 文檔處理
    print("\n文檔處理:")
    doc_deps = [
        ("PyPDF2", "- PDF 讀取"),
        ("chromadb", "- 向量資料庫"),
        ("jieba", "- 中文分詞"),
        ("dotenv", "- 環境變數"),
    ]
    
    doc_ok = 0
    for module, desc in doc_deps:
        if check_module(module, desc):
            doc_ok += 1
    
    # 其他依賴
    print("\n其他依賴:")
    other_deps = [
        ("pydantic", "- 數據驗證"),
        ("requests", "- HTTP 請求"),
        ("sklearn", "- 機器學習"),
    ]
    
    other_ok = 0
    for module, desc in other_deps:
        if check_module(module, desc):
            other_ok += 1
    
    # 總結
    total_deps = len(core_deps) + len(tts_deps) + len(speech_deps) + len(doc_deps) + len(other_deps)
    total_ok = core_ok + tts_ok + speech_ok + doc_ok + other_ok
    
    print("\n" + "="*60)
    print(f"依賴檢查結果: {total_ok}/{total_deps} 個模組可用")
    print("="*60)
    
    if total_ok == total_deps:
        print("🎉 所有依賴都已正確安裝！")
        return True
    else:
        print("⚠️  部分依賴缺失，請安裝缺失的套件。")
        print("\n安裝命令建議:")
        print("pip install langchain-ollama langchain-community langchain-openai")
        print("pip install TTS torch whisper pyaudio simpleaudio")
        print("pip install PyPDF2 chromadb jieba python-dotenv")
        print("pip install pydantic requests scikit-learn")
        return False

if __name__ == "__main__":
    success = main()
    input("\n按 Enter 鍵結束...")
    sys.exit(0 if success else 1) 