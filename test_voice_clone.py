#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
聲音克隆功能測試腳本
使用與 qa_RAGsystem_printChunk_QAsplit_coquiTTSclone.py 相同的設置
"""

import os
import sys
import time
import torch
import sounddevice as sd
import soundfile as sf
from TTS.api import TTS

# 設定參數
COQUI_TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
SPEAKER_WAV_FILE = "./voice/speaker_oreo1min.wav"
TTS_TEMPERATURE = 0.7
TTS_SAMPLE_RATE = 24000

def check_dependencies():
    """檢查所需依賴"""
    missing_deps = []
    
    try:
        import torch
        print(f"✅ PyTorch 版本: {torch.__version__}")
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import sounddevice as sd
        print(f"✅ sounddevice 可用")
    except ImportError:
        missing_deps.append("sounddevice")
        
    try:
        import soundfile as sf
        print(f"✅ soundfile 可用")
    except ImportError:
        missing_deps.append("soundfile")
    
    try:
        from TTS.api import TTS
        print(f"✅ TTS 可用")
    except ImportError:
        missing_deps.append("TTS")
    
    if missing_deps:
        print(f"❌ 缺少依賴: {', '.join(missing_deps)}")
        print("請運行: pip install " + " ".join(missing_deps))
        return False
    
    return True

def initialize_tts():
    """初始化TTS模型"""
    print("正在初始化 Coqui TTS 模型...")
    
    # PyTorch 兼容性修復
    try:
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import XttsAudioConfig
        from TTS.config.shared_configs import BaseDatasetConfig
        from TTS.tts.models.xtts import XttsArgs

        safe_classes = [XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs]
        torch.serialization.add_safe_globals(safe_classes)
        print("✅ PyTorch 安全全局列表已設置")
    except ImportError as e:
        print(f"⚠️ 導入 XTTS 類時出錯: {e}")
    except Exception as e:
        print(f"⚠️ 設置 PyTorch 安全全局列表時出錯: {e}")
    
    # 檢查GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用設備: {device}")
    
    # 初始化模型
    try:
        start_time = time.time()
        tts = TTS(model_name=COQUI_TTS_MODEL_NAME, gpu=(device == "cuda"))
        load_time = time.time() - start_time
        print(f"✅ TTS 模型載入成功，耗時: {load_time:.2f} 秒")
        return tts
    except Exception as e:
        print(f"❌ TTS 模型載入失敗: {e}")
        return None

def test_voice_clone(tts_model):
    """測試聲音克隆功能"""
    # 檢查說話人音頻文件
    if not os.path.exists(SPEAKER_WAV_FILE):
        print(f"❌ 說話人音頻文件不存在: {SPEAKER_WAV_FILE}")
        return False
    
    print(f"✅ 使用說話人音頻文件: {SPEAKER_WAV_FILE}")
    
    # 測試文本
    test_texts = [
        ("你好，這是聲音克隆測試！", "zh"),
        ("Hello, this is a voice cloning test!", "en"),
        ("謝謝您使用我們的語音系統。", "zh")
    ]
    
    for i, (text, lang) in enumerate(test_texts, 1):
        print(f"\n--- 測試 {i}: {lang.upper()} ---")
        print(f"文本: {text}")
        
        try:
            # 生成音頻
            print("🎵 生成音頻...")
            start_time = time.time()
            wav = tts_model.tts(
                text=text,
                language=lang,
                speaker_wav=SPEAKER_WAV_FILE,
                temperature=TTS_TEMPERATURE
            )
            gen_time = time.time() - start_time
            print(f"✅ 音頻生成完成，耗時: {gen_time:.2f} 秒")
            
            # 播放音頻
            print("🔊 播放音頻...")
            sd.play(wav, samplerate=TTS_SAMPLE_RATE, blocking=True)
            print("✅ 播放完成")
            
            # 等待用戶確認
            user_input = input("按回車繼續下一個測試，或輸入 'q' 退出: ")
            if user_input.lower() == 'q':
                break
                
        except Exception as e:
            print(f"❌ 測試失敗: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return True

def main():
    """主函數"""
    print("🎤 聲音克隆功能測試")
    print("=" * 50)
    
    # 檢查依賴
    if not check_dependencies():
        return
    
    print("\n" + "=" * 50)
    
    # 初始化TTS
    tts_model = initialize_tts()
    if not tts_model:
        return
    
    print("\n" + "=" * 50)
    
    # 測試聲音克隆
    print("開始聲音克隆測試...")
    test_voice_clone(tts_model)
    
    print("\n" + "=" * 50)
    print("測試完成！")

if __name__ == "__main__":
    main() 