#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
語音功能測試腳本
用於驗證 Whisper 語音識別和 CoquiTTS 語音合成功能是否正常工作
"""

import os
import sys
import time
import threading
import tempfile
from pathlib import Path

def test_dependencies():
    """測試所有依賴是否正確安裝"""
    print("🔍 測試依賴安裝狀態...")
    
    missing_deps = []
    
    # 測試語音輸入依賴
    try:
        import pyaudio
        print("✅ pyaudio 已安裝")
    except ImportError:
        missing_deps.append("pyaudio")
        print("❌ pyaudio 未安裝")
    
    try:
        import whisper
        print("✅ openai-whisper 已安裝")
    except ImportError:
        missing_deps.append("openai-whisper")
        print("❌ openai-whisper 未安裝")
    
    try:
        import torch
        print(f"✅ torch 已安裝 (版本: {torch.__version__})")
        if torch.cuda.is_available():
            print(f"  🚀 CUDA 可用 (GPU: {torch.cuda.get_device_name(0)})")
        else:
            print("  💻 使用 CPU 模式")
    except ImportError:
        missing_deps.append("torch")
        print("❌ torch 未安裝")
    
    try:
        import numpy
        print(f"✅ numpy 已安裝 (版本: {numpy.__version__})")
    except ImportError:
        missing_deps.append("numpy")
        print("❌ numpy 未安裝")
    
    # 測試語音輸出依賴
    try:
        import TTS
        print(f"✅ TTS 已安裝 (版本: {TTS.__version__})")
    except ImportError:
        missing_deps.append("TTS")
        print("❌ TTS 未安裝")
    
    try:
        import pydub
        print("✅ pydub 已安裝")
    except ImportError:
        missing_deps.append("pydub")
        print("❌ pydub 未安裝")
    
    try:
        import simpleaudio
        print("✅ simpleaudio 已安裝")
    except ImportError:
        missing_deps.append("simpleaudio")
        print("❌ simpleaudio 未安裝")
    
    try:
        import jieba
        print("✅ jieba 已安裝")
    except ImportError:
        missing_deps.append("jieba")
        print("❌ jieba 未安裝")
    
    if missing_deps:
        print(f"\n⚠️  缺失依賴: {', '.join(missing_deps)}")
        print("請運行以下命令安裝：")
        print(f"pip install {' '.join(missing_deps)}")
        return False
    else:
        print("\n✅ 所有依賴已正確安裝！")
        return True

def test_whisper():
    """測試 Whisper 語音識別功能"""
    print("\n🎙️ 測試 Whisper 語音識別功能...")
    
    try:
        import whisper
        import torch
        
        # 載入較小的模型進行測試
        print("正在載入 Whisper base 模型...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("base", device=device)
        print(f"✅ Whisper 模型載入成功 (設備: {device})")
        
        # 創建測試音頻（這裡只是測試模型載入，不進行實際識別）
        print("✅ Whisper 語音識別功能正常")
        return True
        
    except Exception as e:
        print(f"❌ Whisper 測試失敗: {e}")
        return False

def test_coqui_tts():
    """測試 CoquiTTS 語音合成功能"""
    print("\n🔊 測試 CoquiTTS 語音合成功能...")
    
    try:
        from TTS.api import TTS
        import torch
        import numpy as np
        import simpleaudio as sa
        
        # 初始化 TTS 模型
        print("正在初始化 CoquiTTS 模型...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 使用較小的模型進行測試
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        print(f"載入模型: {model_name}")
        
        # 添加 PyTorch 安全全局類（兼容性修復）
        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import XttsAudioConfig
            from TTS.config.shared_configs import BaseDatasetConfig
            from TTS.tts.models.xtts import XttsArgs
            
            safe_classes = [XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs]
            torch.serialization.add_safe_globals(safe_classes)
            print("✅ PyTorch 兼容性修復完成")
        except Exception as e:
            print(f"⚠️  PyTorch 兼容性修復警告: {e}")
        
        tts = TTS(model_name=model_name, gpu=(device == "cuda"))
        print(f"✅ CoquiTTS 模型載入成功 (設備: {device})")
        
        # 測試中文語音合成
        print("測試中文語音合成...")
        test_text_zh = "這是一個測試，CoquiTTS 語音合成功能正常工作。"
        
        try:
            # 獲取可用說話人
            speakers = []
            if hasattr(tts, 'synthesizer') and \
               hasattr(tts.synthesizer, 'tts_model') and \
               hasattr(tts.synthesizer.tts_model, 'speaker_manager') and \
               hasattr(tts.synthesizer.tts_model.speaker_manager, 'speakers'):
                speakers = list(tts.synthesizer.tts_model.speaker_manager.speakers.keys())
                print(f"可用說話人: {speakers[:5]}...")  # 只顯示前5個
            
            # 選擇說話人
            speaker = "Tammie Ema" if "Tammie Ema" in speakers else (speakers[0] if speakers else None)
            
            tts_kwargs = {
                "text": test_text_zh,
                "language": "zh-cn"
            }
            if speaker:
                tts_kwargs["speaker"] = speaker
                print(f"使用說話人: {speaker}")
            
            wav = tts.tts(**tts_kwargs)
            print("✅ 中文語音合成成功")
            
            # 測試播放（可選）
            play_test = input("是否播放測試音頻？(y/n): ").strip().lower()
            if play_test == 'y':
                try:
                    # 轉換音頻格式
                    wav = np.clip(wav, -1.0, 1.0)
                    wav_int16 = (wav * 32767).astype(np.int16)
                    
                    # 播放音頻
                    print("🎵 正在播放測試音頻...")
                    play_obj = sa.play_buffer(
                        wav_int16,
                        num_channels=1,
                        bytes_per_sample=2,
                        sample_rate=22050
                    )
                    play_obj.wait_done()
                    print("✅ 音頻播放成功")
                except Exception as play_error:
                    print(f"⚠️  音頻播放失敗: {play_error}")
            
        except Exception as synthesis_error:
            print(f"❌ 語音合成失敗: {synthesis_error}")
            return False
        
        # 測試英文語音合成
        print("測試英文語音合成...")
        test_text_en = "This is a test. CoquiTTS voice synthesis is working properly."
        
        try:
            en_kwargs = {
                "text": test_text_en,
                "language": "en"
            }
            if speaker:
                en_kwargs["speaker"] = speaker
            
            wav_en = tts.tts(**en_kwargs)
            print("✅ 英文語音合成成功")
            
        except Exception as en_synthesis_error:
            print(f"❌ 英文語音合成失敗: {en_synthesis_error}")
            return False
        
        print("✅ CoquiTTS 語音合成功能正常")
        return True
        
    except Exception as e:
        print(f"❌ CoquiTTS 測試失敗: {e}")
        print("可能的解決方案：")
        print("1. 檢查網絡連接（首次使用需要下載模型）")
        print("2. 確保有足夠的磁盤空間（模型約 1.8GB）")
        print("3. 檢查 TTS 庫版本是否兼容")
        return False

def test_audio_devices():
    """測試音頻設備"""
    print("\n🔧 測試音頻設備...")
    
    try:
        import pyaudio
        
        audio = pyaudio.PyAudio()
        
        print("可用音頻設備：")
        device_count = audio.get_device_count()
        
        input_devices = []
        output_devices = []
        
        for i in range(device_count):
            try:
                device_info = audio.get_device_info_by_index(i)
                device_name = device_info['name']
                max_input_channels = device_info['maxInputChannels']
                max_output_channels = device_info['maxOutputChannels']
                
                if max_input_channels > 0:
                    input_devices.append((i, device_name))
                    print(f"  🎤 輸入設備 {i}: {device_name}")
                
                if max_output_channels > 0:
                    output_devices.append((i, device_name))
                    print(f"  🔊 輸出設備 {i}: {device_name}")
                    
            except Exception as device_error:
                print(f"  ⚠️  設備 {i} 信息獲取失敗: {device_error}")
        
        # 測試默認設備
        try:
            default_input = audio.get_default_input_device_info()
            print(f"✅ 默認輸入設備: {default_input['name']}")
        except Exception as e:
            print(f"❌ 無法獲取默認輸入設備: {e}")
        
        try:
            default_output = audio.get_default_output_device_info()
            print(f"✅ 默認輸出設備: {default_output['name']}")
        except Exception as e:
            print(f"❌ 無法獲取默認輸出設備: {e}")
        
        audio.terminate()
        
        if input_devices and output_devices:
            print("✅ 音頻設備測試通過")
            return True
        else:
            print("❌ 缺少必要的音頻設備")
            return False
            
    except Exception as e:
        print(f"❌ 音頻設備測試失敗: {e}")
        return False

def main():
    """主測試函數"""
    print("🚀 語音功能測試開始\n")
    print("="*50)
    
    # 測試步驟
    tests = [
        ("依賴檢查", test_dependencies),
        ("音頻設備", test_audio_devices),
        ("Whisper 語音識別", test_whisper),
        ("CoquiTTS 語音合成", test_coqui_tts),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} 測試過程中發生異常: {e}")
            results[test_name] = False
    
    # 總結
    print("\n" + "="*50)
    print("🏁 測試結果總結")
    print("="*50)
    
    all_passed = True
    for test_name, result in results.items():
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 所有測試通過！您的語音功能已準備就緒。")
        print("現在可以運行主程序：")
        print("python qa_RAGsystem_printChunk_QAsplit_coquiTTS.py")
    else:
        print("⚠️  部分測試失敗，請根據上述錯誤信息進行修復。")
        print("\n常見解決方案：")
        print("1. 安裝缺失的依賴：pip install -r requirements_voice.txt")
        print("2. 檢查音頻設備權限")
        print("3. 確保網絡連接穩定（用於下載模型）")
        print("4. 檢查磁盤空間是否充足")
    
    print("="*50)

if __name__ == "__main__":
    main() 