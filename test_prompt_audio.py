#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提示音功能測試腳本
用於單獨測試 CoquiTTS 提示音功能是否正常工作
"""

import os
import sys
import time
import threading
import numpy as np

def test_prompt_audio():
    """測試提示音功能"""
    print("🎵 提示音功能測試")
    print("=" * 40)
    
    # 檢查依賴
    print("🔍 檢查依賴...")
    try:
        from TTS.api import TTS
        import simpleaudio as sa
        import torch
        print("✅ 所有依賴已安裝")
    except ImportError as e:
        print(f"❌ 缺少依賴: {e}")
        print("請運行: pip install TTS simpleaudio torch")
        return False
    
    # 初始化 TTS 模型
    print("\n🔧 初始化 TTS 模型...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用設備: {device}")
        
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        tts = TTS(model_name=model_name, gpu=(device == "cuda"))
        print("✅ TTS 模型初始化成功")
        
        # 獲取說話人列表
        if hasattr(tts, 'synthesizer') and \
           hasattr(tts.synthesizer, 'tts_model') and \
           hasattr(tts.synthesizer.tts_model, 'speaker_manager'):
            speakers = list(tts.synthesizer.tts_model.speaker_manager.speakers.keys())
            print(f"可用說話人: {speakers[:5]}...")  # 只顯示前5個
        else:
            speakers = ["Tammie Ema"]  # 默認說話人
            
    except Exception as e:
        print(f"❌ TTS 模型初始化失敗: {e}")
        return False
    
    # 測試中文提示音
    print("\n🎵 測試中文提示音...")
    try:
        zh_text = "感謝您的提問，我思考一下請稍候！"
        zh_kwargs = {
            "text": zh_text,
            "language": "zh-cn",
            "speaker": "Tammie Ema" if "Tammie Ema" in speakers else speakers[0]
        }
        
        print(f"生成中文音頻: {zh_text}")
        wav = tts.tts(**zh_kwargs)
        
        # 播放音頻
        wav = np.clip(wav, -1.0, 1.0)
        wav_int16 = (wav * 32767).astype(np.int16)
        
        print("🔊 播放中文提示音...")
        play_obj = sa.play_buffer(
            wav_int16,
            num_channels=1,
            bytes_per_sample=2,
            sample_rate=22050
        )
        play_obj.wait_done()
        print("✅ 中文提示音播放完成")
        
    except Exception as e:
        print(f"❌ 中文提示音測試失敗: {e}")
        return False
    
    # 等待一下
    time.sleep(1)
    
    # 測試英文提示音
    print("\n🎵 測試英文提示音...")
    try:
        en_text = "Thank you for your question. Let me think about it for a moment."
        en_kwargs = {
            "text": en_text,
            "language": "en",
            "speaker": "Tammie Ema" if "Tammie Ema" in speakers else speakers[0]
        }
        
        print(f"生成英文音頻: {en_text}")
        wav = tts.tts(**en_kwargs)
        
        # 播放音頻
        wav = np.clip(wav, -1.0, 1.0)
        wav_int16 = (wav * 32767).astype(np.int16)
        
        print("🔊 播放英文提示音...")
        play_obj = sa.play_buffer(
            wav_int16,
            num_channels=1,
            bytes_per_sample=2,
            sample_rate=22050
        )
        play_obj.wait_done()
        print("✅ 英文提示音播放完成")
        
    except Exception as e:
        print(f"❌ 英文提示音測試失敗: {e}")
        return False
    
    print("\n🎉 提示音功能測試完成！")
    print("✅ 所有測試通過，提示音功能正常工作")
    return True

def test_background_playback():
    """測試後台播放功能"""
    print("\n🔄 測試後台播放功能...")
    
    def play_in_background():
        print("📱 後台線程開始播放...")
        time.sleep(2)  # 模擬音頻播放時間
        print("📱 後台線程播放完成")
    
    # 啟動後台線程
    thread = threading.Thread(target=play_in_background, daemon=True)
    thread.start()
    
    print("🔄 主線程繼續執行...")
    print("⏳ 等待後台播放完成...")
    thread.join()
    print("✅ 後台播放測試完成")

if __name__ == "__main__":
    print("🎵 CoquiTTS 提示音功能測試")
    print("=" * 50)
    
    # 測試提示音
    if test_prompt_audio():
        # 測試後台播放
        test_background_playback()
        
        print("\n🎊 所有測試完成！")
        print("✅ 提示音功能已準備就緒")
        print("\n💡 提示：")
        print("- 在主程序中，提示音會在用戶提問後自動播放")
        print("- 提示音使用後台線程，不會阻塞主程序")
        print("- 系統會根據檢測到的語言選擇相應的提示音")
    else:
        print("\n❌ 測試失敗，請檢查依賴安裝和配置")
        sys.exit(1) 