#!/usr/bin/env python3
"""
IndexTTS 提示音功能測試腳本
用於單獨測試 IndexTTS 提示音功能是否正常工作
"""

import os
import sys
import pathlib

# 添加 index_tts 模組路徑
sys.path.append(str(pathlib.Path(__file__).parent / "index_tts"))

try:
    from index_tts.indextts.infer import IndexTTS
    import sounddevice as sd
    import threading
    print("✅ 成功導入必要的模組")
except ImportError as e:
    print(f"❌ 導入模組失敗: {e}")
    print("請確保 IndexTTS 和相關依賴已正確安裝")
    sys.exit(1)

# 配置參數
TTS_MODEL_DIR = "./index_tts/checkpoints"
TTS_CONFIG_PATH = "./index_tts/checkpoints/config.yaml"
TTS_VOICE_PATH = "./voice/Dr_Lee.wav"

def test_indextts_prompt_audio():
    """測試IndexTTS提示音功能"""
    print("🎵 IndexTTS 提示音功能測試")
    print("=" * 50)
    
    # 檢查必要文件
    required_files = [
        (TTS_MODEL_DIR, "TTS模型目錄"),
        (TTS_CONFIG_PATH, "TTS配置文件"),
        (TTS_VOICE_PATH, "語音參考文件")
    ]
    
    print("📁 檢查必要文件...")
    for file_path, description in required_files:
        if os.path.exists(file_path):
            print(f"✅ {description}: {file_path}")
        else:
            print(f"❌ {description}不存在: {file_path}")
            return False
    
    # 初始化 IndexTTS 系統
    print("\n🔧 初始化 IndexTTS 系統...")
    try:
        tts_system = IndexTTS(model_dir=TTS_MODEL_DIR, cfg_path=TTS_CONFIG_PATH)
        print("✅ IndexTTS 系統初始化成功")
    except Exception as e:
        print(f"❌ IndexTTS 系統初始化失敗: {e}")
        return False
    
    # 測試中文提示音
    print("\n🎵 測試中文提示音...")
    try:
        prompt_text = "讓我想想..."
        print(f"📝 提示音文本: '{prompt_text}'")
        
        # 生成音頻
        print("🔊 生成中文提示音...")
        sampling_rate, wav_data = tts_system.infer(
            TTS_VOICE_PATH, 
            prompt_text, 
            output_path=None
        )
        
        print(f"📊 音頻信息: 採樣率={sampling_rate}Hz, 時長={len(wav_data)/sampling_rate:.2f}秒")
        
        # 播放音頻
        print("🔊 播放中文提示音...")
        sd.play(wav_data, sampling_rate)
        sd.wait()
        
        print("✅ 中文提示音播放完成")
        
    except Exception as e:
        print(f"❌ 中文提示音測試失敗: {e}")
        return False
    
    print("\n" + "="*30)
    
    # 測試英文提示音
    print("\n🎵 測試英文提示音...")
    try:
        prompt_text = "Let me think about that..."
        print(f"📝 提示音文本: '{prompt_text}'")
        
        # 生成音頻
        print("🔊 生成英文提示音...")
        sampling_rate, wav_data = tts_system.infer(
            TTS_VOICE_PATH, 
            prompt_text, 
            output_path=None
        )
        
        print(f"📊 音頻信息: 採樣率={sampling_rate}Hz, 時長={len(wav_data)/sampling_rate:.2f}秒")
        
        # 播放音頻
        print("🔊 播放英文提示音...")
        sd.play(wav_data, sampling_rate)
        sd.wait()
        
        print("✅ 英文提示音播放完成")
        
    except Exception as e:
        print(f"❌ 英文提示音測試失敗: {e}")
        return False
    
    return True

def test_recording_prompt_audio():
    """測試錄音開始提示音功能"""
    print("\n🎤 測試錄音開始提示音功能...")
    
    # 初始化 IndexTTS 系統
    try:
        tts_system = IndexTTS(model_dir=TTS_MODEL_DIR, cfg_path=TTS_CONFIG_PATH)
        print("✅ IndexTTS 系統初始化成功")
    except Exception as e:
        print(f"❌ IndexTTS 系統初始化失敗: {e}")
        return False
    
    # 測試中文錄音提示音
    print("\n🎤 測試中文錄音提示音...")
    try:
        prompt_text = "請說。"
        print(f"📝 錄音提示音文本: '{prompt_text}'")
        
        # 生成音頻
        print("🔊 生成中文錄音提示音...")
        sampling_rate, wav_data = tts_system.infer(
            TTS_VOICE_PATH, 
            prompt_text, 
            output_path=None
        )
        
        print(f"📊 音頻信息: 採樣率={sampling_rate}Hz, 時長={len(wav_data)/sampling_rate:.2f}秒")
        
        # 播放音頻
        print("🔊 播放中文錄音提示音...")
        sd.play(wav_data, sampling_rate)
        sd.wait()
        
        print("✅ 中文錄音提示音播放完成")
        
    except Exception as e:
        print(f"❌ 中文錄音提示音測試失敗: {e}")
        return False
    
    print("\n" + "="*30)
    
    # 測試英文錄音提示音
    print("\n🎤 測試英文錄音提示音...")
    try:
        prompt_text = "Please speak now."
        print(f"📝 錄音提示音文本: '{prompt_text}'")
        
        # 生成音頻
        print("🔊 生成英文錄音提示音...")
        sampling_rate, wav_data = tts_system.infer(
            TTS_VOICE_PATH, 
            prompt_text, 
            output_path=None
        )
        
        print(f"📊 音頻信息: 採樣率={sampling_rate}Hz, 時長={len(wav_data)/sampling_rate:.2f}秒")
        
        # 播放音頻
        print("🔊 播放英文錄音提示音...")
        sd.play(wav_data, sampling_rate)
        sd.wait()
        
        print("✅ 英文錄音提示音播放完成")
        
    except Exception as e:
        print(f"❌ 英文錄音提示音測試失敗: {e}")
        return False
    
    return True

def test_threaded_prompt_audio():
    """測試多線程提示音播放"""
    print("\n🧵 測試多線程提示音播放...")
    
    def play_prompt_in_background():
        try:
            # 初始化 TTS 系統
            tts_system = IndexTTS(model_dir=TTS_MODEL_DIR, cfg_path=TTS_CONFIG_PATH)
            
            # 生成並播放提示音
            prompt_text = "正在後台播放提示音..."
            sampling_rate, wav_data = tts_system.infer(
                TTS_VOICE_PATH, 
                prompt_text, 
                output_path=None
            )
            
            print("🎵 後台線程開始播放提示音...")
            sd.play(wav_data, sampling_rate)
            sd.wait()
            print("✅ 後台提示音播放完成")
            
        except Exception as e:
            print(f"❌ 後台提示音播放失敗: {e}")
    
    # 啟動後台線程
    try:
        prompt_thread = threading.Thread(target=play_prompt_in_background)
        prompt_thread.daemon = True
        prompt_thread.start()
        
        print("🔄 主線程繼續執行其他任務...")
        import time
        time.sleep(1)  # 模擬主線程的其他工作
        
        # 等待線程完成
        prompt_thread.join(timeout=10)  # 最多等待10秒
        
        if prompt_thread.is_alive():
            print("⚠️ 提示音線程仍在運行")
        else:
            print("✅ 多線程提示音測試完成")
            
    except Exception as e:
        print(f"❌ 多線程測試失敗: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🎵 IndexTTS 提示音功能測試")
    print("請確保您的音響設備正常工作")
    
    # 測試提示音
    if test_indextts_prompt_audio():
        print("\n🎉 基本提示音功能測試完成！")
        
        # 測試錄音提示音
        if test_recording_prompt_audio():
            print("\n🎉 錄音提示音測試完成！")
        else:
            print("\n⚠️ 錄音提示音測試失敗")
        
        # 測試多線程
        if test_threaded_prompt_audio():
            print("\n🎉 所有測試通過！IndexTTS 提示音功能正常工作")
        else:
            print("\n⚠️ 多線程測試失敗，但基本功能正常")
    else:
        print("\n❌ 提示音功能測試失敗")
        sys.exit(1) 