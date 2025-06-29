from TTS.api import TTS
import torch, sounddevice as sd, soundfile as sf

# --- 針對 PyTorch >= 2.6 加載 XTTS 的問題修復 --- #
try:
    # 嘗試導入特定類別
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsAudioConfig
    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.models.xtts import XttsArgs

    # 將所有必要類別加入安全全局列表
    safe_classes = [XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs]
    torch.serialization.add_safe_globals(safe_classes)
    print(f"已將 {', '.join(c.__name__ for c in safe_classes)} 加入 PyTorch 安全全局列表。")
except ImportError as imp_err:
    print(f"警告：無法導入 XTTS 或共享配置/參數類 ({imp_err})，如果您未使用 XTTS 模型或 PyTorch 版本較舊，可以忽略此訊息。")
except Exception as safe_global_err:
    print(f"嘗試將 TTS 相關類加入安全全局列表時出錯: {safe_global_err}")

# 載入模型（建議 gpu=True）
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# 說話者音訊檔案路徑
speaker_wav = "./voice/speaker_oreo1min.wav"

# XTTS 的標準採樣率是 24000 Hz
SAMPLE_RATE = 24000

def say(text: str, lang: str = "zh"):
    print(f"正在合成音訊: {text}")
    wav = tts.tts(
        text=text,
        language=lang,
        speaker_wav=speaker_wav,     # 直接提供說話人音訊檔案
        temperature=0.7             # 可微調語氣隨機度
    )
    # 使用固定的採樣率，而不是嘗試從 tts 對象獲取
    sd.play(wav, samplerate=SAMPLE_RATE, blocking=True)

# 測試合成與播放
print("開始測試 TTS...")
say("Rovii是一家公司，Rovii賣咖啡")
say("cafe很好喝")
say("大家早安，列車即將進站！")
#say("大家早安，列車即將進站！下一班往桃園的列車在十分鐘後發車。")