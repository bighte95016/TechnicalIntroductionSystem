import sys, pathlib# 把 ./index-tts 加進模組搜尋路徑
sys.path.append(str(pathlib.Path(__file__).parent / "index_tts"))
from index_tts.indextts.infer import IndexTTS
import numpy as np
import sounddevice as sd
import time

tts = IndexTTS(model_dir="./index_tts/checkpoints",cfg_path="./index_tts/checkpoints/config.yaml")
voice="./voice/Dr_Lee.wav"
text="大家好，往桃園的下一班次是14:20。"
text2="P-HUD是一種透明顯示投影技術!"

# 直接返回音頻數據（不保存到文件）
print("正在生成音頻...")
sampling_rate, wav_data = tts.infer(voice, text2, output_path=None)

print(f"音頻生成完成！採樣率: {sampling_rate} Hz")
print(f"音頻數據形狀: {wav_data.shape}")
print(f"音頻時長: {len(wav_data) / sampling_rate:.2f} 秒")

# 直接播放音頻
print("正在播放音頻...")
sd.play(wav_data, sampling_rate)
sd.wait()  # 等待播放完成
print("播放完成！")