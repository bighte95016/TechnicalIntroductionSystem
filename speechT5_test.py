from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Config
import torch
import soundfile as sf
from transformers import logging

# 設置調試日誌級別
logging.set_verbosity_debug()

try:
    # 使用本地路徑加載模型
    processor = SpeechT5Processor.from_pretrained("/root/autodl-tmp/speecht5_tts", local_files_only=True)
    print("Processor loaded successfully.")

    model = SpeechT5ForTextToSpeech.from_pretrained("/root/autodl-tmp/speecht5_tts", local_files_only=True)
    print("TTS model loaded successfully.")

    vocoder = SpeechT5HifiGan.from_pretrained("/root/autodl-tmp/microsoft/speecht5_hifigan", local_files_only=True)
    print("Vocoder loaded successfully.")

    # 標記輸入文本
    inputs = processor(text="please go straight! ", return_tensors="pt")

    # 獲取模型配置中的 speaker embedding 維度
    config = SpeechT5Config.from_pretrained("/root/autodl-tmp/speecht5_tts", local_files_only=True)
    speaker_embeddings = torch.randn(1, config.speaker_embedding_dim)

    # 生成語音的聲譜圖
    spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)

    # 將聲譜圖對像傳遞給 generate_speech 時，它會直接輸出語音波形
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    # 保存生成的語音
    sf.write("tts_example.wav", speech.numpy(), samplerate=16000)

    print("Speech synthesis completed successfully.")
except Exception as e:
    print(f"An error occurred: {e}")
