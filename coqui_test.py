import os
import sys # <-- Make sure sys is imported
import torch
import sounddevice as sd
import soundfile as sf
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig


# 添加 XttsConfig 到安全白名單
torch.serialization.add_safe_globals([XttsConfig])


# --- Configuration ---
# Model Selection: Using a high-quality multilingual model
# This model requires a speaker reference audio file.
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# Speaker Reference Audio:
# !!! IMPORTANT !!!
# You MUST provide a short (5-15 seconds) clear audio recording (.wav format)
# of the desired speaker's voice. Save it as 'speaker.wav' in the same
# directory as this script, or change the path below.
SPEAKER_WAV_PATH = "speaker.wav" 

# Output file for synthesized speech
OUTPUT_WAV_PATH = "coqui_output.wav" # Reverted output filename

def synthesize_and_play(tts_model: TTS, text: str, speaker_wav: str, language: str, file_path: str):
    """
    Synthesizes text to speech using the provided XTTS model and plays it back.

    Args:
        tts_model: Initialized Coqui TTS object.
        text: The text to synthesize.
        speaker_wav: Path to the reference speaker audio file (.wav).
        language: Language code (e.g., 'en', 'zh-cn').
        file_path: Path to save the generated audio file.
    """
    print(f"\nSynthesizing ({language}): \"{text[:60]}...\"")
    try:
        # Synthesize using tts_to_file
        tts_model.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language=language,
            file_path=file_path,
            # split_sentences=True # Optional: Helps with longer texts
        )
        print(f"Audio saved to: {file_path}")

        # Play the generated audio
        print("Playing audio...")
        try:
            data, samplerate = sf.read(file_path)
            sd.play(data, samplerate)
            sd.wait()  # Wait until audio playback is finished
            print("Playback finished.")
        except Exception as play_error:
            print(f"Error playing audio file {file_path}: {play_error}")
            print("You might need to install an audio playback backend (e.g., PortAudio for sounddevice).")
            print(f"You can still find the generated audio at: {file_path}")

    except FileNotFoundError:
        print(f"Error: Speaker WAV file not found at '{speaker_wav}'")
        print("Please make sure the speaker reference audio file exists.")
    except Exception as synthesis_error:
        print(f"Error during TTS synthesis: {synthesis_error}")

if __name__ == "__main__":
    print("--- Coqui TTS (XTTS v2) Test ---") # Updated title

    # 1. Check if speaker WAV exists
    if not os.path.exists(SPEAKER_WAV_PATH):
        print(f"\nError: Speaker reference file '{SPEAKER_WAV_PATH}' not found!")
        print("Please create a short (5-15s) WAV file with the desired voice")
        print("and save it as 'speaker.wav' in this directory, then run the script again.")
        sys.exit(1) # Exit if speaker file is missing

    # 2. Check for GPU
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("\nGPU detected. Using GPU for TTS.")
    else:
        print("\nGPU not detected. Using CPU for TTS (might be slow).")

    # 3. Initialize TTS model
    # This will download the model files on the first run (can take time!)
    print(f"\nInitializing TTS model: {MODEL_NAME}...")
    tts = None # Initialize tts to None
    try:
        tts = TTS(model_name=MODEL_NAME, progress_bar=True, gpu=use_gpu)
        print("TTS model initialized successfully.")
    except ValueError as ve: # Catch the specific ValueError
        print(f"Error initializing TTS model: {ve}")
        if "Model file not found" in str(ve):
            print("This often means the local cache is corrupted or incomplete.")
            print("Try deleting the model cache directory and run again:")
            print(f"Cache dir likely in: ~/.local/share/tts/ or AppData/Local/tts/ for model {MODEL_NAME.replace('/', '--')}")
        elif "Weights only load failed" in str(ve):
             print("This is often due to incompatibility between the model file and your PyTorch version (>=2.6).")
             print("Try updating the TTS library: pip install TTS --upgrade")
             print("If updating TTS doesn't work, consider using a different model or managing PyTorch versions.")
        else:
             print("Please check your internet connection and required dependencies (PyTorch).")
        sys.exit(1)
    except Exception as model_init_error:
        print(f"Unexpected error initializing TTS model: {model_init_error}")
        sys.exit(1)
        
    # Ensure tts was initialized before proceeding
    if tts is None:
        print("TTS model could not be initialized. Exiting.")
        sys.exit(1)

    # 4. Define Test Texts
    english_text = "Hello, this is a test using Coqui TTS with the XTTS version 2 model. It should sound quite natural."
    chinese_text = "你好，这是一个使用 Coqui TTS 的 XTTS 第二版模型进行的中文语音合成测试。效果应该相当自然。"

    # 5. Run Synthesis and Playback Tests
    # Test English
    synthesize_and_play(
        tts_model=tts,
        text=english_text,
        speaker_wav=SPEAKER_WAV_PATH,
        language="en",
        file_path=OUTPUT_WAV_PATH
    )

    # Test Chinese
    synthesize_and_play(
        tts_model=tts,
        text=chinese_text,
        speaker_wav=SPEAKER_WAV_PATH,
        language="zh-cn", # Use 'zh-cn' for Mandarin Chinese with XTTS v2
        file_path=OUTPUT_WAV_PATH 
        # Overwrites the previous file, change path if you want to keep both
    )

    # 6. Optional Cleanup
    # try:
    #     if os.path.exists(OUTPUT_WAV_PATH):
    #         os.remove(OUTPUT_WAV_PATH)
    #         print(f"\nRemoved temporary output file: {OUTPUT_WAV_PATH}")
    # except Exception as e:
    #     print(f"Error removing output file: {e}")

    print("\n--- Coqui TTS Testing Complete ---")
