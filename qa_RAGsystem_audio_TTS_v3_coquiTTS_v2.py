import os
import sys
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI # 新增 OpenAI LLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import shutil # 用於刪除目錄
from dotenv import load_dotenv # 新增 用於讀取 .env 文件
import numpy as np
import time  # 添加time模組用於計時

# 語音處理相關導入
import pyaudio
import wave
import threading
import whisper # 本地版本 Whisper
import torch # PyTorch

# TTS語音合成相關導入
# from gtts import gTTS      # Google Text-to-Speech - REMOVED
from TTS.api import TTS    # ADDED: Coqui TTS
from pydub import AudioSegment     # For loading audio files
import simpleaudio as sa           # For playing audio data
import tempfile
import speech_recognition as sr
import queue
from pathlib import Path

# --- 在所有 import 之後，第一次訪問環境變數之前調用 load_dotenv --- #
load_dotenv()

# --- 基本設定 ---
PDF_DIRECTORY = "./technical_file/PDF"
VECTORSTORE_DIR = "./chroma_db" # Chroma DB 持久化儲存目錄

# --- LLM 設定 ---
LLM_PROVIDER = "ollama" # 可選 'ollama' 或 'openai'

# Ollama 設定(gemma3:12b、qwen2.5:7b、qwen2.5:3b、ycchen/breeze-7b-instruct-v1_0 )
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "ycchen/breeze-7b-instruct-v1_0"                  

# OpenAI 設定
OPENAI_MODEL_NAME = "gpt-4-turbo" # 或其他您想使用的 GPT 模型
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # 從環境變數獲取

# --- 嵌入模型設定 (保持不變，仍然使用本地 Ollama) ---
EMBEDDING_MODEL = "mxbai-embed-large" # 使用較小的嵌入模型以提高速度

# --- 語音相關設定 ---
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # 採樣率
CHUNK = 1024  # 每次讀取的音訊片段大小
RECORD_SECONDS = 10  # 預設錄音時間，可由用戶中止
WAVE_OUTPUT_FILENAME = "temp_recording.wav"  # 臨時錄音文件
SPEECH_LANGUAGE = "auto"  # 默認為自動檢測語言，其他選項: "zh"為中文，"en"為英文

# --- TTS設定 ---
TTS_RATE = 175  # 語速 (words per minute)
TTS_VOLUME = 0.8  # 音量 (0.0 到 1.0)

# --- Whisper 模型設定 ---
WHISPER_MODEL_SIZE = "medium" # 可選: "tiny", "base", "small", "medium", "large"
whisper_model = None # 全局變量，用於存儲加載的Whisper模型

# --- Coqui TTS 設定 --- ADDED
# 可以選擇不同的 Coqui TTS 模型，XTTS v2 是個不錯的多語言選擇
# 查看可用模型: tts --list_models
COQUI_TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
# COQUI_TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2" # 備選中文模型
# COQUI_TTS_MODEL_NAME = "tts_models/zh-CN/baker/tacotron2-DDC-GST" # 備選中文模型
coqui_tts_model = None # 全局變量，用於存儲加載的 Coqui TTS 模型
playback_lock = threading.Lock()  # Lock for playback control

# TTS 模型性能調整參數
TTS_ENABLE_GPU = True  # 是否啟用 GPU 加速
TTS_ENABLE_CACHE = True  # 是否啟用音頻緩存
TTS_SAMPLE_RATE = 22050  # 音頻采樣率 (24000 是 XTTS 的原始值，22050 稍低但仍保持良好音質)

# 固定使用的說話人（可以根據偏好修改）
FIXED_ZH_SPEAKER = "Tammie Ema"  # 中文固定說話人     Tammie Ema
FIXED_EN_SPEAKER = "Tammie Ema"  # 英文固定說話人 Daisy Studious
AVAILABLE_SPEAKERS = []  # 全局變量，存儲可用的說話人列表，保留用於驗證固定說話人是否可用

# 音頻緩存字典 {(text, language, speaker): audio_data}
TTS_CACHE = {}  # 音頻緩存

# 終端文字樣式
BOLD = "\033[1m"
RESET = "\033[0m"

# --- TTS引擎初始化 ---
tts_lock = threading.Lock()  # 保留鎖

# 檢查 Python 版本
if sys.version_info < (3, 8):
    print("警告：您的 Python 版本較低，建議使用 Python 3.8 或更高版本以獲得最佳相容性。")

# --- 1. 載入文件 ---
def load_documents(directory):
    """從指定目錄載入 PDF 文件"""
    if not os.path.isdir(directory):
        print(f"錯誤：找不到目錄 {directory}")
        return None
    print(f"正在從目錄載入文件: {directory}")
    # 使用 show_progress=True 來顯示進度 (如果檔案較多)
    loader = DirectoryLoader(
        directory,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True, # 加速載入
        use_multithreading=True, # 加速載入
        silent_errors=True, # 忽略無法載入的檔案
    )
    try:
        documents = loader.load()
        print(f"成功載入 {len(documents)} 個 PDF 文件。")
        if not documents:
            print(f"警告：目錄 {directory} 中沒有找到 PDF 文件。")
            return []
        return documents
    except Exception as e:
        print(f"載入文件時發生錯誤: {e}")
        return []

# --- 2. 分割文件 ---
def split_documents(documents):
    """將文件分割成較小的文字區塊"""
    print("正在分割文件...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,  # 減小文字區塊大小以提高處理速度
        chunk_overlap=100, # 減小重疊以提高處理速度
        length_function=len,
        add_start_index=True, # 添加起始索引元數據
    )
    texts = text_splitter.split_documents(documents)
    print(f"文件被分割成 {len(texts)} 個區塊。")
    return texts

# --- 3. 建立向量儲存 ---
def create_vector_store(texts, persist_directory):
    """建立向量儲存 (強制重建) 並返回 vectorstore"""
    print("正在建立嵌入向量與向量儲存...")
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        print(f"使用嵌入模型: {EMBEDDING_MODEL}")

        # 由於我們總是在啟動時刪除，這裡直接建立新的
        print(f"正在於 {persist_directory} 建立新的向量儲存...")
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        print("正在持久化向量儲存...")
 #       vectorstore.persist() # 確保儲存
        print("向量儲存已成功建立。")
        return vectorstore # <-- 只返回 vectorstore
    except Exception as e:
        print(f"建立向量儲存時發生錯誤: {e}")
        print("請檢查 Ollama 服務是否正在運行，以及指定的嵌入模型是否可用。")
        return None # <-- 只返回 None

# --- 4. 初始化 LLM (根據 LLM_PROVIDER 進行切換) ---
def initialize_llm():
    """根據設定初始化選擇的 LLM"""
    if LLM_PROVIDER == "ollama":
        print(f"正在初始化本地 LLM: {OLLAMA_MODEL} (透過 Ollama: {OLLAMA_BASE_URL})")
        try:
            llm = OllamaLLM(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                #temperature=0.1,   # 調整溫度
                #stream=True,
                max_tokens=100    # 限制最大 token 數量
            )
            llm.invoke("你好") # 測試連接
            print(f"本地 LLM ({OLLAMA_MODEL}) 初始化成功並可連線。")
            return llm
        except Exception as e:
            print(f"初始化本地 LLM ({OLLAMA_MODEL}) 時發生錯誤: {e}")
            print("請確認 Ollama 服務正在運行且模型已下載並可用。")
            return None

    elif LLM_PROVIDER == "openai":
        # 檢查 OPENAI_API_KEY 是否存在 (load_dotenv 應已加載)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("錯誤：找不到 OpenAI API Key。請確保您已將 OPENAI_API_KEY 加入 .env 文件或設置為環境變數。")
            return None

        print(f"正在初始化 OpenAI LLM: {OPENAI_MODEL_NAME}")
        try:
            llm = ChatOpenAI(model_name=OPENAI_MODEL_NAME, temperature=0)
            # 可以在這裡添加一個簡單的測試調用，但會消耗 token
            # llm.invoke("Hello")
            print(f"OpenAI LLM ({OPENAI_MODEL_NAME}) 初始化成功。")
            return llm
        except Exception as e:
            print(f"初始化 OpenAI LLM ({OPENAI_MODEL_NAME}) 時發生錯誤: {e}")
            print("請檢查您的 API Key 是否有效以及網路連接。")
            return None

    else:
        print(f"錯誤：無效的 LLM_PROVIDER 設定 '{LLM_PROVIDER}'。請選擇 'ollama' 或 'openai'。")
        return None

# ★★★ 重新添加 record_audio 函數定義 ★★★
def record_audio(stop_event):
    """接收音頻並保存為臨時文件"""
    audio = pyaudio.PyAudio()
    print("正在準備接收問題...")
    
    # 打開音頻流
    stream = None # 初始化 stream 為 None
    try:
        stream = audio.open(format=AUDIO_FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
        
        print("開始讀取問題...（按下Enter鍵停止讀取問題）")
        frames = []
        
        # 實時音量顯示
        while not stop_event.is_set():
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                
                # 顯示音量指示器（可選）
                audio_data = np.frombuffer(data, dtype=np.int16)
                volume = np.abs(audio_data).mean()
                bars = int(50 * volume / 4000)
                print("\r[" + "=" * bars + " " * (50 - bars) + "]", end="", flush=True)
            except IOError as e:
                # 忽略 PyAudio 在某些情況下可能拋出的輸入溢出錯誤
                if e.errno == pyaudio.paInputOverflowed:
                    print("\n警告：音頻輸入溢出，忽略部分數據。", end="")
                else:
                    raise # 重新拋出其他IOError
                
    except KeyboardInterrupt:
        print("\n錄音被中斷。")
    except Exception as e:
        print(f"\n錄音過程中發生錯誤: {e}")
    finally:
        print("\n問題讀取結束，正在停止錄音並保存...")
        
        # 停止和關閉流
        if stream is not None:
            try:
                if stream.is_active(): # 僅在流活躍時停止
                    stream.stop_stream()
                stream.close()
            except Exception as e:
                print(f"關閉音頻流時出錯: {e}")
                
        audio.terminate()
        
        # 只有在錄到數據時才保存文件
        if frames:
            try:
                wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(AUDIO_FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                print(f"問題已保存為 {WAVE_OUTPUT_FILENAME}")
            except Exception as e:
                print(f"保存音頻文件時出錯: {e}")
        else:
            print("未錄製到有效音頻數據，不保存文件。")

# --- TTS 語音合成功能 (使用 Coqui TTS + pydub + simpleaudio) ---
def initialize_coqui_tts():
    """載入 Coqui TTS 模型並進行必要的預熱"""
    global coqui_tts_model, AVAILABLE_SPEAKERS
    if coqui_tts_model is None:
        print(f"正在初始化 Coqui TTS 引擎 (模型: {COQUI_TTS_MODEL_NAME})...")
        try:
            device = "cuda" if torch.cuda.is_available() and TTS_ENABLE_GPU else "cpu"
            print(f"使用 {'GPU' if device == 'cuda' else 'CPU'} 運行 Coqui TTS 模型")

            # --- ADDED: Fix for PyTorch >= 2.6 loading issue with XTTS --- #
            try:
                # Attempt to import the specific classes mentioned in the errors
                from TTS.tts.configs.xtts_config import XttsConfig
                from TTS.tts.models.xtts import XttsAudioConfig
                from TTS.config.shared_configs import BaseDatasetConfig
                # Attempt to import the FOURTH class from the latest error
                from TTS.tts.models.xtts import XttsArgs

                # Add ALL necessary classes to the safe globals list
                safe_classes = [XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs]
                torch.serialization.add_safe_globals(safe_classes)
                print(f"已將 {', '.join(c.__name__ for c in safe_classes)} 加入 PyTorch 安全全局列表。")
            except ImportError as imp_err:
                print(f"警告：無法導入 XTTS 或共享配置/參數類 ({imp_err})，如果您未使用 XTTS 模型或 PyTorch 版本較舊，可以忽略此訊息。")
            except Exception as safe_global_err:
                print(f"嘗試將 TTS 相關類加入安全全局列表時出錯: {safe_global_err}")
            # --- END OF ADDED FIX --- #

            # 配置並初始化 TTS 模型
            start_time = time.time()
            coqui_tts_model = TTS(model_name=COQUI_TTS_MODEL_NAME, gpu=(device == "cuda"))
            load_time = time.time() - start_time
            print(f"Coqui TTS 引擎加載耗時: {load_time:.2f} 秒")
            
            # --- 獲取並保存可用的說話人列表 --- #
            try:
                if hasattr(coqui_tts_model, 'synthesizer') and \
                   hasattr(coqui_tts_model.synthesizer, 'tts_model') and \
                   hasattr(coqui_tts_model.synthesizer.tts_model, 'speaker_manager') and \
                   hasattr(coqui_tts_model.synthesizer.tts_model.speaker_manager, 'speakers'):
                    
                    AVAILABLE_SPEAKERS = list(coqui_tts_model.synthesizer.tts_model.speaker_manager.speakers.keys())
                    print(f"可用的說話人列表: {AVAILABLE_SPEAKERS}")
                    print(f"默認中文說話人: {FIXED_ZH_SPEAKER}")
                    print(f"默認英文說話人: {FIXED_EN_SPEAKER}")
                    
                    # 驗證默認說話人是否可用
                    if FIXED_ZH_SPEAKER not in AVAILABLE_SPEAKERS:
                        print(f"警告: 默認中文說話人 '{FIXED_ZH_SPEAKER}' 不在可用列表中。將使用第一個可用說話人。")
                    if FIXED_EN_SPEAKER not in AVAILABLE_SPEAKERS:
                        print(f"警告: 默認英文說話人 '{FIXED_EN_SPEAKER}' 不在可用列表中。將使用第一個可用說話人。")
            except Exception as spk_err:
                print(f"獲取說話人列表時出錯: {spk_err}")
                AVAILABLE_SPEAKERS = []
            
            # --- 模型預熱，降低首次生成延遲 --- #
            try:
                print("進行 TTS 模型預熱...")
                warmup_start = time.time()
                
                # 選擇預熱說話人
                warmup_speaker = None
                if FIXED_ZH_SPEAKER in AVAILABLE_SPEAKERS:
                    warmup_speaker = FIXED_ZH_SPEAKER
                elif FIXED_EN_SPEAKER in AVAILABLE_SPEAKERS:
                    warmup_speaker = FIXED_EN_SPEAKER
                elif AVAILABLE_SPEAKERS:
                    warmup_speaker = AVAILABLE_SPEAKERS[0]
                
                # 進行短文本預熱
                if warmup_speaker:
                    # 中文預熱
                    warmup_text_zh = "這是一個預熱測試。"
                    zh_kwargs = {
                        "text": warmup_text_zh,
                        "language": "zh-cn"
                    }
                    if warmup_speaker:
                        zh_kwargs["speaker"] = warmup_speaker
                    _ = coqui_tts_model.tts(**zh_kwargs)
                    
                    # 英文預熱
                    warmup_text_en = "This is a warmup test."
                    en_kwargs = {
                        "text": warmup_text_en,
                        "language": "en"
                    }
                    if warmup_speaker:
                        en_kwargs["speaker"] = warmup_speaker
                    _ = coqui_tts_model.tts(**en_kwargs)
                
                warmup_time = time.time() - warmup_start
                print(f"TTS 模型預熱完成，耗時: {warmup_time:.2f} 秒")
            except Exception as warmup_err:
                print(f"模型預熱過程中出錯: {warmup_err}")
                print("繼續執行，但首次生成可能較慢。")
            
            print("Coqui TTS 引擎初始化成功。")
            return True
        except Exception as e:
            print(f"初始化 Coqui TTS 引擎時發生錯誤: {e}")
            print("請確保 TTS 庫已正確安裝，且模型文件存在或可下載。")
            print("如果使用 GPU，請確認 CUDA 環境配置正確。")
            # --- ADDED: Print traceback for detailed error info --- #
            import traceback
            print("詳細錯誤追蹤：")
            traceback.print_exc()
            # --- END OF ADDED TRACEBACK --- #
            coqui_tts_model = None # Ensure it's None if init failed
            return False
    return True # Already initialized

def text_to_speech(text, language):
    """使用 Coqui TTS 生成語音數據並播放，無需保存臨時文件或進行分段處理
    
    參數:
        text (str): 要播放的文字內容
        language (str): 語言代碼，'zh' 表示中文，'en' 表示英文
    """
    global coqui_tts_model, AVAILABLE_SPEAKERS, TTS_CACHE
    if not coqui_tts_model:
        print("錯誤：Coqui TTS 模型未初始化，無法生成語音。")
        return False

    if not text or len(text.strip()) == 0:
        print("警告：嘗試播放空文本")
        return False

    # Map our language codes ('zh', 'en') to Coqui TTS language codes if needed
    # XTTS v2 often works well with auto-detect or simple codes like 'zh-cn', 'en'
    coqui_lang = 'zh-cn' if language == 'zh' else 'en'
    
    # 根據語言選擇固定的說話人
    if language == 'zh':
        selected_speaker = FIXED_ZH_SPEAKER if FIXED_ZH_SPEAKER in AVAILABLE_SPEAKERS else (AVAILABLE_SPEAKERS[0] if AVAILABLE_SPEAKERS else None)
    else:  # 'en'
        selected_speaker = FIXED_EN_SPEAKER if FIXED_EN_SPEAKER in AVAILABLE_SPEAKERS else (AVAILABLE_SPEAKERS[0] if AVAILABLE_SPEAKERS else None)
    
    # 如果沒有可用的說話人，提示錯誤
    if not selected_speaker:
        print("錯誤: 找不到可用的說話人。")
        return False

    # 生成緩存鍵
    cache_key = (text, coqui_lang, selected_speaker)
    
    print(f"使用 Coqui TTS ({coqui_lang}, 說話人: {selected_speaker}) 播放: {text[:30]}...")
    with playback_lock:
        print(f"準備使用 Coqui TTS 生成音頻...")
        play_success = False
        
        try:
            # 檢查緩存中是否已有此文本的音頻
            if TTS_ENABLE_CACHE and cache_key in TTS_CACHE:
                print("從緩存中獲取音頻...")
                wav = TTS_CACHE[cache_key]
                print(f"成功從緩存獲取 {len(wav)} 樣本的音頻")
            else:
                # 直接生成完整文本的音頻數據（不保存到文件，不分段處理）
                print(f"生成完整音頻數據中...")
                
                # 設置 TTS 參數
                tts_kwargs = {
                    "text": text,
                    "language": coqui_lang,
                    "speaker": selected_speaker
                }
                
                # 直接獲取音頻波形數據
                start_time = time.time()
                wav = coqui_tts_model.tts(**tts_kwargs)
                tts_time = time.time() - start_time
                print(f"音頻生成完成，耗時: {tts_time:.2f} 秒")
                
                # 緩存生成的音頻
                if TTS_ENABLE_CACHE:
                    TTS_CACHE[cache_key] = wav
            
            # 獲取采樣率
            sample_rate = TTS_SAMPLE_RATE  # 使用設定的采樣率
            if hasattr(coqui_tts_model, 'synthesizer') and hasattr(coqui_tts_model.synthesizer, 'output_sample_rate'):
                sample_rate = coqui_tts_model.synthesizer.output_sample_rate
            
            print(f"音頻數據已準備，采樣率: {sample_rate} Hz, 長度: {len(wav)} 樣本")
            
            # 通過 simpleaudio 直接播放
            try:
                # 將浮點數組轉換為 int16 格式 (simpleaudio 需要 int16 或 int8)
                # 首先確保數據範圍在 [-1, 1] 內
                wav = np.clip(wav, -1.0, 1.0)
                # 然後縮放到 int16 的範圍 [-32768, 32767]
                wav_int16 = (wav * 32767).astype(np.int16)
                
                # 播放音頻
                print("使用 simpleaudio 播放...")
                play_obj = sa.play_buffer(
                    wav_int16,
                    num_channels=1,  # XTTS 通常輸出單聲道
                    bytes_per_sample=2,  # 2 bytes for int16
                    sample_rate=sample_rate
                )
                
                print("等待播放完成...")
                play_obj.wait_done()
                print("播放完成。")
                play_success = True
                
            except sa.exceptions.SimpleaudioError as sa_error:
                print(f"Simpleaudio 播放時發生錯誤: {sa_error}")
                print("可能是音頻設備問題或 simpleaudio 配置問題。")
                import traceback
                traceback.print_exc()
                return False
                
        except Exception as e:
            print(f"TTS 生成或播放過程中發生錯誤: {e}")
            import traceback
            traceback.print_exc()
        
        # --- Release Lock ---
        print("釋放播放鎖...")
        return play_success

def load_whisper_model():
    """載入Whisper模型"""
    global whisper_model
    
    if whisper_model is None:
        print(f"正在載入本地 Whisper 模型 ({WHISPER_MODEL_SIZE})...")
        try:
            # 檢查是否有GPU可用
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                print(f"使用 GPU 加速 Whisper 模型")
            else:
                print(f"未檢測到 GPU，使用 CPU 運行 Whisper 模型")
                
            whisper_model = whisper.load_model(WHISPER_MODEL_SIZE, device=device)
            print(f"Whisper 模型載入成功")
        except Exception as e:
            print(f"載入 Whisper 模型時發生錯誤: {e}")
            return False
    
    return True

def speech_to_text():
    """使用本地Whisper將語音轉換為文字，並返回偵測到的語言"""
    try:
        if not os.path.exists(WAVE_OUTPUT_FILENAME):
            return "", "zh"  # 默認返回中文作為語言，空字串作為文字
        
        # 載入Whisper模型
        if not load_whisper_model():
            return "", "zh"
            
        print(f"正在使用本地Whisper模型轉換語音為文字 (語言: {'自動檢測' if SPEECH_LANGUAGE == 'auto' else SPEECH_LANGUAGE})...")
        
        # 使用Whisper模型進行語音識別
        transcribe_options = {
            "fp16": torch.cuda.is_available()  # 如果有GPU，使用fp16加速
        }
        
        # 只有在非自動模式下才指定語言
        if SPEECH_LANGUAGE != "auto":
            transcribe_options["language"] = SPEECH_LANGUAGE
        
        # 使用動態參數調用transcribe
        result = whisper_model.transcribe(
            WAVE_OUTPUT_FILENAME,
            **transcribe_options
        )
            
        # 刪除臨時音頻文件
        if os.path.exists(WAVE_OUTPUT_FILENAME):
            os.remove(WAVE_OUTPUT_FILENAME)
            
        text = result["text"].strip()
        detected_language = result.get("language", "zh")  # 獲取偵測到的語言代碼，默認中文
        print(f"\n語音識別結果 (檢測到的語言: {detected_language})：「{text}」")
        return text, detected_language
    
    except Exception as e:
        print(f"語音轉文字時發生錯誤: {e}")
        if os.path.exists(WAVE_OUTPUT_FILENAME):
            os.remove(WAVE_OUTPUT_FILENAME)
        return "", "zh"  # 錯誤時默認返回中文作為語言

def map_whisper_language_to_supported(detected_lang):
    """將 Whisper 檢測的語言代碼映射到我們支持的語言"""
    lang_map = {
        "zh": "zh", "cn": "zh", "ja": "zh", "ko": "zh",  # 亞洲語言使用中文回答
        "en": "en", "fr": "en", "de": "en", "es": "en", "my": "en",  # 西方語言使用英文回答
    }
    # 默認使用中文回答
    return lang_map.get(detected_lang, "zh")

def process_user_query(question, qa_chain, language):
    """處理用戶查詢並返回答案"""
    print(f"正在處理您的問題 (使用 {LLM_PROVIDER.upper()} LLM, 回答語言: {language})...")
    
    # 調用 qa_chain.invoke
    result = qa_chain.invoke({"query": question})
    # 從結果中提取 'result'
    answer = result.get('result', '抱歉，無法生成答案。').strip()
    source_docs = result.get('source_documents', []) # 獲取來源文檔 (可選)
    
    print("\n答案：")
    print(answer)
    
    # 可選：顯示來源文件資訊
    if source_docs:
        print("--- 參考來源片段 ---")
        seen_sources = set()
        for i, doc in enumerate(source_docs):
            source = doc.metadata.get('source', '未知來源')
            page = doc.metadata.get('page', -1)
            source_key = f"{source}_p{page}"
            if source_key not in seen_sources:
                print(f"  來源: {os.path.basename(source)}, 頁碼: {page + 1 if page != -1 else '未知'}")
                content_preview = doc.page_content[:150].replace('\n', ' ') + "..."
                print(f"    片段: {content_preview}")
                seen_sources.add(source_key)
        print("--------------------")
    
    return answer

# 添加用於保存和播放問候音頻的函數
def generate_and_save_greeting_audio():
    """生成並保存問候音頻，然後播放"""
    global coqui_tts_model, AVAILABLE_SPEAKERS, TTS_CACHE
    
    if not coqui_tts_model:
        print("錯誤：Coqui TTS 模型未初始化，無法生成問候音頻。")
        return False
    
    # 準備保存目錄
    audio_dir = Path("audio_files")
    if not audio_dir.exists():
        audio_dir.mkdir(parents=True)
    
    # 問候語音文件路徑
    greeting_path = audio_dir / "greeting_hello.wav"
    
    # 檢查問候語音文件是否已存在
    if greeting_path.exists():
        print(f"問候語音文件已存在於 {greeting_path}，直接播放...")
        # 使用wave模塊讀取現有的音頻文件
        try:
            with wave.open(str(greeting_path), 'rb') as wf:
                # 獲取音頻參數
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                
                # 將二進制音頻數據轉換為整數數組
                audio_int16 = np.frombuffer(frames, dtype=np.int16)
                
                # 播放音頻
                play_obj = sa.play_buffer(
                    audio_int16,
                    num_channels=channels,
                    bytes_per_sample=sample_width,
                    sample_rate=sample_rate
                )
                play_obj.wait_done()
                return True
        except Exception as e:
            print(f"播放現有問候語音文件時發生錯誤: {e}")
            # 如果播放失敗，則重新生成
    
    print("生成問候語音「您好！」...")
    
    # 根據系統設置的中文說話人生成音頻
    selected_speaker = FIXED_ZH_SPEAKER if FIXED_ZH_SPEAKER in AVAILABLE_SPEAKERS else (AVAILABLE_SPEAKERS[0] if AVAILABLE_SPEAKERS else None)
    
    if not selected_speaker:
        print("錯誤: 找不到可用的說話人。")
        return False
    
    # 生成問候語音
    try:
        tts_kwargs = {
            "text": "!Hello！",
            "language": "zh-cn",
            "speaker": selected_speaker
        }
        
        wav = coqui_tts_model.tts(**tts_kwargs)
        
        # 檢查 wav 的類型，並確保它是 numpy 數組
        print(f"檢查音頻數據類型: {type(wav)}")
        if isinstance(wav, list):
            # 如果是列表，首先檢查第一個元素是否就是我們需要的音頻數據
            if len(wav) == 1 and isinstance(wav[0], np.ndarray):
                wav = wav[0]
            else:
                # 嘗試將列表轉換為 numpy 數組
                try:
                    wav = np.array(wav, dtype=np.float32)
                except:
                    print(f"警告：無法將列表轉換為 numpy 數組，列表結構：{[type(x) for x in wav]}")
                    if len(wav) > 0:
                        # 使用第一個元素如果可能
                        wav = np.array(wav[0], dtype=np.float32) if isinstance(wav[0], (list, np.ndarray)) else np.array(wav[0])
        
        # 確保 wav 是 numpy 數組
        if not isinstance(wav, np.ndarray):
            print(f"錯誤：音頻數據不是 numpy 數組，而是 {type(wav)}")
            # 進一步檢查和處理
            if hasattr(wav, '__array__'):  # 檢查是否可以轉換為 numpy 數組
                wav = np.array(wav)
            else:
                print("無法處理的音頻數據類型，嘗試另一種方法生成問候語音")
                return False
        
        # 將浮點音頻轉換為int16
        wav_int16 = (wav * 32767).astype(np.int16)
        
        # 使用wave模塊保存為WAV文件
        with wave.open(str(greeting_path), 'wb') as wf:
            wf.setnchannels(1)  # 單聲道
            wf.setsampwidth(2)  # 2字節 (16位)
            wf.setframerate(TTS_SAMPLE_RATE)
            wf.writeframes(wav_int16.tobytes())
        
        print(f"成功保存問候語音到 {greeting_path}")
        
        # 播放剛剛生成的音頻
        play_obj = sa.play_buffer(
            wav_int16,
            num_channels=1,
            bytes_per_sample=2,
            sample_rate=TTS_SAMPLE_RATE
        )
        play_obj.wait_done()
        return True
    
    except Exception as e:
        print(f"生成或保存問候語音時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

# 添加直接播放WAV文件的函數
def play_wav_file(wav_file_path):
    """直接播放WAV文件"""
    try:
        if not os.path.exists(wav_file_path):
            print(f"錯誤：找不到音頻文件 {wav_file_path}")
            return False
            
        print(f"正在播放音頻文件: {wav_file_path}")
        
        # 使用wave模塊讀取音頻文件
        with wave.open(wav_file_path, 'rb') as wf:
            # 獲取音頻參數
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            
            # 將二進制音頻數據轉換為整數數組
            audio_int16 = np.frombuffer(frames, dtype=np.int16)
            
            # 播放音頻
            play_obj = sa.play_buffer(
                audio_int16,
                num_channels=channels,
                bytes_per_sample=sample_width,
                sample_rate=sample_rate
            )
            play_obj.wait_done()
            print("音頻播放完成")
            return True
    except Exception as e:
        print(f"播放音頻文件時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主程序入口"""
    global qa_chains
    qa_chains = {}
    
    # 記錄是否已播放過問候音頻
    greeting_played = False
    
    # 初始化模型和文檔處理器
    print("歡迎使用語音問答系統！")
    print("初始化中... 請稍候")
    print("正在初始化文件載入器和模型，這可能需要一些時間...")
    
    try:
        # 初始化 Coqui TTS 引擎
        if not initialize_coqui_tts(): # Call the new init function
             print("TTS 引擎初始化失敗，將無法播放語音。")
             # Decide if you want to exit or continue without TTS
             # return # Exit if TTS is critical

        # --- 步驟 0: 清理舊的向量儲存 ---
        if os.path.exists(VECTORSTORE_DIR):
            print(f"偵測到舊的向量儲存目錄 {VECTORSTORE_DIR}，正在刪除...")
            try:
                shutil.rmtree(VECTORSTORE_DIR)
                print(f"目錄 {VECTORSTORE_DIR} 已成功刪除。")
            except OSError as e:
                print(f"刪除目錄 {VECTORSTORE_DIR} 時發生錯誤: {e}")
                print("請檢查權限或手動刪除該目錄。程式終止。")
                return
        else:
            print(f"向量儲存目錄 {VECTORSTORE_DIR} 不存在，無需刪除。")
        
        # --- 步驟 1-3: 載入、分割文件並建立向量儲存 ---
        documents = load_documents(PDF_DIRECTORY)
        if not documents:
            print("無法載入文件，程式終止。")
            return
        
        texts = split_documents(documents)
        if not texts:
            print("文件分割失敗，程式終止。")
            return
        
        vectorstore = create_vector_store(texts, VECTORSTORE_DIR)
        if not vectorstore:
            print("向量儲存建立失敗，程式終止。")
            return
        
        # --- 步驟 4: 初始化 LLM ---
        llm = initialize_llm()
        if not llm:
            print("LLM 初始化失敗，程式終止。")
            return
        
        # --- 步驟 5: 建立QA鏈 ---
        print("正在建立 QA 鏈...")
        k=3
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        print(f"檢索器將檢索 top {k} 個區塊。")
        
        # 定義雙語 Prompt 模板 Panoramic HUD
        #回答：「這個問題不在我的回答範疇，請詢問一旁的專家」
        template_zh = """
        你是一個介紹全景抬頭式顯示器的工研院專家，
        請根據以下提供的「技術資訊」，簡潔地回答問題。
        使用者的問題皆圍繞在P-HUD相關，
        如果你在提供的技術資訊中找不到答案或相關性不高，回答：「這個問題不在我的回答範疇，請詢問一旁的專家」
        你必須使用繁體中文回答。

        技術資訊：
        {context}

        問題：{question}

        繁體中文回答（請根據技術資訊盡可能簡潔回答，小於25字），並將P-HUD皆改成全景抬頭式顯示器：
        """

        template_en = """
        You are an expert from the ITRI specializing in introducing Panoramic Head-Up Displays(P-HUD). 
        Based on the "technical information" provided below, please answer questions succinctly. 
        All user questions revolve around topics related to P-HUD. 
        If you cannot find the answer or relevant information in the provided technical details, 
        please respond with: "This question is outside the scope of my expertise. Please consult the expert next to you."
        You must answer in English.

        Technical information:
        {context}

        Question: {question}

        English answer (Please answer as concisely as possible based on the technical information, within 25 words.), and replace P-HUD with P-H-U-D:
        """
        
        # 創建兩種語言的提示模板
        ZH_CHAIN_PROMPT = PromptTemplate.from_template(template_zh)
        EN_CHAIN_PROMPT = PromptTemplate.from_template(template_en)
        
        # 建立多語言 QA 鏈
        qa_chains = {
            "zh": RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                return_source_documents=True,
                chain_type_kwargs={"prompt": ZH_CHAIN_PROMPT}
            ),
            "en": RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                return_source_documents=True,
                chain_type_kwargs={"prompt": EN_CHAIN_PROMPT}
            )
        }
        
        print("多語言 QA 鏈建立成功。")
        
        # --- 預載入Whisper模型 (可選) ---
        print("預載入語音識別模型（可能需要一些時間）...")
        load_whisper_model()
        
        print("\n系統初始化完成！準備接受語音輸入。")
        print("按下 Enter 開始錄音，再次按下 Enter 停止錄音...")

        # --- 步驟 6: 查詢迴圈 ---
        print("===================================")
        print(f" 全語音問答系統已就緒 (LLM: {LLM_PROVIDER.upper()})！")
        print(" 系統將使用語音輸入和語音輸出 (TTS: Coqui TTS)") # Updated TTS info
        print(" 特殊指令 (輸入文字):")
        print(" - 輸入 'exit' 來結束程式")
        print(" - 輸入 'help' 顯示說明")
        print(f" 當前語音識別設定: 自動檢測語言")
        print(" 按下 Enter 開始問問題，再次按下 Enter 結束問問題")
        print("===================================")
        
        # 指定問候音頻文件路徑
        greeting_audio_path = "./audio_files/please_speak.wav"
        
        while True:
            try:
                # 顯示提示並等待用戶輸入
                command = input("\n按下 Enter 開始問問題，或輸入命令 > ")
                
                # 處理文字命令 (保留少數文字命令以方便操作)
                if command.strip().lower() == 'exit':
                    print("正在結束程式...")
                    break
                    
                elif command.strip().lower() == 'help':
                    print("===================================")
                    print(" 全語音問答系統使用說明:")
                    print(" 1. 按下 Enter 開始問問題")
                    print(" 2. 開始說話提問")
                    print(" 3. 再次按下 Enter 問完問題")
                    print(" 4. 系統將自動識別您的問題並以相同語言回答")
                    print(" 5. 系統將以語音回答 (使用 Coqui TTS)") 
                    print(" 特殊文字命令:")
                    print(" - 'exit': 結束程式")
                    print(" - 'help': 顯示此說明")
                    print("===================================")
                    continue
                    
                # 播放問候音頻（替代開始錄音的步驟）
                print("播放問候音頻...")
                play_wav_file(greeting_audio_path)
                
                # 開始錄音
                print(f"{BOLD}開始錄音...{RESET} (再次按下 Enter 停止)")
                # 創建停止事件
                stop_event = threading.Event()
                
                # 創建錄音線程
                record_thread = threading.Thread(target=record_audio, args=(stop_event,))
                record_thread.start()
                
                # 等待用戶按下Enter鍵停止錄音
                input("按下Enter鍵，結束問問題...")
                stop_event.set()
                record_thread.join()
                
                # 語音轉文字，並獲取偵測到的語言
                question, detected_lang = speech_to_text()
                if not question:
                    print("無法識別語音，請重試。")
                    continue
                
                # 處理識別出的文字，檢查是否為退出命令
                if question.lower().strip() in ["退出", "結束", "exit", "quit"]:
                    print("語音指令: 退出程式")
                    print("正在結束程式...")
                    break
                    
                # 忽略空問題
                if not question.strip():
                    continue

                # 將 Whisper 偵測的語言代碼映射到我們支持的語言
                lang = map_whisper_language_to_supported(detected_lang)

                # 創建一個隊列來存儲LLM的回答結果
                answer_queue = queue.Queue()

                # 定義LLM處理函數
                def process_query_in_background():
                    try:
                        # 選擇對應的 QA 鏈
                        qa_chain = qa_chains.get(lang, qa_chains["zh"])
                        answer = process_user_query(question, qa_chain, lang)
                        answer_queue.put(("success", answer))
                    except Exception as e:
                        print(f"LLM處理過程中發生錯誤: {e}")
                        answer_queue.put(("error", str(e)))

                # 創建一個非阻塞的提示音播放函數
                def play_prompt_audio():
                    # 這裡創建一個新的鎖，避免與主要playback_lock衝突
                    prompt_lock = threading.Lock()
                    with prompt_lock:
                        try:
                            # 使用簡單的提示音
                            prompt_text = "感謝您的提問，我思考一下請稍候!"
                            tts_kwargs = {
                                "text": prompt_text,
                                "language": "zh-cn",
                                "speaker": FIXED_ZH_SPEAKER if FIXED_ZH_SPEAKER in AVAILABLE_SPEAKERS else (AVAILABLE_SPEAKERS[0] if AVAILABLE_SPEAKERS else None)
                            }
                            
                            if not tts_kwargs["speaker"]:
                                print("警告：找不到可用的說話人，無法播放提示音")
                                return
                                
                            # 直接使用TTS模型生成音頻
                            if coqui_tts_model:
                                wav = coqui_tts_model.tts(**tts_kwargs)
                                
                                # 獲取采樣率
                                sample_rate = TTS_SAMPLE_RATE
                                if hasattr(coqui_tts_model, 'synthesizer') and hasattr(coqui_tts_model.synthesizer, 'output_sample_rate'):
                                    sample_rate = coqui_tts_model.synthesizer.output_sample_rate
                                    
                                # 播放音頻
                                wav = np.clip(wav, -1.0, 1.0)
                                wav_int16 = (wav * 32767).astype(np.int16)
                                play_obj = sa.play_buffer(
                                    wav_int16,
                                    num_channels=1,
                                    bytes_per_sample=2,
                                    sample_rate=sample_rate
                                )
                                play_obj.wait_done()
                        except Exception as e:
                            print(f"播放提示音時發生錯誤: {e}")

                # 啟動LLM處理線程
                print("啟動LLM處理線程...")
                llm_thread = threading.Thread(target=process_query_in_background)
                llm_thread.daemon = True
                llm_thread.start()
                
                # 啟動提示音播放線程（使用獨立的playback機制，不阻塞main的playback_lock）
                print("啟動提示音播放線程...")
                prompt_thread = threading.Thread(target=play_prompt_audio)
                prompt_thread.daemon = True
                prompt_thread.start()
                
                # 等待LLM處理完成
                print("等待LLM處理結果...")
                llm_thread.join()
                
                # 獲取處理結果
                status, result = answer_queue.get()
                
                # 如果提示音仍在播放，試圖優雅地停止它
                if prompt_thread.is_alive():
                    print("提示音仍在播放，但我們將繼續處理回答...")
                    # 注意：我們不再等待提示音完成，而是直接繼續
                
                if status == "success":
                    answer = result
                    
                    # 立即播放回答，不等待提示音
                    print(f"啟動線程播放回答 {lang} 語音 (使用 Coqui TTS)... ")
                    answer_thread = threading.Thread(
                        target=text_to_speech,
                        args=(answer, lang)
                    )
                    answer_thread.start()
                    
                    # 等待回答語音播放完成
                    print("等待回答語音播放完成...")
                    answer_thread.join() 
                    print("回答語音播放線程結束。")
                    
                    # 在控制台顯示問答
                    print(f"\n問題：{question}")
                    print(f"回答：{answer}\n")
                else:
                    error_message = f"處理查詢時發生錯誤: {result}"
                    print(error_message)
                    
                    # 播放錯誤提示
                    print("正在播放錯誤提示 (使用 Coqui TTS)... ")
                    error_lang = "zh" if lang == "zh" else "en"
                    error_text = "處理您的問題時發生了錯誤，請重試。" if error_lang == "zh" else "An error occurred while processing your question. Please try again."
                    
                    error_thread = threading.Thread(
                        target=text_to_speech,
                        args=(error_text, error_lang)
                    )
                    error_thread.start()
                    error_thread.join()
                    print("錯誤提示播放完成。")
                
            except KeyboardInterrupt: # 允許 Ctrl+C 中斷
                print("\n偵測到中斷指令，正在結束程式...")
                break
            except Exception as e:
                print(f"處理問題時發生未預期的錯誤: {e}")
                print("請檢查輸入、LLM 狀態或程式邏輯。")

        # 清理臨時文件
        if os.path.exists(WAVE_OUTPUT_FILENAME):
            os.remove(WAVE_OUTPUT_FILENAME)

        print("--- 全語音問答系統已關閉 ---")

    except Exception as e:
        print(f"主程序執行時發生錯誤: {e}")
        print("請檢查輸入、LLM 狀態或程式邏輯。")

if __name__ == "__main__":
    main() 