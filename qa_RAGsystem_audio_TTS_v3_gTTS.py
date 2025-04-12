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

# 語音處理相關導入
import pyaudio
import wave
import threading
import numpy as np
import time
import whisper # 本地版本 Whisper
import torch # PyTorch

# TTS語音合成相關導入
from gtts import gTTS      # Google Text-to-Speech
from pydub import AudioSegment     # ADDED: For loading audio files
import simpleaudio as sa           # ADDED: For playing audio data
import tempfile             
import numpy as np # pydub might implicitly need numpy sometimes, ensure imported
import speech_recognition as sr
import queue

# --- 在所有 import 之後，第一次訪問環境變數之前調用 load_dotenv --- #
load_dotenv()

# --- 基本設定 ---
PDF_DIRECTORY = "./technical_file/PDF"
VECTORSTORE_DIR = "./chroma_db" # Chroma DB 持久化儲存目錄

# --- LLM 設定 ---
LLM_PROVIDER = "ollama" # 可選 'ollama' 或 'openai'

# Ollama 設定(gemma3:12b、qwen2.5:7b、qwen2.5:3b、phi4、phi4-mini)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:12b"                  

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

# --- TTS設定 (pyttsx3 specific rate/volume removed) ---
# TTS_RATE = 175  # 語速 (words per minute) - REMOVED
# TTS_VOLUME = 0.8  # 音量 (0.0 到 1.0) - REMOVED

# --- Whisper 模型設定 ---
WHISPER_MODEL_SIZE = "medium" # 可選: "tiny", "base", "small", "medium", "large"
whisper_model = None # 全局變量，用於存儲加載的Whisper模型

# --- TTS引擎初始化 ---
# No global engine needed for gTTS
# tts_engine = None # REMOVED
playback_lock = threading.Lock()  # Renamed lock for playback control

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
        show_progress=True,
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
                max_tokens=200    # 限制最大 token 數量
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

# --- TTS 語音合成功能 (使用 gTTS + pydub + simpleaudio) --- 
def initialize_tts_engine():
    """Placeholder function, no complex initialization needed for gTTS/pydub/simpleaudio.
       Could add a check for FFmpeg here if desired.
    """
    print("TTS 使用 gTTS (在線服務) 生成，pydub/simpleaudio 播放，無需特定初始化。")
    # Optional: Check for FFmpeg availability needed by pydub for non-wav formats
    # try:
    #     AudioSegment.converter = "path/to/ffmpeg.exe" # Or ensure it's in PATH
    #     AudioSegment.ffmpeg = "path/to/ffmpeg.exe"
    #     AudioSegment.ffprobe ="path/to/ffprobe.exe"
    #     # A simple check like loading a tiny dummy mp3 might work
    #     print("FFmpeg check (basic): OK") 
    # except Exception as ffmpeg_err:
    #     print(f"Warning: FFmpeg might not be found or configured correctly for pydub: {ffmpeg_err}")
    #     print("MP3 playback might fail. Ensure FFmpeg is installed and in PATH.")
    return True

def text_to_speech(text, language):
    """使用 gTTS 生成語音, pydub 載入, simpleaudio 播放"""
    if not text or len(text.strip()) == 0:
        print("警告：嘗試播放空文本")
        return False

    # Map internal language codes to gTTS language codes
    # Prioritize Traditional Chinese for 'zh' based on prompt content
    gtts_lang = 'zh-tw' if language == 'zh' else 'en'

    print(f"等待播放鎖以使用 gTTS 播放 {gtts_lang} 語音: {text[:30]}...")
    with playback_lock: # Assuming playback_lock is defined globally
        print(f"獲取播放鎖，準備使用 gTTS ({gtts_lang})...")
        temp_filename = None # Initialize to None
        play_success = False
        try:
            # 1. Create gTTS object and save to temporary MP3
            tts = gTTS(text=text, lang=gtts_lang, slow=False)
            # Using NamedTemporaryFile ensures it's automatically cleaned up (usually, but we handle explicit delete too)
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as fp:
                temp_filename = fp.name
                print(f"Saving gTTS audio to temporary file: {temp_filename}")
                tts.save(temp_filename)
            print(f"gTTS 音訊已儲存。")

            # 2. Load the temporary MP3 file using pydub
            #    Requires FFmpeg to be installed and in PATH!
            print(f"使用 pydub 載入: {temp_filename}")
            # Explicitly tell pydub where FFmpeg might be if not in PATH (Optional)
            # AudioSegment.converter = "C:/path/to/ffmpeg/bin/ffmpeg.exe" # Example
            audio = AudioSegment.from_mp3(temp_filename)
            print("pydub 載入成功。")

            # 3. Prepare data for simpleaudio
            samples = np.array(audio.get_array_of_samples()) # Convert to numpy array
            num_channels = audio.channels
            bytes_per_sample = audio.sample_width
            sample_rate = audio.frame_rate
            print(f"Audio details: Channels={num_channels}, Rate={sample_rate}, Bytes={bytes_per_sample}")

            # Ensure correct dtype if needed (simpleaudio expects native endianness)
            if audio.sample_width == 2:
                samples = samples.astype(np.int16)
            elif audio.sample_width == 1:
                samples = samples.astype(np.int8)
            # Add more types if necessary (e.g., int32)

            # 4. Play using simpleaudio
            print("使用 simpleaudio 播放...")
            play_obj = sa.play_buffer(
                samples, 
                num_channels=num_channels, 
                bytes_per_sample=bytes_per_sample, 
                sample_rate=sample_rate
            )
            
            # Wait for playback to finish
            print("等待 simpleaudio 播放完成...")
            play_obj.wait_done()
            print("simpleaudio 播放完成。")
            play_success = True # Mark playback as successful

        except FileNotFoundError as ffmpeg_error:
            # This specific error is often related to FFmpeg not found by pydub
            print(f"錯誤: pydub 無法處理 MP3 - 很可能未找到 FFmpeg/FFprobe. {ffmpeg_error}")
            print("請確保 FFmpeg 已安裝、其 bin 目錄已加入系統 PATH，並重新啟動終端/IDE。")
        except sa.exceptions.SimpleaudioError as sa_error:
             print(f"Simpleaudio 播放時發生錯誤: {sa_error}")
             print("可能是音頻設備問題或 simpleaudio 配置問題。")
        except Exception as e:
            # Catch other potential errors from gTTS, pydub, simpleaudio
            print(f"TTS 或播放過程中發生錯誤: {e}")
        finally:
            # --- Cleanup --- Ensure temporary file is always deleted
            if temp_filename and os.path.exists(temp_filename):
                print(f"準備清理臨時文件: {temp_filename}")
                try:
                    os.remove(temp_filename)
                    print(f"已刪除臨時文件: {temp_filename}")
                except OSError as e_del:
                    print(f"警告：清理過程中無法刪除臨時文件 {temp_filename}: {e_del}")
            # --- Release Lock --- 
            print("釋放播放鎖...")
            # Return True only if playback seemed successful
            # We might return False even if gTTS worked but playback failed
            return play_success 

def load_whisper_model():
    """載入Whisper模型"""
    global whisper_model
    
    if whisper_model is None:
        print(f"正在載入本地 Whisper 模型 ({WHISPER_MODEL_SIZE})...")
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"使用 {'GPU' if device == 'cuda' else 'CPU'} 運行 Whisper 模型")
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
        "en": "en", "fr": "en", "de": "en", "es": "en",  # 西方語言使用英文回答
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

def main():
    """主程序入口"""
    # 初始化模型和文檔處理器
    # global tts_engine # Removed
    global qa_chains # Keep qa_chains global if needed elsewhere, or make local
    qa_chains = {}
    
    print("歡迎使用語音問答系統！")
    print("正在載入模型和處理文檔...")
    
    try:
        # 初始化TTS引擎 (Now just a placeholder or check)
        initialize_tts_engine() # Call the placeholder/checker
        
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
        
        # ★★★ 修改：打印 Chunk 內容並同時寫入 TXT 文件 ★★★
        output_txt_filename = "chunks_output.txt"
        print(f"\n{ '='*30 } 打印分割後的 Chunks (同時寫入 {output_txt_filename}) { '='*30 }")
        try:
            with open(output_txt_filename, 'w', encoding='utf-8') as f_out:
                f_out.write(f"{'='*30} 分割後的 Chunks {'='*30}\n\n") # 文件標題
                for i, doc in enumerate(texts):
                    chunk_header = f"--- Chunk {i+1} ---"
                    print(chunk_header)
                    f_out.write(chunk_header + "\n")
                    
                    # 嘗試打印/寫入來源和頁碼元數據
                    source = doc.metadata.get('source', '未知來源')
                    page = doc.metadata.get('page', -1)
                    start_index = doc.metadata.get('start_index', '未知索引')
                    metadata_line = f"來源: {os.path.basename(source)}, 頁碼: {page + 1 if isinstance(page, int) and page != -1 else '未知'}, 起始索引: {start_index}"
                    print(metadata_line)
                    f_out.write(metadata_line + "\n")
                    
                    print("內容:")
                    f_out.write("內容:\n")
                    
                    print(doc.page_content)
                    f_out.write(doc.page_content + "\n") # 寫入內容
                    
                    separator_line = "-"*60
                    print(separator_line + "\n")
                    f_out.write(separator_line + "\n\n") # 寫入分隔符
                
                end_message = f"{'='*30} Chunks 打印/寫入結束 {'='*30}"
                print(end_message + "\n")
                f_out.write(end_message + "\n")
                
        except IOError as e:
            print(f"錯誤：無法寫入文件 {output_txt_filename}: {e}")
            # 即使寫入失敗，仍然繼續執行後續步驟
            
        # ★★★ 處理結束 ★★★
        
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
        k=2
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        print(f"檢索器將檢索 top {k} 個區塊。")
        
        # 定義雙語 Prompt 模板 Panoramic HUD
        #回答：「這個問題不在我的回答範疇，請詢問一旁的專家」
        template_zh = """
        你是一個介紹全景抬頭式顯示器(P-HUD)的工研院專家，
        請根據以下提供的「技術資訊」，簡潔地回答問題。
        使用者的問題皆圍繞在P-HUD相關，
        如果你在提供的技術資訊中找不到答案或相關性不高，回答：「這個問題不在我的回答範疇，請詢問一旁的專家」
        你必須使用繁體中文回答。

        技術資訊：
        {context}

        問題：{question}

        繁體中文回答（請根據技術資訊簡潔回答），並將P-HUD皆改成P-H-U-D：
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

        English answer (please answer concisely based on the Technical information), and replace P-HUD with P-H-U-D:
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
        print(" 系統將使用語音輸入和語音輸出 (TTS: gTTS)") # Updated TTS info
        print(" 特殊指令 (輸入文字):")
        print(" - 輸入 'exit' 來結束程式")
        print(" - 輸入 'help' 顯示說明")
        print(f" 當前語音識別設定: 自動檢測語言")
        print(" 按下 Enter 開始問問題，再次按下 Enter 結束問問題")
        print("===================================")

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
                    print(" 5. 系統將以語音回答 (使用 Google TTS)")
                    print(" 特殊文字命令:")
                    print(" - 'exit': 結束程式")
                    print(" - 'help': 顯示此說明")
                    print("===================================")
                    continue
                    
                # 如果不是特殊命令，則開始語音輸入 (無論用戶輸入什麼或只是按Enter)
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

                # 將提示音的播放放到背景線程中
                print("啟動背景線程播放提示音 (使用 gTTS)... ")
                prompt_thread = threading.Thread(
                    target=text_to_speech, # Uses the new gTTS+playsound function
                    args=("請稍等一下，我思考一下這個問題", "zh"), # Prompt text
                    daemon=True # 設置為守護線程，這樣即使它沒播完主程序也能退出
                )
                prompt_thread.start()
                # 主線程不等待 prompt_thread，立即繼續執行

                # 將 Whisper 偵測的語言代碼映射到我們支持的語言
                lang = map_whisper_language_to_supported(detected_lang)
                
                # 選擇對應的 QA 鏈
                qa_chain = qa_chains.get(lang, qa_chains["zh"])
                
                try:
                    # ★★★ 處理問題與提示音播放並行 ★★★
                    answer = process_user_query(question, qa_chain, lang)
                    
                    # 將回答的播放也放到獨立線程中
                    print(f"啟動線程播放回答 {lang} 語音 (使用 gTTS)... ")
                    answer_thread = threading.Thread(
                        target=text_to_speech, # Uses the new gTTS+playsound function
                        args=(answer, lang)
                    )
                    answer_thread.start()
                    
                    # ★★★ 等待回答語音播放完成 ★★★
                    print("等待回答語音播放完成...")
                    answer_thread.join() 
                    print("回答語音播放線程結束。")
                    
                    # 在控制台顯示問答 (在語音之後顯示)
                    print(f"\n問題：{question}")
                    print(f"回答：{answer}\n")
                    
                except Exception as e:
                    error_message = f"處理查詢時發生錯誤: {e}"
                    print(error_message)
                    
                    # 播放錯誤提示（也放入線程並等待）
                    print("正在播放錯誤提示 (使用 gTTS)... ")
                    error_lang = "zh" if lang == "zh" else "en"
                    error_text = "處理您的問題時發生了錯誤，請重試。" if error_lang == "zh" else "An error occurred while processing your question. Please try again."
                    
                    error_thread = threading.Thread(
                        target=text_to_speech, # Uses the new gTTS+playsound function
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