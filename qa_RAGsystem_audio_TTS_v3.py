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
import pyttsx3  # 用於文字轉語音
import speech_recognition as sr
import queue

# --- 在所有 import 之後，第一次訪問環境變數之前調用 load_dotenv --- #
load_dotenv()

# --- 基本設定 ---
PDF_DIRECTORY = "./technical_file/PDF"
VECTORSTORE_DIR = "./chroma_db" # Chroma DB 持久化儲存目錄

# --- LLM 設定 ---
LLM_PROVIDER = "ollama" # 可選 'ollama' 或 'openai'

# Ollama 設定(gemma3:12b、qwen2.5:7b、qwen2.5:3b)
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

# --- TTS設定 ---
TTS_RATE = 200  # 語速 (words per minute)
TTS_VOLUME = 0.8  # 音量 (0.0 到 1.0)

# --- Whisper 模型設定 ---
WHISPER_MODEL_SIZE = "medium" # 可選: "tiny", "base", "small", "medium", "large"
whisper_model = None # 全局變量，用於存儲加載的Whisper模型

# --- TTS引擎初始化 ---
# tts_engine = None # 不再需要全局引擎
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
        chunk_size=300,  # 減小文字區塊大小以提高處理速度
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
                #max_tokens=300    # 限制最大 token 數量
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

# --- TTS 語音合成功能 ---
def initialize_tts_engine():
    """檢查TTS可用性並打印語音信息"""
    # 不再初始化全局引擎，僅做檢查
    with tts_lock: # 鎖定以防萬一（雖然主要在主線程調用）
        temp_engine = None
        try:
            print("檢查TTS可用性...")
            temp_engine = pyttsx3.init() # 創建臨時引擎進行檢查
            if temp_engine is None:
                 print("錯誤：無法初始化臨時TTS引擎進行檢查。")
                 return False
            temp_engine.setProperty('rate', TTS_RATE)
            temp_engine.setProperty('volume', TTS_VOLUME)
            voices = temp_engine.getProperty('voices')
            print(f"檢測到 {len(voices)} 個語音引擎")
            for i, voice in enumerate(voices):
                languages = voice.languages if hasattr(voice, 'languages') else ['未知']
                print(f"語音 #{i}: {voice.name} - 語言: {languages}")
            
            print("TTS可用性檢查完成。")
            # 可以在這裡加一個短暫的測試說話，但可能不是必須
            # temp_engine.say("TTS Check OK")
            # temp_engine.runAndWait()
            return True
        except Exception as e:
            print(f"檢查 TTS 可用性時發生錯誤: {e}")
            print("請確保您已安裝pyttsx3及其依賴")
            return False
        finally:
            # 確保臨時引擎被停止（如果它成功初始化了）
            if temp_engine is not None:
                try:
                    # 嘗試停止，但不強制要求成功，因為runAndWait可能已處理
                    pass # 暫時不調用 stop()，觀察效果
                except Exception as stop_e:
                    print(f"停止臨時檢查引擎時出錯: {stop_e}")

def text_to_speech(text, language):
    """創建新引擎實例並在鎖保護下播放語音"""
    # global tts_engine # 不再引用全局引擎
    if not text or len(text.strip()) == 0:
        print("警告：嘗試播放空文本")
        return False

    # 使用鎖確保同一時間只有一個TTS引擎實例在運行 runAndWait
    print(f"等待TTS鎖以播放 {language} 語音: {text[:30]}...")
    with tts_lock:
        print(f"獲取TTS鎖，準備播放 {language} 語音...")
        engine = None # 確保在 try 之前 engine 是 None
        try:
            print(f"創建新的TTS引擎實例...")
            engine = pyttsx3.init() 
            if engine is None:
                print("錯誤：無法初始化新的TTS引擎實例。")
                # 釋放鎖（雖然with語句會自動處理，但明確一下）
                return False
                
            engine.setProperty('rate', TTS_RATE)
            engine.setProperty('volume', TTS_VOLUME)
            print(f"引擎實例 {id(engine)} 創建並配置完成。")
            
            # 根據語言選擇合適的語音
            voices = engine.getProperty('voices')
            target_voice_id = None
            if language == "zh":
                chinese_voices = [v for v in voices if any(lang.startswith('zh') for lang in getattr(v, 'languages', []))]
                if chinese_voices:
                    target_voice_id = chinese_voices[0].id
            else:  
                english_voices = [v for v in voices if any(lang.startswith('en') for lang in getattr(v, 'languages', []))]
                if english_voices:
                    target_voice_id = english_voices[0].id
            
            if target_voice_id:
                try:
                    engine.setProperty('voice', target_voice_id)
                    # print(f"引擎 {id(engine)} 已設置語音為: {target_voice_id}")
                except Exception as voice_error:
                    print(f"引擎 {id(engine)} 設置語音時出錯: {voice_error}")
            # else:
                # print(f"引擎 {id(engine)} 未找到特定語言語音，使用默認。")
                
            # 播放語音
            engine.say(text)
            print(f"引擎 {id(engine)} 調用 runAndWait()...")
            engine.runAndWait()
            print(f"引擎 {id(engine)} runAndWait() 完成。")
            
            # 考慮是否需要 engine.stop()。runAndWait 後引擎狀態可能不確定。
            # 暫時省略 stop()，因為它本身也可能引發問題。
            
            print(f"釋放TTS鎖...")
            return True
        except RuntimeError as e:
             print(f"文字轉語音時發生 RuntimeError (引擎 {id(engine)}): {e}")
             # 這裡可以考慮是否需要 stop()，但鑑於引擎狀態未知，可能不安全
             print(f"釋放TTS鎖 (因 RuntimeError)...")
             return False
        except Exception as e:
            print(f"文字轉語音時發生未預期錯誤 (引擎 {id(engine)}): {e}")
            # 同上，對 stop() 保持謹慎
            print(f"釋放TTS鎖 (因 Exception)...")
            return False
        # finally:
            # with 語句會自動釋放鎖，無需手動處理
            # print("TTS 鎖已釋放 (finally)")

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
    global tts_engine, qa_chains
    qa_chains = {}
    
    print("歡迎使用語音問答系統！")
    print("正在載入模型和處理文檔...")
    
    try:
        # 初始化TTS引擎
        initialize_tts_engine()
        
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
        k=2
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        print(f"檢索器將檢索 top {k} 個區塊。")
        
        # 定義雙語 Prompt 模板 Panoramic HUD
        template_zh = """
        你是一個介紹全景抬頭式顯示器(P-HUD)的專家，請根據以下提供的技術資訊，簡潔地回答問題。
        使用者的問題皆圍繞在P-HUD相關，
        如果你在提供的技術資訊中找不到答案，不要嘗試編造或使用外部知識。
        你必須使用繁體中文回答。

        技術資訊：
        {context}

        問題：{question}

        繁體中文回答（請根據技術資訊簡潔回答）：
        """

        template_en = """
        You are an expert on P-HUD (Panoramic HUD). Please answer the question concisely based on the Technical information provided below.
        User questions are all related to P-HUD.
        If you cannot find the answer in the provided Technical information, please clearly state that you cannot find the answer in the Technical information. Do not make up information or use external knowledge.
        You must answer in English.

        Technical information:
        {context}

        Question: {question}

        English answer (please answer concisely based on the Technical information):
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
        print(" 系統將使用語音輸入和語音輸出")
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
                    print(" 5. 系統將以語音回答")
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
                print("啟動背景線程播放提示音...")
                prompt_thread = threading.Thread(
                    target=text_to_speech,
                    args=("請稍等一下，我思考一下這個問題", "zh"),
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
                    print(f"啟動線程播放回答 {lang} 語音...")
                    answer_thread = threading.Thread(
                        target=text_to_speech,
                        args=(answer, lang)
                        # daemon 不設置或設為 False，因為我們需要等待它完成
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
                    print("正在播放錯誤提示...")
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

        print("--- 全語音問答系統已關閉 ---")

    except Exception as e:
        print(f"主程序執行時發生錯誤: {e}")
        print("請檢查輸入、LLM 狀態或程式邏輯。")

if __name__ == "__main__":
    main() 