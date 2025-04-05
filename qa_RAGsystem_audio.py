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

# --- 在所有 import 之後，第一次訪問環境變數之前調用 load_dotenv --- #
load_dotenv()

# --- 基本設定 ---
PDF_DIRECTORY = "./technical_file/PDF"
VECTORSTORE_DIR = "./chroma_db" # Chroma DB 持久化儲存目錄

# --- LLM 設定 ---
LLM_PROVIDER = "ollama" # 可選 'ollama' 或 'openai'

# Ollama 設定(gemma3:12b、qwen2.5:7b、qwen2.5:3b)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"                  

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

# --- Whisper 模型設定 ---
WHISPER_MODEL_SIZE = "medium" # 可選: "tiny", "base", "small", "medium", "large"
whisper_model = None # 全局變量，用於存儲加載的Whisper模型

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
        chunk_size=500,  # 減小文字區塊大小以提高處理速度
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

# --- 5. 建立 QA 鏈 (使用 RetrievalQA) ---
def create_qa_chain(llm, vectorstore):
    """建立問答鏈"""
    print("正在建立 QA 鏈...")
    try:
        k=3
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        print(f"檢索器將檢索 top {k} 個區塊。")
        
        # 定義雙語 Prompt 模板
        template_zh = """
        你是一個介紹P-HUD(Panoramic HUD)的專家，請根據以下提供的上下文資訊，簡潔地回答問題。
        使用者的問題皆圍繞在P-HUD相關，
        如果你在提供的上下文中找不到答案，請明確說明你無法從文件中找到答案，不要嘗試編造或使用外部知識。
        你必須使用繁體中文回答。

        上下文：
        {context}

        問題：{question}

        繁體中文回答（請根據上下文簡潔回答）：
        """

        template_en = """
        You are an expert on P-HUD (Panoramic HUD). Please answer the question concisely based on the context information provided below.
        User questions are all related to P-HUD.
        If you cannot find the answer in the provided context, please clearly state that you cannot find the answer in the document. Do not make up information or use external knowledge.
        You must answer in English.

        Context:
        {context}

        Question: {question}

        English answer (please answer concisely based on the context):
        """
        
        # 創建兩種語言的提示模板
        ZH_CHAIN_PROMPT = PromptTemplate.from_template(template_zh)
        EN_CHAIN_PROMPT = PromptTemplate.from_template(template_en)
        
        # 默認使用中文提示
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff", # 使用 Stuff 方法
            return_source_documents=True,
            chain_type_kwargs={"prompt": ZH_CHAIN_PROMPT}
        )
        
        # 創建一個字典存儲不同語言的鏈
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
        return qa_chains
    except Exception as e:
        print(f"建立 QA 鏈時發生錯誤: {e}")
        return None

# --- 語音處理功能 ---
def record_audio(stop_event):
    """錄製音頻並保存為臨時文件"""
    audio = pyaudio.PyAudio()
    print("正在準備錄音...")
    
    # 打開音頻流
    stream = audio.open(format=AUDIO_FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    
    print("開始讀取問題...（按下Enter鍵停止讀取問題）")
    frames = []
    
    # 實時音量顯示
    try:
        while not stop_event.is_set():
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            
            # 顯示音量指示器（可選）
            audio_data = np.frombuffer(data, dtype=np.int16)
            volume = np.abs(audio_data).mean()
            bars = int(50 * volume / 4000)
            print("\r[" + "=" * bars + " " * (50 - bars) + "]", end="", flush=True)
            
    except KeyboardInterrupt:
        pass
    finally:
        print("\n問題讀取結束，正在保存...")
        
        # 停止和關閉流
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # 保存為WAV文件
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(AUDIO_FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        print(f"錄音已保存為 {WAVE_OUTPUT_FILENAME}")

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

# --- 主要執行流程 (僅使用語音版本) ---
if __name__ == "__main__":
    # --- 步驟 0: 清理舊的向量儲存 ---
    if os.path.exists(VECTORSTORE_DIR):
        print(f"偵測到舊的向量儲存目錄 {VECTORSTORE_DIR}，正在刪除...")
        try:
            shutil.rmtree(VECTORSTORE_DIR)
            print(f"目錄 {VECTORSTORE_DIR} 已成功刪除。")
        except OSError as e:
            print(f"刪除目錄 {VECTORSTORE_DIR} 時發生錯誤: {e}")
            print("請檢查權限或手動刪除該目錄。程式終止。")
            sys.exit(1)
    else:
        print(f"向量儲存目錄 {VECTORSTORE_DIR} 不存在，無需刪除。")

    # --- 步驟 1-3: 載入、分割文件並建立向量儲存 ---
    documents = load_documents(PDF_DIRECTORY)
    if not documents:
        print("無法載入文件，程式終止。")
        sys.exit(1) # 退出程式

    texts = split_documents(documents)
    if not texts:
        print("文件分割失敗，程式終止。")
        sys.exit(1)

    # 現在 create_vector_store 會返回兩個值
    vectorstore = create_vector_store(texts, VECTORSTORE_DIR) # <-- 只接收一個返回值
    if not vectorstore: # <-- 只檢查 vectorstore
        print("向量儲存建立失敗，程式終止。")
        sys.exit(1)

    # --- 步驟 4-5: 初始化 LLM 並建立 QA 鏈 ---
    llm = initialize_llm()
    if not llm:
        print("LLM 初始化失敗，程式終止。")
        sys.exit(1)

    qa_chains = create_qa_chain(llm, vectorstore)
    if not qa_chains:
        print("QA 鏈建立失敗，程式終止。")
        sys.exit(1)

    # --- 預載入Whisper模型 (可選) ---
    print("預載入語音識別模型（可能需要一些時間）...")
    load_whisper_model()

    # --- 步驟 6: 查詢迴圈 ---
    print("===================================")
    print(f" 語音問答系統已就緒 (LLM: {LLM_PROVIDER.upper()})！")
    print(" 系統將使用語音輸入模式")
    print(" 特殊指令 (輸入文字):")
    print(" - 輸入 'exit' 來結束程式")
    print(" - 輸入 'help' 顯示說明")
    print(f" 當前語音識別設定: 自動檢測語言")
    print(" 按下 Enter 開始錄音，再次按下 Enter 停止錄音")
    print("===================================")

    while True:
        try:
            # 顯示提示並等待用戶輸入
            command = input("\n按下 Enter 開始錄音，或輸入命令 > ")
            
            # 處理文字命令 (保留少數文字命令以方便操作)
            if command.strip().lower() == 'exit':
                print("正在結束程式...")
                break
                
            elif command.strip().lower() == 'help':
                print("===================================")
                print(" 語音問答系統使用說明:")
                print(" 1. 按下 Enter 開始問問題")
                print(" 2. 開始說話提問")
                print(" 3. 再次按下 Enter 問完問題")
                print(" 4. 系統將自動識別您的問題並以相同語言回答")
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

            # 根據偵測到的語言選擇對應的 QA 鏈
            # 將 Whisper 偵測的語言代碼映射到我們支持的語言
            lang_map = {
                "zh": "zh", "cn": "zh", "ja": "zh", "ko": "zh",  # 亞洲語言使用中文回答
                "en": "en", "fr": "en", "de": "en", "es": "en",  # 西方語言使用英文回答
            }
            
            # 默認使用中文回答
            answer_lang = lang_map.get(detected_lang, "zh")
            
            print(f"正在處理您的問題 (使用 {LLM_PROVIDER.upper()} LLM, 回答語言: {answer_lang})...")
            
            # 選擇對應語言的 QA 鏈
            # 如果沒有對應語言的鏈，則退回到中文
            qa_chain = qa_chains.get(answer_lang, qa_chains["zh"])
            
            # 調用 qa_chain.invoke
            result = qa_chain.invoke({"query": question})
            # 從結果中提取 'result'
            answer = result.get('result', '抱歉，無法生成答案。').strip()
            source_docs = result.get('source_documents', []) # 獲取來源文檔 (可選)

            print("\n答案：")
            print(answer)

            # 可選：顯示來源文件資訊 (保持註解)
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

        except KeyboardInterrupt: # 允許 Ctrl+C 中斷
             print("\n偵測到中斷指令，正在結束程式...")
             break
        except Exception as e:
            print(f"處理問題時發生未預期的錯誤: {e}")
            print("請檢查輸入、LLM 狀態或程式邏輯。")

    # 清理臨時文件
    if os.path.exists(WAVE_OUTPUT_FILENAME):
        os.remove(WAVE_OUTPUT_FILENAME)
        
    print("--- 語音問答系統已關閉 ---") 