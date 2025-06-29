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
from langchain.schema import Document
import re
# 新增：BM25和混合檢索相關imports
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import pickle
import jieba  # 中文分詞
import threading  # 新增：線程處理
import time  # 新增：時間處理

# 新增：語音處理相關導入
import pyaudio  # 音頻錄製
import wave  # 音頻文件處理
import whisper  # Whisper 語音識別
import torch  # PyTorch
import numpy as np  # 數值計算
import queue  # 線程間通信

# 新增：TTS語音合成相關導入
from TTS.api import TTS    # Coqui TTS
from pydub import AudioSegment     # For loading audio files
import simpleaudio as sa           # For playing audio data
import tempfile
from pathlib import Path

# --- 在所有 import 之後，第一次訪問環境變數之前調用 load_dotenv --- #
load_dotenv()

# --- 基本設定 ---
PDF_DIRECTORY = "./technical_file/PDF"
VECTORSTORE_DIR = "./chroma_db" # Chroma DB 持久化儲存目錄
BM25_CACHE_DIR = "./bm25_cache" # BM25檢索器快取目錄

# --- LLM 設定 ---
LLM_PROVIDER = "ollama" # 可選 'ollama' 或 'openai'

# Ollama 設定(gemma3:12b、qwen2.5:7b、qwen2.5:3b)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:12b"                  

# OpenAI 設定
OPENAI_MODEL_NAME = "gpt-4-turbo" # 或其他您想使用的 GPT 模型

# --- 嵌入模型設定 (保持不變，仍然使用本地 Ollama) ---
EMBEDDING_MODEL = "mxbai-embed-large" # 使用較小的嵌入模型以提高速度

# --- 混合檢索設定 ---
USE_HYBRID_RETRIEVAL = True  # 是否使用混合檢索
DENSE_WEIGHT = 0.5  # Dense retriever權重 (0.0-1.0)
SPARSE_WEIGHT = 0.5  # BM25 retriever權重 (0.0-1.0)
RETRIEVAL_K = 5  # 統一的檢索數量設定
SIMILARITY_THRESHOLD = 0.05  # 統一的相似度閾值設定 (0.0-1.0，越高越嚴格)

# --- 問答分離設定 ---
USE_QA_SEPARATION = True  # 是否將問題和答案分離存儲（問題用於檢索，答案存在metadata中）

# --- 新增：語音相關設定 ---
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # 採樣率
CHUNK = 1024  # 每次讀取的音訊片段大小
RECORD_SECONDS = 10  # 預設錄音時間，可由用戶中止
WAVE_OUTPUT_FILENAME = "temp_recording.wav"  # 臨時錄音文件

# --- Whisper 模型設定 ---
WHISPER_MODEL_SIZE = "medium"  # 可選: "tiny", "base", "small", "medium", "large"
SPEECH_LANGUAGE = "auto"  # 默認為自動檢測語言，其他選項: "zh"為中文，"en"為英文
whisper_model = None  # 全局變量，用於存儲加載的Whisper模型

# --- 語音功能開關 ---
ENABLE_VOICE_INPUT = True  # 是否啟用語音輸入功能

# --- 新增：自動靜音檢測設定 ---
SILENCE_THRESHOLD = 500  # 靜音閾值（音量低於此值視為靜音）
SILENCE_DURATION = 2.0   # 連續靜音時間（秒），超過此時間自動停止錄音
MIN_RECORD_DURATION = 1.0  # 最小錄音時間（秒），避免過短錄音

# --- 新增：CoquiTTS設定 ---
# 可以選擇不同的 Coqui TTS 模型，XTTS v2 是個不錯的多語言選擇
# 查看可用模型: tts --list_models
COQUI_TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
coqui_tts_model = None  # 全局變量，用於存儲加載的 Coqui TTS 模型
playback_lock = threading.Lock()  # Lock for playback control

# TTS 模型性能調整參數
TTS_ENABLE_GPU = True  # 是否啟用 GPU 加速
TTS_ENABLE_CACHE = True  # 是否啟用音頻緩存
TTS_SAMPLE_RATE = 22050  # 音頻采樣率

# 固定使用的說話人（可以根據偏好修改）
FIXED_ZH_SPEAKER = "Tammie Ema"  # 中文固定說話人
FIXED_EN_SPEAKER = "Tammie Ema"  # 英文固定說話人
AVAILABLE_SPEAKERS = []  # 全局變量，存儲可用的說話人列表

# 音頻緩存字典 {(text, language, speaker): audio_data}
TTS_CACHE = {}  # 音頻緩存

# --- 語音輸出功能開關 ---
ENABLE_VOICE_OUTPUT = True  # 是否啟用語音輸出功能

# --- 提示音功能開關 ---
ENABLE_PROMPT_AUDIO = True  # 是否啟用提示音功能（在用戶提問後播放友好提示）

# 檢查 Python 版本
if sys.version_info < (3, 8):
    print("警告：您的 Python 版本較低，建議使用 Python 3.8 或更高版本以獲得最佳相容性。")


# --- 新增: 按段落處理PDF文件 ---
def load_pdf_by_paragraph(pdf_path: str):
    """讀 PDF → 清理軟換行 → 以行首 Qxx 切塊 → 分離問答"""
    print(f"[load] {pdf_path}")
    try:
        # 使用 PyPDF2 直接讀取完整 PDF
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        full_text = "\n".join(p.extract_text() or "" for p in reader.pages)
        
        # 獲取基本元數據
        base_md = {
            "source": pdf_path,
            "file_name": os.path.basename(pdf_path)
        }

        # 換行正規化
        # a) 把所有非常規換行(LS‧PS‧CR‧VT‧FF) → \n
        full_text = full_text.translate(
            dict.fromkeys(map(ord, "\r\u2028\u2029\v\f"), "\n")
        )
        # b) 只把「真正單行軟換行」換成空格
        full_text = re.sub(r"(?<!\n)\n(?!\s*\n)", " ", full_text)
        # c) 把 3 行以上空白壓成 2 行
        full_text = re.sub(r"\n{3,}", "\n\n", full_text)
        # d) 開頭補 \n，方便偵測第一題
        if not full_text.startswith("\n"):
            full_text = "\n" + full_text

        # 以行首 Qxx 切塊
        q_head = re.compile(r"\n\s*Q\s*\d+\s*[：:]", re.IGNORECASE)
        starts = [m.start()+1 for m in q_head.finditer(full_text)] + [len(full_text)]
        
        # 如果找到問題標記
        if starts and len(starts) > 1:
            if USE_QA_SEPARATION:
                print(f"  ➜ 偵測到 Q 格式，以問題標記切分並分離問答")
            else:
                print(f"  ➜ 偵測到 Q 格式，以問題標記切分（不分離問答）")
            chunks = [
                full_text[starts[i]:starts[i+1]].strip()
                for i in range(len(starts)-1)
            ]
            print(f"  ➜ 找到 {len(chunks)} 個 QA 組合")
            
            # 根據配置決定是否分離問題和答案
            docs = []
            for i, chunk in enumerate(chunks):
                if USE_QA_SEPARATION:
                    # 嘗試分離問題和答案
                    question, answer = separate_question_answer(chunk)
                    
                    if question and answer:
                        # 只將問題存入 page_content，答案存入 metadata
                        doc = Document(
                            page_content=question,  # 只存問題用於檢索
                            metadata={
                                **base_md, 
                                "paragraph_index": i, 
                                "type": "qa_separated",
                                "answer": answer,  # 答案存在metadata中
                                "original_qa": chunk  # 保留原始QA組合以備調試
                            }
                        )
                        docs.append(doc)
                        print(f"    Q{i+1}: 問題({len(question)}字) + 答案({len(answer)}字)")
                    else:
                        # 如果無法分離，退回到原始方式
                        doc = Document(
                            page_content=chunk,
                            metadata={**base_md, "paragraph_index": i, "type": "qa_pair"}
                        )
                        docs.append(doc)
                        print(f"    Q{i+1}: 無法分離問答，使用原始格式")
                else:
                    # 不分離，使用原始QA組合
                    doc = Document(
                        page_content=chunk,
                        metadata={**base_md, "paragraph_index": i, "type": "qa_pair"}
                    )
                    docs.append(doc)
                    print(f"    Q{i+1}: 使用原始QA組合格式")
        else:
            # 退回到依雙換行切段（一般文件）
            print("  ➜ 未偵測到問題標記，依雙換行切分")
            paras = [p.strip() for p in full_text.split("\n\n") if p.strip()]
            docs = [Document(
                page_content=p,
                metadata={**base_md, "paragraph_index": i, "type": "paragraph"}
            ) for i, p in enumerate(paras)]

        print(f"  ✓ 共產生 {len(docs)} 個 chunk")
        return docs
    except Exception as e:
        print(f"處理 {pdf_path} 時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return []

def separate_question_answer(qa_chunk: str):
    """分離問題和答案"""
    try:
        # 多種分離模式嘗試
        patterns = [
            # 模式1: Q數字: 問題內容 A數字: 答案內容
            r'^(Q\s*\d+\s*[：:]\s*.*?)\s+(A\s*\d+\s*[：:]\s*.*)$',
            # 模式2: Q數字: 問題內容 A: 答案內容  
            r'^(Q\s*\d+\s*[：:]\s*.*?)\s+(A\s*[：:]\s*.*)$',
            # 模式3: 以 A 或 答 開頭分割
            r'^(.*?)\s+((?:A\s*[：:]|答\s*[：:]|答案\s*[：:]).*?)$',
            # 模式4: 尋找明顯的答案開始標記
            r'^(.*?)\s+((?:答案|解答|回答)[：:].*?)$'
        ]
        
        for pattern in patterns:
            match = re.match(pattern, qa_chunk.strip(), re.DOTALL | re.IGNORECASE)
            if match:
                question = match.group(1).strip()
                answer = match.group(2).strip()
                
                # 清理問題：移除Q標記，只保留問題內容
                question = re.sub(r'^Q\s*\d+\s*[：:]\s*', '', question, flags=re.IGNORECASE).strip()
                
                # 清理答案：移除A標記，只保留答案內容
                answer = re.sub(r'^(?:A\s*\d*\s*[：:]|答\s*[：:]|答案\s*[：:]|解答\s*[：:]|回答\s*[：:])\s*', '', answer, flags=re.IGNORECASE).strip()
                
                # 驗證分離結果
                if len(question) > 5 and len(answer) > 10:  # 基本長度檢查
                    return question, answer
        
        # 如果所有模式都失敗，嘗試簡單的按行分割
        lines = qa_chunk.strip().split('\n')
        if len(lines) >= 2:
            # 假設第一行是問題，其餘是答案
            question_line = lines[0]
            answer_lines = lines[1:]
            
            # 清理問題
            question = re.sub(r'^Q\s*\d+\s*[：:]\s*', '', question_line, flags=re.IGNORECASE).strip()
            
            # 組合答案
            answer = '\n'.join(answer_lines).strip()
            # 清理答案開頭的標記
            answer = re.sub(r'^(?:A\s*\d*\s*[：:]|答\s*[：:]|答案\s*[：:])\s*', '', answer, flags=re.IGNORECASE).strip()
            
            if len(question) > 5 and len(answer) > 10:
                return question, answer
        
        return None, None
        
    except Exception as e:
        print(f"分離問答時發生錯誤: {e}")
        return None, None

# --- 1. 載入文件 ---
def load_documents(directory):
    """從指定目錄載入 PDF 文件並按段落分割"""
    if not os.path.isdir(directory):
        print(f"錯誤：找不到目錄 {directory}")
        return []
    
    print(f"正在從目錄載入文件: {directory}")
    
    # 獲取所有PDF文件的路徑
    pdf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    if not pdf_files:
        print(f"警告：目錄 {directory} 中沒有找到 PDF 文件。")
        return []
    
    print(f"發現 {len(pdf_files)} 個 PDF 文件...")
    
    # 處理每個PDF文件，直接按段落分割
    all_documents = []
    for pdf_path in pdf_files:
        print(f"處理文件: {os.path.basename(pdf_path)}")
        chunks = load_pdf_by_paragraph(pdf_path)
        all_documents.extend(chunks)
        print(f"  從 {os.path.basename(pdf_path)} 中提取了 {len(chunks)} 個段落")
    
    print(f"成功載入並處理 {len(pdf_files)} 個 PDF 文件，共提取 {len(all_documents)} 個段落。")
    return all_documents

# --- 2. 分割文件 (不再需要額外分割，因為已經在載入時按段落分割) ---
def split_documents(documents):
    """由於已在加載階段按段落分割，此函數只進行簡單處理"""
    print("文檔已在載入階段按段落分割，進行最終處理...")
    
    # 進行簡單的過濾，例如去除過短的段落
    filtered_docs = []
    for doc in documents:
        # 過濾太短的段落（例如少於10個字符的）
        if len(doc.page_content.strip()) < 10:
            continue
        
        # 保留有效段落
        filtered_docs.append(doc)
    
    print(f"最終得到 {len(filtered_docs)} 個有效段落。")
    return filtered_docs

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

# --- 5. 建立 QA 鏈 (使用 RetrievalQA，支援混合檢索) ---
def create_qa_chain(llm, vectorstore, texts=None):
    """建立問答鏈，支援混合檢索"""
    print("正在建立 QA 鏈...")
    try:
        # 建立Dense檢索器(向量搜索)
        dense_retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",  # 使用相似度閾值搜索
            search_kwargs={
                "k": RETRIEVAL_K,
                "score_threshold": SIMILARITY_THRESHOLD  # 使用統一的相似度閾值設定
            }
        )
        print(f"Dense檢索器將檢索 top {RETRIEVAL_K} 個區塊（相似度閾值: {SIMILARITY_THRESHOLD}）。")
        
        # 根據設定決定是否使用混合檢索
        if USE_HYBRID_RETRIEVAL and texts is not None:
            print("使用混合檢索模式（Dense + BM25）")
            
            # 建立BM25檢索器
            bm25_retriever = create_bm25_retriever(texts)
            if bm25_retriever is None:
                print("BM25檢索器建立失敗，回退到純Dense檢索")
                retriever = dense_retriever
            else:
                # 建立混合檢索器
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[dense_retriever, bm25_retriever],
                    weights=[DENSE_WEIGHT, SPARSE_WEIGHT]  # Dense和BM25的權重
                )
                retriever = ensemble_retriever
                print(f"混合檢索器建立成功 (Dense權重: {DENSE_WEIGHT}, BM25權重: {SPARSE_WEIGHT})")
        else:
            print("使用純Dense檢索模式")
            retriever = dense_retriever
        
        # 優化 Prompt 模板 - 支援分離的問答格式
        template = """你是一個專業的全景抬頭式顯示器(P-HUD)技術專家。請根據以下提供的問答對簡短回答用戶問題。

說明：以下每個項目包含一個相關問題和對應答案。請基於這些資訊回答用戶的問題。

要求：
1. 回答要簡潔明確，不超過3句話
2. 不要使用任何特殊符號如星號、破折號、項目符號等
3. 直接回答重點，避免冗長說明
4. 優先使用完全匹配或最相關的問答對來回答
5. 如果找到相關的問答對就基於其答案回答，沒有相關資訊才說超出文檔範疇

相關問答對：
{context}

用戶問題：{question}

簡短回答："""
        
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        
        # 自定義文檔格式化函數
        def format_docs(docs):
            formatted_parts = []
            for i, doc in enumerate(docs, 1):
                # 檢查是否為分離的問答格式
                if doc.metadata.get('type') == 'qa_separated':
                    question = doc.page_content
                    answer = doc.metadata.get('answer', '答案未找到')
                    formatted_parts.append(f"問答對{i}：\n問題：{question}\n答案：{answer}")
                else:
                    # 原始格式
                    formatted_parts.append(f"文檔{i}：\n{doc.page_content}")
            return "\n\n".join(formatted_parts)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff", # 使用 Stuff 方法
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": QA_CHAIN_PROMPT,
                "document_variable_name": "context"
            }
        )
        
        # 包裝QA鏈以使用自定義格式化
        class CustomQAChain:
            def __init__(self, qa_chain, format_func):
                self.qa_chain = qa_chain
                self.format_func = format_func
            
            def invoke(self, inputs):
                # 先獲取檢索結果
                query = inputs["query"]
                docs = retriever.get_relevant_documents(query)
                
                # 格式化文檔
                formatted_context = self.format_func(docs)
                
                # 直接調用LLM
                prompt_text = QA_CHAIN_PROMPT.format(context=formatted_context, question=query)
                answer = llm.invoke(prompt_text)
                
                return {
                    "result": answer,
                    "source_documents": docs
                }
        
        wrapped_qa_chain = CustomQAChain(qa_chain, format_docs)
        
        print("QA 鏈建立成功。")
        return wrapped_qa_chain
    except Exception as e:
        print(f"建立 QA 鏈時發生錯誤: {e}")
        return None

# 添加一個函數用於打印所有chunks的詳細信息
def print_all_chunks(texts):
    """打印所有文本塊的詳細資訊"""
    print("\n=== 所有文本塊詳細資訊 ===\n")
    
    for i, chunk in enumerate(texts):
        # 獲取元數據
        source = chunk.metadata.get('source', '未知來源')
        paragraph_index = chunk.metadata.get('paragraph_index', -1)
        chunk_type = chunk.metadata.get('type', 'unknown')
        
        # 獲取內容
        content = chunk.page_content
        
        # 計算字數和行數
        char_count = len(content)
        line_count = content.count('\n') + 1
        
        # 打印塊信息
        print(f"塊 #{i+1}")
        print(f"  來源: {os.path.basename(source)}")
        print(f"  段落索引: {paragraph_index}")
        print(f"  類型: {chunk_type}")
        print(f"  字符數: {char_count}")
        print(f"  行數: {line_count}")
        
        # 根據類型顯示不同格式
        if chunk_type == 'qa_separated':
            # 分離的問答格式
            answer = chunk.metadata.get('answer', '答案未找到')
            print(f"  問題: {content}")
            print(f"  答案: {answer}")
            print(f"  答案字數: {len(answer)}")
        else:
            # 原始格式
            content_preview = content.replace('\n', ' ')
            if len(content_preview) > 100:
                content_preview = content_preview[:97] + "..."
            print(f"  內容預覽: {content_preview}")
            print(f"  完整內容: {content}")
        
        print("\n" + "-"*50 + "\n")

# 添加保存chunks到txt文件的函數
def save_chunks_to_txt(texts, filename="chunks_data.txt"):
    """保存chunks數據到txt文件"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("===== 文本塊詳細資訊 =====\n\n")
            
            for i, chunk in enumerate(texts):
                # 獲取元數據
                source = chunk.metadata.get('source', '未知來源')
                paragraph_index = chunk.metadata.get('paragraph_index', -1)
                chunk_type = chunk.metadata.get('type', 'unknown')
                
                # 獲取內容
                content = chunk.page_content
                
                # 計算字數和行數
                char_count = len(content)
                line_count = content.count('\n') + 1
                
                # 寫入塊信息
                f.write(f"塊 #{i+1}\n")
                f.write(f"  來源: {os.path.basename(source)}\n")
                f.write(f"  段落索引: {paragraph_index}\n")
                f.write(f"  類型: {chunk_type}\n")
                f.write(f"  字符數: {char_count}\n")
                f.write(f"  行數: {line_count}\n")
                
                # 根據類型寫入不同格式
                if chunk_type == 'qa_separated':
                    # 分離的問答格式
                    answer = chunk.metadata.get('answer', '答案未找到')
                    f.write(f"  問題: {content}\n")
                    f.write(f"  答案: {answer}\n")
                    f.write(f"  答案字數: {len(answer)}\n")
                else:
                    # 原始格式
                    f.write(f"  內容:\n{content}\n")
                
                f.write("\n" + "-"*50 + "\n\n")
            
        print(f"\n文本塊數據已成功保存到 {filename}")
        return True
    except Exception as e:
        print(f"保存txt文件時發生錯誤: {e}")
        return False

def load_qa_chunks(file_path):
    """從文件中讀取內容，並將 Q 與 A 組合成一個 chunk，返回一個列表。
    假設每組內容以 'Q:' 開始，並包含 'A:' 的答案。"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正則表達式：以 'Q:' 作為起始，匹配到 'A:' 之後的內容，直到遇到下一個 'Q:' 或文件結束
    pattern = re.compile(r'(Q:.*?A:.*?)(?=Q:|$)', re.DOTALL)
    chunks = pattern.findall(content)
    
    # 去除首尾空白，並過濾空字串
    return [chunk.strip() for chunk in chunks if chunk.strip()]

def print_optimization_suggestions():
    """打印RAG系統優化建議"""
    print("\n" + "="*60)
    print("RAG系統優化建議")
    print("="*60)
    print(f"""
當前配置：
- 檢索模式: {'混合檢索 (Dense + BM25)' if USE_HYBRID_RETRIEVAL else '純Dense檢索'}
- Dense權重: {DENSE_WEIGHT if USE_HYBRID_RETRIEVAL else '1.0'}
- BM25權重: {SPARSE_WEIGHT if USE_HYBRID_RETRIEVAL else '0.0'}

如果您的系統經常回復請詢問專家，可以嘗試以下優化：

1. 調整相似度閾值：
   目前設定 {SIMILARITY_THRESHOLD} 較嚴格
   建議嘗試 0.1-0.2 較寬松
   在檔案頂部修改 SIMILARITY_THRESHOLD 參數

2. 增加檢索數量：
   目前設定 k={RETRIEVAL_K}
   可嘗試 k=8-10 提供更多上下文
   在檔案頂部修改 RETRIEVAL_K 參數

3. 混合檢索權重調整：
   目前 Dense:{DENSE_WEIGHT}, BM25:{SPARSE_WEIGHT}
   可嘗試 Dense:0.5, BM25:0.5 平衡權重
   或 Dense:0.7, BM25:0.3 偏重語義搜索
   在檔案頂部修改 DENSE_WEIGHT 和 SPARSE_WEIGHT

4. 切換檢索模式：
   目前: USE_HYBRID_RETRIEVAL = {USE_HYBRID_RETRIEVAL}
   可嘗試切換為 {'False' if USE_HYBRID_RETRIEVAL else 'True'}

5. 改進文檔切分策略：
   考慮按主題而非單純按QA切分
   允許文檔塊之間有重疊內容
   使用更小的chunk size以提高檢索精度

6. 嵌入模型優化：
   嘗試使用更大的嵌入模型
   考慮使用專門的中文嵌入模型
   評估使用OpenAI的text-embedding-ada-002

7. BM25優化（僅混合檢索模式）：
   調整BM25參數 k1 和 b
   改進中文分詞策略
   考慮使用自定義停用詞

8. Prompt工程：
   進一步調整prompt中的指導原則
   添加更多示例來引導LLM回答風格
   考慮使用少樣本學習技術

9. 添加評估機制：
   記錄問題和回答配對
   統計請詢問專家的頻率
   建立測試問題集進行評估
   比較不同檢索模式的效果
""")
    print("="*60 + "\n")

# --- 新增: 創建BM25檢索器 ---
def create_bm25_retriever(texts, cache_dir=BM25_CACHE_DIR):
    """創建BM25檢索器並支援快取，適配分離的問答格式"""
    print("正在建立BM25檢索器...")
    
    # 確保快取目錄存在
    os.makedirs(cache_dir, exist_ok=True)
    bm25_cache_file = os.path.join(cache_dir, "bm25_retriever.pkl")
    
    try:
        # 嘗試載入快取的BM25檢索器
        if os.path.exists(bm25_cache_file):
            print("載入快取的BM25檢索器...")
            with open(bm25_cache_file, 'rb') as f:
                bm25_retriever = pickle.load(f)
            print("BM25檢索器載入成功")
            return bm25_retriever
    except Exception as e:
        print(f"載入BM25快取失敗: {e}，將重新建立...")
    
    # 建立新的BM25檢索器
    try:
        # 為中文文本進行分詞預處理
        def preprocess_text(text):
            """中文文本預處理：分詞"""
            # 使用jieba進行中文分詞
            words = jieba.cut(text)
            return " ".join(words)
        
        # 預處理所有文檔
        print("正在進行中文分詞預處理...")
        processed_texts = []
        for doc in texts:
            # 對於分離的問答格式，只對問題部分進行BM25檢索
            if doc.metadata.get('type') == 'qa_separated':
                # 只處理問題部分，保持原有的metadata結構
                processed_content = preprocess_text(doc.page_content)  # doc.page_content已經是問題
            else:
                # 原始格式，處理完整內容
                processed_content = preprocess_text(doc.page_content)
            
            processed_doc = Document(
                page_content=processed_content,
                metadata=doc.metadata
            )
            processed_texts.append(processed_doc)
        
        # 建立BM25檢索器
        bm25_retriever = BM25Retriever.from_documents(processed_texts)
        bm25_retriever.k = RETRIEVAL_K  # 使用統一的檢索數量設定
        
        # 儲存到快取
        print("儲存BM25檢索器到快取...")
        with open(bm25_cache_file, 'wb') as f:
            pickle.dump(bm25_retriever, f)
        
        print("BM25檢索器建立成功（已適配分離問答格式）")
        return bm25_retriever
        
    except Exception as e:
        print(f"建立BM25檢索器時發生錯誤: {e}")
        return None

# --- 新增: Whisper 語音識別功能 ---
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

def record_audio(stop_event):
    """接收音頻並保存為臨時文件"""
    try:
        audio = pyaudio.PyAudio()
    except Exception as e:
        print(f"❌ 初始化音頻設備失敗: {e}")
        print("請確保：")
        print("1. 已安裝 pyaudio: pip install pyaudio")
        print("2. 系統有可用的音頻輸入設備")
        print("3. 在 Windows 上可能需要安裝 Microsoft Visual C++ Redistributable")
        return
        
    # 打開音頻流
    stream = None  # 初始化 stream 為 None
    try:
        # 檢查可用的音頻設備
        device_count = audio.get_device_count()
        
        # 尋找默認輸入設備
        default_input_device = None
        try:
            default_input_device = audio.get_default_input_device_info()
        except Exception as e:
            # 嘗試尋找第一個可用的輸入設備
            for i in range(device_count):
                try:
                    device_info = audio.get_device_info_by_index(i)
                    if device_info['maxInputChannels'] > 0:
                        break
                except:
                    continue
        
        stream = audio.open(format=AUDIO_FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
        
        frames = []
        print("🎤 正在錄音中...")
        print("📊 音量指示器：")
        
        # 實時音量顯示
        while not stop_event.is_set():
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                
                # 顯示音量指示器
                audio_data = np.frombuffer(data, dtype=np.int16)
                volume = np.abs(audio_data).mean()
                bars = int(50 * volume / 4000)
                volume_bar = "🔊 [" + "=" * bars + " " * (50 - bars) + "]"
                print(f"\r{volume_bar} 📢 再按 Enter 停止錄音", end="", flush=True)
                
            except IOError as e:
                # 忽略 PyAudio 在某些情況下可能拋出的輸入溢出錯誤
                if hasattr(e, 'errno') and e.errno == pyaudio.paInputOverflowed:
                    print("\n⚠️  音頻輸入溢出，忽略部分數據。", end="")
                else:
                    print(f"\n❌ 音頻讀取錯誤: {e}")
                    break
                
    except KeyboardInterrupt:
        print("\n⏹️ 錄音被中斷。")
    except Exception as e:
        print(f"\n❌ 錄音過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n⏹️ 錄音結束，正在處理音頻...")
        
        # 停止和關閉流
        if stream is not None:
            try:
                if stream.is_active():  # 僅在流活躍時停止
                    stream.stop_stream()
                stream.close()
            except Exception as e:
                print(f"❌ 關閉音頻流時出錯: {e}")
                
        audio.terminate()
        
        # 只有在錄到數據時才保存文件
        if 'frames' in locals() and frames:
            try:
                wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(AUDIO_FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                print("✅ 音頻文件保存成功")
            except Exception as e:
                print(f"❌ 保存音頻文件時出錯: {e}")
        else:
            print("⚠️  未錄製到有效音頻數據。")

def speech_to_text():
    """使用本地Whisper將語音轉換為文字，並返回偵測到的語言"""
    try:
        if not os.path.exists(WAVE_OUTPUT_FILENAME):
            return "", "zh"  # 默認返回中文作為語言，空字串作為文字
        
        # 載入Whisper模型
        if not load_whisper_model():
            return "", "zh"
            
        print(f"🤖 正在識別語音...")
        
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
        
        # 根據檢測到的語言顯示不同的 emoji
        lang_emoji = "🇨🇳" if detected_language in ["zh", "cn"] else "🇺🇸" if detected_language == "en" else "🌐"
        print(f"{lang_emoji} 語音識別完成 ({detected_language})")
        
        return text, detected_language
    
    except Exception as e:
        print(f"❌ 語音轉文字時發生錯誤: {e}")
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

def process_voice_input(qa_chain):
    """處理語音輸入的完整流程"""
    try:
        print("🎙️ 語音輸入模式")
        print("📝 使用方法：")
        print("   🎯 按 Enter 開始錄音並說出您的問題")
        print("   ⏹️ 再按 Enter 停止錄音")
        print("   ⚡ 系統將自動識別並回答")
        
        # 直接開始錄音，不需要額外確認
        print("🚀 開始錄音！請說出您的問題...")
        
        # 創建停止事件
        stop_event = threading.Event()
        
        # 創建錄音線程
        record_thread = threading.Thread(target=record_audio, args=(stop_event,))
        record_thread.start()
        
        # 等待用戶按下Enter鍵停止錄音
        input()  # 等待按Enter停止錄音
        stop_event.set()
        record_thread.join()
        
        # 語音轉文字，並獲取偵測到的語言
        print("🤖 正在識別語音，請稍候...")
        question, detected_lang = speech_to_text()
        if not question:
            print("❌ 無法識別語音內容，請重試。")
            print("💡 建議：")
            print("   - 確保麥克風工作正常")
            print("   - 說話聲音清晰")
            print("   - 環境相對安靜")
            return None
        
        # 處理識別出的文字，檢查是否為退出命令
        if question.lower().strip() in ["退出", "結束", "exit", "quit"]:
            print("🔊 語音指令: 退出程式")
            return "exit"
            
        # 忽略空問題
        if not question.strip():
            print("⚠️ 識別到空內容，請重試。")
            return None
        
        print(f"✅ 語音識別成功！")
        print(f"📝 您的問題：{question}")
        return question, detected_lang  # 返回問題和檢測到的語言
        
    except Exception as e:
        print(f"❌ 處理語音輸入時發生錯誤: {e}")
        return None

# --- 新增: 語音功能依賴檢查 ---
def check_voice_dependencies():
    """檢查語音功能所需的依賴是否安裝"""
    missing_deps = []
    
    try:
        import pyaudio
    except ImportError:
        missing_deps.append("pyaudio")
    
    try:
        import whisper
    except ImportError:
        missing_deps.append("openai-whisper")
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    # 檢查TTS相關依賴
    try:
        import TTS
    except ImportError:
        missing_deps.append("TTS")
    
    try:
        import pydub
    except ImportError:
        missing_deps.append("pydub")
    
    try:
        import simpleaudio
    except ImportError:
        missing_deps.append("simpleaudio")
    
    if missing_deps:
        print("⚠️  語音功能依賴缺失：")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n要安裝缺失的依賴，請運行：")
        print("pip install " + " ".join(missing_deps))
        print("\n或者安裝所有語音功能依賴：")
        print("pip install -r requirements_voice.txt")
        return False
    
    return True

def initialize_voice_features():
    """初始化語音功能"""
    global ENABLE_VOICE_INPUT, ENABLE_VOICE_OUTPUT
    
    if not ENABLE_VOICE_INPUT and not ENABLE_VOICE_OUTPUT:
        return True
    
    print("正在檢查語音功能依賴...")
    if not check_voice_dependencies():
        print("語音功能將被禁用，您仍可以使用文字輸入輸出功能。")
        ENABLE_VOICE_INPUT = False
        ENABLE_VOICE_OUTPUT = False
        return False
    
    print("語音功能依賴檢查通過！")
    
    # 初始化語音輸入功能
    if ENABLE_VOICE_INPUT:
        try:
            print("正在預載入 Whisper 模型...")
            if load_whisper_model():
                print("Whisper 模型預載入成功！")
            else:
                print("Whisper 模型預載入失敗，將在需要時再次嘗試載入。")
        except Exception as e:
            print(f"預載入 Whisper 模型時出現錯誤: {e}")
            print("語音輸入功能可能無法正常工作。")
    
    # 初始化TTS功能
    if ENABLE_VOICE_OUTPUT:
        try:
            print("正在初始化 Coqui TTS...")
            if initialize_coqui_tts():
                print("Coqui TTS 初始化成功！")
            else:
                print("Coqui TTS 初始化失敗，TTS功能將被禁用。")
                ENABLE_VOICE_OUTPUT = False
        except Exception as e:
            print(f"初始化 Coqui TTS 時出現錯誤: {e}")
            print("TTS功能將被禁用。")
            ENABLE_VOICE_OUTPUT = False
    
    return True

# --- 新增：CoquiTTS 語音合成功能 ---
def initialize_coqui_tts():
    """載入 Coqui TTS 模型並進行必要的預熱"""
    global coqui_tts_model, AVAILABLE_SPEAKERS
    if coqui_tts_model is None:
        print(f"正在初始化 Coqui TTS 引擎 (模型: {COQUI_TTS_MODEL_NAME})...")
        try:
            device = "cuda" if torch.cuda.is_available() and TTS_ENABLE_GPU else "cpu"
            print(f"使用 {'GPU' if device == 'cuda' else 'CPU'} 運行 Coqui TTS 模型")

            # --- Fix for PyTorch >= 2.6 loading issue with XTTS ---
            try:
                from TTS.tts.configs.xtts_config import XttsConfig
                from TTS.tts.models.xtts import XttsAudioConfig
                from TTS.config.shared_configs import BaseDatasetConfig
                from TTS.tts.models.xtts import XttsArgs

                safe_classes = [XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs]
                torch.serialization.add_safe_globals(safe_classes)
                print(f"已將 TTS 相關類加入 PyTorch 安全全局列表")
            except ImportError as imp_err:
                print(f"警告：無法導入 XTTS 相關類 ({imp_err})，如果您未使用 XTTS 模型，可以忽略此訊息")
            except Exception as safe_global_err:
                print(f"嘗試將 TTS 相關類加入安全全局列表時出錯: {safe_global_err}")

            # 配置並初始化 TTS 模型
            start_time = time.time()
            coqui_tts_model = TTS(model_name=COQUI_TTS_MODEL_NAME, gpu=(device == "cuda"))
            load_time = time.time() - start_time
            print(f"Coqui TTS 引擎加載耗時: {load_time:.2f} 秒")
            
            # 獲取並保存可用的說話人列表
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
            
            # 模型預熱，降低首次生成延遲
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
            import traceback
            print("詳細錯誤追蹤：")
            traceback.print_exc()
            coqui_tts_model = None
            return False
    return True

def text_to_speech(text, language="zh"):
    """使用 Coqui TTS 生成語音數據並播放
    
    參數:
        text (str): 要播放的文字內容
        language (str): 語言代碼，'zh' 表示中文，'en' 表示英文
    """
    global coqui_tts_model, AVAILABLE_SPEAKERS, TTS_CACHE
    
    if not ENABLE_VOICE_OUTPUT:
        return True  # 如果語音輸出被禁用，直接返回成功
        
    if not coqui_tts_model:
        print("⚠️ Coqui TTS 模型未初始化，無法生成語音。")
        return False

    if not text or len(text.strip()) == 0:
        print("⚠️ 嘗試播放空文本")
        return False

    # 映射語言代碼
    coqui_lang = 'zh-cn' if language == 'zh' else 'en'
    
    # 根據語言選擇固定的說話人
    if language == 'zh':
        selected_speaker = FIXED_ZH_SPEAKER if FIXED_ZH_SPEAKER in AVAILABLE_SPEAKERS else (AVAILABLE_SPEAKERS[0] if AVAILABLE_SPEAKERS else None)
    else:  # 'en'
        selected_speaker = FIXED_EN_SPEAKER if FIXED_EN_SPEAKER in AVAILABLE_SPEAKERS else (AVAILABLE_SPEAKERS[0] if AVAILABLE_SPEAKERS else None)
    
    if not selected_speaker:
        print("❌ 找不到可用的說話人。")
        return False

    # 生成緩存鍵
    cache_key = (text, coqui_lang, selected_speaker)
    
    print(f"🔊 使用 Coqui TTS ({coqui_lang}, 說話人: {selected_speaker}) 播放: {text[:30]}...")
    
    with playback_lock:
        try:
            # 檢查緩存中是否已有此文本的音頻
            if TTS_ENABLE_CACHE and cache_key in TTS_CACHE:
                print("📦 從緩存中獲取音頻...")
                wav = TTS_CACHE[cache_key]
            else:
                # 生成音頻數據
                print(f"🎵 生成音頻數據中...")
                
                tts_kwargs = {
                    "text": text,
                    "language": coqui_lang,
                    "speaker": selected_speaker
                }
                
                start_time = time.time()
                wav = coqui_tts_model.tts(**tts_kwargs)
                tts_time = time.time() - start_time
                print(f"✅ 音頻生成完成，耗時: {tts_time:.2f} 秒")
                
                # 緩存生成的音頻
                if TTS_ENABLE_CACHE:
                    TTS_CACHE[cache_key] = wav
            
            # 獲取采樣率
            sample_rate = TTS_SAMPLE_RATE
            if hasattr(coqui_tts_model, 'synthesizer') and hasattr(coqui_tts_model.synthesizer, 'output_sample_rate'):
                sample_rate = coqui_tts_model.synthesizer.output_sample_rate
            
            # 通過 simpleaudio 播放
            try:
                # 將浮點數組轉換為 int16 格式
                wav = np.clip(wav, -1.0, 1.0)
                wav_int16 = (wav * 32767).astype(np.int16)
                
                # 播放音頻
                print("🎵 正在播放...")
                play_obj = sa.play_buffer(
                    wav_int16,
                    num_channels=1,
                    bytes_per_sample=2,
                    sample_rate=sample_rate
                )
                
                play_obj.wait_done()
                print("✅ 播放完成")
                return True
                
            except sa.exceptions.SimpleaudioError as sa_error:
                print(f"❌ Simpleaudio 播放時發生錯誤: {sa_error}")
                return False
                
        except Exception as e:
            print(f"❌ TTS 生成或播放過程中發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            return False

def play_prompt_audio(detected_language="zh"):
    """播放提示音，在用戶提問後、系統開始思考時播放
    
    參數:
        detected_language (str): 檢測到的語言，'zh' 表示中文，'en' 表示英文
    """
    global coqui_tts_model, AVAILABLE_SPEAKERS, TTS_CACHE
    
    if not ENABLE_VOICE_OUTPUT or not coqui_tts_model:
        return True  # 如果語音輸出被禁用或模型未初始化，直接返回
    
    # 創建一個非阻塞的提示音播放函數
    def play_prompt_in_background():
        # 這裡創建一個新的鎖，避免與主要playback_lock衝突
        prompt_lock = threading.Lock()
        with prompt_lock:
            try:
                # 根據檢測到的語言選擇提示音文本
                if detected_language == "en":
                    # 英文提示音
                    prompt_text = "Thank you for your question. Let me think about it for a moment."
                    prompt_lang = "en"
                else:
                    # 中文提示音
                    prompt_text = "感謝您的提問，我思考一下請稍候！"
                    prompt_lang = "zh-cn"
                    
                # 選擇相應語言的說話人
                if prompt_lang == "zh-cn":
                    selected_speaker = FIXED_ZH_SPEAKER if FIXED_ZH_SPEAKER in AVAILABLE_SPEAKERS else (AVAILABLE_SPEAKERS[0] if AVAILABLE_SPEAKERS else None)
                else:
                    selected_speaker = FIXED_EN_SPEAKER if FIXED_EN_SPEAKER in AVAILABLE_SPEAKERS else (AVAILABLE_SPEAKERS[0] if AVAILABLE_SPEAKERS else None)
                    
                if not selected_speaker:
                    print("⚠️ 找不到可用的說話人，無法播放提示音")
                    return
                
                # 生成緩存鍵
                cache_key = (prompt_text, prompt_lang, selected_speaker)
                
                print(f"🎵 播放提示音 ({prompt_lang})...")
                
                # 檢查緩存
                if TTS_ENABLE_CACHE and cache_key in TTS_CACHE:
                    wav = TTS_CACHE[cache_key]
                else:
                    # 生成提示音
                    tts_kwargs = {
                        "text": prompt_text,
                        "language": prompt_lang,
                        "speaker": selected_speaker
                    }
                    
                    wav = coqui_tts_model.tts(**tts_kwargs)
                    
                    # 緩存生成的音頻
                    if TTS_ENABLE_CACHE:
                        TTS_CACHE[cache_key] = wav
                
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
                print("✅ 提示音播放完成")
                
            except Exception as e:
                print(f"❌ 播放提示音時發生錯誤: {e}")
    
    # 在後台線程中播放提示音，避免阻塞主程序
    try:
        prompt_thread = threading.Thread(target=play_prompt_in_background, daemon=True)
        prompt_thread.start()
        return True
    except Exception as e:
        print(f"⚠️ 啟動提示音線程時發生錯誤: {e}")
        return False

def check_tts_dependencies():
    """檢查TTS功能所需的依賴是否安裝"""
    missing_deps = []
    
    try:
        import TTS
    except ImportError:
        missing_deps.append("TTS")
    
    try:
        import pydub
    except ImportError:
        missing_deps.append("pydub")
    
    try:
        import simpleaudio
    except ImportError:
        missing_deps.append("simpleaudio")
    
    if missing_deps:
        print("⚠️  TTS功能依賴缺失：")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n要安裝缺失的依賴，請運行：")
        print("pip install " + " ".join(missing_deps))
        return False
    
    return True

def initialize_tts_features():
    """初始化TTS功能"""
    global ENABLE_VOICE_OUTPUT
    
    if not ENABLE_VOICE_OUTPUT:
        return True
    
    print("正在檢查TTS功能依賴...")
    if not check_tts_dependencies():
        print("TTS功能將被禁用，您仍可以使用文字輸出功能。")
        ENABLE_VOICE_OUTPUT = False
        return False
    
    print("TTS功能依賴檢查通過！")
    
    # 初始化 Coqui TTS
    try:
        print("正在初始化 Coqui TTS...")
        if initialize_coqui_tts():
            print("Coqui TTS 初始化成功！")
            return True
        else:
            print("Coqui TTS 初始化失敗，TTS功能將被禁用。")
            ENABLE_VOICE_OUTPUT = False
            return False
    except Exception as e:
        print(f"初始化 Coqui TTS 時出現錯誤: {e}")
        print("TTS功能將被禁用。")
        ENABLE_VOICE_OUTPUT = False
        return False

# --- 新增: 語音功能依賴檢查 ---

# --- 主要執行流程 ---
if __name__ == "__main__":
    # --- 步驟 0: 清理舊的向量儲存和BM25快取 ---
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
    
    # 清理BM25快取
    if os.path.exists(BM25_CACHE_DIR):
        print(f"偵測到舊的BM25快取目錄 {BM25_CACHE_DIR}，正在刪除...")
        try:
            shutil.rmtree(BM25_CACHE_DIR)
            print(f"目錄 {BM25_CACHE_DIR} 已成功刪除。")
        except OSError as e:
            print(f"刪除目錄 {BM25_CACHE_DIR} 時發生錯誤: {e}")
            print("將繼續執行，但BM25快取可能不是最新的。")
    else:
        print(f"BM25快取目錄 {BM25_CACHE_DIR} 不存在，無需刪除。")

    # --- 步驟 1-3: 載入、分割文件並建立向量儲存 ---
    documents = load_documents(PDF_DIRECTORY)
    if not documents:
        print("無法載入文件，程式終止。")
        sys.exit(1) # 退出程式

    texts = split_documents(documents)
    if not texts:
        print("文件分割失敗，程式終止。")
        sys.exit(1)
        
    # 打印所有文本塊的詳細信息
    print_all_chunks(texts)
    
    # 保存為txt文件
    save_txt = input("\n是否保存文本塊數據到txt文件? (y/n): ").strip().lower()
    if save_txt == 'y':
        filename = input("請輸入文件名 (默認為 chunks_data.txt): ").strip()
        if not filename:
            filename = "chunks_data.txt"
        if not filename.endswith('.txt'):
            filename += '.txt'
        save_chunks_to_txt(texts, filename)
    
    # 用戶確認是否繼續
    continue_execution = input("\n已打印所有chunks，是否繼續執行建立向量數據庫和QA系統? (y/n): ").strip().lower() == 'y'
    if not continue_execution:
        print("程序已中止，感謝使用！")
        sys.exit(0)

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

    qa_chain = create_qa_chain(llm, vectorstore, texts)
    if not qa_chain:
        print("QA 鏈建立失敗，程式終止。")
        sys.exit(1)

    # --- 初始化語音功能 (如果啟用) ---
    if ENABLE_VOICE_INPUT or ENABLE_VOICE_OUTPUT:
        initialize_voice_features()

    # --- 步驟 6: 查詢迴圈 ---
    print("===================================")
    print(f" RAG 系統已就緒 (LLM: {LLM_PROVIDER.upper()})！")
    print(f" 檢索模式: {'混合檢索 (Dense + BM25)' if USE_HYBRID_RETRIEVAL else '純Dense檢索'}")
    if USE_HYBRID_RETRIEVAL:
        print(f" 權重配置: Dense={DENSE_WEIGHT}, BM25={SPARSE_WEIGHT}")
    
    if ENABLE_VOICE_INPUT:
        print(" 🎙️ 語音輸入模式已啟用（主要模式）")
        print(" 🎯 語音輸入流程：")
        print("    🎤 按 Enter 開始錄音並說出問題")
        print("    ⏹️ 再按 Enter 結束錄音")
    
    if ENABLE_VOICE_OUTPUT:
        print(" 🔊 語音輸出模式已啟用")
        print("    系統將自動播放答案語音")
        if ENABLE_PROMPT_AUDIO:
            print("    🎵 提示音功能已啟用（問題處理時播放友好提示）")
    
    if ENABLE_VOICE_INPUT or ENABLE_VOICE_OUTPUT:
        print(" 特殊指令：")
        if ENABLE_VOICE_INPUT:
            print(" - 輸入 'text' 或 't' 切換到文字輸入模式")
        print(" - 輸入 'quit' 或 'exit' 來結束程式")
        print(" - 輸入 'debug' 切換調試模式（顯示檢索詳情）")
        print(" - 輸入 'help' 查看優化建議")
    else:
        print(" 請輸入您的問題（關於 PDF 文件的內容）。")
        print(" 輸入 'quit' 或 'exit' 來結束程式。")
        print(" 輸入 'debug' 切換調試模式（顯示檢索詳情）。")
        print(" 輸入 'help' 查看優化建議。")
    print("===================================")

    debug_mode = False  # 調試模式開關
    text_mode = False   # 文字輸入模式開關

    while True:
        try:
            if ENABLE_VOICE_INPUT and not text_mode:
                # 語音輸入模式（默認模式）
                print("\n🎙️ 語音輸入模式")
                command = input("🎤 按 Enter 開始錄音說話，或輸入指令 > ").strip()
                
                # 處理特殊指令
                if command.lower() in ['quit', 'exit']:
                    break
                elif command.lower() == 'debug':
                    debug_mode = not debug_mode
                    print(f"調試模式已{'開啟' if debug_mode else '關閉'}")
                    continue
                elif command.lower() == 'help':
                    print_optimization_suggestions()
                    continue
                elif command.lower() in ['text', 't']:
                    text_mode = True
                    print("已切換到文字輸入模式")
                    print("輸入 'voice' 或 'v' 可切換回語音模式")
                    continue
                elif command:  # 如果輸入了其他文字，當作文字問題處理
                    question = command
                    detected_lang = "zh"  # 文字輸入默認中文
                else:
                    # 空輸入，進行語音輸入
                    voice_result = process_voice_input(qa_chain)
                    if voice_result == "exit":
                        break
                    elif voice_result is None:
                        continue
                    else:
                        question, detected_lang = voice_result  # 獲取問題和檢測到的語言
                        if isinstance(question, tuple):  # 如果返回的是元組
                            question, detected_lang = question
                        else:  # 如果只返回問題
                            detected_lang = "zh"  # 默認中文
            else:
                # 文字輸入模式
                print("\n📝 文字輸入模式")
                question = input("請輸入您的問題 > ")
                
                if question.lower() in ['quit', 'exit']:
                    break
                if question.lower() == 'debug':
                    debug_mode = not debug_mode
                    print(f"調試模式已{'開啟' if debug_mode else '關閉'}")
                    continue
                if question.lower() == 'help':
                    print_optimization_suggestions()
                    continue
                if ENABLE_VOICE_INPUT and question.lower() in ['voice', 'v']:
                    text_mode = False
                    print("已切換到語音輸入模式")
                    continue
                
                # 為文字輸入設置默認語言
                detected_lang = "zh"  # 文字輸入默認中文
                    
            if not question.strip(): # 忽略空輸入
                continue

            print(f"正在處理您的問題 (使用 {LLM_PROVIDER.upper()} LLM)...")
            
            # 新增：播放提示音
            if ENABLE_VOICE_OUTPUT and ENABLE_PROMPT_AUDIO and not debug_mode:  # 非調試模式才播放提示音
                # 根據檢測到的語言播放相應的提示音
                tts_language = "zh"
                if 'detected_lang' in locals() and detected_lang:
                    tts_language = map_whisper_language_to_supported(detected_lang)
                
                play_prompt_audio(tts_language)
            
            # 如果開啟調試模式，先顯示檢索結果
            if debug_mode:
                print("\n=== 調試資訊：檢索結果 ===")
                
                if USE_HYBRID_RETRIEVAL:
                    print("混合檢索模式調試：")
                    
                    # Dense檢索結果
                    print("1. Dense檢索結果：")
                    dense_retriever = vectorstore.as_retriever(
                        search_type="similarity_score_threshold",
                        search_kwargs={"k": RETRIEVAL_K, "score_threshold": SIMILARITY_THRESHOLD}
                    )
                    dense_docs = dense_retriever.get_relevant_documents(question)
                    
                    if dense_docs:
                        print(f"   檢索到 {len(dense_docs)} 個Dense相關文檔")
                        for i, doc in enumerate(dense_docs[:3]):  # 只顯示前3個
                            source = doc.metadata.get('source', '未知來源')
                            paragraph_index = doc.metadata.get('paragraph_index', -1)
                            chunk_type = doc.metadata.get('type', 'unknown')
                            
                            print(f"   Dense-{i+1}: {os.path.basename(source)}, 段落 {paragraph_index}, 類型: {chunk_type}")
                            
                            if chunk_type == 'qa_separated':
                                # 分離的問答格式
                                question = doc.page_content
                                answer = doc.metadata.get('answer', '答案未找到')
                                print(f"      問題: {question[:50]}{'...' if len(question) > 50 else ''}")
                                print(f"      答案: {answer[:50]}{'...' if len(answer) > 50 else ''}")
                            else:
                                # 原始格式
                                content_preview = doc.page_content[:100].replace('\n', ' ') + "..."
                                print(f"      預覽: {content_preview}")
                    else:
                        print("   Dense檢索無結果")
                    
                    # BM25檢索結果（如果可用）
                    print("\n2. BM25檢索結果：")
                    try:
                        bm25_retriever = create_bm25_retriever(texts)
                        if bm25_retriever:
                            # 對查詢進行分詞
                            import jieba
                            processed_query = " ".join(jieba.cut(question))
                            bm25_docs = bm25_retriever.get_relevant_documents(processed_query)
                            
                            if bm25_docs:
                                print(f"   檢索到 {len(bm25_docs)} 個BM25相關文檔")
                                for i, doc in enumerate(bm25_docs[:3]):  # 只顯示前3個
                                    source = doc.metadata.get('source', '未知來源')
                                    paragraph_index = doc.metadata.get('paragraph_index', -1)
                                    chunk_type = doc.metadata.get('type', 'unknown')
                                    
                                    print(f"   BM25-{i+1}: {os.path.basename(source)}, 段落 {paragraph_index}, 類型: {chunk_type}")
                                    
                                    if chunk_type == 'qa_separated':
                                        # 分離的問答格式
                                        question = doc.page_content
                                        answer = doc.metadata.get('answer', '答案未找到')
                                        print(f"      問題: {question[:50]}{'...' if len(question) > 50 else ''}")
                                        print(f"      答案: {answer[:50]}{'...' if len(answer) > 50 else ''}")
                                    else:
                                        # 原始格式
                                        content_preview = doc.page_content[:100].replace('\n', ' ') + "..."
                                        print(f"      預覽: {content_preview}")
                            else:
                                print("   BM25檢索無結果")
                        else:
                            print("   BM25檢索器不可用")
                    except Exception as e:
                        print(f"   BM25檢索調試失敗: {e}")
                    
                else:
                    print("純Dense檢索模式調試：")
                    # 直接使用檢索器來查看檢索結果
                    retriever = vectorstore.as_retriever(
                        search_type="similarity_score_threshold",
                        search_kwargs={"k": RETRIEVAL_K, "score_threshold": SIMILARITY_THRESHOLD}
                    )
                    retrieved_docs = retriever.get_relevant_documents(question)
                    
                    if retrieved_docs:
                        print(f"檢索到 {len(retrieved_docs)} 個相關文檔塊：")
                        for i, doc in enumerate(retrieved_docs):
                            source = doc.metadata.get('source', '未知來源')
                            paragraph_index = doc.metadata.get('paragraph_index', -1)
                            chunk_type = doc.metadata.get('type', 'unknown')
                            
                            print(f"  文檔 {i+1}: {os.path.basename(source)}, 段落 {paragraph_index}, 類型: {chunk_type}")
                            
                            if chunk_type == 'qa_separated':
                                # 分離的問答格式
                                question_text = doc.page_content
                                answer = doc.metadata.get('answer', '答案未找到')
                                print(f"    問題: {question_text[:100]}{'...' if len(question_text) > 100 else ''}")
                                print(f"    答案: {answer[:100]}{'...' if len(answer) > 100 else ''}")
                            else:
                                # 原始格式
                                content_preview = doc.page_content[:200].replace('\n', ' ') + "..."
                                print(f"    內容預覽: {content_preview}")
                            print()
                    else:
                        print("警告：沒有檢索到任何滿足相似度閾值的文檔！")
                        print("這可能是導致回復'請詢問專家'的原因。")
                
                print("=" * 40)
            
            # <-- 直接調用 qa_chain.invoke
            result = qa_chain.invoke({"query": question})
            # <-- 從結果中提取 'result'
            answer = result.get('result', '抱歉，無法生成答案。').strip()
            source_docs = result.get('source_documents', []) # 獲取來源文檔 (可選)

            print("\n答案：")
            print(answer)
            
            # 新增：語音播放答案
            if ENABLE_VOICE_OUTPUT:
                print("\n🔊 正在播放語音答案...")
                # 根據Whisper檢測到的語言決定TTS語言
                if 'detected_lang' in locals():
                    tts_language = map_whisper_language_to_supported(detected_lang)
                else:
                    tts_language = "zh"  # 默認中文
                
                # 播放答案
                text_to_speech(answer, tts_language)

            # 可選：顯示來源文件資訊 (保持註解)
            if source_docs and not debug_mode:  # 非調試模式才顯示簡化版本
                print("--- 參考來源片段 ---")
                seen_sources = set()
                for i, doc in enumerate(source_docs):
                    source = doc.metadata.get('source', '未知來源')
                    paragraph_index = doc.metadata.get('paragraph_index', -1)
                    source_key = f"{source}_p{paragraph_index}"
                    if source_key not in seen_sources:
                        print(f"  來源: {os.path.basename(source)}, 段落索引: {paragraph_index}")
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

    # 清理臨時音頻文件
    if os.path.exists(WAVE_OUTPUT_FILENAME):
        try:
            os.remove(WAVE_OUTPUT_FILENAME)
            print(f"已清理臨時音頻文件: {WAVE_OUTPUT_FILENAME}")
        except Exception as e:
            print(f"清理臨時音頻文件時發生錯誤: {e}")

    print("--- RAG 系統已關閉 ---")

 