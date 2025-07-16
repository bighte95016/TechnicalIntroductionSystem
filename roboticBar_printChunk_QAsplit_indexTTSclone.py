import os
import sys

# 禁用 Chroma 遙測功能以避免錯誤
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
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

# 新增：語音輸入相關imports
import pyaudio
import wave
import threading
import whisper # 本地版本 Whisper
import torch # PyTorch
import numpy as np


# 新增：TTS語音輸出相關imports
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent / "index_tts"))
from index_tts.indextts.infer import IndexTTS
import sounddevice as sd

# --- 在所有 import 之後，第一次訪問環境變數之前調用 load_dotenv --- #
load_dotenv()

# 【關鍵修正】將分詞器定義在頂層，使其可以被pickle序列化
def jieba_tokenizer(text):
    """
    使用jieba進行分詞，並返回詞語列表，供BM25Retriever使用
    """
    return list(jieba.cut(text))

# --- 基本設定 ---
PDF_DIRECTORY = "./bar_file/PDF"
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

# 終端文字樣式
BOLD = "\033[1m"
RESET = "\033[0m"

# --- 混合檢索設定 ---
USE_HYBRID_RETRIEVAL = True  # 是否使用混合檢索
DENSE_WEIGHT = 0.4  # Dense retriever權重 (0.0-1.0) - 降低語義搜索權重
SPARSE_WEIGHT = 0.6  # BM25 retriever權重 (0.0-1.0) - 提高關鍵詞匹配權重
RETRIEVAL_K = 5  # 統一的檢索數量設定 - 增加到6個，確保BM25完美匹配不會被過濾
SIMILARITY_THRESHOLD = 0.1  # 統一的相似度閾值設定 (0.0-1.0，越高越嚴格)

# --- 問答分離設定 ---
USE_QA_SEPARATION = True  # 是否將問題和答案分離存儲（問題用於檢索，答案存在metadata中）

# --- 語音輸入設定 ---
ENABLE_VOICE_INPUT = True  # 是否啟用語音輸入功能

# --- TTS語音輸出設定 ---
ENABLE_TTS_OUTPUT = True  # 是否啟用TTS語音輸出功能
TTS_MODEL_DIR = "./index_tts/checkpoints"  # TTS模型目錄
TTS_CONFIG_PATH = "./index_tts/checkpoints/config.yaml"  # TTS配置文件路徑
TTS_VOICE_PATH = "./voice/Senior_Shiu.wav"  # 語音參考文件路徑
tts_system = None  # 全局變量，用於存儲TTS系統

# --- 提示音功能設定 ---
ENABLE_PROMPT_AUDIO = True  # 是否啟用提示音功能（在用戶提問後播放友好提示）

# 檢查 Python 版本
if sys.version_info < (3, 8):
    print("警告：您的 Python 版本較低，建議使用 Python 3.8 或更高版本以獲得最佳相容性。")

# --- 新增：語音輸入相關函數 ---
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
        
        print("開始錄音...（按下Enter鍵停止錄音）")
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
        print("\n錄音結束，正在停止錄音並保存...")
        
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
                print(f"音頻已保存為 {WAVE_OUTPUT_FILENAME}")
            except Exception as e:
                print(f"保存音頻文件時出錯: {e}")
        else:
            print("未錄製到有效音頻數據，不保存文件。")

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

# --- 新增：TTS語音輸出相關函數 ---
def load_tts_system():
    """載入TTS系統"""
    global tts_system
    
    if tts_system is None:
        print(f"正在載入TTS語音合成系統...")
        try:
            # 檢查必要的文件是否存在
            if not os.path.exists(TTS_MODEL_DIR):
                print(f"錯誤：TTS模型目錄不存在: {TTS_MODEL_DIR}")
                return False
            
            if not os.path.exists(TTS_CONFIG_PATH):
                print(f"錯誤：TTS配置文件不存在: {TTS_CONFIG_PATH}")
                return False
                
            if not os.path.exists(TTS_VOICE_PATH):
                print(f"錯誤：語音參考文件不存在: {TTS_VOICE_PATH}")
                return False
            
            # 初始化TTS系統
            tts_system = IndexTTS(model_dir=TTS_MODEL_DIR, cfg_path=TTS_CONFIG_PATH)
            print(f"TTS系統載入成功 (使用語音: {os.path.basename(TTS_VOICE_PATH)})")
            return True
            
        except Exception as e:
            print(f"載入TTS系統時發生錯誤: {e}")
            print("請檢查TTS模型和相關文件是否正確安裝。")
            return False
    
    return True

def text_to_speech(text):
    """將文本轉換為語音並播放"""
    if not ENABLE_TTS_OUTPUT:
        return False
        
    try:
        # 載入TTS系統
        if not load_tts_system():
            return False
        
        # 清理文本，去除特殊字符和過長內容
        cleaned_text = clean_text_for_tts(text)
        if not cleaned_text:
            print("文本清理後為空，跳過語音合成")
            return False
        
        print(f"正在生成語音... (文本長度: {len(cleaned_text)})")
        
        # 生成音頻數據
        sampling_rate, wav_data = tts_system.infer(
            TTS_VOICE_PATH, 
            cleaned_text, 
            output_path=None  # 不保存到文件，直接返回音頻數據
        )
        
        print(f"語音生成完成！採樣率: {sampling_rate} Hz, 時長: {len(wav_data) / sampling_rate:.2f} 秒")
        
        # 播放音頻
        print("正在播放語音...")
        sd.play(wav_data, sampling_rate)
        sd.wait()  # 等待播放完成
        print("語音播放完成！")
        
        return True
        
    except Exception as e:
        print(f"TTS語音合成時發生錯誤: {e}")
        return False

def clean_text_for_tts(text):
    """清理文本以適合TTS合成"""
    if not text:
        return ""
    
    # 移除多餘的空白字符
    text = re.sub(r'\s+', ' ', text.strip())
    
    # 移除特殊符號，但保留基本標點
    text = re.sub(r'[^\w\s，。！？：；、（）「」『』\-]', '', text)
    
    # 限制文本長度（避免過長的文本）
    max_length = 500  # 最大字符數
    if len(text) > max_length:
        # 在句號處截斷
        sentences = re.split(r'[。！？]', text[:max_length])
        if len(sentences) > 1:
            text = '。'.join(sentences[:-1]) + '。'
        else:
            text = text[:max_length] + '...'
    
    return text

def ask_tts_preference():
    """詢問用戶是否要啟用TTS語音輸出"""
    if not ENABLE_TTS_OUTPUT:
        return False
        
    try:
        choice = input("是否啟用TTS語音輸出功能？(y/n，默認為y): ").strip().lower()
        return choice != 'n'
    except:
        return True  # 默認啟用

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
        q_matches = list(q_head.finditer(full_text))
        
        # 詳細診斷輸出
        print(f"  📊 診斷資訊：")
        print(f"     - 原始文本長度: {len(full_text)} 字符")
        print(f"     - 找到的Q標記數量: {len(q_matches)}")
        
        if q_matches:
            print(f"     - Q標記位置和內容:")
            for i, match in enumerate(q_matches[:10]):  # 只顯示前10個
                start_pos = match.start()
                end_pos = min(start_pos + 50, len(full_text))
                context = full_text[start_pos:end_pos].replace('\n', '\\n')
                print(f"       Q{i+1}: 位置{start_pos} -> \"{context}...\"")
            if len(q_matches) > 10:
                print(f"       ... 還有 {len(q_matches) - 10} 個Q標記")
        
        starts = [m.start()+1 for m in q_matches] + [len(full_text)]
        
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
            
            # 統計分離成功/失敗的數量
            successful_separations = 0
            failed_separations = 0
            filtered_out = 0
            
            # 根據配置決定是否分離問題和答案
            docs = []
            for i, chunk in enumerate(chunks):
                # 檢查chunk長度
                if len(chunk.strip()) < 5:
                    filtered_out += 1
                    print(f"    Q{i+1}: 過短被過濾 (長度: {len(chunk.strip())})")
                    continue
                
                if USE_QA_SEPARATION:
                    # 嘗試分離問題和答案
                    question, answer = separate_question_answer(chunk)
                    
                    if question and answer:
                        successful_separations += 1
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
                        print(f"    Q{i+1}: ✅ 分離成功 - 問題({len(question)}字) + 答案({len(answer)}字)")
                    else:
                        failed_separations += 1
                        # 如果無法分離，退回到原始方式
                        doc = Document(
                            page_content=chunk,
                            metadata={**base_md, "paragraph_index": i, "type": "qa_pair"}
                        )
                        docs.append(doc)
                        print(f"    Q{i+1}: ❌ 分離失敗，使用原始格式 (長度: {len(chunk)})")
                        # 顯示無法分離的原因
                        chunk_preview = chunk[:100].replace('\n', '\\n')
                        print(f"           內容預覽: \"{chunk_preview}...\"")
                else:
                    # 不分離，使用原始QA組合
                    doc = Document(
                        page_content=chunk,
                        metadata={**base_md, "paragraph_index": i, "type": "qa_pair"}
                    )
                    docs.append(doc)
                    print(f"    Q{i+1}: 使用原始QA組合格式 (長度: {len(chunk)})")
            
            # 統計摘要
            print(f"  📈 處理統計：")
            print(f"     - 原始Q標記數量: {len(q_matches)}")
            print(f"     - 切分的chunk數量: {len(chunks)}")
            print(f"     - 長度過濾掉: {filtered_out}")
            if USE_QA_SEPARATION:
                print(f"     - 問答分離成功: {successful_separations}")
                print(f"     - 問答分離失敗: {failed_separations}")
            print(f"     - 最終chunk數量: {len(docs)}")
            
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
                
                # 清理答案開頭的標記
                answer = re.sub(r'^(?:A\s*\d*\s*[：:]|答\s*[：:]|答案\s*[：:])\s*', '', answer, flags=re.IGNORECASE).strip()
                
                # 驗證分離結果 - 統一長度檢查條件
                if len(question) > 2 and len(answer) > 5:
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
            
            # 統一長度檢查條件 - 修復Q58分離失敗問題
            if len(question) > 2 and len(answer) > 5:
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
    
    # 統計變量
    original_count = len(documents)
    filtered_count = 0
    
    # 進行簡單的過濾，例如去除過短的段落
    filtered_docs = []
    for i, doc in enumerate(documents):
        content_length = len(doc.page_content.strip())
        
        # 對於分離的問答格式，考慮問題和答案的總長度
        if doc.metadata.get('type') == 'qa_separated':
            answer = doc.metadata.get('answer', '')
            total_length = content_length + len(answer)
            # 對於問答格式，使用更寬鬆的過濾條件
            if total_length < 8:  # 問題+答案總長度小於10才過濾
                filtered_count += 1
                print(f"  ⚠️  過濾第{i+1}個chunk (問題+答案總長度僅{total_length}字符): 問題=\"{doc.page_content.strip()[:20]}...\" 答案=\"{answer[:20]}...\"")
                continue
        else:
            # 對於非問答格式，使用原來的過濾條件
            if content_length < 3:
                filtered_count += 1
                print(f"  ⚠️  過濾第{i+1}個chunk (長度僅{content_length}字符): \"{doc.page_content.strip()[:30]}...\"")
                continue
        
        # 保留有效段落
        filtered_docs.append(doc)
    
    # 詳細統計
    print(f"  📊 最終過濾統計：")
    print(f"     - 輸入chunk數量: {original_count}")
    print(f"     - 因長度過短被過濾: {filtered_count}")
    print(f"     - 最終保留chunk數量: {len(filtered_docs)}")
    
    if filtered_count > 0:
        print(f"  💡 提示：有{filtered_count}個chunk因長度小於5字符被過濾")
        print(f"     如需保留這些短chunk，可修改過濾條件")
    
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
    """建立問答鏈，支援混合檢索和多語言回復"""
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
        
        # 多語言 Prompt 模板
        chinese_template = """你是一個專業的robotic bar客服機器人。請根據以下提供的問答對回答用戶問題。

說明：以下每個項目包含一個相關問題和對應答案。請基於這些資訊回答用戶的問題。

要求：
1. 回答要簡潔明確，不超過3句話
2. 不要使用任何特殊符號如星號、破折號、項目符號等
3. 直接回答重點，避免冗長說明
4. 優先使用完全匹配或最相關的問答對來回答
5. 如果找到相關的問答對就基於其答案回答，沒有相關資訊才說"您的問題不在我的回答範疇，請詢問櫃檯專員"
6. 必須使用繁體中文回答

相關問答對：
{context}

用戶問題：{question}

簡短回答："""

        english_template = """You are a professional robotic bar customer service robot. Please answer user questions based on the provided Q&A pairs below.

Instructions: Each item below contains a relevant question and corresponding answer. Please base your response on this information.

Requirements:
1. Keep answers concise and clear, no more than 3 sentences
2. Do not use special symbols like asterisks, dashes, or bullet points
3. Answer directly to the point, avoid lengthy explanations
4. Prioritize using exactly matching or most relevant Q&A pairs for answers
5. If relevant Q&A pairs are found, base your answer on them. Only say "Your question is beyond my scope of knowledge, please consult the counter staff" if no relevant information is found
6. Must answer in English

Relevant Q&A pairs:
{context}

User question: {question}

Concise answer:"""
        
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
        
        def format_docs_english(docs):
            formatted_parts = []
            for i, doc in enumerate(docs, 1):
                # 檢查是否為分離的問答格式
                if doc.metadata.get('type') == 'qa_separated':
                    question = doc.page_content
                    answer = doc.metadata.get('answer', 'Answer not found')
                    formatted_parts.append(f"Q&A Pair {i}:\nQuestion: {question}\nAnswer: {answer}")
                else:
                    # 原始格式
                    formatted_parts.append(f"Document {i}:\n{doc.page_content}")
            return "\n\n".join(formatted_parts)
        
        # 包裝QA鏈以支持多語言
        class MultilingualQAChain:
            def __init__(self, llm, retriever, chinese_template, english_template, format_func_zh, format_func_en):
                self.llm = llm
                self.retriever = retriever
                self.chinese_prompt = PromptTemplate.from_template(chinese_template)
                self.english_prompt = PromptTemplate.from_template(english_template)
                self.format_func_zh = format_func_zh
                self.format_func_en = format_func_en
            
            def invoke(self, inputs):
                # 獲取查詢和語言參數
                query = inputs["query"]
                language = inputs.get("language", "zh")  # 默認中文
                
                # 標準化查詢中的標點符號，提高檢索精度
                normalized_query = normalize_punctuation(query)
                
                # 檢索相關文檔
                docs = self.retriever.get_relevant_documents(normalized_query)
                
                # 根據語言選擇格式化函數和提示模板
                if language == "en":
                    formatted_context = self.format_func_en(docs)
                    prompt_text = self.english_prompt.format(context=formatted_context, question=query)
                else:
                    formatted_context = self.format_func_zh(docs)
                    prompt_text = self.chinese_prompt.format(context=formatted_context, question=query)
                
                # 調用LLM
                answer = self.llm.invoke(prompt_text)
                
                return {
                    "result": answer,
                    "source_documents": docs
                }
        
        multilingual_qa_chain = MultilingualQAChain(
            llm, retriever, chinese_template, english_template, format_docs, format_docs_english
        )
        
        print("多語言QA鏈建立成功。")
        return multilingual_qa_chain
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
   建議嘗試 0.1-0.2 較寬鬆
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
    
    # 【注意】由於我們修改了BM25的創建邏輯，即使快取存在也可能需要重建。
    # 這裡的邏輯保持不變，但主程序啟動時會刪除舊快取，確保新邏輯生效。
    try:
        # 嘗試載入快取的BM25檢索器
        if os.path.exists(bm25_cache_file):
            print("載入快取的BM25檢索器...")
            with open(bm25_cache_file, 'rb') as f:
                bm25_retriever = pickle.load(f)
            # 確保k值與當前設置一致
            bm25_retriever.k = RETRIEVAL_K
            print("BM25檢索器載入成功")
            return bm25_retriever
    except Exception as e:
        print(f"載入BM25快取失敗: {e}，將重新建立...")
    
    # 建立新的BM25檢索器
    try:
        # 準備用於BM25的文檔，只包含需要被索引的內容。
        # 這樣做可以確保元數據與原始文檔完全一致。
        bm25_docs = []
        for doc in texts:
            # 對於分離的問答格式，只對問題部分進行BM25索引
            if doc.metadata.get('type') == 'qa_separated':
                content_to_index = doc.page_content  # page_content 已經是問題
            else:
                content_to_index = doc.page_content
            
            # 創建一個新的Document對象，只包含要索引的文本和原始元數據
            # 這樣可以避免對答案或其他元數據字段進行不必要的分詞和索引
            bm25_doc = Document(page_content=content_to_index, metadata=doc.metadata)
            bm25_docs.append(bm25_doc)

        print("正在使用jieba分詞器建立BM25檢索器...")
        # 【關鍵修正】直接傳入原始文檔和分詞函數。
        # from_documents會自動對文檔內容和後續的所有查詢應用此函數。
        bm25_retriever = BM25Retriever.from_documents(
            documents=bm25_docs,
            preprocess_func=jieba_tokenizer
        )
        bm25_retriever.k = RETRIEVAL_K  # 使用統一的檢索數量設定
        
        # 儲存到快取
        print("儲存BM25檢索器到快取...")
        with open(bm25_cache_file, 'wb') as f:
            pickle.dump(bm25_retriever, f)
        
        print("BM25檢索器建立成功（已內置jieba分詞功能）")
        return bm25_retriever
        
    except Exception as e:
        print(f"建立BM25檢索器時發生錯誤: {e}")
        return None

def play_prompt_audio(detected_language="zh"):
    """播放提示音，在用戶提問後、系統開始思考時播放（IndexTTS版本）
    
    Args:
        detected_language: 檢測到的語言 ('zh' 或 'en')
    """
    if not ENABLE_PROMPT_AUDIO or not ENABLE_TTS_OUTPUT:
        return
    
    # 檢查語音參考文件是否存在
    if not os.path.exists(TTS_VOICE_PATH):
        print("⚠️ 語音參考文件不存在，無法播放提示音")
        return
    
    # 創建一個非阻塞的提示音播放函數
    def play_prompt_in_background():
        try:
            # 載入TTS系統
            if not load_tts_system():
                print("⚠️ TTS系統載入失敗，無法播放提示音")
                return
            
            # 根據檢測到的語言選擇提示音文本
            if detected_language == "en":
                # 英文提示音
                prompt_text = "Thank you for your question. Let me think about it and get back to you shortly."
                prompt_lang = "English"
            else:
                # 中文提示音
                prompt_text = "感謝您的提問，我思考一下，請稍後。"
                prompt_lang = "中文"
            
            print(f"🎵 播放提示音 ({prompt_lang})...")
            
            # 生成並播放提示音
            sampling_rate, wav_data = tts_system.infer(
                TTS_VOICE_PATH, 
                prompt_text, 
                output_path=None  # 不保存到文件
            )
            
            # 播放提示音
            sd.play(wav_data, sampling_rate)
            sd.wait()  # 等待播放完成
            
            print("✅ 提示音播放完成")
            
        except Exception as e:
            print(f"❌ 播放提示音時發生錯誤: {e}")
    
    # 在後台線程中播放提示音，避免阻塞主程序
    try:
        import threading
        prompt_thread = threading.Thread(target=play_prompt_in_background)
        prompt_thread.daemon = True  # 設為守護線程
        prompt_thread.start()
    except Exception as e:
        print(f"⚠️ 啟動提示音線程時發生錯誤: {e}")

# --- 新增：檢索後處理函數 ---
def normalize_punctuation(text):
    """
    標準化標點符號，將半角標點符號轉換為全角標點符號
    這可以提高檢索精度，因為文檔中使用的是全角標點符號
    """
    # 半角到全角的標點符號映射
    punctuation_map = {
        "?": "？",
        "!": "！",
        ",": "，",
        ".": "。",
        ":": "：",
        ";": "；",
        "(": "（",
        ")": "）",
        "[": "［",
        "]": "］",
        "{": "｛",
        "}": "｝",
        '"': '"',
        "'": "'",
        "<": "＜",
        ">": "＞"
    }
    
    for half_width, full_width in punctuation_map.items():
        text = text.replace(half_width, full_width)
    
    return text

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

    # --- 步驟 6: 查詢迴圈 ---
    print("===================================")
    print(f" RAG 系統已就緒 (LLM: {LLM_PROVIDER.upper()})！")
    print(f" 檢索模式: {'混合檢索 (Dense + BM25)' if USE_HYBRID_RETRIEVAL else '純Dense檢索'}")
    if USE_HYBRID_RETRIEVAL:
        print(f" 權重配置: Dense={DENSE_WEIGHT}, BM25={SPARSE_WEIGHT}")

    # 預載入Whisper模型 (如果啟用語音輸入)
    if ENABLE_VOICE_INPUT:
        print("預載入語音識別模型（可能需要一些時間）...")
        load_whisper_model()
        print(f" 語音輸入功能: 已啟用 (Whisper模型: {WHISPER_MODEL_SIZE})")
        print("    🌐 多語言支持：檢測到英文時將以英文回復，檢測到中文時以中文回復")
    else:
        print(" 語音輸入功能: 未啟用")

    # 預載入TTS系統 (如果啟用語音輸出)
    tts_enabled = False
    if ENABLE_TTS_OUTPUT:
        print("正在初始化TTS語音輸出功能...")
        print("預載入TTS語音合成系統（可能需要一些時間）...")
        if load_tts_system():
            print(f" TTS語音輸出功能: 已啟用 (語音: {os.path.basename(TTS_VOICE_PATH)})")
            if ENABLE_PROMPT_AUDIO:
                print("    🎵 提示音功能已啟用（問題處理時播放友好提示）")
            tts_enabled = True
        else:
            print(" TTS語音輸出功能: 載入失敗，已禁用")
            tts_enabled = False
    else:
        print(" TTS語音輸出功能: 未啟用")

    print(" 系統默認使用語音輸入模式。")
    print(" 特殊指令:")
    print(" - 按 Enter 開始語音提問")
    print(" - 輸入 'text' 或 't' 切換到文字輸入模式")
    print(" - 輸入 'quit' 或 'exit' 來結束程式")
    print(" - 輸入 'debug' 切換調試模式（顯示檢索詳情）")
    print(" - 輸入 'help' 查看優化建議")
    if tts_enabled:
        print(" - 輸入 'tts' 手動播放上一次答案的語音")
    print("===================================")

    debug_mode = False  # 調試模式開關
    text_mode = False   # 文字模式開關，默認為False（語音模式）
    last_answer = ""    # 保存上一次的答案，用於TTS重播

    while True:
        try:
            if ENABLE_VOICE_INPUT and not text_mode:
                # 默認語音輸入模式
                print(f"\n{BOLD}=== 語音輸入模式 ==={RESET}")
                command = input("按 Enter 開始語音提問，或輸入指令 > ")
                
                # 處理特殊命令
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
                    continue
                elif tts_enabled and command.lower() == 'tts':
                    if last_answer:
                        print("正在重播上一次答案的語音...")
                        text_to_speech(last_answer)
                    else:
                        print("沒有可播放的答案")
                    continue
                elif command.strip() != '':
                    # 如果輸入了其他文字，當作文字問題處理
                    question = command
                else:
                    # 空輸入，開始語音錄音
                    print("開始語音錄音...")
                    
                    # 創建停止事件
                    stop_event = threading.Event()
                    
                    # 創建錄音線程
                    print(f"{BOLD}正在錄音...{RESET} (按下 Enter 停止)")
                    record_thread = threading.Thread(target=record_audio, args=(stop_event,))
                    record_thread.start()
                    
                    # 等待用戶按下Enter鍵停止錄音
                    input("按下Enter鍵停止錄音...")
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
                    
                    # 將 Whisper 偵測的語言代碼映射到我們支持的語言
                    detected_language = map_whisper_language_to_supported(detected_lang)
                    print(f"檢測到的語言: {detected_language} ({'中文' if detected_language == 'zh' else '英文'})")
            else:
                # 文字輸入模式
                print(f"\n{BOLD}=== 文字輸入模式 ==={RESET}")
                question = input("請輸入您的問題 (或輸入 'voice'/'v' 切換回語音模式" + 
                                (", 'tts' 重播語音)" if tts_enabled else "") + " > ")
                
                # 處理特殊命令
                if question.lower() in ['quit', 'exit']:
                    break
                elif question.lower() == 'debug':
                    debug_mode = not debug_mode
                    print(f"調試模式已{'開啟' if debug_mode else '關閉'}")
                    continue
                elif question.lower() == 'help':
                    print_optimization_suggestions()
                    continue
                elif ENABLE_VOICE_INPUT and question.lower() in ['voice', 'v']:
                    text_mode = False
                    print("已切換到語音輸入模式")
                    continue
                elif tts_enabled and question.lower() == 'tts':
                    if last_answer:
                        print("正在重播上一次答案的語音...")
                        text_to_speech(last_answer)
                    else:
                        print("沒有可播放的答案")
                    continue

            if not question.strip(): # 忽略空輸入
                continue

            # 確定語言（用於提示音和TTS）
            if ENABLE_VOICE_INPUT and not text_mode and 'detected_language' in locals():
                # 語音模式且有檢測到的語言
                tts_language = detected_language
            else:
                # 文字模式，使用默認中文
                tts_language = "zh"

            print(f"正在處理您的問題 (使用 {LLM_PROVIDER.upper()} LLM)...")
            
            # 新增：播放提示音
            if tts_enabled and ENABLE_PROMPT_AUDIO and not debug_mode:  # 非調試模式才播放提示音
                play_prompt_audio(tts_language)
            
            # 如果開啟調試模式，先顯示檢索結果
            if debug_mode:
                print("\n=== 調試資訊：檢索結果 ===")
                
                # 【修正】在調試模式下也使用標準化後的查詢
                normalized_question_for_debug = normalize_punctuation(question)
                print(f"調試查詢 (標準化後): '{normalized_question_for_debug}'")

                if USE_HYBRID_RETRIEVAL:
                    print("混合檢索模式調試：")
                    
                    # Dense檢索結果
                    print("1. Dense檢索結果：")
                    dense_retriever = vectorstore.as_retriever(
                        search_type="similarity_score_threshold",
                        search_kwargs={"k": RETRIEVAL_K, "score_threshold": SIMILARITY_THRESHOLD}
                    )
                    dense_docs = dense_retriever.get_relevant_documents(normalized_question_for_debug) # 使用修正後的查詢
                    
                    if dense_docs:
                        print(f"   檢索到 {len(dense_docs)} 個Dense相關文檔")
                        for i, doc in enumerate(dense_docs[:3]):  # 只顯示前3個
                            source = doc.metadata.get('source', '未知來源')
                            paragraph_index = doc.metadata.get('paragraph_index', -1)
                            chunk_type = doc.metadata.get('type', 'unknown')
                            
                            print(f"   Dense-{i+1}: {os.path.basename(source)}, 段落 {paragraph_index}, 類型: {chunk_type}")
                            
                            if chunk_type == 'qa_separated':
                                # 分離的問答格式
                                doc_question = doc.page_content
                                answer = doc.metadata.get('answer', '答案未找到')
                                print(f"      問題: {doc_question[:50]}{'...' if len(doc_question) > 50 else ''}")
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
                            # 【關鍵修正】不再需要手動分詞，因為分詞功能已內置於檢索器中
                            bm25_docs = bm25_retriever.get_relevant_documents(normalized_question_for_debug)
                            
                            if bm25_docs:
                                print(f"   檢索到 {len(bm25_docs)} 個BM25相關文檔")
                                for i, doc in enumerate(bm25_docs[:3]):  # 只顯示前3個
                                    source = doc.metadata.get('source', '未知來源')
                                    paragraph_index = doc.metadata.get('paragraph_index', -1)
                                    chunk_type = doc.metadata.get('type', 'unknown')
                                    
                                    print(f"   BM25-{i+1}: {os.path.basename(source)}, 段落 {paragraph_index}, 類型: {chunk_type}")
                                    
                                    if chunk_type == 'qa_separated':
                                        # 分離的問答格式
                                        # 注意：這裡顯示的page_content是未分詞的原始問題，因為我們只傳遞了分詞函數
                                        # BM25內部處理分詞，但返回的Document對象保持原樣
                                        doc_question = doc.page_content
                                        answer = doc.metadata.get('answer', '答案未找到')
                                        print(f"      問題: {doc_question[:50]}{'...' if len(doc_question) > 50 else ''}")
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
                    retrieved_docs = retriever.get_relevant_documents(normalized_question_for_debug) # 使用修正後的查詢
                    
                    if retrieved_docs:
                        print(f"檢索到 {len(retrieved_docs)} 個相關文檔塊：")
                        for i, doc in enumerate(retrieved_docs):
                            source = doc.metadata.get('source', '未知來源')
                            paragraph_index = doc.metadata.get('paragraph_index', -1)
                            chunk_type = doc.metadata.get('type', 'unknown')
                            
                            print(f"  文檔 {i+1}: {os.path.basename(source)}, 段落 {paragraph_index}, 類型: {chunk_type}")
                            
                            if chunk_type == 'qa_separated':
                                # 分離的問答格式
                                doc_question = doc.page_content
                                answer = doc.metadata.get('answer', '答案未找到')
                                print(f"    問題: {doc_question[:100]}{'...' if len(doc_question) > 100 else ''}")
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
            
            # 新增：顯示混合檢索的最終結果（僅在調試模式）
            if debug_mode:
                print("\n=== 混合檢索最終結果 ===")
                if USE_HYBRID_RETRIEVAL:
                    # 【修正】確保調試也使用標準化查詢
                    normalized_question_for_debug = normalize_punctuation(question)
                    # 使用QA鏈的檢索器獲取最終結果
                    final_retriever = qa_chain.retriever
                    final_docs = final_retriever.get_relevant_documents(normalized_question_for_debug) # 使用修正後的查詢
                    
                    print(f"混合檢索最終排序結果（實際傳給LLM的文檔）：")
                    for i, doc in enumerate(final_docs):
                        source = doc.metadata.get('source', '未知來源')
                        paragraph_index = doc.metadata.get('paragraph_index', -1)
                        chunk_type = doc.metadata.get('type', 'unknown')
                        
                        print(f"  最終-{i+1}: {os.path.basename(source)}, 段落 {paragraph_index}, 類型: {chunk_type}")
                        
                        if chunk_type == 'qa_separated':
                            doc_question = doc.page_content
                            answer = doc.metadata.get('answer', '答案未找到')
                            print(f"      問題: {doc_question[:50]}{'...' if len(doc_question) > 50 else ''}")
                            print(f"      答案: {answer[:50]}{'...' if len(answer) > 50 else ''}")
                        else:
                            content_preview = doc.page_content[:100].replace('\n', ' ') + "..."
                            print(f"      預覽: {content_preview}")
                else:
                    print("純Dense檢索模式，最終結果與Dense檢索結果相同")
                print("=" * 50)
            
            # <-- 直接調用 qa_chain.invoke，傳遞語言參數
            result = qa_chain.invoke({"query": question, "language": tts_language})
            # <-- 從結果中提取 'result'
            answer = result.get('result', '抱歉，無法生成答案。').strip()
            source_docs = result.get('source_documents', []) # 獲取來源文檔 (可選)

            print("\n答案：")
            print(answer)

            # 保存答案供TTS使用並自動播放語音
            if tts_enabled:
                last_answer = answer
                # 自動播放語音，不需要用戶確認
                print("\n正在播放語音答案...")
                text_to_speech(answer)

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

    print("--- RAG 系統已關閉 ---")

 