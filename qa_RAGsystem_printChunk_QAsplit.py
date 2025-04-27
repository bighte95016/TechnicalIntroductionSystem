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

# --- 嵌入模型設定 (保持不變，仍然使用本地 Ollama) ---
EMBEDDING_MODEL = "mxbai-embed-large" # 使用較小的嵌入模型以提高速度

# 檢查 Python 版本
if sys.version_info < (3, 8):
    print("警告：您的 Python 版本較低，建議使用 Python 3.8 或更高版本以獲得最佳相容性。")


# --- 新增: 按段落處理PDF文件 ---
def load_pdf_by_paragraph(pdf_path: str):
    """讀 PDF → 清理軟換行 → 依雙換行切段，QA 格式自動合併"""
    print(f"[load] {pdf_path}")
    try:
        # ① 直接一次讀完整本（❗重點：split_pages=False）
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # 將所有頁面內容合併
        full_text = "\n".join([doc.page_content for doc in documents])
        
        # 獲取基本元數據（從第一頁）
        base_md = {}
        if documents:
            for key, value in documents[0].metadata.items():
                if key != 'page':  # 排除頁碼信息
                    base_md[key] = value
        
        # 添加文件名到元數據
        base_md.update({"source": pdf_path,
                        "file_name": os.path.basename(pdf_path)})

        # ② 正規化換行
        full_text = full_text.replace("\r\n", "\n").replace("\r", "\n")
        full_text = re.sub(r"(?<!\n)\n(?!\n)", " ", full_text)   # 軟換行→空格
        full_text = re.sub(r"\n{3,}", "\n\n", full_text)         # 3+ 行→2 行

        # ③ 依空白行切段
        paras = [p.strip() for p in full_text.split("\n\n") if p.strip()]

        # ④ 判斷是否為 QA 文件（抓 Axx：行）
        a_cnt = sum(1 for p in paras if re.match(r"^A\d+[:：]", p))
        is_qa = a_cnt > 3
        docs = []

        if is_qa:
            print("  ➜ 偵測到 QA 格式，合併 Q+A")
            buf = []
            for p in paras:
                buf.append(p)
                if re.match(r"^A\d+[:：]", p):       # 遇到回答就封包
                    docs.append("\n".join(buf))
                    buf.clear()
            if buf:  docs.append("\n".join(buf))     # 殘餘處理
            docs = [Document(page_content=qa,
                             metadata={**base_md,
                                       "paragraph_index": i,
                                       "type": "qa_pair"})
                    for i, qa in enumerate(docs)]
        else:
            print("  ➜ 一般段落文件")
            docs = [Document(page_content=p,
                             metadata={**base_md,
                                       "paragraph_index": i,
                                       "type": "paragraph"})
                    for i, p in enumerate(paras)]

        print(f"  ✓ 共產生 {len(docs)} 個 chunk")
        return docs
    except Exception as e:
        print(f"處理 {pdf_path} 時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return []

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

# --- 5. 建立 QA 鏈 (使用 RetrievalQA) ---
def create_qa_chain(llm, vectorstore):
    """建立問答鏈"""
    print("正在建立 QA 鏈...")
    try:
        k=3
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        print(f"檢索器將檢索 top {k} 個區塊。")
        
        # 定義 Prompt 模板 (極簡版本)
        template = """
        你是一個介紹全景抬頭式顯示器的工研院專家，請根據以下提供的上下文資訊，簡潔地回答問題。
        使用者的問題皆圍繞在P-HUD相關，
        如果你在提供的上下文中找不到答案或相關性不高，回答：「這個問題不在我的回答範疇，請詢問一旁的專家」

        上下文：
        {context}

        問題：{question}

        答案（請根據上下文簡潔回答）：
        """
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff", # 使用 Stuff 方法
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        print("QA 鏈建立成功。")
        return qa_chain
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
        
        # 獲取內容
        content = chunk.page_content
        
        # 計算字數和行數
        char_count = len(content)
        line_count = content.count('\n') + 1
        
        # 格式化内容預覽(限制在100字符内)
        content_preview = content.replace('\n', ' ')
        if len(content_preview) > 100:
            content_preview = content_preview[:97] + "..."
        
        # 打印塊信息
        print(f"塊 #{i+1}")
        print(f"  來源: {os.path.basename(source)}")
        print(f"  段落索引: {paragraph_index}")
        print(f"  字符數: {char_count}")
        print(f"  行數: {line_count}")
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
                
                # 獲取內容
                content = chunk.page_content
                
                # 計算字數和行數
                char_count = len(content)
                line_count = content.count('\n') + 1
                
                # 寫入塊信息
                f.write(f"塊 #{i+1}\n")
                f.write(f"  來源: {os.path.basename(source)}\n")
                f.write(f"  段落索引: {paragraph_index}\n")
                f.write(f"  字符數: {char_count}\n")
                f.write(f"  行數: {line_count}\n")
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

# --- 主要執行流程 ---
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

    qa_chain = create_qa_chain(llm, vectorstore)
    if not qa_chain:
        print("QA 鏈建立失敗，程式終止。")
        sys.exit(1)

    # --- 步驟 6: 查詢迴圈 ---
    print("===================================")
    print(f" RAG 系統已就緒 (LLM: {LLM_PROVIDER.upper()})！")
    print(" 請輸入您的問題（關於 PDF 文件的內容）。")
    print(" 輸入 'quit' 或 'exit' 來結束程式。")
    print("===================================")

    while True:
        try:
            question = input("請輸入您的問題 > ")
            if question.lower() in ['quit', 'exit']:
                break
            if not question.strip(): # 忽略空輸入
                continue

            print(f"正在處理您的問題 (使用 {LLM_PROVIDER.upper()} LLM)...")
            
            # <-- 直接調用 qa_chain.invoke
            result = qa_chain.invoke({"query": question})
            # <-- 從結果中提取 'result'
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

 