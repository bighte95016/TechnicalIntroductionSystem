import os
import sys
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI # 新增 OpenAI LLM
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_core.runnables import RunnableParallel
import shutil # 用於刪除目錄
from dotenv import load_dotenv # 新增 用於讀取 .env 文件

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

# --- 嵌入模型設定 (保持不變，仍然使用本地 Ollama) ---
EMBEDDING_MODEL = "mxbai-embed-large" # 使用較小的嵌入模型以提高速度

# 檢查 Python 版本
if sys.version_info < (3, 8):
    print("警告：您的 Python 版本較低，建議使用 Python 3.8 或更高版本以獲得最佳相容性。")

# --- 載入文件函數 ---
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

# --- 初始化 LLM (根據 LLM_PROVIDER 進行切換) ---
def initialize_llm():
    """根據設定初始化選擇的 LLM"""
    if LLM_PROVIDER == "ollama":
        print(f"正在初始化本地 LLM: {OLLAMA_MODEL} (透過 Ollama: {OLLAMA_BASE_URL})")
        try:
            llm = OllamaLLM(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
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
            print(f"OpenAI LLM ({OPENAI_MODEL_NAME}) 初始化成功。")
            return llm
        except Exception as e:
            print(f"初始化 OpenAI LLM ({OPENAI_MODEL_NAME}) 時發生錯誤: {e}")
            print("請檢查您的 API Key 是否有效以及網路連接。")
            return None

    else:
        print(f"錯誤：無效的 LLM_PROVIDER 設定 '{LLM_PROVIDER}'。請選擇 'ollama' 或 'openai'。")
        return None

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

    # --- 步驟 1: 載入文件 ---
    documents = load_documents(PDF_DIRECTORY)
    if not documents:
        print("無法載入文件，程式終止。")
        sys.exit(1) # 退出程式

    # --- 步驟 2: 分割文件 ---
    print("正在分割文件...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # 減小文字區塊大小以提高處理速度
        chunk_overlap=100, # 減小重疊以提高處理速度
        length_function=len,
        add_start_index=True, # 添加起始索引元數據
    )
    texts = text_splitter.split_documents(documents)
    print(f"文件被分割成 {len(texts)} 個區塊。")
    if not texts:
        print("文件分割失敗，程式終止。")
        sys.exit(1)

    # --- 步驟 3: 建立向量儲存 ---
    print("正在建立嵌入向量與向量儲存...")
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        print(f"使用嵌入模型: {EMBEDDING_MODEL}")

        print(f"正在於 {VECTORSTORE_DIR} 建立新的向量儲存...")
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=VECTORSTORE_DIR
        )
        print("向量儲存已成功建立。")
    except Exception as e:
        print(f"建立向量儲存時發生錯誤: {e}")
        print("請檢查 Ollama 服務是否正在運行，以及指定的嵌入模型是否可用。")
        sys.exit(1)

    # --- 步驟 4: 初始化 LLM ---
    llm = initialize_llm()
    if not llm:
        print("LLM 初始化失敗，程式終止。")
        sys.exit(1)

    # --- 步驟 5: 建立 LCEL RAG 鏈 ---
    print("正在建立 LCEL RAG 鏈...")
    try:
        # 設定檢索器
        k = 3
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        print(f"檢索器將檢索 top {k} 個區塊。")
        
        # 定義 Prompt 模板
        template = """
        你是一個介紹P-HUD(Panoramic HUD)的專家，請根據以下提供的上下文資訊，簡潔地回答問題。
        使用者的問題皆圍繞在P-HUD相關，
        如果你在提供的上下文中找不到答案，請明確說明你無法從文件中找到答案，不要嘗試編造或使用外部知識。

        上下文：
        {context}

        問題：{question}

        答案（請根據上下文簡潔回答）：
        """
        prompt = PromptTemplate.from_template(template)
        
        # 定義格式化檢索結果的函數
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # 使用 LCEL 運算符串連接鏈的各個部分
        rag_chain = (
            # 創建帶有兩個鍵 "context" 和 "question" 的字典
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            }
            # 將字典傳遞給提示模板
            | prompt 
            # 將提示輸出傳遞給 LLM
            | llm 
            # 將 LLM 輸出轉換為字符串
            | StrOutputParser() 
        )
        
        # 創建並行運行的鏈，同時返回回答和源文檔
        rag_chain_with_sources = RunnableParallel(
            {
                "answer": rag_chain,
                "source_documents": retriever
            }
        )
        
        print("LCEL RAG 鏈建立成功。")
        
    except Exception as e:
        print(f"建立 LCEL RAG 鏈時發生錯誤: {e}")
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
            
            # 使用 LCEL 鏈處理問題
            result = rag_chain_with_sources.invoke(question)
            
            answer = result.get('answer', '抱歉，無法生成答案。').strip()
            source_docs = result.get('source_documents', [])

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

        except KeyboardInterrupt: # 允許 Ctrl+C 中斷
             print("\n偵測到中斷指令，正在結束程式...")
             break
        except Exception as e:
            print(f"處理問題時發生未預期的錯誤: {e}")
            print("請檢查輸入、LLM 狀態或程式邏輯。")

    print("--- RAG 系統已關閉 ---") 