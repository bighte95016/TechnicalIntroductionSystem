import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
# 更新導入路徑以解決棄用警告
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
import os
import getpass
from dotenv import load_dotenv

# 載入.env文件
load_dotenv()

# 檢查GPU可用性
print("\n=== 系統資訊 ===")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"當前GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU數量: {torch.cuda.device_count()}")
    print(f"當前GPU記憶體使用: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
    print(f"當前GPU記憶體總量: {torch.cuda.get_device_properties(0).total_memory/1024**2:.2f} MB")
else:
    print("警告: 未檢測到GPU，將使用CPU運行，這可能會非常慢！")
print("================\n")

# ---------------------------
# 0. 設置Hugging Face Token
# ---------------------------
# 從環境變數獲取，如果沒有則讓用戶輸入
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

if not HUGGINGFACE_TOKEN:
    print("\n=== 需要Hugging Face Token ===")
    print("使用google/gemma-3-12b-it模型需要授權訪問")
    print("請在Hugging Face網站接受此模型的使用條款: https://huggingface.co/google/gemma-3-12b-it")
    print("然後提供您的Hugging Face Token")
    print("或者將token添加到.env文件中: HUGGINGFACE_TOKEN=your_token_here")
    HUGGINGFACE_TOKEN = getpass.getpass("請輸入您的Hugging Face Token (輸入不會顯示): ")

if HUGGINGFACE_TOKEN:
    print("正在使用提供的Token登入Hugging Face Hub...")
    login(token=HUGGINGFACE_TOKEN)
    print("登入成功!")
else:
    print("警告: 未提供Hugging Face Token! 無法訪問google/gemma-3-12b-it模型。")
    print("程序將退出。")
    exit(1)

# ---------------------------
# 1. 加載LLM
# ---------------------------
# 使用Gemma 3.0 模型，需要Hugging Face授權
model_name = "google/gemma-3-4b-it"

# 其他可選模型（如遇問題可啟用）
# model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # 開放模型
# model_name = "google/gemma-7b-it"  # 較小的Gemma版本

try:
    print(f"正在載入模型: {model_name}")
    print("這可能需要幾分鐘時間，具體取決於您的網絡和硬件...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 明確設置設備
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用設備: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # 使用半精度以減少記憶體使用
        # device_map="auto",         # 移除 device_map="auto"
        low_cpu_mem_usage=True     # 低記憶體使用模式
    ).to(device) # <-- 將整個模型明確移動到檢測到的設備 (GPU或CPU)
    
    # 打印 accelerate 創建的設備映射 (現在應該不存在或為空)
    # print("Accelerate 設備映射:", model.hf_device_map) 
    # 檢查模型實際在哪個設備
    print(f"模型實際加載設備: {model.device}")
    
    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,        # 設置最大長度
        max_new_tokens=512,     # 設置生成的新token數量
        do_sample=True,
        temperature=0.3,        # 降低溫度使回答更加確定
        top_p=0.95,
        truncation=True,        # 明確啟用截斷
        padding=True,           # 啟用填充
        device=0 if torch.cuda.is_available() else -1 # 在 Pipeline 中仍然指定設備
    )
    local_llm = HuggingFacePipeline(pipeline=gen_pipeline)
    print("模型載入成功!")
    
except Exception as e:
    print(f"載入模型時發生錯誤: {e}")
    print("\n可能的問題:")
    print("1. 您的Hugging Face Token無效或未在網站上接受模型使用條款")
    print("2. 您的系統沒有足夠的GPU/RAM資源載入12B參數的模型")
    print("3. 網絡連接問題導致模型下載失敗")
    print("\n解決方案:")
    print("1. 確保您已在Hugging Face網站接受模型使用條款: https://huggingface.co/google/gemma-3-12b-it")
    print("2. 檢查您的Token是否正確")
    print("3. 考慮使用較小的模型，如google/gemma-7b-it")
    print("4. 確保您有強大的GPU和至少16GB的GPU內存")
    print("5. 檢查CUDA驅動程式是否正確安裝")
    exit(1)

# ---------------------------
# 2. PDF 文檔加載與預處理
# ---------------------------
print("正在載入PDF文檔...")
# 使用 PyPDFLoader 從指定目錄加載所有 PDF 文件
try:
    loader = DirectoryLoader('./technical_file/PDF', glob='*.pdf', loader_cls=PyPDFLoader)
    docs = loader.load()
    print(f"成功載入 {len(docs)} 個文檔")

    # 使用 CharacterTextSplitter 將長文檔分割成小片段
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(docs)
    print(f"文檔分割為 {len(docs)} 個片段")
except Exception as e:
    print(f"載入或分割文檔時發生錯誤: {e}")
    exit(1)

# ---------------------------
# 3. 向量化與向量庫構建
# ---------------------------
print("正在建立向量資料庫...")
# 使用更輕量級且效能良好的嵌入模型
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
try:
    # 強制在CPU上運行Embedding，避免與GPU上的LLM衝突
    embedding_device = 'cpu'
    print(f"Initializing embeddings on device: {embedding_device}")
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={'device': embedding_device} # 指定設備
    )
    # 創建 FAISS 向量庫 (FAISS 會在其內部處理數據，通常在CPU上建立索引)
    print("Creating FAISS index...")
    vector_store = FAISS.from_documents(docs, embeddings)
    print("向量資料庫建立成功")
except Exception as e:
    print(f"建立向量資料庫時發生錯誤: {e}")
    exit(1)

# ---------------------------
# 4. 建立檢索增強問答鏈
# ---------------------------
print("正在建立RAG問答鏈...")
qa_chain = RetrievalQA.from_chain_type(
    llm=local_llm,
    chain_type="stuff",  # 可根據需求選擇不同的整合模式，如 "map_reduce"
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# ---------------------------
# 5. 問題過濾與回答接口
# ---------------------------
def answer_question(question: str) -> str:
    try:
        print("\n--- Debug: Entering answer_question ---")
        print(f"原始問題: {question}")
        
        # --- 手動獲取和格式化輸入以進行調試 ---
        try:
            print("調試：正在手動檢索文檔...")
            retrieved_docs = vector_store.similarity_search(question, k=3)
            print(f"調試：檢索到 {len(retrieved_docs)} 個文檔")
            
            # 獲取 StuffDocumentsChain 使用的提示模板
            # 注意：這依賴於 LangChain 的內部結構，未來版本可能改變
            prompt_template = qa_chain.combine_documents_chain.llm_chain.prompt
            
            # 準備輸入變量
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            inputs_for_prompt = {
                "context": context,
                "question": question
            }
            
            # 格式化最終提示
            final_prompt = prompt_template.format(**inputs_for_prompt)
            print("\n==================== 調試：最終 LLM 輸入提示 ====================")
            print(final_prompt)
            print("=================================================================\n")
            
        except Exception as e:
            print(f"調試：在準備最終提示時出錯: {e}")
            # 即使準備提示出錯，仍然嘗試調用原始鏈
        # -------------------------------------------
            
        print("調用 qa_chain.invoke...")
        # 使用新的invoke方法替代直接調用
        result = qa_chain.invoke({"query": question})
        print("--- Debug: qa_chain.invoke Succeeded ---")
        
        if 'result' in result:
            return result['result']
        else:
            print("警告: 結果中未找到 'result' 鍵。完整結果:", result)
            return "處理您的問題時出現錯誤，結果格式不正確，請重試。"
            
    except Exception as e:
        print(f"處理問題 'answer_question' 時發生嚴重錯誤: {e}")
        # 打印更詳細的錯誤追蹤 (如果設置了 CUDA_LAUNCH_BLOCKING=1)
        import traceback
        traceback.print_exc()
        return f"處理您的問題時發生技術錯誤: {str(e)}"

# ---------------------------
# 6. 主程序：接收問題並生成回答
# ---------------------------
if __name__ == "__main__":
    # --- 添加簡單的 Embedding 直接調用測試 ---
    try:
        print("\n--- Debug: 直接測試 Embedding ---")
        test_query = "什麼是p-hud" # 使用導致錯誤的查詢
        print(f"測試查詢: {test_query}")
        # 確保 embedding 在正確的設備上運行
        embedding_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用 Embedding 設備: {embedding_device}")
        # embeddings.client.to(embedding_device) # SentenceTransformer 可能不需要手動移動
        
        query_embedding = embeddings.embed_query(test_query)
        print(f"查詢嵌入向量 (前10維): {query_embedding[:10]}")
        print(f"查詢嵌入向量維度: {len(query_embedding)}")
        print("--- Debug: Embedding 測試成功 ---")
    except Exception as e:
        print(f"直接測試 Embedding 時出錯: {e}")
        import traceback
        traceback.print_exc()
        print("如果此處出錯，問題就在 Embedding 模型或其 GPU 執行。")
    # -------------------------------------

    # --- 添加簡單的 LLM 直接調用測試 ---
    try:
        print("\n--- Debug: 直接測試 LLM Pipeline ---")
        test_prompt = "你好嗎?"
        print(f"測試提示: {test_prompt}")
        test_output = gen_pipeline(test_prompt)
        print(f"LLM直接輸出: {test_output}")
        print("--- Debug: LLM Pipeline 測試成功 ---")
    except Exception as e:
        print(f"直接測試 LLM Pipeline 時出錯: {e}")
        import traceback
        traceback.print_exc()
        print("如果此處出錯，問題可能在模型加載或基礎 Pipeline 配置。")
    # -------------------------------------
        
    print("\n=== RAG 問答系統已就緒 ===")
    print(f"使用模型: {model_name}")
    print(f"使用嵌入模型: {embedding_model_name}")
    print("輸入'exit'或'quit'退出")
    print("===========================\n")
    
    while True:
        question = input("\n請輸入您的問題: ")
        if question.lower() in ['exit', 'quit']:
            print("感謝使用，再見！")
            break
            
        print("正在處理您的問題...")
        answer = answer_question(question)
        print("\n回答:", answer)

