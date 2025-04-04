import time
from langchain_ollama import OllamaEmbeddings

# 測試函數
def test_embedding_speed():
    embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:11434")
    
    # 準備較大文本進行測試
    test_text = "這是一個測試文本" * 100
    
    # 計時開始
    start_time = time.time()
    
    # 執行嵌入
    vector = embeddings.embed_query(test_text)
    
    # 計算耗時
    elapsed = time.time() - start_time
    
    print(f"嵌入處理耗時: {elapsed:.2f} 秒")
    print(f"向量維度: {len(vector)}")

if __name__ == "__main__":
    test_embedding_speed()