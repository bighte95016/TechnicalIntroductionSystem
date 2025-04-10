import time
import subprocess
from langchain_ollama import OllamaEmbeddings

def get_gpu_usage():
    try:
        # 執行 nvidia-smi 並取得部分輸出 (例如 GPU 利用率和記憶體使用)
        output = subprocess.check_output(
            "nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader",
            shell=True
        )
        return output.decode("utf-8").strip()
    except Exception as e:
        return f"無法取得 GPU 使用狀態: {e}"

def test_embedding_speed():
    embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:11434")
    
    # 建立較長的文本作為測試
    test_text = "這是一個測試文本" * 100
    
    # 在進行嵌入前查看 GPU 使用狀態
    print("=== 嵌入前 GPU 狀態 ===")
    print(get_gpu_usage())
    
    start_time = time.time()    
    # 執行嵌入
    vector = embeddings.embed_query(test_text)    
    elapsed = time.time() - start_time
    
    # 嵌入後，檢查 GPU 使用狀態
    print("=== 嵌入後 GPU 狀態 ===")
    print(get_gpu_usage())
    
    print(f"嵌入處理耗時: {elapsed:.2f} 秒")
    print(f"向量維度: {len(vector)}")

if __name__ == "__main__":
    test_embedding_speed()
