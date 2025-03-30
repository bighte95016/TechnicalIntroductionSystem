from ctransformers import AutoModelForCausalLM
import sys

# --- 設定 ---
# GGUF 模型檔案的路徑 (相對於此腳本執行的位置)
MODEL_PATH = "./model/Llama-Breeze2-8B-Instruct-text-only.Q4_K_M.gguf"
# 模型類型 (通常是 llama, mistral, gptneox 等，需要根據模型基礎決定)
# Breeze 是基於 Llama2，所以設定為 'llama' 通常是正確的
MODEL_TYPE = "llama"
# 要分配給 GPU 的層數 (如果不安裝 CUDA 版本或設為 0，則使用 CPU)
# 將此值設為一個較大的數字 (例如 50) 以最大化 GPU 使用率 (如果有的話)
GPU_LAYERS = 0
# --- 設定結束 ---

llm = None

def initialize_model():
    """載入 GGUF 模型"""
    global llm
    try:
        print(f"[系統] 正在從 {MODEL_PATH} 載入模型...")
        print(f"[系統] 模型類型: {MODEL_TYPE}, GPU 層數: {GPU_LAYERS}")
        llm = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            model_file=MODEL_PATH, # GGUF/GGML 必須明確指定 model_file
            model_type=MODEL_TYPE,
            gpu_layers=GPU_LAYERS,
            # context_length=2048, # 可選：設定上下文長度
            # hf=True # 如果模型是從 Hugging Face Hub 下載並希望自動轉換，可以設置
        )
        print("[系統] 模型載入成功！")
        return True
    except Exception as e:
        print(f"\n[系統] 載入模型時發生錯誤: {e}", file=sys.stderr)
        print("[系統] 請確認：", file=sys.stderr)
        print(f"  1. ctransformers 已正確安裝 (若需 GPU，請安裝 CUDA 版本)。", file=sys.stderr)
        print(f"  2. 模型檔案路徑 '{MODEL_PATH}' 正確無誤。", file=sys.stderr)
        print(f"  3. MODEL_TYPE '{MODEL_TYPE}' 與模型架構相符。", file=sys.stderr)
        return False

def generate_response(prompt):
    """使用載入的模型生成回應"""
    if not llm:
        print("[系統] 模型尚未初始化。", file=sys.stderr)
        return None
    try:
        print("[系統] 正在生成回應...")
        # 這裡直接傳遞 prompt，ctransformers 會處理它
        # 您可以調整 max_new_tokens, temperature 等參數
        response = llm(prompt, max_new_tokens=512, temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.1)
        print(f"[模型]: {response}")
        return response
    except Exception as e:
        print(f"\n[系統] 生成回應時發生錯誤: {e}", file=sys.stderr)
        return None

if __name__ == "__main__":
    if not initialize_model():
        sys.exit(1) # 如果模型載入失敗，則退出程式

    print("-" * 30)
    print(f"歡迎使用 GGUF 模型問答系統 (模型路徑: {MODEL_PATH})")
    print("輸入您的問題，或輸入 'quit' / 'exit' 離開。")
    print("-" * 30)

    while True:
        try:
            user_input = input("[您]: ")
            if user_input.lower() in ['quit', 'exit']:
                print("[系統] 感謝使用，再見！")
                break

            generate_response(user_input)

        except KeyboardInterrupt:
            print("\n[系統] 收到中斷指令，正在離開...")
            break
        except EOFError:
            print("\n[系統] 收到輸入結束信號，正在離開...")
            break
