import ollama
import sys

# 指定要使用的模型名稱 (使用您 ollama create 時指定的名稱)
#ycchen/breeze-7b-instruct-v1_0 
#breeze-8b-gguf-q8 (Llama-Breeze2-8B-Instruct-text-only.Q8_K_M.gguf )
#breeze-8b-gguf-q4 (Llama-Breeze2-8B-Instruct-text-only.Q4_K_M.gguf )
MODEL_NAME = 'ycchen/breeze-7b-instruct-v1_0'

# 初始化對話歷史，包含系統提示
conversation_history = [
    {'role': 'system', 'content': '你是一位知識淵博的科技技術介紹專家。請用簡潔明瞭且專業的方式回答使用者關於各種科技技術的問題。'},
]

def ask_tech_expert(history):
    """
    根據提供的對話歷史與 Ollama 模型互動。

    Args:
        history: 包含對話內容的列表，每個元素是一個字典，包含 'role' 和 'content'。

    Returns:
        模型的文字回應，如果發生錯誤則返回 None。
    """
    try:
        print(f"\n[系統] 正在與模型 {MODEL_NAME} 互動...")
        response = ollama.chat(
            model=MODEL_NAME,
            messages=history
        )
        # 從回應中提取內容
        message_content = response['message']['content']
        print(f"[科技專家]: {message_content}")
        return message_content
    except Exception as e:
        print(f"\n[系統] 與 Ollama 互動時發生錯誤: {e}", file=sys.stderr)
        print(f"[系統] 請確認 Ollama 服務正在運行，並且模型 '{MODEL_NAME}' 已下載。", file=sys.stderr)
        print(f"[系統] 您可以嘗試執行 'ollama pull {MODEL_NAME}' 來下載模型。", file=sys.stderr)
        return None

if __name__ == "__main__":
    print("-" * 30)
    print(f"歡迎使用科技技術問答系統 (模型: {MODEL_NAME})")
    print("輸入您的技術問題，或輸入 'quit' / 'exit' 離開。")
    print("-" * 30)

    while True:
        try:
            user_input = input("[您]: ")
            if user_input.lower() in ['quit', 'exit']:
                print("[系統] 感謝使用，再見！")
                break

            # 將使用者輸入加入歷史
            conversation_history.append({'role': 'user', 'content': user_input})

            # 取得模型回應
            model_response = ask_tech_expert(conversation_history)

            if model_response:
                # 將模型回應加入歷史
                conversation_history.append({'role': 'assistant', 'content': model_response})
            else:
                # 如果發生錯誤，移除剛剛加入的使用者輸入，避免影響後續對話
                conversation_history.pop()
                print("[系統] 無法從模型獲取回應，請重試。")

        except KeyboardInterrupt:
            print("\n[系統] 收到中斷指令，正在離開...")
            break
        except EOFError: # 在某些終端中按 Ctrl+D 會觸發 EOFError
            print("\n[系統] 收到輸入結束信號，正在離開...")
            break
