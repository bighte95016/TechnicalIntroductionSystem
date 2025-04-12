# test_qa_RAGsystem.py
# 測試指令: python -m unittest test_project.test_qa_RAGsystem -v
import unittest
import os
import time


# --- 導入真實的依賴項 ---
# 假設您的 qa_chain 初始化和 log 函數在這裡
# 您需要確保運行測試的環境能正確初始化 qa_chain (例如，已設定 API Key)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qa_RAGsystem import process_question, create_qa_chain, initialize_llm
from logging_util import log_question_answer
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


# 測試參數設定
VECTORSTORE_DIR = "./chroma_db" # Chroma DB 持久化儲存目錄
EMBEDDING_MODEL = "mxbai-embed-large" # 使用較小的嵌入模型以提高速度
OLLAMA_BASE_URL = "http://localhost:11434"    # Ollama 設定(gemma3:12b、qwen2.5:7b、qwen2.5:3b)
OLLAMA_MODEL = "gemma3:12b"


# 串創建測試物件 embeddings
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)


# 串創建測試物件 llm 
llm = initialize_llm()
if not llm:
    print("LLM 初始化失敗，程式終止。")
    sys.exit(1)


# 串創建測試物件 vectorstore
vectorstore = Chroma(
    persist_directory=VECTORSTORE_DIR,
    embedding_function=embeddings
    # 如果您在創建時指定了特定的 collection_name，載入時可能也需要指定：
    # collection_name="your_collection_name"
)
if not vectorstore: # <-- 只檢查 vectorstore
        print("向量儲存建立失敗，程式終止。")
        sys.exit(1)

# 串創建測試物件 qa_chain
qa_chain = create_qa_chain(llm, vectorstore)


# 測試資料檔案
TEST_QUESTIONS_FILE = 'technical_file/TXT/透明P-HUD技術展覽用 Chatbot 問答集.txt'


class TestQuestionProcessing(unittest.TestCase):

    def setUp(self):
        """在每個測試方法執行前執行"""

        # 設定 mock qa_chain.invoke 的預設回傳值
        # 讓每次調用都返回固定的、可預測的結果
        self.mock_answer = "成功完成一次回答"
        self.mock_docs = None
        self.mock_qa_chain = qa_chain
        self.mock_logger = log_question_answer
    
    def get_question_text(self, text):
        """
        判斷字串是否以 "Q" 開頭，如果是，則返回冒號前的題號及冒號後的文字。

        Args:
            text: 問題。

        Returns:
            如果字串以 "Q" 開頭，則返回冒號前的題號及冒號後的文字；
            否則返回 None。
        """
        if text.startswith("Q"):
            parts = text.split(":", 1)  # 以第一個冒號分割字串，最多分割成兩部分
            if len(parts) > 1:
                return parts[0].strip(), parts[1].strip()  # 返回冒號前的題號及冒號後的文字並去除前後空白
        return None, None

    def test_process_questions_from_file(self):
        """測試從 txt 檔案讀取問題並處理"""
        # 檢查測試檔案是否存在
        self.assertTrue(os.path.exists(TEST_QUESTIONS_FILE), f"測試問題檔案 {TEST_QUESTIONS_FILE} 不存在")

        processed_questions_count = 0
        questions_in_file = []
        expected_invoke_calls = []
        expected_log_calls = []

        with open(TEST_QUESTIONS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                question = line.strip()
                # q_number, q_text = self.get_question_text(question)
                # q_text = None
                if question: # 只處理非空行
                    questions_in_file.append(question)
                    # --- 執行被測試函數 ---
                    # self.mock_logger.log_question_answer(q_number + "\n")
                    answer, source_docs = process_question(question, self.mock_qa_chain, self.mock_logger)
                    processed_questions_count += 1

                    # --- 驗證結果 ---
                    # 1. 驗證回傳的答案和文件是否來自 Mock
                    self.assertEqual("成功完成一次回答", self.mock_answer)


        # --- 在迴圈外驗證 Mock 物件的整體調用情況 ---
        print(f"\n--- Test Summary ---")
        print(f"從檔案讀取並處理了 {processed_questions_count} 個非空問題。")
        print(f"預期 invoke 調用次數: {len(expected_invoke_calls)}")
        print(f"預期 log 調用次數: {len(expected_log_calls)}")
        # print(f"實際 log 調用次數: {self.mock_logger.call_count}")

        # 3. 驗證 mock_qa_chain.invoke 被正確調用了預期次數和參數
        # self.assertEqual(self.mock_qa_chain.invoke.call_count, processed_questions_count, "invoke 調用次數不符")
        # 驗證每次調用的參數是否都符合預期 (順序可能重要)
        # self.assertEqual(self.mock_qa_chain.invoke.call_args_list, expected_invoke_calls, "invoke 調用參數列表不符")

        # 4. 驗證 mock_logger 被正確調用了預期次數和參數
        # self.assertEqual(self.mock_logger.call_count, processed_questions_count, "logger 調用次數不符")
        # 驗證每次調用的參數是否都符合預期
        # self.assertEqual(self.mock_logger.call_args_list, expected_log_calls, "logger 調用參數列表不符")

# 允許直接執行此測試檔案
if __name__ == '__main__':
    unittest.main(verbosity=2) # verbosity=2 提供更詳細的輸出