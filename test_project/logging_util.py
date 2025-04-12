import logging
import datetime
import os

# 0. 建立 Log 檔案名稱 
LOG_DIRECTORY = "logs"

# 1. 確保 logs 目錄存在
if not os.path.exists(LOG_DIRECTORY):
    try:
        os.makedirs(LOG_DIRECTORY)
        print(f"目錄 '{LOG_DIRECTORY}' 已創建。")
    except OSError as e:
        print(f"錯誤：無法創建目錄 '{LOG_DIRECTORY}': {e}")
        LOG_DIRECTORY = "." # 退回使用當前目錄

# 2-1. 依照當前日期時間產生 log 檔名
current_time = datetime.datetime.now()
log_filename = current_time.strftime("%Y%m%d_%H%M%S") + "_qa.log"
log_filepath = os.path.join(LOG_DIRECTORY, log_filename)

# 2-2. 設定 log 格式
# Level: DEBUG, INFO, WARNING, ERROR, CRITICAL  
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# 3. 設定特定的 Logger
#    我們創建一個獨立的 logger，避免影響或被根 logger 干擾
qa_logger = logging.getLogger('QuestionAnswerLogger')
qa_logger.setLevel(logging.INFO) # 設定處理的最低級別，INFO 即可

# 4. 創建 FileHandler，指定輸出檔案路徑與編碼
#    注意：這裡我們只給這個特定的 logger 添加 handler
file_handler = logging.FileHandler(log_filepath, encoding='utf-8')


# 5. 創建 Formatter，只輸出訊息本身 (%(message)s)
#    這是為了精確模擬原始 logging_util.py 的輸出格式
raw_formatter = logging.Formatter('%(message)s')

# 6. 將 Formatter 設定給 FileHandler
file_handler.setFormatter(raw_formatter)

# 7. 將 FileHandler 添加給我們的特定 Logger
#    確保這個 logger 有地方輸出
if not qa_logger.handlers: # 避免重複添加 handler (如果模組被重載)
    qa_logger.addHandler(file_handler)

# 8. (重要) 防止日誌訊息傳遞給根 Logger (propagate=False)
#    這樣可以確保只有我們定義的格式被寫入檔案，
#    並且不會意外地輸出到控制台 (除非根 logger 也被配置輸出到控制台)
qa_logger.propagate = False

print(f"標準 logging 已設定，日誌檔案將寫入：{log_filepath}")

# --- 功能函數 ---
def log_question_answer(question: str, answer: str):
    """
    使用配置好的 logger 將問題和答案記錄到指定的日誌檔案。
    每個字串會被記錄為單獨的一行。

    Args:
        question (str): 要記錄的發問問題。
        answer (str): 對應問題的回覆。
    """
    try:
        # 使用 logger 記錄訊息，由於 formatter 只取 message，所以會直接寫入字串
        qa_logger.info("Question:\n")
        qa_logger.info(question + "\n") # 使用 .info() 或任何 >= INFO 級別的方法
        qa_logger.info("Answer:\n")
        qa_logger.info(answer + "\n")
        qa_logger.info("----------------------------------------------------------------")
    except Exception as e:
        # 可以加入更通用的日誌記錄錯誤處理，例如打印到stderr
        import sys
        print(f"寫入日誌時發生錯誤：{e}", file=sys.stderr)


def get_current_log_filepath():
    """
    返回當前正在使用的 log 檔案的完整路徑。
    """
    return log_filepath