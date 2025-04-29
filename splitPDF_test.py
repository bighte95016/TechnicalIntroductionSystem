import os, re, unicodedata
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from PyPDF2 import PdfReader



PDF_PATH = "./technical_file/PDF/透明P-HUD_Chatbot__Q1-Q100_修正版.pdf"

# ------------ 1. 讀完整本 ------------
reader = PdfReader(PDF_PATH)
full_text = "\n".join(p.extract_text() or "" for p in reader.pages)

# ------------ 2. 換行正規化 ------------
#   a) 把所有非常規換行(LS‧PS‧CR‧VT‧FF) → \n
full_text = full_text.translate(
    dict.fromkeys(map(ord, "\r\u2028\u2029\v\f"), "\n")
)
#   b) 只把「真正單行軟換行」換成空格  
#      ⇒ 若同一行附近還有第 2 個 \n（允許夾空格）就留下
full_text = re.sub(r"(?<!\n)\n(?!\s*\n)", " ", full_text)
#   c) 把 3 行以上空白壓成 2 行
full_text = re.sub(r"\n{3,}", "\n\n", full_text)
#   d) 開頭補 \n，方便偵測第一題
if not full_text.startswith("\n"):
    full_text = "\n" + full_text

# ------------ 3. 以行首 Qxx 切塊 ------------
q_head = re.compile(r"\n\s*Q\s*\d+\s*[：:]", re.IGNORECASE)
starts = [m.start()+1 for m in q_head.finditer(full_text)] + [len(full_text)]

chunks = [
    full_text[starts[i]:starts[i+1]].strip()
    for i in range(len(starts)-1)
]

print("切出 chunk 數 =", len(chunks))         # ← 應該是 100

# ------------ 4. 包成 LangChain Document ------------
docs = [
    Document(
        page_content=chunk,
        metadata={"source": PDF_PATH,
                  "paragraph_index": i,
                  "type": "qa_pair"}
    )
    for i, chunk in enumerate(chunks)
]
