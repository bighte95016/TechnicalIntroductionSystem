# Cache Augmented Generation (CAG) 系統使用說明

## 概述

Cache Augmented Generation (CAG) 系統是一個專門用於處理 `./technical_file/PDF` 目錄中PDF文件的智能問答系統。該系統使用緩存技術來提高處理效率，支持混合搜索（向量搜索 + BM25）來提供更準確的檢索結果。

## 主要功能

### 1. 緩存機制
- **文本緩存**: 緩存從PDF提取的文本內容
- **分塊緩存**: 緩存文本分塊結果
- **向量緩存**: 緩存向量嵌入和ChromaDB集合
- **BM25緩存**: 緩存BM25索引
- **響應緩存**: 緩存生成的回答

### 2. 混合搜索
- **向量搜索**: 使用語義相似性進行搜索
- **BM25搜索**: 使用關鍵詞匹配進行搜索
- **結果融合**: 結合兩種搜索結果提供更準確的檢索

### 3. 智能問答
- 基於檢索到的相關文檔生成準確回答
- 支持中文問答
- 提供文檔來源信息

## 系統架構

```
CAG系統
├── PDF文件處理
│   ├── 文本提取 (PyPDF2)
│   ├── 文本清理
│   └── 文本分塊 (LangChain)
├── 索引建立
│   ├── 向量索引 (ChromaDB + Ollama Embeddings)
│   └── BM25索引 (rank-bm25 + jieba)
├── 緩存系統
│   ├── 文本緩存 (JSON)
│   ├── 向量緩存 (ChromaDB持久化)
│   ├── BM25緩存 (pickle)
│   └── 響應緩存 (JSON)
└── 查詢處理
    ├── 混合搜索
    ├── 結果排序
    └── 回答生成 (Ollama)
```

## 安裝依賴

確保已安裝以下依賴包：

```bash
pip install langchain==0.3.26
pip install langchain-community==0.3.26
pip install langchain-ollama==0.3.3
pip install chromadb==0.6.3
pip install PyPDF2==3.0.1
pip install jieba==0.42.1
pip install rank-bm25==0.2.2
pip install numpy==1.26.4
pip install ollama==0.5.1
```

## 使用方法

### 1. 啟動系統

```bash
python cache_augmented_generation.py
```

### 2. 基本操作

系統提供5個主要操作：

1. **處理所有PDF文件（建立緩存）**
   - 掃描 `./technical_file/PDF` 目錄中的所有PDF文件
   - 提取文本並建立索引
   - 創建緩存以提高後續查詢效率

2. **查詢問題**
   - 輸入您的問題
   - 系統會搜索相關文檔並生成回答
   - 顯示相關文檔片段和來源信息

3. **顯示緩存統計**
   - 查看各種緩存的大小和狀態
   - 監控系統性能

4. **清除緩存**
   - 清除所有緩存數據
   - 重新開始處理

5. **退出系統**

### 3. 程式化使用

您也可以在其他Python程式中使用CAG系統：

```python
from cache_augmented_generation import CacheAugmentedGeneration

# 初始化系統
cag = CacheAugmentedGeneration(
    pdf_directory="./technical_file/PDF",
    cache_directory="./cag_cache",
    model_name="qwen2.5:7b",
    embedding_model="bge-m3:latest"
)

# 處理PDF文件
cag.process_all_pdfs()

# 查詢問題
result = cag.query("什麼是機器學習？")
print(f"回答: {result['answer']}")
print(f"相關文檔數: {result['total_documents']}")
```

## 配置選項

### 初始化參數

```python
CacheAugmentedGeneration(
    pdf_directory="./technical_file/PDF",  # PDF文件目錄
    cache_directory="./cag_cache",         # 緩存目錄
    model_name="qwen2.5:7b",              # Ollama模型名稱
    embedding_model="bge-m3:latest",       # 嵌入模型名稱
    chunk_size=500,                        # 文本分塊大小
    chunk_overlap=50                       # 文本分塊重疊大小
)
```

### 模型要求

確保Ollama中已安裝以下模型：

```bash
# 安裝主要模型
ollama pull qwen2.5:7b

# 安裝嵌入模型
ollama pull bge-m3:latest
```

## 緩存結構

```
cag_cache/
├── text_cache/
│   └── text_cache.json          # 文本緩存
├── vector_cache/
│   ├── pdf_[hash]/              # ChromaDB向量數據庫
│   └── ...
├── bm25_cache/
│   ├── [hash]_500_50.pkl       # BM25索引文件
│   └── ...
└── response_cache/
    └── response_cache.json      # 回答緩存
```

## 性能優化

### 1. 緩存優勢
- **首次處理**: 建立完整的索引和緩存
- **後續查詢**: 直接從緩存載入，大幅提升速度
- **增量更新**: 只處理新增或修改的PDF文件

### 2. 混合搜索優勢
- **語義搜索**: 理解問題的語義含義
- **關鍵詞搜索**: 精確匹配重要關鍵詞
- **結果融合**: 提供更全面的檢索結果

### 3. 批量處理
- 支持批量處理多個PDF文件
- 自動處理文件哈希和緩存管理
- 智能跳過已處理的文件

## 故障排除

### 1. 常見問題

**Q: 系統初始化時報錯**
A: 檢查Ollama是否正在運行，並確保已安裝所需模型

**Q: PDF文件無法處理**
A: 確保PDF文件格式正確，檢查文件權限

**Q: 查詢速度很慢**
A: 首次查詢需要建立索引，後續查詢會很快

**Q: 緩存占用空間過大**
A: 可以定期清除緩存，或調整chunk_size參數

### 2. 日誌調試

系統提供詳細的日誌輸出，可以幫助診斷問題：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 3. 性能監控

使用緩存統計功能監控系統性能：

```python
stats = cag.get_cache_stats()
print(stats)
```

## 擴展功能

### 1. 自定義模型

```python
# 使用不同的模型
cag = CacheAugmentedGeneration(
    model_name="llama3.2:3b",
    embedding_model="nomic-embed-text:latest"
)
```

### 2. 自定義分塊策略

```python
# 調整分塊參數
cag = CacheAugmentedGeneration(
    chunk_size=1000,
    chunk_overlap=100
)
```

### 3. 批量查詢

```python
questions = ["問題1", "問題2", "問題3"]
results = [cag.query(q) for q in questions]
```

## 注意事項

1. **模型依賴**: 需要Ollama服務運行
2. **內存使用**: 大量PDF文件可能需要較多內存
3. **磁盤空間**: 緩存會占用一定磁盤空間
4. **首次處理**: 第一次處理PDF文件會比較慢
5. **中文支持**: 系統針對中文進行了優化

## 版本信息

- **版本**: 1.0.0
- **作者**: AI Assistant
- **更新日期**: 2024年
- **兼容性**: Python 3.8+

## 許可證

本項目遵循MIT許可證。 