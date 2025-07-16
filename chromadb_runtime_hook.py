"""
ChromaDB Runtime Hook for PyInstaller
解決Chroma嵌入器動態載入問題，特別是ONNXMiniLM_L6_V2類未定義的問題
"""

import sys
import os
import importlib
import warnings

# 禁用警告以避免干擾
warnings.filterwarnings('ignore')

def patch_chromadb_embeddings():
    """修補ChromaDB嵌入器模組，解決動態載入問題"""
    try:
        # 嘗試導入chromadb.embeddings模組
        if 'chromadb.embeddings' not in sys.modules:
            try:
                # 使用動態導入避免 Pylance 錯誤
                importlib.import_module('chromadb.embeddings')
                print("✅ chromadb.embeddings 模組已載入")
            except ImportError:
                print("⚠️ chromadb.embeddings 模組不可用")
                # 即使模組不可用，我們也要創建替代類
                class DummyONNXMiniLM_L6_V2:
                    def __init__(self, *args, **kwargs):
                        raise NotImplementedError("ONNXMiniLM_L6_V2 嵌入器在當前環境中不可用")
                
                # 將替代類註冊到全局和模組命名空間
                globals()['ONNXMiniLM_L6_V2'] = DummyONNXMiniLM_L6_V2
                
                # 嘗試將類註冊到 sys.modules 中
                try:
                    # 創建一個虛擬的 chromadb.embeddings 模組
                    import types
                    dummy_embeddings_module = types.ModuleType('chromadb.embeddings')
                    dummy_embeddings_module.ONNXMiniLM_L6_V2 = DummyONNXMiniLM_L6_V2
                    sys.modules['chromadb.embeddings'] = dummy_embeddings_module
                    
                    # 創建虛擬的 chromadb.embeddings.onnx_mini_lm_l6_v2 模組
                    dummy_onnx_module = types.ModuleType('chromadb.embeddings.onnx_mini_lm_l6_v2')
                    dummy_onnx_module.ONNXMiniLM_L6_V2 = DummyONNXMiniLM_L6_V2
                    sys.modules['chromadb.embeddings.onnx_mini_lm_l6_v2'] = dummy_onnx_module
                    
                    print("✅ 已創建虛擬的 chromadb.embeddings 模組和 ONNXMiniLM_L6_V2 替代類")
                except Exception as e:
                    print(f"⚠️ 創建虛擬模組失敗: {e}")
                
                return
        
        # 嘗試導入並註冊ONNXMiniLM_L6_V2嵌入器
        try:
            # 使用字符串導入避免 Pylance 錯誤
            onnx_module = importlib.import_module('chromadb.embeddings.onnx_mini_lm_l6_v2')
            ONNXMiniLM_L6_V2 = getattr(onnx_module, 'ONNXMiniLM_L6_V2')
            # 將類註冊到全局命名空間
            globals()['ONNXMiniLM_L6_V2'] = ONNXMiniLM_L6_V2
            print("✅ ONNXMiniLM_L6_V2 嵌入器已成功載入")
        except (ImportError, AttributeError) as e:
            print(f"⚠️ 無法導入ONNXMiniLM_L6_V2: {e}")
            # 創建一個替代的嵌入器類
            class DummyONNXMiniLM_L6_V2:
                def __init__(self, *args, **kwargs):
                    raise NotImplementedError("ONNXMiniLM_L6_V2 嵌入器在打包版本中不可用")
            globals()['ONNXMiniLM_L6_V2'] = DummyONNXMiniLM_L6_V2
            print("⚠️ 已創建ONNXMiniLM_L6_V2替代類")
        
        # 嘗試導入其他常用嵌入器
        embedding_modules = [
            'chromadb.embeddings.sentence_transformer_embedding_function',
            'chromadb.embeddings.openai_embedding_function',
            'chromadb.embeddings.huggingface_embedding_function',
            'chromadb.embeddings.default_ef',
        ]
        
        for module_name in embedding_modules:
            try:
                importlib.import_module(module_name)
                print(f"✅ 已載入嵌入器模組: {module_name}")
            except ImportError as e:
                print(f"⚠️ 無法載入嵌入器模組 {module_name}: {e}")
        
        # 修補chromadb.utils.embedding_functions模組
        try:
            importlib.import_module('chromadb.utils.embedding_functions')
            print("✅ chromadb.utils.embedding_functions 已載入")
        except ImportError as e:
            print(f"⚠️ 無法載入chromadb.utils.embedding_functions: {e}")
        
        print("✅ ChromaDB嵌入器修補完成")
        
    except Exception as e:
        print(f"❌ ChromaDB嵌入器修補失敗: {e}")
        import traceback
        traceback.print_exc()

def patch_sentence_transformers():
    """修補sentence_transformers模組，確保相關類可用"""
    try:
        # 嘗試導入sentence_transformers
        if 'sentence_transformers' not in sys.modules:
            try:
                # 使用動態導入避免 Pylance 錯誤
                importlib.import_module('sentence_transformers')
                print("✅ sentence_transformers 已載入")
            except ImportError:
                print("⚠️ sentence_transformers 不可用")
                return
        
        # 嘗試導入關鍵類
        try:
            st_module = importlib.import_module('sentence_transformers')
            SentenceTransformer = getattr(st_module, 'SentenceTransformer')
            globals()['SentenceTransformer'] = SentenceTransformer
            print("✅ SentenceTransformer 類已載入")
        except (ImportError, AttributeError):
            print("⚠️ SentenceTransformer 類不可用")
            # 創建替代類
            class DummySentenceTransformer:
                def __init__(self, *args, **kwargs):
                    raise NotImplementedError("SentenceTransformer 在打包版本中不可用")
            globals()['SentenceTransformer'] = DummySentenceTransformer
            print("⚠️ 已創建SentenceTransformer替代類")
        
    except Exception as e:
        print(f"❌ sentence_transformers 修補失敗: {e}")

def patch_transformers():
    """修補transformers模組，確保相關類可用"""
    try:
        # 嘗試導入transformers
        if 'transformers' not in sys.modules:
            try:
                # 使用動態導入避免 Pylance 錯誤
                importlib.import_module('transformers')
                print("✅ transformers 已載入")
            except ImportError:
                print("⚠️ transformers 不可用")
                return
        
        # 嘗試導入關鍵類
        try:
            transformers_module = importlib.import_module('transformers')
            AutoTokenizer = getattr(transformers_module, 'AutoTokenizer')
            AutoModel = getattr(transformers_module, 'AutoModel')
            globals()['AutoTokenizer'] = AutoTokenizer
            globals()['AutoModel'] = AutoModel
            print("✅ transformers 關鍵類已載入")
        except (ImportError, AttributeError):
            print("⚠️ transformers 關鍵類不可用")
        
    except Exception as e:
        print(f"❌ transformers 修補失敗: {e}")

def patch_onnxruntime():
    """修補onnxruntime模組，確保ONNX推理可用"""
    try:
        # 嘗試導入onnxruntime
        if 'onnxruntime' not in sys.modules:
            try:
                # 使用動態導入避免 Pylance 錯誤
                importlib.import_module('onnxruntime')
                print("✅ onnxruntime 已載入")
            except ImportError:
                print("⚠️ onnxruntime 不可用")
                return
        
        # 檢查推理會話
        try:
            onnx_module = importlib.import_module('onnxruntime')
            InferenceSession = getattr(onnx_module, 'InferenceSession')
            globals()['InferenceSession'] = InferenceSession
            print("✅ onnxruntime InferenceSession 已載入")
        except (ImportError, AttributeError):
            print("⚠️ onnxruntime InferenceSession 不可用")
        
    except Exception as e:
        print(f"❌ onnxruntime 修補失敗: {e}")

def patch_huggingface_hub():
    """修補huggingface_hub模組，確保模型下載功能可用"""
    try:
        # 嘗試導入huggingface_hub
        if 'huggingface_hub' not in sys.modules:
            try:
                # 使用動態導入避免 Pylance 錯誤
                importlib.import_module('huggingface_hub')
                print("✅ huggingface_hub 已載入")
            except ImportError:
                print("⚠️ huggingface_hub 不可用")
                return
        
        # 嘗試導入關鍵函數
        try:
            hf_module = importlib.import_module('huggingface_hub')
            hf_hub_download = getattr(hf_module, 'hf_hub_download')
            snapshot_download = getattr(hf_module, 'snapshot_download')
            globals()['hf_hub_download'] = hf_hub_download
            globals()['snapshot_download'] = snapshot_download
            print("✅ huggingface_hub 關鍵函數已載入")
        except (ImportError, AttributeError):
            print("⚠️ huggingface_hub 關鍵函數不可用")
        
    except Exception as e:
        print(f"❌ huggingface_hub 修補失敗: {e}")

def patch_langchain_embeddings():
    """修補langchain嵌入器模組"""
    try:
        # 嘗試導入langchain_ollama
        if 'langchain_ollama' not in sys.modules:
            try:
                # 使用動態導入避免 Pylance 錯誤
                importlib.import_module('langchain_ollama')
                print("✅ langchain_ollama 已載入")
            except ImportError:
                print("⚠️ langchain_ollama 不可用")
                return
        
        try:
            ollama_module = importlib.import_module('langchain_ollama')
            OllamaEmbeddings = getattr(ollama_module, 'OllamaEmbeddings')
            globals()['OllamaEmbeddings'] = OllamaEmbeddings
            print("✅ OllamaEmbeddings 已載入")
        except (ImportError, AttributeError):
            print("⚠️ OllamaEmbeddings 不可用")
        
    except Exception as e:
        print(f"❌ langchain_ollama 修補失敗: {e}")

def main():
    """主要的修補函數"""
    print("🔧 開始執行ChromaDB Runtime Hook...")
    
    # 執行各種修補
    patch_chromadb_embeddings()
    patch_sentence_transformers()
    patch_transformers()
    patch_onnxruntime()
    patch_huggingface_hub()
    patch_langchain_embeddings()
    
    print("✅ ChromaDB Runtime Hook 執行完成")

# 執行修補
if __name__ == "__main__":
    main()
else:
    # 當作為runtime hook被導入時自動執行
    main() 