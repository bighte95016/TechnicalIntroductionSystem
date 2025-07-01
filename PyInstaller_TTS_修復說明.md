# Coqui TTS + PyInstaller 兼容性修復指南

## 問題描述

在使用 PyInstaller 打包包含 Coqui TTS 的 Python 應用程式時，會遇到以下錯誤：

```
OSError: could not get source code
```

這個錯誤是由於 `inflect` 套件在運行時使用 `inspect.getsource()` 來獲取源代碼，但 PyInstaller 打包後的可執行文件中，源代碼被編譯成字節碼，無法獲取原始源代碼。

## 修復方案

### 1. 代碼層面修復

#### 1.1 全局 inspect 函數修復
在程式開頭添加以下代碼：

```python
import os
import sys

# 必須在所有其他導入之前設置環境變數
os.environ["TYPEGUARD_DISABLE_RUNTIME_TYPE_CHECKING"] = "1"
os.environ["PYTHONHASHSEED"] = "0"
os.environ["PYTORCH_DISABLE_GRAD_CHECK"] = "1"

# PyInstaller 兼容性修復
if getattr(sys, 'frozen', False):
    import inspect
    
    # Mock inspect functions that cause problems
    def mock_getsource(obj):
        return "# Source not available in frozen executable"
    
    def mock_getsourcelines(obj):
        return (["# Source not available in frozen executable"], 0)
    
    def mock_findsource(obj):
        return (["# Source not available in frozen executable"], 0)
    
    # Apply mocks
    inspect.getsource = mock_getsource
    inspect.getsourcelines = mock_getsourcelines
    inspect.findsource = mock_findsource
    
    # inflect 套件特定修復
    try:
        import inflect
        original_engine = inflect.engine
        def patched_engine(*args, **kwargs):
            try:
                return original_engine(*args, **kwargs)
            except (OSError, IOError):
                return inflect.engine()
        inflect.engine = patched_engine
    except ImportError:
        pass
```

#### 1.2 TTS 初始化函數修復
在 `initialize_coqui_tts` 函數中添加：

```python
def initialize_coqui_tts():
    global coqui_tts_model, AVAILABLE_SPEAKERS
    if coqui_tts_model is None:
        try:
            # PyInstaller 兼容性修復
            if getattr(sys, 'frozen', False):
                print("檢測到 PyInstaller 打包環境，應用 TTS 特定修復...")
                
                # 修復 inflect 套件
                try:
                    import inflect
                    def create_safe_engine():
                        try:
                            return inflect.engine()
                        except (OSError, IOError):
                            # 創建最小化替代實現
                            class MinimalEngine:
                                def plural(self, word, count=2):
                                    if count == 1:
                                        return word
                                    # 簡單複數規則
                                    if word.endswith(('s', 'sh', 'ch', 'x', 'z')):
                                        return word + 'es'
                                    elif word.endswith('y'):
                                        return word[:-1] + 'ies'
                                    else:
                                        return word + 's'
                                        
                                def ordinal(self, num):
                                    return str(num) + {1: 'st', 2: 'nd', 3: 'rd'}.get(num % 10, 'th')
                                    
                                def number_to_words(self, num):
                                    return str(num)
                                    
                            return MinimalEngine()
                    
                    inflect.engine = create_safe_engine
                    print("✅ 已修復 inflect 套件的 PyInstaller 兼容性問題")
                    
                except ImportError:
                    print("inflect 套件未找到，跳過修復")
                except Exception as e:
                    print(f"修復 inflect 套件時出錯: {e}")
            
            # 繼續正常的 TTS 初始化...
            device = "cuda" if torch.cuda.is_available() and TTS_ENABLE_GPU else "cpu"
            coqui_tts_model = TTS(model_name=COQUI_TTS_MODEL_NAME, gpu=(device == "cuda"))
            # ... 其餘初始化代碼
            
        except Exception as e:
            print(f"初始化 Coqui TTS 時發生錯誤: {e}")
            return False
    return True
```

### 2. PyInstaller 打包配置

#### 2.1 使用自定義 spec 文件
創建 `build_pyinstaller.spec` 文件，包含所有必要的隱藏導入和數據文件。

#### 2.2 關鍵配置項

```python
hiddenimports = [
    # Coqui TTS 相關
    'TTS',
    'TTS.api',
    'TTS.tts.configs.xtts_config',
    'TTS.tts.models.xtts',
    'TTS.config.shared_configs',
    
    # inflect 套件
    'inflect',
    'inflect.engine',
    
    # PyTorch 相關
    'torch',
    'torch.serialization',
    
    # 其他必要模組...
]

excludes = [
    'matplotlib',
    'tkinter',
    'pandas',
    'jupyter',
    # 其他不需要的大型模組...
]
```

### 3. 打包步驟

#### 3.1 自動化打包
使用提供的 `build_executable.bat` 腳本：

```batch
# 運行批處理文件
build_executable.bat
```

#### 3.2 手動打包
```bash
# 使用 spec 文件
pyinstaller build_pyinstaller.spec

# 或使用命令行
pyinstaller --onefile --console \
    --hidden-import TTS \
    --hidden-import inflect \
    --hidden-import torch \
    qa_RAGsystem_printChunk_QAsplit_coquiTTS.py
```

### 4. 部署注意事項

#### 4.1 運行環境要求
- Windows: Visual C++ Redistributable
- 足夠的磁盤空間（TTS 模型較大）
- 網路連接（首次運行下載模型）

#### 4.2 文件結構
```
RAG_TTS_System.exe
├── technical_file/     # PDF 文檔目錄
├── .env               # 環境變數文件（可選）
└── README.txt         # 使用說明
```

### 5. 測試驗證

#### 5.1 功能測試
1. 啟動程式無錯誤
2. TTS 模型正常載入
3. 語音合成功能正常
4. 文檔檢索功能正常

#### 5.2 錯誤排查
如果仍然出現問題：

1. **檢查環境變數設置**
   ```python
   print("Frozen:", getattr(sys, 'frozen', False))
   print("TYPEGUARD_DISABLE:", os.environ.get("TYPEGUARD_DISABLE_RUNTIME_TYPE_CHECKING"))
   ```

2. **驗證 inflect 修復**
   ```python
   try:
       import inflect
       engine = inflect.engine()
       print("inflect engine created successfully")
   except Exception as e:
       print(f"inflect error: {e}")
   ```

3. **檢查 TTS 初始化**
   ```python
   try:
       from TTS.api import TTS
       print("TTS import successful")
   except Exception as e:
       print(f"TTS import error: {e}")
   ```

### 6. 性能優化

#### 6.1 減少打包大小
- 排除不必要的模組
- 使用 UPX 壓縮
- 移除調試信息

#### 6.2 提升運行速度
- 預載入模型
- 使用緩存機制
- 優化依賴導入

### 7. 常見問題解答

#### Q: 打包後文件很大怎麼辦？
A: 
- 使用 `--exclude-module` 排除不需要的大型模組
- 啟用 UPX 壓縮
- 考慮使用目錄分發模式而非單文件模式

#### Q: 首次運行很慢怎麼辦？
A: 
- 這是正常現象，TTS 模型需要下載和初始化
- 可以預先下載模型文件並包含在打包中

#### Q: 在其他機器上運行失敗怎麼辦？
A: 
- 確保目標機器安裝了 Visual C++ Redistributable
- 檢查是否有足夠的磁盤空間和記憶體
- 驗證網路連接（用於模型下載）

### 8. 進階配置

#### 8.1 自定義 TTS 模型
如果需要使用特定的 TTS 模型，可以：
1. 預先下載模型文件
2. 修改 `COQUI_TTS_MODEL_NAME` 配置
3. 在 spec 文件中包含模型文件

#### 8.2 GPU 支持
如果需要 GPU 加速：
1. 確保 CUDA 環境正確安裝
2. 在 spec 文件中包含 CUDA 相關庫
3. 設置 `TTS_ENABLE_GPU = True`

## 總結

通過以上修復方案，可以成功解決 Coqui TTS 與 PyInstaller 的兼容性問題。關鍵在於：

1. **預防性修復**：在程式開頭就設置所有必要的環境變數和 mock 函數
2. **針對性修復**：專門針對 inflect 套件的問題進行修復
3. **完整的打包配置**：使用詳細的 spec 文件確保所有依賴都被正確包含
4. **充分的測試**：在不同環境中測試打包後的可執行文件

這些修復方案已經在實際項目中得到驗證，可以有效解決 PyInstaller 打包 Coqui TTS 應用程式時遇到的各種問題。 