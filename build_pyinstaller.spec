# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# 隱藏導入列表 - 包含所有動態導入的模組
hiddenimports = [
    # Coqui TTS 相關
    'TTS',
    'TTS.api',
    'TTS.tts.configs.xtts_config',
    'TTS.tts.models.xtts',
    'TTS.config.shared_configs',
    'TTS.tts.models.xtts.xtts',
    'TTS.tts.layers.xtts.xtts_manager',
    'TTS.tts.layers.xtts.tokenizer',
    'TTS.utils.audio',
    'TTS.vocoder.utils.generic_utils',
    
    # inflect 套件（解決 PyInstaller 兼容性問題）
    'inflect',
    'inflect.engine',
    
    # PyTorch 相關
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torch.utils.data',
    'torchaudio',
    'torchvision',
    
    # 音頻處理相關
    'pyaudio',
    'wave',
    'simpleaudio',
    'simpleaudio.shiny',
    'numpy',
    'scipy',
    'librosa',
    'soundfile',
    'resampy',
    
    # Whisper 語音識別
    'whisper',
    'whisper.model',
    'whisper.audio',
    'whisper.decoding',
    'whisper.tokenizer',
    
    # LangChain Ollama 相關 - 擴展修復
    'langchain_ollama',
    'langchain_ollama.llms',
    'langchain_ollama.llms.ollama',
    'langchain_ollama.embeddings',
    'langchain_ollama.embeddings.ollama',
    'langchain_ollama.chat_models',
    'langchain_ollama.chat_models.ollama',
    'langchain_ollama.llms.base',
    'langchain_ollama.embeddings.base',
    'langchain_ollama.chat_models.base',
    'langchain_ollama.utils',
    'langchain_ollama._api',
    'langchain_ollama._client',
    'langchain_ollama._types',
    
    # LangChain 核心
    'langchain',
    'langchain.llms',
    'langchain.llms.base',
    'langchain.llms.ollama',
    'langchain.embeddings',
    'langchain.embeddings.base',
    'langchain.embeddings.ollama',
    'langchain.chains',
    'langchain.chains.retrieval_qa',
    'langchain.chains.retrieval_qa.base',
    'langchain.prompts',
    'langchain.prompts.base',
    'langchain.prompts.prompt',
    'langchain.schema',
    'langchain.schema.document',
    'langchain.schema.embeddings',
    'langchain.schema.language_model',
    'langchain.schema.retriever',
    'langchain.retrievers',
    'langchain.retrievers.ensemble',
    'langchain.retrievers.multi_query',
    'langchain.text_splitter',
    'langchain.vectorstores',
    'langchain.vectorstores.base',
    'langchain.vectorstores.chroma',
    'langchain.document_loaders',
    'langchain.document_loaders.base',
    'langchain.document_loaders.pdf',
    
    # LangChain Community 相關
    'langchain_community',
    'langchain_community.vectorstores',
    'langchain_community.vectorstores.chroma',
    'langchain_community.vectorstores.base',
    'langchain_community.retrievers',
    'langchain_community.retrievers.bm25',
    'langchain_community.document_loaders',
    'langchain_community.document_loaders.pdf',
    'langchain_community.llms',
    'langchain_community.llms.ollama',
    'langchain_community.embeddings',
    'langchain_community.embeddings.ollama',
    
    # LangChain OpenAI
    'langchain_openai',
    'langchain_openai.llms',
    'langchain_openai.chat_models',
    'langchain_openai.embeddings',
    
    # 文檔處理
    'PyPDF2',
    'PyPDF2.pdf',
    'chromadb',
    'chromadb.config',
    'chromadb.utils.embedding_functions',
    'chromadb.api',
    'chromadb.api.models',
    'chromadb.api.types',
    
    # 中文處理
    'jieba',
    'jieba.posseg',
    'jieba.analyse',
    
    # HTTP 和 API 相關
    'requests',
    'requests.adapters',
    'requests.auth',
    'requests.cookies',
    'requests.exceptions',
    'requests.models',
    'requests.sessions',
    'requests.structures',
    'requests.utils',
    'urllib3',
    'urllib3.util',
    'urllib3.util.retry',
    'urllib3.poolmanager',
    'certifi',
    'charset_normalizer',
    'idna',
    
    # 環境變數和配置
    'dotenv',
    'python_dotenv',
    
    # 其他必要模組
    'pickle',
    'threading',
    'queue',
    'tempfile',
    'pathlib',
    
    # 科學計算
    'sklearn',
    'sklearn.feature_extraction',
    'sklearn.feature_extraction.text',
    'sklearn.metrics',
    'sklearn.metrics.pairwise',
    
    # 系統相關
    'platform',
    'subprocess',
    'multiprocessing',
    'concurrent.futures',
    
    # 日期時間
    'datetime',
    'time',
    'calendar',
    
    # 正則表達式和文本處理
    're',
    'string',
    'unicodedata',
    
    # JSON 和配置
    'json',
    'yaml',
    'configparser',
    
    # 網路相關
    'socket',
    'ssl',
    'http.client',
    'urllib.parse',
    'urllib.request',
    
    # 壓縮和歸檔
    'zipfile',
    'tarfile',
    'gzip',
    
    # 加密和哈希
    'hashlib',
    'hmac',
    'secrets',
    
    # 文件系統操作
    'shutil',
    'glob',
    'fnmatch',
    
    # 進程和線程
    'signal',
    'atexit',
    'weakref',
    
    # 數據結構
    'collections',
    'collections.abc',
    'heapq',
    'bisect',
    
    # 數學和統計
    'math',
    'statistics',
    'random',
    'decimal',
    'fractions',
    
    # 內存和垃圾回收
    'gc',
    'sys',
    'os',
    'os.path',
    'io',
    
    # 類型檢查
    'typing',
    'typing_extensions',
    
    # 異常處理
    'traceback',
    'warnings',
    
    # 國際化
    'locale',
    'gettext',
    
    # Pydantic（LangChain依賴）
    'pydantic',
    'pydantic.fields',
    'pydantic.main',
    'pydantic.types',
    'pydantic.validators',
    'pydantic.v1',
    'pydantic.v1.fields',
    'pydantic.v1.main',
    'pydantic.v1.types',
    
    # SQLAlchemy（Chroma依賴）
    'sqlalchemy',
    'sqlalchemy.engine',
    'sqlalchemy.sql',
    
    # 其他可能的依賴
    'tenacity',
    'aiohttp',
    'httpx',
    'anyio',
    'sniffio',
    
    # 額外的 LangChain 相關模組
    'langchain_core',
    'langchain_core.embeddings',
    'langchain_core.language_models',
    'langchain_core.language_models.base',
    'langchain_core.language_models.llms',
    'langchain_core.language_models.chat_models',
    'langchain_core.retrievers',
    'langchain_core.vectorstores',
    'langchain_core.documents',
    'langchain_core.prompts',
    'langchain_core.output_parsers',
    'langchain_core.runnables',
    'langchain_core.callbacks',
    'langchain_core.messages',
    
    # 額外的 Ollama 相關
    'ollama',
    'ollama._client',
    'ollama._types',
    'ollama.client',
    
    # 額外的 HTTP 客戶端
    'httpcore',
    'httpcore._backends',
    'httpcore._sync',
    'httpcore._async',
    'h11',
    'h2',
]

# 排除的模組（減少打包大小）
excludes = [
    'matplotlib',
    'tkinter',
    'pandas',
    'jupyter',
    'notebook',
    'IPython',
    'pytest',
    'sphinx',
    'setuptools',
    'pip',
    'wheel',
    'twine',
    'flake8',
    'black',
    'mypy',
    'pylint',
    'coverage',
    'tox',
    'virtualenv',
    'conda',
    'anaconda',
]

# 二進制文件包含
binaries = []

# 數據文件包含
datas = [
    ('technical_file', 'technical_file'),  # 包含 PDF 文件目錄
]

a = Analysis(
    ['qa_RAGsystem_printChunk_QAsplit_coquiTTS.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='RAG_TTS_System',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
) 