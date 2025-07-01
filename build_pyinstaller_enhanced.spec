# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# 增強的隱藏導入列表，包含 jaraco 相關依賴
hiddenimports = [
    # jaraco 相關依賴（解決 pkg_resources 問題）
    'jaraco',
    'jaraco.text',
    'jaraco.functools',
    'jaraco.collections',
    'jaraco.classes',
    'jaraco.context',
    'jaraco.itertools',
    'more_itertools',
    'importlib_metadata',
    'zipp',
    
    # setuptools 和 pkg_resources 相關
    'setuptools',
    'pkg_resources',
    'pkg_resources.py2_warn',
    'pkg_resources._vendor',
    'pkg_resources._vendor.packaging',
    'pkg_resources._vendor.packaging.version',
    'pkg_resources._vendor.packaging.specifiers',
    'pkg_resources._vendor.packaging.requirements',
    
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
    
    # LangChain 相關（完整列表）
    'langchain_ollama',
    'langchain_ollama.llms',
    'langchain_ollama.llms.ollama',
    'langchain_ollama.embeddings',
    'langchain_ollama.embeddings.ollama', 
    'langchain_ollama.chat_models',
    'langchain_ollama.chat_models.ollama',
    'langchain_core',
    'langchain_core.embeddings',
    'langchain_core.embeddings.base',
    'langchain_core.language_models',
    'langchain_core.language_models.llms',
    'langchain_core.language_models.base',
    'langchain_core.language_models.chat_models',
    'langchain_core.prompts',
    'langchain_core.prompts.prompt',
    'langchain_core.schema',
    'langchain_core.schema.document',
    'langchain_core.vectorstores',
    'langchain_core.vectorstores.base',
    'langchain_core.retrievers',
    'langchain_core.retrievers.base',
    'langchain_core.chains',
    'langchain_core.chains.base',
    'langchain_community',
    'langchain_community.vectorstores',
    'langchain_community.vectorstores.chroma',
    'langchain_community.retrievers',
    'langchain_community.retrievers.bm25',
    'langchain_openai',
    'langchain_openai.llms',
    'langchain_openai.chat_models',
    'langchain_openai.embeddings',
    'langchain.chains',
    'langchain.chains.retrieval_qa',
    'langchain.chains.retrieval_qa.base',
    'langchain.prompts',
    'langchain.schema',
    'langchain.schema.document',
    'langchain.retrievers',
    'langchain.retrievers.ensemble',
    
    # Ollama 客戶端
    'ollama',
    'ollama.client',
    
    # ChromaDB 向量數據庫
    'chromadb',
    'chromadb.api',
    'chromadb.config',
    'chromadb.utils',
    
    # PDF 處理
    'PyPDF2',
    'PyPDF2.pdf',
    
    # 中文處理
    'jieba',
    'jieba.analyse',
    'jieba.posseg',
    
    # 其他依賴
    'requests',
    'urllib3',
    'certifi',
    'charset_normalizer',
    'idna',
    'dotenv',
    'python_dotenv',
    'pydantic',
    'pydantic_core',
    'typing_extensions',
    'sqlalchemy',
    'sqlite3',
    'threading',
    'queue',
    'tempfile',
    'pathlib',
    'shutil',
    'pickle',
    're',
    'time',
    'datetime',
    'os',
    'sys',
]

# 排除不需要的模組以減少文件大小
excludes = [
    'tkinter',
    'matplotlib',
    'IPython',
    'jupyter',
    'notebook',
    'pandas',
    'sklearn',
    'seaborn',
    'plotly',
    'bokeh',
    'streamlit',
    'flask',
    'django',
    'fastapi',
    'uvicorn',
    'gunicorn',
]

a = Analysis(
    ['qa_RAGsystem_printChunk_QAsplit_coquiTTS.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('technical_file', 'technical_file'),
    ],
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
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
