#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修復 jaraco.text 模組錯誤的腳本
解決 PyInstaller 打包時 pkg_resources 相關的依賴問題
"""

import sys
import subprocess
import importlib

def check_and_install_package(package_name, description=""):
    """檢查並安裝套件"""
    try:
        importlib.import_module(package_name)
        print(f"✅ {package_name} 已安裝 - {description}")
        return True
    except ImportError:
        print(f"❌ {package_name} 未安裝 - {description}")
        print(f"正在安裝 {package_name}...")
        try:
            # 嘗試安裝套件
            install_name = package_name.replace('_', '-').replace('.', '-')
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', 
                install_name, '--upgrade'
            ], capture_output=True, text=True, check=True)
            print(f"✅ {package_name} 安裝成功")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ {package_name} 安裝失敗: {e}")
            return False

def fix_setuptools_dependencies():
    """修復 setuptools 相關依賴"""
    print("🔧 修復 setuptools 和相關依賴...")
    
    packages_to_fix = [
        ('setuptools', 'Python 套件管理工具'),
        ('wheel', 'Python 套件構建工具'),
        ('jaraco.text', 'jaraco 文本處理工具'),
        ('jaraco.functools', 'jaraco 函數工具'),
        ('jaraco.collections', 'jaraco 集合工具'),
        ('jaraco.classes', 'jaraco 類工具'),
        ('more-itertools', '增強的迭代器工具'),
        ('importlib-metadata', '導入元數據工具'),
        ('zipp', 'ZIP 文件處理工具'),
    ]
    
    success_count = 0
    for package, description in packages_to_fix:
        if check_and_install_package(package, description):
            success_count += 1
    
    print(f"\n📊 修復結果: {success_count}/{len(packages_to_fix)} 套件處理成功")
    return success_count == len(packages_to_fix)

def upgrade_core_tools():
    """升級核心打包工具"""
    print("\n🚀 升級核心打包工具...")
    
    core_tools = [
        'pip',
        'setuptools', 
        'wheel',
        'pyinstaller'
    ]
    
    for tool in core_tools:
        try:
            print(f"升級 {tool}...")
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', 
                tool, '--upgrade'
            ], check=True, capture_output=True)
            print(f"✅ {tool} 升級成功")
        except subprocess.CalledProcessError as e:
            print(f"❌ {tool} 升級失敗: {e}")

def generate_enhanced_spec():
    """生成增強的 spec 文件以解決 jaraco 問題"""
    print("\n📝 生成增強的 PyInstaller spec 文件...")
    
    enhanced_spec = '''# -*- mode: python ; coding: utf-8 -*-

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
'''
    
    with open('build_pyinstaller_enhanced.spec', 'w', encoding='utf-8') as f:
        f.write(enhanced_spec)
    
    print("✅ 增強的 spec 文件已生成: build_pyinstaller_enhanced.spec")

def main():
    """主函數"""
    print("🚀 jaraco.text 錯誤修復工具")
    print("=" * 50)
    
    # 步驟 1: 升級核心工具
    upgrade_core_tools()
    
    # 步驟 2: 修復 setuptools 依賴
    print("\n" + "=" * 50)
    if fix_setuptools_dependencies():
        print("✅ setuptools 依賴修復成功")
    else:
        print("⚠️ 部分依賴修復失敗，但繼續執行...")
    
    # 步驟 3: 生成增強的 spec 文件
    print("\n" + "=" * 50)
    generate_enhanced_spec()
    
    # 步驟 4: 提供使用指導
    print("\n" + "=" * 50)
    print("🎯 下一步操作指導:")
    print("1. 使用增強的 spec 文件重新打包:")
    print("   pyinstaller build_pyinstaller_enhanced.spec --clean --noconfirm")
    print()
    print("2. 或者運行修復後的批處理文件:")
    print("   fix_and_rebuild_enhanced.bat")
    print()
    print("3. 如果仍有問題，嘗試手動安裝:")
    print("   pip install jaraco.text jaraco.functools more-itertools --upgrade")
    print()
    print("4. 檢查虛擬環境是否正確激活")
    
    print("\n" + "=" * 50)
    print("✅ 修復腳本執行完成")

if __name__ == "__main__":
    main() 