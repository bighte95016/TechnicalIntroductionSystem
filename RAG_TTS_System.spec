# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['qa_RAGsystem_printChunk_QAsplit_coquiTTS.py'],
    pathex=[],
    binaries=[],
    datas=[('technical_file', 'technical_file')],
    hiddenimports=['TTS', 'TTS.api', 'TTS.tts.configs.xtts_config', 'TTS.tts.models.xtts', 'TTS.config.shared_configs', 'inflect', 'torch', 'whisper', 'pyaudio', 'simpleaudio', 'langchain_community', 'langchain_ollama', 'jieba'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'tkinter', 'pandas', 'jupyter'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
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
