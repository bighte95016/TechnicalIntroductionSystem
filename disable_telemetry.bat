@echo off
echo 禁用遙測數據收集...
set DISABLE_TELEMETRY=1
set POSTHOG_DISABLED=1
set TTS_CACHE_PATH=./tts_cache

echo 運行 RAG 系統...
python qa_RAGsystem_printChunk_QAsplit_coquiTTSclone.py

pause 