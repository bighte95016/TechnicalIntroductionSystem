@echo off
echo Quick fix for langchain_ollama module issue...
echo.

echo Cleaning old build files...
if exist "dist" rmdir /s /q "dist"
if exist "build" rmdir /s /q "build"

echo.
echo Building with enhanced hidden imports...
pyinstaller --onefile --console --name "RAG_TTS_System_Fixed" ^
    --hidden-import langchain_ollama ^
    --hidden-import langchain_ollama.llms ^
    --hidden-import langchain_ollama.embeddings ^
    --hidden-import langchain_ollama.chat_models ^
    --hidden-import langchain_community ^
    --hidden-import langchain_community.vectorstores ^
    --hidden-import langchain_community.vectorstores.chroma ^
    --hidden-import langchain_community.retrievers ^
    --hidden-import langchain_community.retrievers.bm25 ^
    --hidden-import langchain_openai ^
    --hidden-import langchain.chains ^
    --hidden-import langchain.prompts ^
    --hidden-import langchain.schema ^
    --hidden-import TTS ^
    --hidden-import TTS.api ^
    --hidden-import inflect ^
    --hidden-import torch ^
    --hidden-import whisper ^
    --hidden-import PyPDF2 ^
    --hidden-import chromadb ^
    --hidden-import jieba ^
    --hidden-import dotenv ^
    --hidden-import pydantic ^
    --add-data "technical_file;technical_file" ^
    qa_RAGsystem_printChunk_QAsplit_coquiTTS.py

if exist "dist\RAG_TTS_System_Fixed.exe" (
    echo SUCCESS: Fixed executable created!
    echo Location: dist\RAG_TTS_System_Fixed.exe
) else (
    echo ERROR: Build failed
)

echo.
echo Cleaning temporary files...
if exist "build" rmdir /s /q "build"

pause 