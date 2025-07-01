@echo off
echo Testing PyInstaller build environment...
echo.

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python not found
    pause
    exit /b 1
)

echo.
echo Checking PyInstaller installation...
python -c "import PyInstaller; print('PyInstaller version:', PyInstaller.__version__)"
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

echo.
echo Checking main program file...
if exist "qa_RAGsystem_printChunk_QAsplit_coquiTTS.py" (
    echo Main program file found
) else (
    echo ERROR: Main program file not found
    pause
    exit /b 1
)

echo.
echo Checking dependencies...
python -c "import TTS; print('TTS OK')" 2>nul || echo WARNING: TTS not found
python -c "import torch; print('PyTorch OK')" 2>nul || echo WARNING: PyTorch not found
python -c "import whisper; print('Whisper OK')" 2>nul || echo WARNING: Whisper not found
python -c "import langchain_community; print('LangChain Community OK')" 2>nul || echo WARNING: LangChain Community not found

echo.
echo Starting simple PyInstaller test...
pyinstaller --onefile --console --name "Test_RAG" qa_RAGsystem_printChunk_QAsplit_coquiTTS.py

if exist "dist\Test_RAG.exe" (
    echo SUCCESS: Test executable created
) else (
    echo ERROR: Test executable creation failed
)

echo.
echo Test completed.
pause 