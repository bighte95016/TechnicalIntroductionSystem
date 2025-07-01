@echo off
chcp 65001 >nul 2>&1
echo ========================================
echo   Coqui TTS RAG System PyInstaller Build
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please ensure Python is installed and added to PATH.
    pause
    exit /b 1
)

REM Check if PyInstaller is installed
python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
    if errorlevel 1 (
        echo Error: Failed to install PyInstaller.
        pause
        exit /b 1
    )
)

REM Check if main program file exists
if not exist "qa_RAGsystem_printChunk_QAsplit_coquiTTS.py" (
    echo Error: Main program file qa_RAGsystem_printChunk_QAsplit_coquiTTS.py not found
    pause
    exit /b 1
)

echo Cleaning old build files...
if exist "dist" rmdir /s /q "dist"
if exist "build" rmdir /s /q "build"
if exist "*.spec" del "*.spec"

echo.
echo Starting PyInstaller build...
echo This may take several minutes, please wait...
echo.

REM Use custom spec file if available
if exist "build_pyinstaller.spec" (
    echo Using custom spec file for build...
    pyinstaller build_pyinstaller.spec
) else (
    echo Using command line parameters for build...
    pyinstaller ^
        --onefile ^
        --console ^
        --name "RAG_TTS_System" ^
        --hidden-import TTS ^
        --hidden-import TTS.api ^
        --hidden-import TTS.tts.configs.xtts_config ^
        --hidden-import TTS.tts.models.xtts ^
        --hidden-import TTS.config.shared_configs ^
        --hidden-import inflect ^
        --hidden-import torch ^
        --hidden-import whisper ^
        --hidden-import pyaudio ^
        --hidden-import simpleaudio ^
        --hidden-import langchain_community ^
        --hidden-import langchain_ollama ^
        --hidden-import jieba ^
        --add-data "technical_file;technical_file" ^
        --exclude-module matplotlib ^
        --exclude-module tkinter ^
        --exclude-module pandas ^
        --exclude-module jupyter ^
        qa_RAGsystem_printChunk_QAsplit_coquiTTS.py
)

if errorlevel 1 (
    echo.
    echo Build failed! Please check error messages.
    echo.
    echo Common solutions:
    echo 1. Ensure all dependency packages are correctly installed
    echo 2. Check if there is enough disk space
    echo 3. Try running this script with administrator privileges
    echo 4. If using GPU, ensure CUDA environment is properly configured
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo          Build Complete!
echo ========================================
echo.

if exist "dist\RAG_TTS_System.exe" (
    echo Success: Executable created at dist\RAG_TTS_System.exe
    echo.
    echo Usage Instructions:
    echo 1. Copy technical_file directory to the same location as executable
    echo 2. If using OpenAI API, create .env file and set OPENAI_API_KEY
    echo 3. Ensure target machine has required runtime libraries (Visual C++ Redistributable)
    echo 4. On first run, program will auto-download Whisper and TTS models
    echo.
    
    REM Ask if user wants to test run
    set /p test_run="Do you want to test run the executable? (y/n): "
    if /i "%test_run%"=="y" (
        echo.
        echo Testing executable...
        echo Note: This is a quick test, full functionality should be verified in target environment.
        echo.
        cd dist
        RAG_TTS_System.exe --version 2>nul || (
            echo Executable created successfully, but may need testing in target environment.
        )
        cd ..
    )
) else (
    echo Failed: Executable creation failed. Please check dist directory.
)

echo.
echo Cleaning temporary files...
if exist "build" rmdir /s /q "build"

echo.
echo Build process completed.
pause 