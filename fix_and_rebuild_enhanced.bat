@echo off
chcp 65001 > nul
echo ==========================================
echo Enhanced PyInstaller Build with jaraco Fix
echo ==========================================

echo Step 1: Running jaraco.text error fix...
python fix_jaraco_error.py
if %errorlevel% neq 0 (
    echo WARNING: jaraco fix script had issues, but continuing...
)

echo.
echo Step 2: Installing additional setuptools dependencies...
echo Installing jaraco packages...
pip install jaraco.text --upgrade
pip install jaraco.functools --upgrade
pip install jaraco.collections --upgrade
pip install jaraco.classes --upgrade
pip install more-itertools --upgrade
pip install importlib-metadata --upgrade
pip install zipp --upgrade

echo.
echo Step 3: Upgrading core build tools...
pip install --upgrade pip setuptools wheel pyinstaller

echo.
echo Step 4: Installing/upgrading LangChain packages...
pip install langchain-ollama --upgrade
pip install langchain-core --upgrade
pip install langchain-community --upgrade
pip install ollama --upgrade

echo.
echo Step 5: Cleaning old build files...
if exist "build" (
    echo Removing build directory...
    rmdir /s /q "build"
)
if exist "dist" (
    echo Removing dist directory...
    rmdir /s /q "dist"
)
if exist "__pycache__" (
    echo Removing __pycache__ directory...
    rmdir /s /q "__pycache__"
)

echo.
echo Step 6: Building with enhanced spec file...
if exist "build_pyinstaller_enhanced.spec" (
    echo Using enhanced spec file...
    pyinstaller build_pyinstaller_enhanced.spec --clean --noconfirm
) else (
    echo Enhanced spec file not found, using original...
    pyinstaller build_pyinstaller.spec --clean --noconfirm
)

if %errorlevel% neq 0 (
    echo ERROR: PyInstaller with spec file failed!
    echo.
    echo Trying direct command with jaraco imports...
    pyinstaller --onefile ^
        --console ^
        --clean ^
        --noconfirm ^
        --hidden-import=jaraco ^
        --hidden-import=jaraco.text ^
        --hidden-import=jaraco.functools ^
        --hidden-import=jaraco.collections ^
        --hidden-import=jaraco.classes ^
        --hidden-import=more_itertools ^
        --hidden-import=importlib_metadata ^
        --hidden-import=zipp ^
        --hidden-import=setuptools ^
        --hidden-import=pkg_resources ^
        --hidden-import=langchain_ollama ^
        --hidden-import=langchain_ollama.llms ^
        --hidden-import=langchain_ollama.embeddings ^
        --hidden-import=langchain_ollama.chat_models ^
        --hidden-import=langchain_core ^
        --hidden-import=langchain_community ^
        --hidden-import=ollama ^
        --hidden-import=chromadb ^
        --hidden-import=TTS ^
        --hidden-import=whisper ^
        --hidden-import=PyPDF2 ^
        --hidden-import=jieba ^
        --hidden-import=inflect ^
        --add-data="technical_file;technical_file" ^
        --name=RAG_TTS_System_Enhanced ^
        qa_RAGsystem_printChunk_QAsplit_coquiTTS.py
    
    if %errorlevel% neq 0 (
        echo ERROR: Both build methods failed!
        echo.
        echo Troubleshooting steps:
        echo 1. Check if Python virtual environment is activated
        echo 2. Run: python -c "import jaraco.text; print('jaraco.text OK')"
        echo 3. Try: pip install --force-reinstall setuptools
        echo 4. Check PyInstaller version: pyinstaller --version
        pause
        exit /b 1
    )
)

echo.
echo Step 7: Verifying build results...
if exist "dist\RAG_TTS_System.exe" (
    echo SUCCESS: RAG_TTS_System.exe was created!
    echo File information:
    dir "dist\RAG_TTS_System.exe"
    set EXECUTABLE_NAME=RAG_TTS_System.exe
) else if exist "dist\RAG_TTS_System_Enhanced.exe" (
    echo SUCCESS: RAG_TTS_System_Enhanced.exe was created!
    echo File information:
    dir "dist\RAG_TTS_System_Enhanced.exe"
    set EXECUTABLE_NAME=RAG_TTS_System_Enhanced.exe
) else (
    echo ERROR: No executable was created!
    echo Please check the PyInstaller output above for errors.
    pause
    exit /b 1
)

echo.
echo Step 8: Quick import test...
echo Testing jaraco.text import in the executable...
echo import jaraco.text; print("jaraco.text import successful") > temp_test.py
dist\%EXECUTABLE_NAME% temp_test.py 2>nul
if %errorlevel% equ 0 (
    echo ✅ jaraco.text import test PASSED
) else (
    echo ⚠️ jaraco.text import test failed, but executable was created
)
if exist temp_test.py del temp_test.py

echo.
echo ==========================================
echo Build completed!
echo.
echo Created executable: dist\%EXECUTABLE_NAME%
echo.
echo Next steps:
echo 1. Test the executable: dist\%EXECUTABLE_NAME%
echo 2. Ensure Ollama service is running
echo 3. Make sure technical_file folder contains PDF files
echo.
echo If you still get jaraco.text errors:
echo 1. Check your Python environment
echo 2. Try: pip install --force-reinstall jaraco.text setuptools
echo 3. Use a fresh virtual environment
echo ==========================================
pause 