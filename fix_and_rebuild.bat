@echo off
chcp 65001 > nul
echo ========================================
echo LangChain Ollama Module Fix and Rebuild
echo ========================================

echo Step 1: Checking Python environment...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python first.
    pause
    exit /b 1
)

echo.
echo Step 2: Running diagnostic script...
python fix_langchain_ollama.py
if %errorlevel% neq 0 (
    echo WARNING: Diagnostic script had issues, but continuing...
)

echo.
echo Step 3: Installing/upgrading required packages...
echo Installing langchain-ollama...
pip install langchain-ollama --upgrade
echo Installing langchain-core...
pip install langchain-core --upgrade  
echo Installing langchain-community...
pip install langchain-community --upgrade
echo Installing ollama...
pip install ollama --upgrade
echo Installing PyInstaller...
pip install pyinstaller --upgrade

echo.
echo Step 4: Cleaning old build files...
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
if exist "*.spec~" (
    echo Removing backup spec files...
    del /q "*.spec~"
)

echo.
echo Step 5: Running PyInstaller with updated spec...
echo Using spec file: build_pyinstaller.spec
pyinstaller build_pyinstaller.spec --clean --noconfirm
if %errorlevel% neq 0 (
    echo ERROR: PyInstaller with spec file failed!
    echo.
    echo Trying alternative approach with direct command...
    pyinstaller --onefile ^
        --console ^
        --clean ^
        --noconfirm ^
        --hidden-import=langchain_ollama ^
        --hidden-import=langchain_ollama.llms ^
        --hidden-import=langchain_ollama.llms.ollama ^
        --hidden-import=langchain_ollama.embeddings ^
        --hidden-import=langchain_ollama.embeddings.ollama ^
        --hidden-import=langchain_ollama.chat_models ^
        --hidden-import=langchain_ollama.chat_models.ollama ^
        --hidden-import=langchain_core ^
        --hidden-import=langchain_core.embeddings ^
        --hidden-import=langchain_core.language_models ^
        --hidden-import=langchain_core.language_models.llms ^
        --hidden-import=langchain_community ^
        --hidden-import=langchain_community.vectorstores ^
        --hidden-import=langchain_community.vectorstores.chroma ^
        --hidden-import=langchain_community.retrievers ^
        --hidden-import=langchain_community.retrievers.bm25 ^
        --hidden-import=ollama ^
        --hidden-import=chromadb ^
        --hidden-import=TTS ^
        --hidden-import=whisper ^
        --hidden-import=PyPDF2 ^
        --hidden-import=jieba ^
        --hidden-import=inflect ^
        --add-data="technical_file;technical_file" ^
        --name=RAG_TTS_System_Fixed ^
        qa_RAGsystem_printChunk_QAsplit_coquiTTS.py
    
    if %errorlevel% neq 0 (
        echo ERROR: Both build methods failed!
        echo.
        echo Possible solutions:
        echo 1. Check if all required packages are installed
        echo 2. Try running: pip install --upgrade pip setuptools wheel
        echo 3. Check if the main Python file exists
        echo 4. Review the error messages above
        pause
        exit /b 1
    )
)

echo.
echo Step 6: Testing the built executable...
if exist "dist\RAG_TTS_System.exe" (
    echo SUCCESS: RAG_TTS_System.exe was created!
    echo File information:
    dir "dist\RAG_TTS_System.exe"
    echo.
    echo The executable is ready for testing.
) else if exist "dist\RAG_TTS_System_Fixed.exe" (
    echo SUCCESS: RAG_TTS_System_Fixed.exe was created!
    echo File information:
    dir "dist\RAG_TTS_System_Fixed.exe"
    echo.
    echo The executable is ready for testing.
) else (
    echo ERROR: No executable was created!
    echo Please check the PyInstaller output above for errors.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build completed successfully!
echo.
echo Next steps:
if exist "dist\RAG_TTS_System.exe" (
    echo 1. Test the executable: dist\RAG_TTS_System.exe
) else (
    echo 1. Test the executable: dist\RAG_TTS_System_Fixed.exe
)
echo 2. Make sure Ollama service is running before testing
echo 3. Ensure the technical_file folder contains your PDF files
echo 4. If you encounter module errors, run this script again
echo.
echo Troubleshooting:
echo - If import errors persist, check langchain_hiddenimports.txt
echo - For TTS issues, verify TTS package installation
echo - For GPU acceleration, ensure CUDA is properly installed
echo ========================================
pause 