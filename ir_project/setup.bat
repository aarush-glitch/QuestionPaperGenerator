@echo off
REM Setup script for Information Retrieval Project (Windows)
REM ========================================================

echo 🚀 Setting up Information Retrieval Project...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8+ first.
    pause
    exit /b 1
)

REM Check if pip is installed
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip is not installed. Please install pip first.
    pause
    exit /b 1
)

REM Install Python dependencies
echo 📦 Installing Python dependencies...
pip install -r requirements.txt

REM Check if Ollama is installed
ollama --version >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Ollama is not installed.
    echo 📥 Please install Ollama from: https://ollama.ai/
    echo 🔽 Then run: ollama pull nomic-embed-text:latest
    echo.
    echo Continue setup anyway? (y/N^)
    set /p response=
    if /i not "%response%"=="y" exit /b 1
) else (
    echo 🤖 Pulling Ollama embedding model...
    ollama pull nomic-embed-text:latest
)

REM Create necessary directories
echo 📁 Creating directories...
if not exist indexes mkdir indexes
if not exist logs mkdir logs

REM Create the FAISS index
echo 🔍 Creating FAISS vector index...
cd embedding
python create_index.py
cd ..

REM Test the installation
echo 🧪 Testing installation...
cd search
python -c "try: from semantic_search import SemanticSearchEngine; print('✅ Search engine import successful')\nexcept Exception as e: print(f'❌ Search engine import failed: {e}')"
cd ..

echo.
echo ✅ Setup complete!
echo.
echo 🎯 Next steps:
echo 1. Run Streamlit demo: cd demo ^&^& streamlit run streamlit_demo.py
echo 2. Run API demo: cd demo ^&^& python api_demo.py
echo 3. Test search: cd search ^&^& python semantic_search.py
echo.
echo 📚 See README.md for detailed usage instructions.
pause