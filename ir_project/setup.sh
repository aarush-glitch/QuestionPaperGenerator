#!/bin/bash
# Setup script for Information Retrieval Project
# =============================================

echo "🚀 Setting up Information Retrieval Project..."

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Please install pip first."
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama is not installed."
    echo "📥 Please install Ollama from: https://ollama.ai/"
    echo "🔽 Then run: ollama pull nomic-embed-text:latest"
    echo ""
    echo "Continue setup anyway? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "🤖 Pulling Ollama embedding model..."
    ollama pull nomic-embed-text:latest
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p indexes
mkdir -p logs

# Create the FAISS index
echo "🔍 Creating FAISS vector index..."
cd embedding
python create_index.py
cd ..

# Test the installation
echo "🧪 Testing installation..."
cd search
python -c "
try:
    from semantic_search import SemanticSearchEngine
    print('✅ Search engine import successful')
except Exception as e:
    print(f'❌ Search engine import failed: {e}')
"
cd ..

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Run Streamlit demo: cd demo && streamlit run streamlit_demo.py"
echo "2. Run API demo: cd demo && python api_demo.py"
echo "3. Test search: cd search && python semantic_search.py"
echo ""
echo "📚 See README.md for detailed usage instructions."