"""
IR Project Validation Script
============================

This script validates that the IR project is properly set up and all
components can be imported and initialized correctly.
"""

import sys
import os
from pathlib import Path

def validate_project_structure():
    """Validate that all required files and directories exist."""
    project_root = Path(__file__).parent
    
    required_dirs = ["data", "embedding", "search", "evaluation", "demo", "indexes"]
    required_files = [
        "requirements.txt",
        "README.md",
        "data/questions.json",
        "embedding/create_index.py",
        "search/semantic_search.py",
        "evaluation/metrics.py",
        "demo/streamlit_demo.py",
        "demo/api_demo.py"
    ]
    
    print("🔍 Validating project structure...")
    
    # Check directories
    missing_dirs = []
    for dir_name in required_dirs:
        if not (project_root / dir_name).exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    # Check files
    missing_files = []
    for file_path in required_files:
        if not (project_root / file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ Project structure is complete")
    return True

def test_imports():
    """Test that all modules can be imported correctly."""
    print("🔧 Testing module imports...")
    
    try:
        # Add project root to path
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        # Test imports
        from embedding.create_index import VectorStoreManager
        print("✅ VectorStoreManager import successful")
        
        from search.semantic_search import SemanticSearchEngine
        print("✅ SemanticSearchEngine import successful")
        
        from evaluation.metrics import IRMetrics, SearchEvaluator
        print("✅ Evaluation metrics import successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    print("📦 Checking dependencies...")
    
    required_packages = [
        "langchain_community",
        "langchain_ollama", 
        "langchain",
        "faiss",
        "numpy",
        "streamlit",
        "fastapi"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "faiss":
                import faiss
            else:
                __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is not installed")
    
    if missing_packages:
        print(f"\n🚨 Missing packages: {missing_packages}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def check_ollama():
    """Check if Ollama is available."""
    print("🤖 Checking Ollama availability...")
    
    try:
        import subprocess
        result = subprocess.run(["ollama", "--version"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Ollama is installed")
            
            # Check if model is available
            result = subprocess.run(["ollama", "list"], 
                                  capture_output=True, text=True, timeout=10)
            if "nomic-embed-text" in result.stdout:
                print("✅ nomic-embed-text model is available")
                return True
            else:
                print("⚠️  nomic-embed-text model not found")
                print("Run: ollama pull nomic-embed-text:latest")
                return False
        else:
            print("❌ Ollama command failed")
            return False
            
    except FileNotFoundError:
        print("❌ Ollama is not installed")
        print("Install from: https://ollama.ai/")
        return False
    except subprocess.TimeoutExpired:
        print("⚠️  Ollama command timed out")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False

def main():
    """Run all validation checks."""
    print("🚀 IR Project Validation")
    print("=" * 50)
    
    checks = [
        ("Project Structure", validate_project_structure),
        ("Module Imports", test_imports),
        ("Dependencies", check_dependencies),
        ("Ollama Setup", check_ollama)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n📋 {check_name}")
        print("-" * 30)
        try:
            if not check_func():
                all_passed = False
        except Exception as e:
            print(f"❌ {check_name} failed with error: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All validation checks passed!")
        print("✅ The IR project is ready to use")
        print("\n🎯 Next steps:")
        print("1. Run: python embedding/create_index.py")
        print("2. Run: streamlit run demo/streamlit_demo.py")
        print("3. Run: python demo/api_demo.py")
    else:
        print("⚠️  Some validation checks failed")
        print("📚 Please check the README.md for setup instructions")
    
    return all_passed

if __name__ == "__main__":
    main()