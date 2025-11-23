"""
Rebuild FAISS index from existing cleaned_questions.json file.

Usage:
  python scripts/rebuild_faiss_index.py --course data_structures
  
This will read data/courses/data_structures/cleaned_questions.json
and rebuild the FAISS index in data/courses/data_structures/faiss_index/
"""
import argparse
import json
import shutil
from pathlib import Path
import sys

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def rebuild_faiss_index(course_slug: str):
    """Rebuild FAISS index from existing cleaned_questions.json"""
    
    course_dir = ROOT / 'data' / 'courses' / course_slug
    cleaned_json = course_dir / 'cleaned_questions.json'
    faiss_dir = course_dir / 'faiss_index'
    
    # Check if cleaned_questions.json exists
    if not cleaned_json.exists():
        print(f"❌ Error: {cleaned_json} does not exist!")
        return False
    
    print(f"📖 Reading questions from: {cleaned_json}")
    
    try:
        cleaned = json.loads(cleaned_json.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"❌ Error reading JSON: {e}")
        return False
    
    total_questions = sum(len(v) for v in cleaned.values())
    print(f"✅ Loaded {total_questions} questions")
    
    # Build FAISS index
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_ollama import OllamaEmbeddings
        from langchain.schema import Document
        
        print("🔄 Creating embeddings and building FAISS index...")
        print("   (This may take a few minutes depending on number of questions)")
        
        docs = []
        for marks_key, questions in cleaned.items():
            for q in questions:
                meta = {
                    'topic': q.get('topic', ''),
                    'subtopic': q.get('subtopic', ''),
                    'marks': q.get('marks', ''),
                    'difficulty': q.get('difficulty', ''),
                    'cognitive_level': q.get('cognitive_level', ''),
                    'time': q.get('time', ''),
                }
                # Enrich page content with metadata for better search
                text = f"{q.get('question', '')}\n\nTopic: {meta['topic']}\nSubtopic: {meta['subtopic']}\nMarks: {meta['marks']}"
                docs.append(Document(page_content=text, metadata=meta))
        
        print(f"   Creating {len(docs)} document embeddings using Ollama (nomic-embed-text)...")
        embeddings = OllamaEmbeddings(model='nomic-embed-text:latest')
        db = FAISS.from_documents(docs, embeddings)
        
        # Remove old index if exists
        if faiss_dir.exists():
            print(f"   Removing old index: {faiss_dir}")
            shutil.rmtree(faiss_dir, ignore_errors=True)
        
        # Save new index
        faiss_dir.mkdir(parents=True, exist_ok=True)
        db.save_local(str(faiss_dir))
        
        print(f"✅ FAISS index successfully built and saved to: {faiss_dir}")
        print(f"   Index files: index.faiss, index.pkl")
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("   Make sure you have installed: langchain-community, langchain-ollama, faiss-cpu")
        return False
    except Exception as e:
        print(f"❌ Error building FAISS index: {e}")
        print("   Make sure Ollama is running (ollama serve)")
        return False


def main():
    parser = argparse.ArgumentParser(description='Rebuild FAISS index from existing cleaned_questions.json')
    parser.add_argument('--course', required=True, help='Course slug (e.g., data_structures)')
    args = parser.parse_args()
    
    print(f"\n🚀 Rebuilding FAISS index for course: {args.course}\n")
    success = rebuild_faiss_index(args.course)
    
    if success:
        print("\n✨ Done! You can now use the updated index in 2_User_Pipeline.py")
    else:
        print("\n❌ Failed to rebuild index. Check errors above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
