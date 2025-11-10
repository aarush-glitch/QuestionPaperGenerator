import json
import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document

# Load Ollama embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

def load_questions(file_path="questions.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []

    for mark_group, questions in data.items():
        for q in questions:
            doc = Document(
                page_content=q["question"],
                metadata={
                    "topic": q.get("topic", "unknown"),
                    "subtopic": q.get("subtopic", "unknown"),
                    "marks": q.get("marks", 0),
                    "type": q.get("question_type", "unknown"),
                    "difficulty": q.get("difficulty_level", "unknown"),
                    "cognitive_level": q.get("cognitive_level", "unknown"),
                    "time": q.get("time", "unknown")
                }
            )
            docs.append(doc)

    return docs

def store_in_faiss(docs, index_path="new_faiss_index"):
    if os.path.exists(index_path):
        print(f"🔄 FAISS index already exists at {index_path}.")
        return

    print("⚙️ Creating FAISS index using Ollama...")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(index_path)
    print(f"✅ Stored in FAISS at: {index_path}")

if __name__ == "__main__":
    questions = load_questions("/Users/sanatwalia/Desktop/Zomato_Showcasing/coe-project/questions.json")
    store_in_faiss(questions)
