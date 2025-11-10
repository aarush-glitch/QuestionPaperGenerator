from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# Load embeddings & FAISS vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest") #changed embedding model from llama3
db = FAISS.load_local("new_faiss_index", embeddings, allow_dangerous_deserialization=True)

# Search configuration
query = "Natural language processing techniques for text classification"
target_marks = 3
target_difficulty = "medium"
target_cognitive = "applying"

def smart_filter(docs, marks, difficulty, cognitive):
    exact = [
        d for d in docs
        if d.metadata.get("marks") == marks
        and d.metadata.get("difficulty_level", "").lower() == difficulty.lower()
        and d.metadata.get("cognitive_level", "").lower() == cognitive.lower()
    ]
    if exact:
        return exact

    fallback = [
        d for d in docs
        if d.metadata.get("difficulty_level", "").lower() == difficulty.lower()
    ]
    if fallback:
        return fallback

    return docs[:3]

similar_questions = db.similarity_search(query, k=20)
filtered = smart_filter(similar_questions, target_marks, target_difficulty, target_cognitive)

# Step 3: Display results
print(f"\n🎯 Filtered {len(filtered)} question(s) for:")
print(f"   ➤ Marks: {target_marks}")
print(f"   ➤ Difficulty: {target_difficulty}")
print(f"   ➤ Cognitive Level: {target_cognitive}\n")

for doc in filtered:
    metadata = doc.metadata
    print(f"❓ Question: {doc.page_content}")
    print(f"🏷️  Topic: {metadata.get('topic', 'unknown')}")
    print(f"🔹 Subtopic: {metadata.get('subtopic', 'unknown')}")
    print(f"🎯 Marks: {metadata.get('marks', '?')}")
    print(f"📈 Difficulty: {metadata.get('difficulty', 'unknown')}")
    print(f"🧠 Cognitive Level: {metadata.get('cognitive_level', 'unknown')}")
    print("-" * 60)
