# import json
# import random
# from transformers import pipeline
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_ollama import OllamaLLM
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# # Load question generator
# question_gen = pipeline("text2text-generation", model="iarfmoose/t5-base-question-generator")

# # Load subtopic LLM
# llm = OllamaLLM(model="llama3")
# output_parser = StrOutputParser()

# # Prompt to predict subtopic
# subtopic_prompt = PromptTemplate.from_template("""
# Given the subject: {topic}
# And the question: "{question}"

# What is the best subtopic this question belongs to?
# Respond with a short academic subtopic label like "Neural Networks", "Backpropagation", etc.
# """)

# subtopic_chain = subtopic_prompt | llm | output_parser

# # Configs
# MARKS_META = {
#     1: {"question_type": "mcq", "difficulty_level": "easy", "time": "1 min", "cognitive_level": "remembering"},
#     2: {"question_type": "short", "difficulty_level": "medium", "time": "2-3 min", "cognitive_level": "understanding"},
#     3: {"question_type": "descriptive", "difficulty_level": "medium", "time": "4-5 min", "cognitive_level": "applying"},
#     5: {"question_type": "long", "difficulty_level": "hard", "time": "6-10 min", "cognitive_level": "evaluating"}
# }

# splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# def is_valid_question(q):
#     q = q.lower().strip()
#     return len(q) > 10 and not any(bad in q for bad in ["true", "false", "not_entailment", "entailment"])

# def detect_topic(text, topic_keywords):
#     for topic, keywords in topic_keywords.items():
#         for kw in keywords:
#             if kw.lower() in text.lower():
#                 return topic
#     return "General"

# def generate_questions(text, topic_keywords, questions_per_category=5):
#     chunks = splitter.split_text(text)
#     seen_questions = set()
#     used_chunks = set()
#     marks_buckets = {1: [], 2: [], 3: [], 5: []}

#     for marks in [1, 2, 3, 5]:
#         while len(marks_buckets[marks]) < questions_per_category and len(used_chunks) < len(chunks):
#             chunk = random.choice(chunks)
#             if chunk in used_chunks:
#                 continue
#             used_chunks.add(chunk)

#             result = question_gen(f"generate question: {chunk}", max_length=160 if marks > 2 else 64, do_sample=False)
#             question = result[0]['generated_text'].strip()

#             if not is_valid_question(question) or question in seen_questions:
#                 continue
#             seen_questions.add(question)

#             topic = detect_topic(chunk, topic_keywords)
#             subtopic = subtopic_chain.invoke({"topic": topic, "question": question}).strip()

#             question_json = {
#                 "question": question,
#                 "topic": topic,
#                 "subtopic": subtopic,
#                 "question_type": MARKS_META[marks]["question_type"],
#                 "difficulty_level": MARKS_META[marks]["difficulty_level"],
#                 "time": MARKS_META[marks]["time"],
#                 "cognitive_level": MARKS_META[marks]["cognitive_level"],
#                 "marks": marks,
#                 "image": None
#             }

#             marks_buckets[marks].append(question_json)

#     return {
#         "1_mark": marks_buckets[1],
#         "2_mark": marks_buckets[2],
#         "3_mark": marks_buckets[3],
#         "5_mark": marks_buckets[5]
#     }

# def save_questions(output, filename="output_questions.json"):
#     with open(filename, "w", encoding="utf-8") as f:
#         json.dump(output, f, indent=2, ensure_ascii=False)
#     print(f"✅ Saved structured questions to {filename}")

# if __name__ == "__main__":
#     with open("C:\\Users\\hp\\projects\\Question_Generator\\Data_Preprocessing\\vector_stores\\extracted_output.txt", "r", encoding="utf-8") as f:
#         text = f.read()

#     with open("C:\\Users\\hp\\projects\\Question_Generator\\Data_Preprocessing\\vector_stores\\keyword.json", "r", encoding="utf-8") as kf:
#         topic_keywords = json.load(kf)

#     print("🚀 Generating questions with subtopics...")
#     final_questions = generate_questions(text, topic_keywords)
#     save_questions(final_questions)
#     print("✅ Question generation completed successfully!")

import json
import random
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load Ollama LLM
llm = OllamaLLM(model="gemma3:1b")
output_parser = StrOutputParser()

# Prompt to generate questions
question_prompt = PromptTemplate.from_template("""
You are a question generator for academic exams.

Given this text:
---
{text}
---

Generate ONE question suitable for {marks}-mark category.
- Question type: {question_type}
- Difficulty level: {difficulty_level}
- Cognitive level: {cognitive_level}

Only return the question, no explanations.
Important: The question must be answerable from the point of view of a university student. Only use facts present in the excerpt. If you are asking user to answer from some specific line of the exerpt do mention the context of the line so that the question has a standalone context if the user is not given the exerpt during questioning.
""")

# Clarify difficulty expectations to the LLM so it produces distinct easy/medium/hard questions:
# - easy: short recall or definition question; answer requires a single fact or phrase from the excerpt.
# - medium: requires understanding or short explanation; may combine 1–2 facts or ask for a brief reason.
# - hard: analytical/application question; requires reasoning, comparing, or multi-step inference from the excerpt.
# Include a one-line example in the prompt to illustrate each difficulty if needed.

question_chain = question_prompt | llm | output_parser

# Prompt to predict subtopic
subtopic_prompt = PromptTemplate.from_template("""
Given the subject: {topic}
And the question: "{question}"

What is the best subtopic this question belongs to?
Respond with a single short academic subtopic label (maximum 4 words). RETURN ONLY THE LABEL (no explanation, no reasoning).
""")

subtopic_chain = subtopic_prompt | llm | output_parser

# Prompt to predict topic using LLM (similar to subtopic)
topic_prompt = PromptTemplate.from_template(
    """
Given the following excerpt:
{text}

Possible topics (numbered):
{candidates}

Respond with the NUMBER (e.g. "3") of the single most appropriate topic from the list above. Return only the number and nothing else. If none match, return "0".
"""
)

topic_chain = topic_prompt | llm | output_parser

# Prompt to predict topic label directly (LLM returns a single short label)
topic_label_prompt = PromptTemplate.from_template("""
Given the following text excerpt:
---
{text}
---

Provide a wide TOPIC label (maximum 4 words) that describes the excerpt or the topic which covers it.
- RETURN ONLY THE LABEL (no explanation, no extra text).
- Keep the label short and academic (e.g. "Artificial Intelligence", "Software Engineering", "Data Structures", "Databases") See how these are wide and vague terms.

""")

topic_label_chain = topic_label_prompt | llm | output_parser


# Configs
MARKS_META = {
    # Use a single integer minute value per question for easy aggregation in dashboards
    1: {"question_type": "mcq", "difficulty_level": "easy", "time": 1, "cognitive_level": "remembering"},
    2: {"question_type": "short", "difficulty_level": "medium", "time": 3, "cognitive_level": "understanding"},
    # make 3-mark questions 'hard' to provide variety in difficulty across buckets
    3: {"question_type": "descriptive", "difficulty_level": "hard", "time": 5, "cognitive_level": "applying"},
    5: {"question_type": "long", "difficulty_level": "hard", "time": 8, "cognitive_level": "evaluating"}
}

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

def normalize_topic(label: str) -> str:
    """Normalize topic label returned by LLM: strip, collapse whitespace, title-case and limit to 4 words."""
    import re
    if not label:
        return "General"
    s = label.strip().strip('"').strip("'")
    s = re.sub(r"\s+", " ", s)
    words = s.split()
    s = " ".join(words[:4])
    return s.title() if s else "General"


def detect_topic(text, topic_keywords=None):
    """LLM-first topic detection that mirrors subtopic generation.
    Falls back to keyword scoring if LLM returns nothing usable.
    Returns a short normalized label (or 'General')."""
    cleaned = text or ""

    # Try LLM-first: ask for a single short label (no index)
    try:
        raw = topic_label_chain.invoke({"text": cleaned}).strip()
        # Normalize the raw label (reuse your normalize_topic)
        sel = normalize_topic(raw)
        # If the LLM returned something meaningful, accept it
        if sel and sel.lower() != "0" and sel.lower() != "none":
            # Optionally, if topic_keywords provided and sel exactly matches a candidate, prefer candidate's case
            if topic_keywords:
                for c in topic_keywords.keys():
                    if sel.lower() == str(c).lower():
                        return c
            return sel
    except Exception:
        # LLM failure — fall through to fallback
        pass

    # Fallback: simple keyword scoring (literal matches) — keep your existing heuristic
    try:
        import re
        cleaned_l = cleaned.lower()
        scores = {}
        for topic, keywords in (topic_keywords or {}).items():
            score = 0
            for kw in keywords:
                if not kw:
                    continue
                pattern = re.escape(str(kw).lower())
                matches = re.findall(pattern, cleaned_l)
                score += len(matches)
            scores[topic] = score

        if scores:
            sorted_topics = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            best_topic, best_score = sorted_topics[0]
            second_score = sorted_topics[1][1] if len(sorted_topics) > 1 else 0
            if best_score >= 2 or (second_score == 0 and best_score >= 1) or (best_score >= 2 * max(1, second_score)):
                return best_topic
    except Exception:
        pass

    return "General"


def generate_questions(text, topic_keywords, questions_per_category=5):
    chunks = splitter.split_text(text)
    seen_questions = set()
    used_chunks = set()
    marks_buckets = {1: [], 2: [], 3: [], 5: []}

    for marks, meta in MARKS_META.items():
        while len(marks_buckets[marks]) < questions_per_category and len(used_chunks) < len(chunks):
            chunk = random.choice(chunks)
            if chunk in used_chunks:
                continue
            used_chunks.add(chunk)

            # Generate question with Ollama
            question = question_chain.invoke({
                "text": chunk,
                "marks": marks,
                "question_type": meta["question_type"],
                "difficulty_level": meta["difficulty_level"],
                "cognitive_level": meta["cognitive_level"]
            }).strip()

            if not question or question in seen_questions:
                continue
            seen_questions.add(question)

            # Detect topic + subtopic
            topic = detect_topic(chunk, topic_keywords)
            subtopic = subtopic_chain.invoke({"topic": topic, "question": question}).strip()

            question_json = {
                "question": question,
                "topic": topic,
                "subtopic": subtopic,
                "question_type": meta["question_type"],
                "difficulty_level": meta["difficulty_level"],
                "time": meta["time"],
                "cognitive_level": meta["cognitive_level"],
                "marks": marks,
                "image": None
            }

            marks_buckets[marks].append(question_json)

    return {
        "1_mark": marks_buckets[1],
        "2_mark": marks_buckets[2],
        "3_mark": marks_buckets[3],
        "5_mark": marks_buckets[5]
    }

def save_questions(output, filename="new_output_questions.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved structured questions to {filename}")

if __name__ == "__main__":
    # Use paths relative to this file so the script works across machines
    base_dir = os.path.dirname(__file__)
    extracted_path = os.path.join(base_dir, "extracted_output.txt")
    keyword_path = os.path.join(base_dir, "keyword.json")

    text = ""
    if os.path.exists(extracted_path):
        with open(extracted_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        print(f"⚠️ extracted_output.txt not found at {extracted_path}; running with empty text")

    topic_keywords = {}
    if os.path.exists(keyword_path):
        try:
            with open(keyword_path, "r", encoding="utf-8") as kf:
                topic_keywords = json.load(kf)
        except Exception as e:
            print(f"⚠️ Failed to load keyword.json: {e}")
            topic_keywords = {}
    else:
        print(f"⚠️ keyword.json not found at {keyword_path}; continuing with empty keywords")

    print("🚀 Generating questions with Ollama...")
    final_questions = generate_questions(text, topic_keywords)
    save_questions(final_questions)
    print("✅ Question generation completed successfully!")
