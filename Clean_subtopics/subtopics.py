import json
import re

def extract_subtopic(text):
    # Match patterns like "Generative Models", "Computer Vision", "NLP", etc.
    match = re.search(r"\*\*([^\*]+)\*\*", text)
    if match:
        return match.group(1).strip()
    match = re.search(r'"([^"]+)"', text)
    if match:
        return match.group(1).strip()
    match = re.search(r"subtopic (is|would be|for this question would be)[:\s]*([A-Za-z \-/()]+)", text)
    if match:
        return match.group(2).strip()
    return text if len(text.split()) <= 4 else "General"

def clean_subtopics(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for marks_key, questions in data.items():
        for q in questions:
            original = q.get("subtopic", "")
            q["subtopic"] = extract_subtopic(original)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Cleaned subtopics saved to: {output_file}")

if __name__ == "__main__":
    clean_subtopics(
        input_file="C:\\Users\\hp\\projects\\Question_Generator\\Data_Preprocessing\\output_questions.json",
        output_file="questions.json"
    )
