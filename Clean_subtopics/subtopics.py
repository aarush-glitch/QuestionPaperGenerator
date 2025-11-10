import json
import re

def extract_subtopic(text):
    # Match bold-marked labels first: **Label**
    match = re.search(r"\*\*([^\*]+)\*\*", text)
    if match:
        return match.group(1).strip()

    # Match quoted labels: "Label"
    match = re.search(r'"([^\"]+)"', text)
    if match:
        return match.group(1).strip()

    # Direct phrasing patterns
    match = re.search(r"subtopic (?:is|would be|for this question would be)[:\s]*([A-Za-z0-9 \-/()]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Otherwise, take the first short line/sentence as a fallback
    # Remove extra whitespace and split on newlines or periods
    cleaned = text.strip()
    lines = [l.strip() for l in re.split(r"[\n\.]+", cleaned) if l.strip()]
    if lines:
        first = lines[0]
        # if short enough, return it
        if len(first.split()) <= 6:
            return first
        # try to return up to first 4 words as a compact label
        return " ".join(first.split()[:4])

    return "General"

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
