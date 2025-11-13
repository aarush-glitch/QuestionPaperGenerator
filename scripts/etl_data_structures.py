"""
ETL for Data Structures CSV -> cleaned_questions.json + FAISS index + manifest

Usage:
  python scripts/etl_data_structures.py 

This script reads `final_database_coe.csv` in the repo root, normalizes fields,
deduplicates by `unique_hash` (if available) or question text, and writes
`data/courses/data_structures/cleaned_questions.json` and `manifest.json`.

If `langchain_community.vectorstores.FAISS` and `langchain_ollama.OllamaEmbeddings`
are available in the environment, it will also attempt to build a FAISS index at
`data/courses/data_structures/faiss_index/`.
"""
from pathlib import Path
import csv
import json
import hashlib
import datetime
import re
from collections import defaultdict


ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "final_database_coe.csv"
OUT_DIR = ROOT / "data" / "courses" / "data_structures"


def parse_time_minutes(val):
    try:
        if val is None:
            return None
        if isinstance(val, int):
            return val
        s = str(val).strip().lower()
        nums = re.findall(r"\d+\.?\d*", s)
        if not nums:
            return None
        if '-' in s and len(nums) >= 2:
            a = float(nums[0]); b = float(nums[1])
            return int(round((a + b) / 2))
        return int(round(float(nums[0])))
    except Exception:
        return None


def normalize_row(row: dict):
    # Map CSV columns to canonical schema used by the app
    q_text = (row.get('question') or '').strip()
    marks_raw = row.get('marks') or row.get('mark') or ''
    try:
        marks = int(float(str(marks_raw)))
    except Exception:
        # try extract integer from string like '2_marks'
        m = re.search(r"(\d+)", str(marks_raw))
        marks = int(m.group(1)) if m else 0

    difficulty = (row.get('difficulty') or '').strip().lower() or None
    cognitive = (row.get('cognitive_level') or row.get('cognitive') or '').strip() or None
    topic = (row.get('topic') or '').strip() or "General"
    subtopic = (row.get('subtopic') or '').strip() or topic
    time_val = parse_time_minutes(row.get('time') or row.get('time_estimate') or row.get('time (min)') or '')
    unique_hash = row.get('unique_hash') or None

    return {
        'question': q_text,
        'marks': marks,
        'difficulty': difficulty,
        'cognitive_level': cognitive,
        'topic': topic,
        'subtopic': subtopic,
        'time': time_val,
        'source_file': row.get('source_file') or '',
        'unique_hash': unique_hash,
    }


def fingerprint_file(path: Path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_cleaned_and_manifest(grouped, out_dir: Path, src_csv: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = out_dir / 'cleaned_questions.json'
    cleaned_path.write_text(json.dumps(grouped, indent=2, ensure_ascii=False), encoding='utf-8')

    manifest = {
        'source': str(src_csv),
        'fingerprint': fingerprint_file(src_csv) if src_csv.exists() else None,
        'model': 'csv-etl-v1',
        'count_questions': sum(len(v) for v in grouped.values()),
        'groups': {k: len(v) for k, v in grouped.items()},
        'updated_at': datetime.datetime.utcnow().isoformat() + 'Z',
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f"Wrote cleaned questions to {cleaned_path}")
    print(f"Wrote manifest to {out_dir / 'manifest.json'}")


def build_faiss(cleaned_questions: dict, out_dir: Path):
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_ollama import OllamaEmbeddings
        from langchain.schema import Document
    except Exception as e:
        print("FAISS or OllamaEmbeddings not available; skipping index build.")
        return False

    docs = []
    for marks_key, questions in cleaned_questions.items():
        for q in questions:
            meta = {
                'topic': q.get('topic'),
                'subtopic': q.get('subtopic'),
                'marks': q.get('marks'),
                'difficulty': q.get('difficulty'),
                'cognitive_level': q.get('cognitive_level'),
                'time': q.get('time'),
            }
            text = f"{q.get('question','')}\n\nTopic: {meta['topic']}\nSubtopic: {meta['subtopic']}\nMarks: {meta['marks']}\nTime: {meta['time'] or ''}"
            docs.append(Document(page_content=text, metadata=meta))

    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    db = FAISS.from_documents(docs, embeddings)
    faiss_dir = out_dir / 'faiss_index'
    if faiss_dir.exists():
        import shutil
        shutil.rmtree(faiss_dir, ignore_errors=True)
    db.save_local(str(faiss_dir))
    print(f"Saved FAISS index to {faiss_dir}")
    return True


def main():
    if not CSV_PATH.exists():
        print(f"CSV not found at {CSV_PATH}. Aborting.")
        return

    rows = []
    with open(CSV_PATH, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for r in reader:
            rows.append(r)

    seen_hashes = set()
    grouped = defaultdict(list)

    for r in rows:
        nr = normalize_row(r)
        uh = nr.get('unique_hash') or hashlib.sha1((nr.get('question') or '').encode('utf-8')).hexdigest()
        if uh in seen_hashes:
            continue
        seen_hashes.add(uh)

        # place into standard buckets by marks (1,2,3,5). fallback to nearest bucket
        marks = nr.get('marks') or 0
        if marks <= 1:
            bucket = '1_mark'
        elif marks == 2:
            bucket = '2_mark'
        elif marks == 3:
            bucket = '3_mark'
        else:
            bucket = '5_mark'

        # canonicalize difficulty to easy/medium/hard if possible
        diff = (nr.get('difficulty') or '').lower()
        if diff in ('easy','e'):
            nr['difficulty'] = 'easy'
        elif diff in ('medium','m'):
            nr['difficulty'] = 'medium'
        elif diff in ('hard','h'):
            nr['difficulty'] = 'hard'
        else:
            # try numeric heuristic: marks 1->easy, 2->medium, 3+->hard
            if nr.get('marks',0) <= 1:
                nr['difficulty'] = 'easy'
            elif nr.get('marks',0) == 2:
                nr['difficulty'] = 'medium'
            else:
                nr['difficulty'] = 'hard'

        grouped[bucket].append({k: v for k, v in nr.items() if k != 'unique_hash'})

    write_cleaned_and_manifest(grouped, OUT_DIR, CSV_PATH)

    # try build faiss index
    try:
        ok = build_faiss(grouped, OUT_DIR)
        if ok:
            print("FAISS index built successfully.")
    except Exception as e:
        print(f"FAISS build failed: {e}")


if __name__ == '__main__':
    main()
