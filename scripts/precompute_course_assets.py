"""
Precompute course assets by running the question generator (offline) and saving
cleaned JSON + FAISS index under `data/courses/<course_slug>/`.

Usage:
  python scripts/precompute_course_assets.py --course "Algorithms" --out-slug algorithms

If `questions_generation.question_gen` is available, this script will call
`generate_questions(text, topic_keywords)` when possible. Otherwise it will
use a deterministic fallback behavior.
"""
import argparse
from pathlib import Path
import json
import hashlib
from datetime import datetime


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / 'scripts'
OUT_BASE = ROOT / 'data' / 'courses'

# Ensure the repository root is on sys.path so package imports like
# `questions_generation` work when the script is run directly.
import sys
try:
    repo_root_str = str(ROOT)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
except Exception:
    pass


def slugify(name: str):
    return ''.join(c if c.isalnum() else '_' for c in name.strip().lower()).strip('_')


def load_topic_keywords():
    try:
        kwf = Path('questions_generation') / 'keyword.json'
        if kwf.exists():
            return json.loads(kwf.read_text(encoding='utf-8'))
    except Exception:
        pass
    return {}


def run_generator_for_course(course_name: str, seed_text: str | None = None, questions_per_category: int = 10):
    """Call the project's question generator (LLM-based) to produce structured
    questions. This function delegates entirely to
    `questions_generation.question_gen.generate_questions` and does NOT apply
    heuristic fallbacks — the generator must provide topic/subtopic/etc via LLM.

    `questions_per_category` controls how many questions per marks bucket are
    requested (default 10 -> 40 total across 1/2/3/5 mark buckets).
    """
    try:
        import questions_generation.question_gen as qgen
    except Exception as e:
        print(f"Question generator module not available: {e}")
        return {}

    topic_keywords = load_topic_keywords()
    text = seed_text or ''

    if not hasattr(qgen, 'generate_questions'):
        print("Generator module does not expose `generate_questions`. Aborting generation.")
        return {}

    try:
        # Call the generator exactly like the `question_gen.py` entrypoint expects.
        generated = qgen.generate_questions(text, topic_keywords, questions_per_category)
        return generated if isinstance(generated, dict) else {}
    except Exception as e:
        print(f"Generator raised an exception: {e}")
        return {}


def save_and_build(generated: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'generated_questions.json').write_text(json.dumps(generated, indent=2, ensure_ascii=False), encoding='utf-8')

    # simple cleaning: copy generated into cleaned_questions (and ensure keys match)
    # Normalize: ensure both 'difficulty' and 'difficulty_level' are present for compatibility.
    cleaned = {}
    for group_key, items in (generated or {}).items():
        cleaned[group_key] = []
        for q in (items or []):
            try:
                # prefer existing 'difficulty' then 'difficulty_level'
                diff = q.get('difficulty') or q.get('difficulty_level') or None
                if diff is not None:
                    q['difficulty'] = diff
                    q['difficulty_level'] = diff
                cleaned[group_key].append(q)
            except Exception:
                cleaned[group_key].append(q)
    (out_dir / 'cleaned_questions.json').write_text(json.dumps(cleaned, indent=2, ensure_ascii=False), encoding='utf-8')

    manifest = {
        'model': 'precompute-v1',
        'count_questions': sum(len(v) for v in cleaned.values()),
        'groups': {k: len(v) for k, v in cleaned.items()},
        'updated_at': datetime.utcnow().isoformat() + 'Z'
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    # attempt to build FAISS using same approach as ETL script
    try:
        from scripts.etl_data_structures import build_faiss
        build_faiss(cleaned, out_dir)
    except Exception:
        # best-effort: attempt inline build similar to ETL
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_ollama import OllamaEmbeddings
            from langchain.schema import Document
            docs = []
            for marks_key, questions in cleaned.items():
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
            embeddings = OllamaEmbeddings(model='nomic-embed-text:latest')
            db = FAISS.from_documents(docs, embeddings)
            faiss_dir = out_dir / 'faiss_index'
            if faiss_dir.exists():
                import shutil
                shutil.rmtree(faiss_dir, ignore_errors=True)
            db.save_local(str(faiss_dir))
            print('FAISS built (fallback path)')
        except Exception:
            print('Skipping FAISS build (not available)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--course', required=True)
    parser.add_argument('--out-slug', required=False)
    parser.add_argument('--seed-text', required=False, help='Path to a text file with seed material')
    args = parser.parse_args()

    course = args.course
    slug = args.out_slug or slugify(course)
    out_dir = OUT_BASE / slug

    seed = None
    if args.seed_text:
        p = Path(args.seed_text)
        if p.exists():
            seed = p.read_text(encoding='utf-8')

    generated = run_generator_for_course(course, seed)
    save_and_build(generated, out_dir)
    print(f'Precomputed assets saved to {out_dir}')


if __name__ == '__main__':
    main()
