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


def run_generator_for_course(course_name: str, seed_text: str | None = None):
    # attempt to import generator
    try:
        import questions_generation.question_gen as qgen
    except Exception:
        qgen = None

    topic_keywords = load_topic_keywords()
    text = seed_text or ''
    generated = None
    if qgen is not None and hasattr(qgen, 'generate_questions'):
        try:
            sig = None
            import inspect
            sig = inspect.signature(qgen.generate_questions)
            params = list(sig.parameters.keys())
            if len(params) >= 2:
                generated = qgen.generate_questions(text, topic_keywords)
            elif len(params) >= 1:
                generated = qgen.generate_questions(text)
            else:
                generated = qgen.generate_questions()
        except Exception as e:
            print(f"Generator raised: {e}; falling back to deterministic generator.")
            generated = None

    # fallback deterministic generator: split seed text into sentences or create placeholders
    if generated is None:
        if not text:
            # create dummy placeholders
            sentences = [f"Placeholder question about {course_name} - topic {i+1}." for i in range(12)]
        else:
            sentences = [s.strip() for s in text.split('.') if s.strip()][:20]

        def make_q(s, marks, topic='General'):
            time_map = {1:1,2:3,3:5,5:8}
            diff_map = {1:'easy',2:'medium',3:'hard',5:'hard'}
            cog_map = {1:'remembering',2:'understanding',3:'applying',5:'evaluating'}
            return {
                'question': s,
                'topic': topic,
                'subtopic': topic,
                'question_type': 'short' if marks<=2 else 'descriptive',
                'difficulty': diff_map.get(marks,'medium'),
                'time': time_map.get(marks,3),
                'cognitive_level': cog_map.get(marks,'understanding'),
                'marks': marks,
            }

        buckets = {1: [], 2: [], 3: [], 5: []}
        for i, s in enumerate(sentences):
            m = [1,2,3,5][i % 4]
            buckets[m].append(make_q(s, m, topic=course_name))

        generated = {
            '1_mark': buckets[1],
            '2_mark': buckets[2],
            '3_mark': buckets[3],
            '5_mark': buckets[5],
        }

    return generated


def save_and_build(generated: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'generated_questions.json').write_text(json.dumps(generated, indent=2, ensure_ascii=False), encoding='utf-8')

    # simple cleaning: copy generated into cleaned_questions (and ensure keys match)
    cleaned = generated
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
