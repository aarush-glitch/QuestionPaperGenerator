import json
import traceback
import sys
from pathlib import Path

# Ensure repo root is on sys.path so imports resolve when running script directly
try:
    repo_root = Path(__file__).resolve().parent.parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
except Exception:
    pass

results = {}

# Check question generator
try:
    import questions_generation.question_gen as qgen
    results['question_gen_import'] = True
    try:
        # call generator with minimal params (1 per category)
        sample = qgen.generate_questions("This is a short test excerpt about algorithms and data structures.", {}, 1)
        results['generator_called'] = True
        # report counts
        if isinstance(sample, dict):
            results['sample_counts'] = {k: len(v) for k, v in sample.items()}
        else:
            results['sample_counts'] = 'unexpected-type'
    except Exception as e:
        results['generator_called'] = False
        results['generator_error'] = traceback.format_exc()
except Exception as e:
    results['question_gen_import'] = False
    results['question_gen_error'] = traceback.format_exc()

# Check FAISS and OllamaEmbeddings imports
try:
    from langchain_community.vectorstores import FAISS
    results['faiss_import'] = True
except Exception:
    results['faiss_import'] = False

try:
    from langchain_ollama import OllamaEmbeddings
    results['ollama_embeddings_import'] = True
    try:
        # attempt to instantiate embeddings (may not require remote)
        emb = OllamaEmbeddings(model='nomic-embed-text:latest')
        results['ollama_emb_instance'] = True
    except Exception as e:
        results['ollama_emb_instance'] = False
        results['ollama_emb_error'] = traceback.format_exc()
except Exception:
    results['ollama_embeddings_import'] = False

print(json.dumps(results, indent=2))
