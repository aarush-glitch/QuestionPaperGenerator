import os
import io
import json
from pathlib import Path

import streamlit as st

# --- Local project imports (from your uploaded files) ---
try:
    from Pytesseract.refinement import handle_image, handle_pdf
except Exception as e:
    handle_image = None
    handle_pdf = None
    _refine_import_error = str(e)

try:
    from Clean_subtopics.subtopics import extract_subtopic
except Exception as e:
    extract_subtopic = None
    _subtopics_import_error = str(e)

# Ollama + FAISS parts
try:
    from langchain_community.vectorstores import FAISS
    from langchain_ollama import OllamaEmbeddings
except Exception as e:
    FAISS = None
    OllamaEmbeddings = None
    _faiss_import_error = str(e)

# Question generator import (optional)
try:
    import questions_generation.question_gen as qgen
except Exception as e:
    qgen = None
    _qgen_import_error = str(e)


st.set_page_config(page_title="LLM Question Paper Pipeline (app2)", layout="wide")

st.title("🧭 LLM Question Paper Generation — End‑to‑End Pipeline (app2)")
st.caption("Pipeline with automatic question generation hook: OCR → (save) → call generator → load generated JSON → Subtopic Cleanup → Vector Store → Search & Filter.")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", [
        "1) OCR & Refinement",
        "2) Subtopic Cleanup",
        "3) Build Vector Store (FAISS)",
        "4) Search & Smart Filter",
        "5) Export Results"
    ])

    st.markdown("---")
    st.subheader("Environment Notes")
    st.write("• Requires: `pytesseract`, `tesseract` binary, `PyMuPDF`, `opencv-python`, `langchain-community`, `langchain-ollama`, `faiss-cpu` (or faiss).")
    st.write("• Ollama must be running locally if you use `OllamaEmbeddings`.")
    st.write("• Make sure your `questions.json` follows the expected format.")

    st.markdown("---")
    st.markdown("**Index Folder (FAISS):**")
    default_index = st.text_input("Index path", value="new_faiss_index")


# --- Shared session state ---
if "refined_text" not in st.session_state:
    st.session_state.refined_text = ""
if "cleaned_questions" not in st.session_state:
    st.session_state.cleaned_questions = None
if "search_results" not in st.session_state:
    st.session_state.search_results = []
# new: hold generated questions dict (before cleaning)
if "generated_questions" not in st.session_state:
    st.session_state.generated_questions = None
if "debug_logs" not in st.session_state:
    st.session_state.debug_logs = []


# =============================
# 1) OCR & Refinement
# =============================
if page.startswith("1"):

    st.header("1) OCR & Refinement")
    st.write("Upload an **image** or **PDF**. We'll run OCR, basic refinement, and optionally call the local question generator to create questions JSON.")

    # Optional raw text input (e.g., if user wants to skip OCR step)
    st.text_area("Optional: paste raw text to refine/analyze", key="manual_text", height=150, help="If provided, this text will be refined/analyzed directly.")

    uploaded = st.file_uploader("Upload an image (png/jpg/jpeg) or a PDF", type=["png", "jpg", "jpeg", "pdf"])

    run_btn = st.button("Run OCR/Refinement (and try generate questions)", type="primary")

    if run_btn:
        output_text = ""
        if st.session_state.manual_text.strip():
            # If manual text provided, try to call same refinement function via handle_image/handle_pdf if available.
            # Fallback: just store the text (since internal refine_text/analyze_text are encapsulated).
            output_text = st.session_state.manual_text.strip()
        elif uploaded is not None:
            suffix = Path(uploaded.name).suffix.lower()
            tmp_path = Path("uploaded_input" + suffix)
            with open(tmp_path, "wb") as f:
                f.write(uploaded.read())

            if suffix in [".png", ".jpg", ".jpeg"]:
                if handle_image is None:
                    st.error("`handle_image` could not be imported from refinement.py")
                else:
                    output_text = handle_image(str(tmp_path))
            elif suffix == ".pdf":
                if handle_pdf is None:
                    st.error("`handle_pdf` could not be imported from refinement.py")
                else:
                    output_text = handle_pdf(str(tmp_path))
        else:
            st.warning("Please paste some text or upload a file.")
        
        if output_text:
            st.session_state.refined_text = output_text
            st.success("Refinement complete.")
            st.code(output_text, language="markdown")

            # save refined text so external question generator can consume it
            try:
                out_dir = Path("questions_generation")
                out_dir.mkdir(parents=True, exist_ok=True)
                in_path = out_dir / "extracted_output.txt"
                in_path.write_text(output_text, encoding="utf-8")
                st.info(f"Saved refined text to {in_path}")
            except Exception as e:
                st.warning(f"Could not save refined text for generator: {e}")
                in_path = None

            # Try to call local question generator (best-effort)
            if qgen is None:
                st.info("Question generator not available to call from app2.")
                if "_qgen_import_error" in globals():
                    st.code(_qgen_import_error)
            else:
                # helper to push debug messages
                def push_log(msg):
                    try:
                        st.session_state.debug_logs.append(msg)
                        # also append to a repo-local file for persistent debugging
                        debug_path = Path("questions_generation") / "generator_debug.log"
                        debug_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(debug_path, "a", encoding="utf-8") as df:
                            df.write(msg + "\n")
                    except Exception:
                        pass

                st.write("Attempting to run question generator (this may take a while)...")
                gen_out = out_dir / "generated_questions.json"
                generated = None
                try:
                    # Try multiple possible call signatures
                    if hasattr(qgen, "generate_questions"):
                        # Robustly call generate_questions by inspecting its signature.
                        # The generator typically expects (text, topic_keywords, ...).
                        # Avoid passing file-path strings as topic_keywords (causes .items error).
                        import inspect

                        sig = inspect.signature(qgen.generate_questions)
                        params = list(sig.parameters.keys())

                        # Prepare text input
                        text_input = None
                        if in_path is not None and in_path.exists():
                            try:
                                text_input = in_path.read_text(encoding="utf-8")
                            except Exception:
                                text_input = None
                        if not text_input and st.session_state.get("refined_text"):
                            text_input = st.session_state.refined_text

                        # Try to load topic keywords if available
                        topic_keywords = None
                        kw_path = Path("questions_generation") / "keyword.json"
                        if kw_path.exists():
                            try:
                                topic_keywords = json.loads(kw_path.read_text(encoding="utf-8"))
                            except Exception:
                                topic_keywords = None

                        generated = None
                        try:
                            push_log(f"Calling generate_questions with text_input={'present' if text_input else 'None'} (len={len(text_input) if text_input else 0}), topic_keywords={'present' if topic_keywords else 'None'}")
                            # Common signature: generate_questions(text, topic_keywords, ...)
                            if len(params) >= 2:
                                # Ensure we have text and a dict for keywords; otherwise try sensible fallbacks
                                q_text = text_input or ""
                                q_kw = topic_keywords or {}
                                result = qgen.generate_questions(q_text, q_kw)
                            elif len(params) == 1:
                                # Single-arg: probably expects raw text (not a path)
                                q_text = text_input or ""
                                result = qgen.generate_questions(q_text)
                            else:
                                # No-arg generator
                                result = qgen.generate_questions()

                            # If function returns the structured questions, save them
                            if isinstance(result, dict):
                                generated = result
                                try:
                                    gen_out.write_text(json.dumps(generated, indent=2, ensure_ascii=False), encoding="utf-8")
                                except Exception:
                                    pass
                            push_log(f"Generator returned type {type(result).__name__}")
                        except TypeError:
                            # Fallbacks: try older path-based signatures
                            try:
                                qgen.generate_questions(str(in_path), str(gen_out))
                            except Exception:
                                try:
                                    qgen.generate_questions(str(in_path))
                                except Exception:
                                    try:
                                        qgen.generate_questions()
                                    except Exception as e:
                                        push_log(f"Generator fallbacks all failed: {e}")
                                        raise
                    elif hasattr(qgen, "main"):
                        try:
                            qgen.main()
                        except TypeError:
                            qgen.main([])
                    else:
                        st.warning("No recognized entry function in questions_generation.question_gen")
                    
                    # attempt to locate produced json
                    candidate_paths = [
                        gen_out,
                        out_dir / "new_output_questions.json",
                        out_dir / "output_questions.json",
                        Path("output_questions.json"),
                        Path("questions.json"),
                    ]
                    for p in candidate_paths:
                        if p.exists():
                            try:
                                generated = json.loads(p.read_text(encoding="utf-8"))
                                st.success(f"Loaded generated JSON from {p}")
                                st.session_state.generated_questions = generated
                                break
                            except Exception:
                                # maybe file not JSON; skip
                                continue

                    if generated is None:
                        st.warning("Generator ran but no output JSON found in expected locations.")
                except Exception as e:
                    st.error(f"Question generator call failed: {e}")

            # offer download
            st.download_button(
                "Download refined text",
                data=output_text.encode("utf-8"),
                file_name="refined_output.txt"
            )

    # Tips / Diagnostics
    with st.expander("Diagnostics & Tips"):
        if handle_image is None or handle_pdf is None:
            st.warning("`refinement.py` import issue: some functions were not available.")
            if "_refine_import_error" in globals():
                st.code(_refine_import_error)
        if qgen is None:
            st.info("Question generator module not imported. Place your generator at questions_generation/question_gen.py")
            if "_qgen_import_error" in globals():
                st.code(_qgen_import_error)
        st.info("If you see Tesseract errors, ensure the binary is installed and accessible on your system.")
        st.markdown("---")
        st.subheader("Generator debug logs")
        if st.session_state.debug_logs:
            for l in st.session_state.debug_logs[-200:]:
                st.text(l)
        else:
            st.info("No debug logs yet. Run step 1 to attempt generation.")


# =============================
# 2) Subtopic Cleanup
# =============================
elif page.startswith("2"):
    st.header("2) Subtopic Cleanup")
    st.write("Clean and normalize `subtopic` fields in your question JSON using rules from `subtopics.py`.")

    uploaded_json = st.file_uploader("Upload questions JSON", type=["json"])
    sample = st.checkbox("Load minimal sample JSON structure", value=False)

    data = None
    if sample:
        sample_data = {
            "1_mark": [
                {"question": "Define overfitting.", "topic": "ML", "subtopic": '**Model Evaluation**', "marks": 1, "difficulty": "easy", "cognitive_level": "remembering"}
            ],
            "3_marks": [
                {"question": "Explain backpropagation.", "topic": "DL", "subtopic": 'subtopic would be: "Neural Networks"', "marks": 3, "difficulty": "medium", "cognitive_level": "understanding"}
            ]
        }
        data = sample_data

    if uploaded_json is not None:
        try:
            data = json.loads(uploaded_json.read().decode("utf-8"))
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

    # new: allow using generated JSON from step 1 (auto-populated)
    if data is None and st.session_state.generated_questions:
        st.info("Using generated questions from OCR+generator run.")
        data = st.session_state.generated_questions

    if data is not None:
        st.subheader("Original JSON")
        st.json(data)

        if extract_subtopic is None:
            st.error("`extract_subtopic` not available from subtopics.py")
        else:
            # Apply cleanup
            for marks_key, questions in data.items():
                for q in questions:
                    original = q.get("subtopic", "")
                    q["subtopic"] = extract_subtopic(original)

            st.success("Subtopics cleaned.")
            st.subheader("Cleaned JSON")
            st.json(data)

            st.session_state.cleaned_questions = data

            st.download_button(
                "Download cleaned questions.json",
                data=json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"),
                file_name="questions.cleaned.json"
            )
    else:
        st.info("Upload your `questions.json`, tick the sample, or run step 1 to auto-generate questions.")


# =============================
# 3) Build Vector Store (FAISS)
# =============================
elif page.startswith("3"):
    st.header("3) Build Vector Store (FAISS)")
    st.write("Embeds questions using **Ollama Embeddings** and stores them in a **FAISS** index.")

    if FAISS is None or OllamaEmbeddings is None:
        st.error("FAISS/Ollama packages not available.")
        if "_faiss_import_error" in globals():
            st.code(_faiss_import_error)
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write("Provide cleaned questions JSON (from step 2).")
            uploaded_json = st.file_uploader("Upload cleaned questions.json", type=["json"], key="faiss_json")
        with col2:
            rebuild = st.checkbox("Overwrite if index exists")

        if st.button("Build Index", type="primary"):
            if uploaded_json is None and st.session_state.cleaned_questions is None:
                st.warning("Please upload cleaned questions.json or run step 2 first.")
            else:
                try:
                    if uploaded_json is not None:
                        data = json.loads(uploaded_json.read().decode("utf-8"))
                    else:
                        data = st.session_state.cleaned_questions

                    # Build Documents list
                    from langchain.schema import Document
                    docs = []
                    for mark_group, questions in data.items():
                        for q in questions:
                            meta = {
                                "topic": q.get("topic"),
                                "subtopic": q.get("subtopic"),
                                "marks": q.get("marks"),
                                "difficulty": q.get("difficulty"),
                                "cognitive_level": q.get("cognitive_level"),
                            }
                            docs.append(Document(page_content=q.get("question", ""), metadata=meta))

                    index_path = Path(default_index)
                    if index_path.exists() and not rebuild:
                        st.info(f"Index already exists at `{index_path}`. Tick overwrite to rebuild.")
                    else:
                        if index_path.exists():
                            # remove previous
                            import shutil
                            shutil.rmtree(index_path, ignore_errors=True)

                        st.write("Creating FAISS index…")
                        embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
                        db = FAISS.from_documents(docs, embeddings)
                        db.save_local(str(index_path))
                        st.success(f"Stored FAISS at `{index_path}`")
                except Exception as e:
                    st.error(f"Failed to build index: {e}")


# =============================
# 4) Search & Smart Filter
# =============================
elif page.startswith("4"):
    st.header("4) Search & Smart Filter")
    st.write("Query the FAISS index, then filter by marks, difficulty, and cognitive level.")

    query = st.text_input("Search query", value="Natural language processing techniques for text classification")

    col1, col2, col3 = st.columns(3)
    with col1:
        marks = st.number_input("Target marks", min_value=1, max_value=20, value=3, step=1)
    with col2:
        difficulty = st.selectbox("Target difficulty", ["easy", "medium", "hard"], index=1)
    with col3:
        cognitive = st.selectbox("Target cognitive", ["remembering", "understanding", "applying", "analyzing", "evaluating", "creating"], index=2)

    run_btn = st.button("Search", type="primary")

    if run_btn:
        if FAISS is None or OllamaEmbeddings is None:
            st.error("FAISS/Ollama not available in this environment.")
        else:
            try:
                embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
                db = FAISS.load_local(default_index, embeddings, allow_dangerous_deserialization=True)

                docs = db.similarity_search(query, k=15)

                def smart_filter(docs, marks, difficulty, cognitive):
                    exact = [d for d in docs if str(d.metadata.get("marks")) == str(marks)
                                           and d.metadata.get("difficulty") == difficulty
                                           and d.metadata.get("cognitive_level") == cognitive]
                    if exact:
                        return exact
                    # relax difficulty
                    relaxed1 = [d for d in docs if str(d.metadata.get("marks")) == str(marks)
                                               and d.metadata.get("cognitive_level") == cognitive]
                    if relaxed1:
                        return relaxed1
                    # relax cognitive
                    relaxed2 = [d for d in docs if str(d.metadata.get("marks")) == str(marks)
                                               and d.metadata.get("difficulty") == difficulty]
                    if relaxed2:
                        return relaxed2
                    # fallback any marks match
                    relaxed3 = [d for d in docs if str(d.metadata.get("marks")) == str(marks)]
                    return relaxed3 or docs

                filtered = smart_filter(docs, marks, difficulty, cognitive)

                st.session_state.search_results = [
                    {
                        "question": d.page_content,
                        "topic": d.metadata.get("topic"),
                        "subtopic": d.metadata.get("subtopic"),
                        "marks": d.metadata.get("marks"),
                        "difficulty": d.metadata.get("difficulty"),
                        "cognitive_level": d.metadata.get("cognitive_level"),
                    }
                    for d in filtered
                ]

                st.success(f"Found {len(filtered)} result(s).")
                for i, item in enumerate(st.session_state.search_results, 1):
                    with st.expander(f"Q{i}: {item['question'][:80]}…"):
                        st.write(f"**Topic**: {item['topic']}  \n"
                                f"**Subtopic**: {item['subtopic']}  \n"
                                f"**Marks**: {item['marks']}  \n"
                                f"**Difficulty**: {item['difficulty']}  \n"
                                f"**Cognitive**: {item['cognitive_level']}")
            except Exception as e:
                st.error(f"Search failed: {e}")


# =============================
# 5) Export Results
# =============================
elif page.startswith("5"):
    st.header("5) Export Results")
    st.write("Download refined text (step 1) and/or the search results (step 4).")

    if st.session_state.refined_text:
        st.download_button(
            "Download refined_output.txt",
            data=st.session_state.refined_text.encode("utf-8"),
            file_name="refined_output.txt"
        )
    else:
        st.info("No refined text in session. Run step 1 first.")

    if st.session_state.search_results:
        st.download_button(
            "Download selected_questions.json",
            data=json.dumps(st.session_state.search_results, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name="selected_questions.json"
        )
    else:
        st.info("No search results yet. Run step 4 first.")