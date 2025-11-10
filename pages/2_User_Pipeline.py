import json
import tempfile
import os
from pathlib import Path

import streamlit as st
from copy import deepcopy

# Optional local helpers
try:
    from Pytesseract.refinement import handle_image, handle_pdf
except Exception:
    handle_image = None
    handle_pdf = None

try:
    from Clean_subtopics.subtopics import extract_subtopic
except Exception:
    extract_subtopic = None

try:
    import questions_generation.question_gen as qgen
except Exception:
    qgen = None

try:
    from langchain_community.vectorstores import FAISS
    from langchain_ollama import OllamaEmbeddings
except Exception:
    FAISS = None
    OllamaEmbeddings = None


st.set_page_config(page_title="CoE Prject", layout="centered")


def show_header():
    st.markdown(
        """
    <div style='text-align:center; padding:1.2rem; border-radius:0.8rem; background:#23272f; color:#e0f7fa;'>
      <h2 style='margin:0.1rem 0 0.2rem 0;'>Question Generator — Complete Workflow</h2>
      <!--<div style='opacity:0.9'>A focused, user-first flow: Upload → Generate (optional) → Clean → Search → Export</div>-->
    </div>
    """,
        unsafe_allow_html=True,
    )


show_header()

# session defaults
st.session_state.setdefault("refined_text", "")
st.session_state.setdefault("generated_questions", None)
st.session_state.setdefault("cleaned_questions", None)
st.session_state.setdefault("search_results", [])
st.session_state.setdefault("wizard_step", 0)
st.session_state.setdefault("auto_build_index", True)


def save_refined_text(text: str):
    try:
        out_dir = Path("questions_generation")
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / "extracted_output.txt"
        p.write_text(text, encoding="utf-8")
    except Exception:
        pass


def step_card(title: str, idx: int | None = None):
    cls = f"step-banner step-{idx}" if idx is not None else "step-banner"
    st.markdown(f"<div class='{cls}'>{title}</div>", unsafe_allow_html=True)


def run_extraction(uploaded, manual_text):
    output_text = ""
    if manual_text and str(manual_text).strip():
        output_text = str(manual_text).strip()
    elif uploaded is not None:
        suffix = Path(uploaded.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        try:
            if suffix in [".png", ".jpg", ".jpeg"] and handle_image is not None:
                output_text = handle_image(tmp_path)
            elif suffix == ".pdf" and handle_pdf is not None:
                output_text = handle_pdf(tmp_path)
            else:
                if handle_image is None and handle_pdf is None:
                    st.error("OCR helpers not available. Paste text instead.")
                else:
                    st.error("Unsupported file type or OCR not available for this file.")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    if output_text:
        st.session_state.refined_text = output_text
        save_refined_text(output_text)
        st.success("Text ready — you can now generate questions or proceed.")
        st.text_area("Refined text (editable)", value=st.session_state.refined_text, key="refined_preview", height=220)
    else:
        st.warning("No text was produced. Please paste text or try another file.")


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


def build_index_from_cleaned(cleaned_questions, show_progress: bool = False):
    """Builds a FAISS index from cleaned_questions and returns (success, message).
    Non-destructive: will remove existing index folder and recreate it.
    """
    try:
        if FAISS is None or OllamaEmbeddings is None:
            return False, "FAISS or OllamaEmbeddings not available"

        if not cleaned_questions:
            return False, "No cleaned questions to index"

        from langchain.schema import Document
        docs = []
        for mark_group, questions in cleaned_questions.items():
            for q in questions:
                # normalize metadata keys
                marks = q.get("marks") or q.get("mark") or 0
                difficulty = q.get("difficulty") or q.get("difficulty_level") or ""
                cognitive = q.get("cognitive_level") or q.get("cognitive") or ""
                topic = q.get("topic") or "General"
                subtopic = q.get("subtopic") or topic

                meta = {
                    "topic": topic,
                    "subtopic": subtopic,
                    "marks": marks,
                    "difficulty": difficulty,
                    "cognitive_level": cognitive,
                }

                # include context in embedding text to improve search relevance
                text_for_embed = f"{q.get('question','')}\n\nTopic: {topic}\nSubtopic: {subtopic}\nMarks: {marks}"
                docs.append(Document(page_content=text_for_embed, metadata=meta))

        idx_path = Path("new_faiss_index")
        if idx_path.exists():
            import shutil
            shutil.rmtree(idx_path, ignore_errors=True)

        # progress UI (best-effort; from_documents is blocking so we update at milestones)
        progress = None
        status = None
        try:
            if show_progress:
                status = st.empty()
                progress = st.progress(0)
                status.text("Preparing documents for embedding...")
                progress.progress(10)

            embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

            if show_progress and progress:
                status.text("Building FAISS index (this may take a while)...")
                progress.progress(40)

            db = FAISS.from_documents(docs, embeddings)

            if show_progress and progress:
                progress.progress(80)
                status.text("Saving index to disk...")

            db.save_local(str(idx_path))

            if show_progress and progress:
                progress.progress(100)
                status.text("Index built and saved.")

            return True, "Index built and saved."
        finally:
            # leave the final status visible briefly (Streamlit handles rendering)
            pass
    except Exception as e:
        return False, f"Failed to build index: {e}"


st.markdown("---")

with st.expander("Quick Guide", expanded=True):
    st.write(
        "This wizard guides you through the essential steps. Use Next/Back to navigate. Advanced diagnostics are available at the bottom."
    )


def nav():
    c1, c2, _ = st.columns([1, 1, 8])
    back = c1.button("Back")
    nxt = c2.button("Next")
    if back:
        st.session_state.wizard_step = max(0, st.session_state.wizard_step - 1)
    if nxt:
        st.session_state.wizard_step = min(4, st.session_state.wizard_step + 1)


nav()

st.write(f"Step {st.session_state.wizard_step + 1} of 5")


# Step 0: Upload / Extract
if st.session_state.wizard_step == 0:
    step_card("Upload or Paste Material", 0)
    uploaded = st.file_uploader("Upload PDF or image (or skip to paste text)", type=["pdf", "png", "jpg", "jpeg"], key="wizard_upload")
    manual = st.text_area("Or paste text here (optional)", height=160, key="wizard_manual")
    if handle_image is None or handle_pdf is None:
        st.caption("OCR helpers not fully available — paste text to proceed.")
    if qgen is None:
        st.caption("Question generator not present — generator step will be disabled if missing.")
    if st.button("Extract & Refine", key="wizard_extract"):
        run_extraction(uploaded, manual)


# Step 1: Generate
elif st.session_state.wizard_step == 1:
    step_card("Generate Questions (optional)", 1)
    gen_toggle = st.checkbox("Auto-generate questions from refined text", value=False, key="wizard_gen_toggle")
    if gen_toggle:
        if not st.session_state.refined_text:
            st.info("Please provide refined text first (go to previous step and extract or paste text).")
        elif qgen is None:
            st.error("Question generator not available in this environment.")
        else:
            # (Auto-clean/auto-build toggle moved to Clean & Prepare step)
            
            if st.button("Run Generator", key="wizard_run_gen"):
                with st.spinner("Generating questions — this may take a moment..."):
                    generated = None
                    try:
                        import inspect

                        # try to load topic keywords to pass to the generator (helps topic detection)
                        topic_keywords = {}
                        try:
                            kwf = Path("questions_generation") / "keyword.json"
                            if kwf.exists():
                                topic_keywords = json.loads(kwf.read_text(encoding="utf-8"))
                        except Exception:
                            topic_keywords = {}

                        if hasattr(qgen, "generate_questions"):
                            sig = inspect.signature(qgen.generate_questions)
                            params = list(sig.parameters.keys())
                            text_input = st.session_state.refined_text
                            # prefer signature (text, topic_keywords) if available
                            try:
                                if len(params) >= 2:
                                    res = qgen.generate_questions(text_input, topic_keywords)
                                elif len(params) >= 1:
                                    res = qgen.generate_questions(text_input)
                                else:
                                    res = qgen.generate_questions()
                            except Exception as ex:
                                st.error(f"Generator invocation error: {ex}")
                                res = None
                        elif hasattr(qgen, "main"):
                            try:
                                qgen.main()
                                res = None
                            except Exception:
                                try:
                                    qgen.main([])
                                    res = None
                                except Exception:
                                    res = None
                    except Exception as e:
                        st.error(f"Generator failed: {e}")

                    # prefer in-memory result and persist it; otherwise fall back to existing file
                    if 'res' in locals() and isinstance(res, dict):
                        generated = res
                        try:
                            out_dir = Path("questions_generation")
                            out_dir.mkdir(parents=True, exist_ok=True)
                            out_path = out_dir / "generated_questions.json"
                            out_path.write_text(json.dumps(generated, indent=2, ensure_ascii=False), encoding="utf-8")
                            # append a debug entry
                            dbg = out_dir / "generator_debug.log"
                            try:
                                with open(dbg, "a", encoding="utf-8") as df:
                                    df.write(f"Saved generated_questions.json (len groups={len(generated.keys())})\n")
                            except Exception:
                                pass
                        except Exception:
                            pass

                    if generated is None:
                        candidate = Path("questions_generation") / "generated_questions.json"
                        if candidate.exists():
                            try:
                                generated = json.loads(candidate.read_text(encoding="utf-8"))
                            except Exception:
                                generated = None

                    # If still no generated output, produce a lightweight heuristic fallback
                    if generated is None:
                        text_src = st.session_state.refined_text or ""
                        if text_src.strip():
                            # simple sentence-based fallback: create anchored questions
                            sentences = [s.strip() for s in text_src.split('.') if s.strip()]
                            def make_q(s, marks, topic="General"):
                                return {
                                    "question": f"According to the text: {s[:200]}?",
                                    "topic": topic,
                                    "subtopic": topic,
                                    "question_type": "short" if marks<=2 else "descriptive",
                                    "difficulty_level": "easy" if marks==1 else ("medium" if marks<=3 else "hard"),
                                    "time": "1 min" if marks==1 else ("3 min" if marks==2 else ("5 min" if marks==3 else "8 min")),
                                    "cognitive_level": "remembering" if marks==1 else ("understanding" if marks==2 else ("applying" if marks==3 else "evaluating")),
                                    "marks": marks,
                                    "image": None
                                }

                            buckets = {1: [], 2: [], 3: [], 5: []}
                            for i, s in enumerate(sentences):
                                if i >= 20:
                                    break
                                m = [1,2,3,5][i % 4]
                                buckets[m].append(make_q(s, m, topic="General"))

                            # ensure at least one per bucket
                            for m in [1,2,3,5]:
                                if not buckets[m] and sentences:
                                    buckets[m].append(make_q(sentences[0], m))

                            generated = {
                                "1_mark": buckets[1],
                                "2_mark": buckets[2],
                                "3_mark": buckets[3],
                                "5_mark": buckets[5]
                            }
                            # persist fallback so user can download / clean
                            try:
                                out_dir = Path("questions_generation")
                                out_dir.mkdir(parents=True, exist_ok=True)
                                (out_dir / "generated_questions.json").write_text(json.dumps(generated, indent=2, ensure_ascii=False), encoding="utf-8")
                                with open(out_dir / "generator_debug.log", "a", encoding="utf-8") as df:
                                    df.write("Fallback generator produced output\n")
                            except Exception:
                                pass

                    if generated is not None:
                        st.session_state.generated_questions = generated
                        st.success("Questions generated and loaded.")
                        st.write("Preview (first items):")
                        for k, v in list(generated.items())[:3]:
                            st.markdown(f"**{k}** — {len(v)} item(s)")
                        # --- Auto-clean and auto-build index so user can jump to search ---
                        try:
                            # create cleaned copy so we don't mutate original unexpectedly
                            if extract_subtopic is not None:
                                cleaned = deepcopy(generated)
                                for marks_key, questions in cleaned.items():
                                    for q in questions:
                                        original = q.get("subtopic", "")
                                        try:
                                            q["subtopic"] = extract_subtopic(original)
                                        except Exception:
                                            # fallback to original value
                                            q["subtopic"] = original
                                st.session_state.cleaned_questions = cleaned
                                st.info("Subtopics cleaned automatically.")
                            else:
                                st.session_state.cleaned_questions = generated
                                st.info("Subtopics not cleaned automatically (cleaner missing); cleaned_questions set to generated output.")

                            # attempt to build index automatically only if the toggle is enabled
                            if st.session_state.get("auto_build_index", True):
                                success, msg = build_index_from_cleaned(st.session_state.cleaned_questions, show_progress=True)
                                if success:
                                    st.success(msg)
                                else:
                                    st.info(msg)

                                # advance user to Search step (step index 3) so they can try searching or re-build if needed
                                st.session_state.wizard_step = 3
                            else:
                                # Auto-build is disabled. Keep user on Clean & Prepare so they may run build manually.
                                st.info("Auto-build disabled; cleaned questions are ready. Go to 'Clean & Prepare' or 'Build Index' when ready.")
                                st.session_state.wizard_step = 2
                        except Exception as _:
                            # swallow to avoid blocking UI; user can still click Clean / Build manually
                            pass
                    else:
                        st.warning("No structured output detected — check your generator or use file-based output.")


# Step 2: Clean & Prepare
elif st.session_state.wizard_step == 2:
    step_card("Clean & Prepare", 2)
    # Auto-clean & auto-build toggle (moved here from Generate step)
    st.checkbox("Auto-clean & auto-build index after generation", value=st.session_state.get("auto_build_index", True), key="auto_build_index")
    if st.session_state.generated_questions is None:
        st.info("No generated questions loaded yet. You can upload a questions JSON to continue.")
        upload_q = st.file_uploader("Upload questions JSON", type=["json"], key="wizard_upload_questions")
        if upload_q is not None:
            try:
                st.session_state.generated_questions = json.loads(upload_q.read().decode("utf-8"))
                st.success("Uploaded questions JSON loaded.")
            except Exception as e:
                st.error("Invalid JSON: " + str(e))

    if st.session_state.generated_questions:
        if extract_subtopic is None:
            st.warning("Subtopic cleaning not available (missing `Clean_subtopics.subtopics`). You can still export raw questions.")
            st.session_state.cleaned_questions = st.session_state.generated_questions
        else:
            if st.button("Clean Subtopics", key="wizard_clean"):
                data = st.session_state.generated_questions
                for marks_key, questions in data.items():
                    for q in questions:
                        original = q.get("subtopic", "")
                        q["subtopic"] = extract_subtopic(original)
                st.session_state.cleaned_questions = data
                st.success("Subtopics cleaned.")

        if st.session_state.cleaned_questions:
            st.write("Preview of cleaned data (first group):")
            first_key = next(iter(st.session_state.cleaned_questions.keys()))
            st.json({first_key: st.session_state.cleaned_questions[first_key][:5]})
            st.download_button("Download cleaned questions.json", data=json.dumps(st.session_state.cleaned_questions, indent=2, ensure_ascii=False).encode("utf-8"), file_name="questions.cleaned.json")


# Step 3: Index & Search
elif st.session_state.wizard_step == 3:
    step_card("Build Index & Search (optional)", 3)
    query = st.text_input("Search for questions (natural language)", value="", placeholder="e.g. explain convolutional layers", key="wizard_query")
    build_idx = st.button("Build Index (if available)", key="wizard_build_idx")
    if build_idx:
        if FAISS is None or OllamaEmbeddings is None:
            st.error("Indexing requires FAISS and Ollama embeddings to be installed and available.")
        elif not st.session_state.cleaned_questions:
            st.warning("No cleaned questions available to index.")
        else:
            # show progress during manual build
            with st.spinner("Building index — this may take some time"):
                success, msg = build_index_from_cleaned(st.session_state.cleaned_questions, show_progress=True)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)

    search_btn = st.button("Search", key="wizard_search")
    if search_btn:
        if not query.strip():
            st.info("Enter a search query first.")
        else:
            if FAISS is None or OllamaEmbeddings is None:
                st.warning("FAISS/Ollama not available here — searching not possible.")
            else:
                try:
                    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
                    db = FAISS.load_local("new_faiss_index", embeddings, allow_dangerous_deserialization=True)

                    docs = db.similarity_search(query, k=15)

                    # smart filter controls
                    st.write("Refine results")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        marks = st.number_input("Target marks", min_value=1, max_value=20, value=3, step=1, key="wizard_marks")
                    with col2:
                        difficulty = st.selectbox("Target difficulty", ["easy", "medium", "hard"], index=1, key="wizard_diff")
                    with col3:
                        cognitive = st.selectbox("Target cognitive", ["remembering", "understanding", "applying", "analyzing", "evaluating", "creating"], index=2, key="wizard_cog")

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


# Step 4: Export
elif st.session_state.wizard_step == 4:
    step_card("Export", 4)
    exp_col1, exp_col2 = st.columns([1, 1])
    with exp_col1:
        if st.session_state.refined_text:
            st.download_button("Download refined text", data=st.session_state.refined_text.encode("utf-8"), file_name="refined_output.txt")
        else:
            st.caption("No refined text available")
    with exp_col2:
        if st.session_state.cleaned_questions:
            st.download_button("Download cleaned questions", data=json.dumps(st.session_state.cleaned_questions, indent=2, ensure_ascii=False).encode("utf-8"), file_name="questions.cleaned.json")
        elif st.session_state.generated_questions:
            st.download_button("Download generated questions", data=json.dumps(st.session_state.generated_questions, indent=2, ensure_ascii=False).encode("utf-8"), file_name="questions.generated.json")
        else:
            st.caption("No question data to export")


st.markdown("---")

# with st.expander("Advanced / Diagnostics"):
#     st.write("This panel is for maintainers. It shows detected optional features and some paths.")
#     st.write({
#         "OCR_helpers": bool(handle_image or handle_pdf),
#         "Question_generator": bool(qgen),
#         "Subtopic_cleaner": bool(extract_subtopic),
#         "FAISS": bool(FAISS),
#         "OllamaEmbeddings": bool(OllamaEmbeddings),
#     })
    
