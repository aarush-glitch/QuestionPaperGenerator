import json
import tempfile
import os
from pathlib import Path

import streamlit as st
from copy import deepcopy

# Courses persistence helpers
def _courses_path():
    p = Path("meta")
    p.mkdir(parents=True, exist_ok=True)
    return p / "courses.json"


def load_courses():
    p = _courses_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_courses(courses):
    try:
        p = _courses_path()
        p.write_text(json.dumps(courses, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

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

# PDF generation via PyMuPDF (optional)
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


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

# show selected course in a small header/banner
def show_selected_course_banner():
    course = st.session_state.get("selected_course")
    if course:
        st.markdown(f"<div style='text-align:center; padding:0.4rem; border-radius:0.4rem; background:#e8f5e9; color:#1b5e20; margin-bottom:0.8rem;'>Selected course: <strong>{course}</strong></div>", unsafe_allow_html=True)


show_selected_course_banner()

# session defaults
st.session_state.setdefault("refined_text", "")
st.session_state.setdefault("generated_questions", None)
st.session_state.setdefault("cleaned_questions", None)
st.session_state.setdefault("search_results", [])
st.session_state.setdefault("wizard_step", 0)
st.session_state.setdefault("auto_build_index", True)
st.session_state.setdefault("selected_questions", [])
st.session_state.setdefault("max_marks_limit", 20)

# load available courses from disk (or sensible defaults)
if "available_courses" not in st.session_state:
    loaded = load_courses()
    if not loaded:
        loaded = ["Data Structures", "OOPs", "Software Engineering", "Computer Networks", "Ethical Hacking"]
    st.session_state["available_courses"] = loaded

st.session_state.setdefault("selected_course", "")


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
    return relaxed3 or []


def build_index_from_cleaned(cleaned_questions, show_progress: bool = False):
    """Builds a FAISS index from cleaned_questions and returns (success, message).
    Non-destructive: will remove existing index folder and recreate it.
    """
    try:
        # helper to coerce various time formats (strings like "2-3 min", "1 min", numeric) to a single integer minutes value
        def parse_time_minutes(val):
            try:
                if val is None:
                    return None
                if isinstance(val, int):
                    return val
                s = str(val).strip().lower()
                import re
                nums = re.findall(r"\d+\.?\d*", s)
                if not nums:
                    return None
                if '-' in s and len(nums) >= 2:
                    a = float(nums[0]); b = float(nums[1])
                    return int(round((a + b) / 2))
                # otherwise use the first number
                return int(round(float(nums[0])))
            except Exception:
                return None
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
                # preserve expected time-to-solve if provided by generator; coerce to integer minutes
                raw_time = q.get("time") or q.get("time_estimate") or ""
                time_est = parse_time_minutes(raw_time)
                topic = q.get("topic") or "General"
                subtopic = q.get("subtopic") or topic

                meta = {
                    "topic": topic,
                    "subtopic": subtopic,
                    "marks": marks,
                    "difficulty": difficulty,
                    "cognitive_level": cognitive,
                    "time": time_est,
                }

                # include context in embedding text to improve search relevance
                text_for_embed = f"{q.get('question','')}\n\nTopic: {topic}\nSubtopic: {subtopic}\nMarks: {marks}\nTime: {time_est if time_est is not None else ''}"
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


def create_pdf_from_selection(selection, title: str = "Question Paper"):
    """Create a PDF bytes object from the selected questions using PyMuPDF (if available).
    Returns (success: bool, bytes_or_msg).
    """
    try:
        content_lines = [title, "", "Generated for JIIT"]
        total_marks = 0
        for i, q in enumerate(selection, 1):
            marks = int(q.get("marks") or 0)
            total_marks += marks
            time_str = q.get("time")
            time_part = f" [Time: {time_str} min]" if time_str not in (None, "") else ""
            content_lines.append(f"Q{i}. ({marks} marks){time_part} {q.get('question', '')}")
            content_lines.append("")

        content_lines.append("")
        content_lines.append(f"Total Marks: {total_marks}")

        text = "\n".join(content_lines)

        if fitz is None:
            # fallback: return text bytes so user can download as .txt
            return False, text.encode("utf-8")

        # create PDF
        doc = fitz.open()
        # page size A4
        page = doc.new_page(width=595, height=842)
        # simple layout: insert text at margin
        text_rect = fitz.Rect(40, 40, 555, 800)
        # choose a default font/size
        page.insert_textbox(text_rect, text, fontsize=11, fontname="helv")

        # write to bytes
        pdf_bytes = doc.write()
        doc.close()
        return True, pdf_bytes
    except Exception as e:
        return False, str(e)


st.markdown("---")

with st.expander("Quick Guide", expanded=True):
    st.write(
        "This wizard guides you through the essential steps. Use Next/Back to navigate. Advanced diagnostics are available at the bottom."
    )


def nav():
    c1, c2, _ = st.columns([1, 1, 8])
    back = c1.button("Back")
    # disable Next on the first step until a course is selected
    disable_next = False
    try:
        if st.session_state.wizard_step == 0 and not st.session_state.get("selected_course"):
            disable_next = True
    except Exception:
        disable_next = False
    nxt = c2.button("Next", disabled=disable_next)
    if back:
        st.session_state.wizard_step = max(0, st.session_state.wizard_step - 1)
    if nxt:
        st.session_state.wizard_step = min(5, st.session_state.wizard_step + 1)


nav()

st.write(f"Step {st.session_state.wizard_step + 1} of 6")


# Step 0: Choose Course
if st.session_state.wizard_step == 0:
    step_card("Select Course", 0)
    st.markdown("#### Select course for this question paper")
    cols = st.columns([3, 2])
    with cols[0]:
        course_options = list(st.session_state.get("available_courses", [])) + ["-- Add new course --"]
        # compute default index so the selectbox doesn't reset to the first option on reruns
        current = st.session_state.get("selected_course")
        try:
            default_index = course_options.index(current) if current in course_options else 0
        except Exception:
            default_index = 0
        chosen = st.selectbox("Choose existing course or add new", options=course_options, index=default_index)
    with cols[1]:
        st.write("\n")
        st.write("\n")
        st.caption(f"Current: {st.session_state.get('selected_course') or 'None'}")

    # handle new course flow
    if chosen == "-- Add new course --":
        new_name = st.text_input("New course name (e.g. Data Structures - 2025)", key="new_course_name")
        if new_name:
            if st.button("Add and select course", key="add_course_btn"):
                if new_name not in st.session_state.available_courses:
                    st.session_state.available_courses.append(new_name)
                    # persist courses to disk
                    try:
                        save_courses(st.session_state.available_courses)
                    except Exception:
                        pass
                st.session_state.selected_course = new_name
                st.success(f"Selected course: {new_name}")
    else:
        # selecting an existing course immediately sets it
        if st.session_state.get("selected_course") != chosen:
            st.session_state.selected_course = chosen

    if not st.session_state.get("selected_course"):
        st.warning("Please choose or add a course before proceeding.")

    # allow user to proceed to upload or skip to existing DB work
    colp1, colp2 = st.columns(2)
    with colp1:
        if st.button("Proceed to Upload / Paste Text", key="goto_upload"):
            if not st.session_state.get("selected_course"):
                st.error("Select a course first.")
            else:
                st.session_state.wizard_step = 1
    with colp2:
        if st.button("Skip upload — Use existing questions", key="goto_existing"):
            # jump to Search (step index 4 after shift)
            st.session_state.wizard_step = 4


# Step 1: Upload / Extract
elif st.session_state.wizard_step == 1:
    step_card("Upload or Paste Material", 1)
    uploaded = st.file_uploader("Upload PDF or image (or skip to paste text)", type=["pdf", "png", "jpg", "jpeg"], key="wizard_upload")
    manual = st.text_area("Or paste text here (optional)", height=160, key="wizard_manual")
    if handle_image is None or handle_pdf is None:
        st.caption("OCR helpers not fully available — paste text to proceed.")
    if qgen is None:
        st.caption("Question generator not present — generator step will be disabled if missing.")
    if st.button("Extract & Refine", key="wizard_extract"):
        if not st.session_state.get("selected_course"):
            st.error("Select a course first.")
        else:
            run_extraction(uploaded, manual)

# Step 2: Generate
elif st.session_state.wizard_step == 2:
    step_card("Generate Questions (optional)", 2)
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
                                # Detect common connection-refused errors (e.g., Ollama daemon not running)
                                msg = str(ex)
                                try:
                                    out_dir = Path("questions_generation")
                                    out_dir.mkdir(parents=True, exist_ok=True)
                                    dbg = out_dir / "generator_debug.log"
                                    with open(dbg, "a", encoding="utf-8") as df:
                                        df.write(f"Generator exception: {repr(ex)}\n")
                                except Exception:
                                    pass

                                if isinstance(ex, ConnectionRefusedError) or "10061" in msg or "Connection refused" in msg:
                                    st.error(
                                        "Generator connection error: connection refused.\n"
                                        "This usually means the local LLM service (e.g. Ollama) is not running or not reachable at localhost:11434.\n"
                                        "Check that the Ollama app or daemon is running, or run `ollama list` / `curl http://127.0.0.1:11434/v1/models` to diagnose.`"
                                    )
                                else:
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
                                # map marks to a single integer minute estimate (1,3,5,8)
                                time_map = {1: 1, 2: 3, 3: 5, 5: 8}
                                diff_map = {1: "easy", 2: "medium", 3: "hard", 5: "hard"}
                                cog_map = {1: "remembering", 2: "understanding", 3: "applying", 5: "evaluating"}
                                # attempt to classify topic using generator's detect_topic (if available)
                                try:
                                    if qgen is not None and hasattr(qgen, "detect_topic"):
                                        # topic_keywords from the surrounding scope (if present) will be used
                                        try:
                                            tk = topic_keywords if 'topic_keywords' in locals() else {}
                                            topic = qgen.detect_topic(s, tk)
                                        except Exception:
                                            topic = topic
                                except Exception:
                                    topic = topic

                                # subtopic can be cleaned if extractor is available
                                subtopic = topic
                                try:
                                    if extract_subtopic is not None:
                                        subtopic = extract_subtopic(subtopic)
                                except Exception:
                                    subtopic = subtopic

                                return {
                                    "question": f"According to the text: {s[:200]}?",
                                    "topic": topic,
                                    "subtopic": subtopic,
                                    "question_type": "short" if marks <= 2 else "descriptive",
                                    "difficulty_level": diff_map.get(marks, "medium"),
                                    "time": int(time_map.get(marks, 3)),
                                    "cognitive_level": cog_map.get(marks, "understanding"),
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

                                # advance user to Search step (step index 4 after shift)
                                st.session_state.wizard_step = 4
                            else:
                                # Auto-build is disabled. Keep user on Clean & Prepare so they may run build manually.
                                st.info("Auto-build disabled; cleaned questions are ready. Go to 'Clean & Prepare' or 'Build Index' when ready.")
                                st.session_state.wizard_step = 3
                        except Exception as _:
                            # swallow to avoid blocking UI; user can still click Clean / Build manually
                            pass
                    else:
                        st.warning("No structured output detected — check your generator or use file-based output.")


# Step 3: Clean & Prepare
elif st.session_state.wizard_step == 3:
    step_card("Clean & Prepare", 3)
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


# Step 4: Index & Search
elif st.session_state.wizard_step == 4:
    step_card("Build Index & Search (optional)", 4)
    query = st.text_input("Search for questions (natural language)", value="", placeholder="e.g. explain convolutional layers", key="wizard_query")

    # layout: left column for search & results, right column for selection basket
    results_col, sel_col = st.columns([2, 1])

    with results_col:
        # show filter controls before running search so users can set them first
        st.write("Refine results")
        col1, col2, col3 = st.columns(3)
        with col1:
            marks = st.number_input("Target marks", min_value=1, max_value=20, value=3, step=1, key="wizard_marks")
        with col2:
            difficulty = st.selectbox("Target difficulty", ["easy", "medium", "hard"], index=1, key="wizard_diff")
        with col3:
            cognitive = st.selectbox("Target cognitive", ["remembering", "understanding", "applying", "analyzing", "evaluating", "creating"], index=2, key="wizard_cog")

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

        # Search button (uses the filter values above). Results are stored in session so
        # changing filters does not automatically clear previously displayed results.
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

                        filtered = smart_filter(docs, marks, difficulty, cognitive)

                        st.session_state.search_results = [
                            {
                                "question": d.page_content,
                                "topic": d.metadata.get("topic"),
                                "subtopic": d.metadata.get("subtopic"),
                                # ensure time is an integer minute value in the UI
                                "time": d.metadata.get("time"),
                                "marks": d.metadata.get("marks"),
                                "difficulty": d.metadata.get("difficulty"),
                                "cognitive_level": d.metadata.get("cognitive_level"),
                            }
                            for d in filtered
                        ]

                        st.success(f"Found {len(filtered)} result(s).")
                    except Exception as e:
                        st.error(f"Search failed: {e}")

        # Always show latest stored search results (if any). This ensures changing filters
        # doesn't immediately blank the UI; user must click Search to update results.
        if st.session_state.get("search_results"):
            for i, item in enumerate(st.session_state.search_results, 1):
                with st.expander(f"Q{i}: {item['question'][:80]}…"):
                    # display time with unit (minutes)
                    display_time = item.get('time')
                    display_time_str = f"{display_time} min" if display_time not in (None, "") else "N/A"
                    st.write(f"**Topic**: {item['topic']}  \n"
                             f"**Subtopic**: {item['subtopic']}  \n"
                             f"**Marks**: {item['marks']}  \n"
                             f"**Time**: {display_time_str}  \n"
                             f"**Difficulty**: {item['difficulty']}  \n"
                             f"**Cognitive**: {item['cognitive_level']}")
                    # add selection control
                    try:
                        q_marks = int(item.get("marks") or 0)
                    except Exception:
                        q_marks = 0

                    add_key = f"add_{i}"
                    remove_key = f"rem_{i}"
                    if any((s.get("question") == item.get("question") and s.get("marks") == item.get("marks")) for s in st.session_state.selected_questions):
                        if st.button("Remove from selection", key=remove_key):
                            # remove first matching
                            for idx_s, s in enumerate(st.session_state.selected_questions):
                                if s.get("question") == item.get("question") and s.get("marks") == item.get("marks"):
                                    st.session_state.selected_questions.pop(idx_s)
                                    st.success("Removed from selection")
                                    break
                    else:
                        if st.button("Add to selection", key=add_key):
                            # check constraint
                            try:
                                current_total = sum(int(s.get("marks") or 0) for s in st.session_state.selected_questions)
                            except Exception:
                                current_total = 0
                            limit = int(st.session_state.get("max_marks_limit", 20) or 0)
                            if current_total + q_marks <= limit:
                                st.session_state.selected_questions.append(item)
                                st.success("Added to selection")
                            else:
                                remaining = limit - current_total
                                if remaining <= 0:
                                    st.warning(f"You have reached the maximum total marks ({limit}). Remove some selected questions to add more.")
                                else:
                                    st.warning(f"Cannot add: only {remaining} mark(s) remaining out of {limit}.")

    with sel_col:
        st.markdown("### Selected Questions")
        # control to set max marks limit
        st.number_input("Max total marks (limit)", min_value=1, max_value=200, value=st.session_state.get("max_marks_limit", 20), step=1, key="max_marks_limit")

        sel = st.session_state.get("selected_questions") or []
        try:
            total_selected = sum(int(s.get("marks") or 0) for s in sel)
        except Exception:
            total_selected = 0
        st.write(f"Total selected marks: {total_selected} / {st.session_state.get('max_marks_limit', 20)}")

        # list selected questions with remove option
        for j, s in enumerate(sel, 1):
            st.write(f"{j}. ({s.get('marks')} m) {s.get('question')[:120]}...")
            if st.button(f"Remove #{j}", key=f"sel_rem_{j}"):
                st.session_state.selected_questions.pop(j-1)
                st.experimental_rerun()

        if sel:
            if st.button("Finalize & Generate PDF"):
                # generate PDF and offer download
                success, payload = create_pdf_from_selection(sel, title="Question Paper")
                if not success and isinstance(payload, bytes):
                    # fallback text bytes
                    st.download_button("Download question paper (txt)", data=payload, file_name="question_paper.txt")
                elif success:
                    st.download_button("Download question paper (pdf)", data=payload, file_name="question_paper.pdf")
                else:
                    st.error(f"Failed to generate paper: {payload}")


# Step 5: Export
elif st.session_state.wizard_step == 5:
    step_card("Export", 5)
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
    
