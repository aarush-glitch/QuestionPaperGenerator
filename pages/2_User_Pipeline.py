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


st.set_page_config(page_title="Question Paper Generator - Pipeline", layout="wide", initial_sidebar_state="expanded")

# Match main container width/padding with Home page
try:
    st.markdown(
        """
        <style>
        /* Broad selectors to cover multiple Streamlit versions and themes */
        section[data-testid="stAppViewContainer"] .main .block-container,
        section[data-testid="stAppViewContainer"] .block-container,
        .reportview-container .main .block-container,
        div.block-container,
        .stApp .block-container,
        .css-1d391kg,
        .css-18e3th9 {
            max-width: 1000px !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
except Exception:
    pass

# --- STYLES (matching Home.py) ---
st.markdown("""
<style>
.gradient-header {
    background: linear-gradient(90deg, #43cea2 0%, #185a9d 100%);
    padding: 2.5rem 0 1.5rem 0;
    border-radius: 1.5rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.15);
}
.soft-card {
    background: linear-gradient(120deg, #23272f 60%, #43cea2 100%);
    border-radius: 1rem;
    padding: 1.7rem;
    margin-bottom: 1.5rem;
    color: #f5f6fa;
    box-shadow: 0 2px 8px #0002;
}
.big-title {
    font-size: 2.8rem;
    font-weight: bold;
    color: #f5f6fa;
    text-align: center;
    margin-bottom: 0.2em;
    font-family: 'Segoe UI', 'Arial', sans-serif;
    letter-spacing: 2px;
    text-shadow: 0 2px 8px #185a9d44;
}
.subtitle {
    font-size: 1.3rem;
    color: #e0f7fa;
    text-align: center;
    margin-bottom: 1.5em;
    font-family: 'Segoe UI', 'Arial', sans-serif;
    letter-spacing: 1px;
}
.step-badge {
    display: inline-block;
    background: #43cea2;
    color: #23272f;
    border-radius: 0.5em;
    padding: 0.3em 0.8em;
    margin: 0.2em 0.4em;
    font-weight: 600;
    font-size: 1.05rem;
    box-shadow: 0 1px 4px #185a9d22;
}
.course-banner {
    text-align: center;
    padding: 0.6rem;
    border-radius: 0.8rem;
    background: linear-gradient(120deg, #e8f5e9 60%, #a5d6a7 100%);
    color: #1b5e20;
    margin-bottom: 1.2rem;
    font-size: 1.1rem;
    font-weight: 800;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)


def show_header():
    st.markdown(
        """
        <div style='padding: 2.5rem 0 1.5rem 0; border-radius: 1.5rem; text-align: center; margin-bottom: 2rem; box-shadow: 0 4px 24px rgba(0,0,0,0.15); background: #23272f;'>
            <div class='big-title'>Question Paper Generator</div>
            <div class='subtitle'>Upload → Generate → Search → Export</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


show_header()

# show selected course in a banner matching Home.py style
def show_selected_course_banner():
    course = st.session_state.get("selected_course")
    if course:
        st.markdown(
            f"<div class='course-banner'>📚 Selected Course: <strong>{course}</strong></div>",
            unsafe_allow_html=True
        )


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
st.session_state.setdefault("dashboard_mode", False)
st.session_state.setdefault("auto_use_prebuilt_index", True)


def _safe_int(v):
    """Convert value to int safely. Returns 0 on failure.
    Accepts numeric types or strings like '5', '5.0', '5 mins'."""
    try:
        if v is None:
            return 0
        # If it's already an int, return it
        if isinstance(v, int):
            return v
        # Try float first to accept '5.0'
        return int(float(str(v).strip()))
    except Exception:
        # fallback: extract first number from string
        try:
            import re

            s = str(v)
            nums = re.findall(r"\d+\.?\d*", s)
            if not nums:
                return 0
            return int(float(nums[0]))
        except Exception:
            return 0


def _add_to_selection(item, q_marks):
    """Callback to add an item to the selection (used with st.button on_click)."""
    try:
        q_marks_int = _safe_int(q_marks)
        new_item = dict(item)
        new_item["marks"] = q_marks_int
        # normalize difficulty keys so both forms are present for UI compatibility
        try:
            if not new_item.get("difficulty") and new_item.get("difficulty_level"):
                new_item["difficulty"] = new_item.get("difficulty_level")
            if not new_item.get("difficulty_level") and new_item.get("difficulty"):
                new_item["difficulty_level"] = new_item.get("difficulty")
        except Exception:
            pass
        st.session_state.setdefault("selected_questions", [])
        st.session_state.selected_questions.append(new_item)
    except Exception:
        # swallow to avoid crashing Streamlit callback
        pass
    finally:
        # toggle a lightweight trigger so Streamlit reliably reruns across versions
        st.session_state["_ui_trigger"] = not st.session_state.get("_ui_trigger", False)


def _remove_from_selection_by_index(idx):
    """Callback to remove a selected item by index."""
    try:
        st.session_state.setdefault("selected_questions", [])
        if 0 <= idx < len(st.session_state.selected_questions):
            st.session_state.selected_questions.pop(idx)
    except Exception:
        pass
    finally:
        st.session_state["_ui_trigger"] = not st.session_state.get("_ui_trigger", False)


def _remove_first_matching(question, marks):
    """Remove first selected question matching question text and marks."""
    try:
        q_marks = _safe_int(marks)
        for idx_s, s in enumerate(st.session_state.get("selected_questions", [])):
            if s.get("question") == question and _safe_int(s.get("marks")) == q_marks:
                st.session_state.selected_questions.pop(idx_s)
                break
    except Exception:
        pass
    finally:
        st.session_state["_ui_trigger"] = not st.session_state.get("_ui_trigger", False)


def _on_course_selected():
    """Callback when a course is selected from the selectbox widget."""
    try:
        chosen = st.session_state.get("selected_course_widget")
        # propagate widget selection into canonical 'selected_course' session key
        st.session_state["selected_course"] = chosen
        if chosen:
            det = detect_course_assets(chosen)
            st.session_state.course_asset_detect = det
        else:
            st.session_state.course_asset_detect = None
    except Exception:
        st.session_state.course_asset_detect = None
    finally:
        st.session_state["_ui_trigger"] = not st.session_state.get("_ui_trigger", False)

# load available courses from disk (or sensible defaults)
if "available_courses" not in st.session_state:
    loaded = load_courses()
    if not loaded:
        loaded = [
            "--Choose A Course--",
            "Data Structures",
            "Algorithms and Problem Solving",
            "Computer Organisation Architecture",
            "Database System and Web",
        ]
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


# Course asset loader: when a course is selected, attempt to load precomputed
# cleaned questions and faiss index from `data/courses/<slug>/` and copy the
# index into `new_faiss_index/` so the existing search UI can use it.
def _slugify(name: str) -> str:
    return ''.join(c if c.isalnum() else '_' for c in name.strip().lower()).strip('_')


def detect_course_assets(course_name: str) -> dict:
    """Detect presence of precomputed assets for a course. Returns a dict:
    { 'has_cleaned': bool, 'has_index': bool, 'manifest': dict|None, 'base': Path }
    Does NOT load or copy anything — just detects and reads manifest if present.
    """
    out = {'has_cleaned': False, 'has_index': False, 'manifest': None, 'base': None}
    try:
        if not course_name:
            return out
        base = Path('data') / 'courses' / _slugify(course_name)
        out['base'] = base
        cleaned = base / 'cleaned_questions.json'
        manifest = base / 'manifest.json'
        faiss_idx = base / 'faiss_index'

        out['has_cleaned'] = cleaned.exists()
        out['has_index'] = faiss_idx.exists()
        if manifest.exists():
            try:
                out['manifest'] = json.loads(manifest.read_text(encoding='utf-8'))
            except Exception:
                out['manifest'] = None
        return out
    except Exception:
        return out


def apply_course_assets(course_name: str, load_cleaned: bool = False, load_index: bool = False):
    """Apply (load) precomputed assets into the running app.
    - load_cleaned: load cleaned_questions.json into `st.session_state.cleaned_questions`
    - load_index: copy `faiss_index/` into `new_faiss_index/`
    """
    try:
        base = Path('data') / 'courses' / _slugify(course_name)
        cleaned = base / 'cleaned_questions.json'
        faiss_idx = base / 'faiss_index'

        if load_cleaned and cleaned.exists():
            try:
                loaded_cleaned = json.loads(cleaned.read_text(encoding='utf-8'))
                # normalize difficulty keys for compatibility with UI
                try:
                    for g, items in (loaded_cleaned or {}).items():
                        for q in items:
                            if not q.get("difficulty") and q.get("difficulty_level"):
                                q["difficulty"] = q.get("difficulty_level")
                            if not q.get("difficulty_level") and q.get("difficulty"):
                                q["difficulty_level"] = q.get("difficulty")
                except Exception:
                    pass
                st.session_state.cleaned_questions = loaded_cleaned
                st.success(f"Loaded questions for '{course_name}'.")
            except Exception as e:
                st.error(f"Failed to load cleaned questions: {e}")

        if load_index and faiss_idx.exists():
            try:
                import shutil
                target = Path('new_faiss_index')
                if target.exists():
                    shutil.rmtree(target, ignore_errors=True)
                shutil.copytree(faiss_idx, target)
                st.success(f"Prebuilt FAISS index copied for '{course_name}'.")
            except Exception as e:
                st.error(f"Failed to copy FAISS index: {e}")

        # ensure UI updates after applying assets
        try:
            st.session_state["_ui_trigger"] = not st.session_state.get("_ui_trigger", False)
        except Exception:
            pass

        return True
    except Exception:
        return False


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
    # ensure UI updates after extraction
    try:
        st.session_state["_ui_trigger"] = not st.session_state.get("_ui_trigger", False)
    except Exception:
        pass


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
    Uses the `sample_question_paper.pdf` cover template (if present) and overlays
    session-state fields and COs. Questions are rendered in Times New Roman, size 12,
    and do not show time-to-solve.
    Returns (success: bool, bytes_or_msg).
    """
    try:
        # quick text fallback when fitz unavailable
        if fitz is None:
            lines = [title, "", "Generated by Question Paper Generator", ""]
            total_marks = sum(int(q.get("marks") or 0) for q in selection)
            lines.append(f"Total Marks: {total_marks}")
            lines.append("")
            for i, q in enumerate(selection, 1):
                marks = int(q.get("marks") or 0)
                lines.append(f"Q{i}. ({marks} m)")
                lines.append(q.get('question', ''))
                lines.append("")
            return False, "\n".join(lines).encode("utf-8")

        import textwrap
        import re

        # constants and assets
        PAGE_WIDTH = 595
        PAGE_HEIGHT = 842
        LEFT_MARGIN = 50
        RIGHT_MARGIN = 20
        TOP_MARGIN = 60
        BOTTOM_MARGIN = 50

        # Course Objectives mapping (provided)
        CO_MAP = {
            "Data Structures": {
                "CO1": "Recall basic data structures and their fundamental operations.",
                "CO2": "Explain the concepts, representations, and use-cases of data structures.",
                "CO3": "Apply suitable data structures to solve programming problems.",
                "CO4": "Analyze the performance of operations across different data structures.",
                "CO5": "Evaluate and justify the choice of data structures for specific problems.",
            },
            "Algorithms and Problem Solving": {
                "CO1": "Recall standard algorithms and core algorithmic concepts.",
                "CO2": "Explain various algorithmic paradigms and their applications.",
                "CO3": "Apply algorithmic techniques to develop solutions to problems.",
                "CO4": "Analyze algorithm correctness and computational complexity.",
                "CO5": "Evaluate different algorithmic solutions to select the most optimal one.",
            },
            "Computer Organisation Architecture": {
                "CO1": "Recall basic components and structures of computer systems.",
                "CO2": "Explain the functioning of CPU, memory hierarchy, and instruction execution.",
                "CO3": "Apply architectural concepts to design basic circuits and instruction formats.",
                "CO4": "Analyze system performance based on pipelining, memory, and throughput factors.",
                "CO5": "Evaluate architectural design trade-offs in terms of performance and cost.",
            },
            "Database System and Web": {
                "CO1": "Recall fundamental database concepts, SQL basics, and web technologies.",
                "CO2": "Explain relational models, normalization, and web architecture principles.",
                "CO3": "Apply SQL and web development concepts to build simple applications.",
                "CO4": "Analyze database schemas, queries, and web workflows for performance issues.",
                "CO5": "Evaluate database and web design alternatives for scalability and security.",
            }
        }

        # Predefined placeholders (from sample template)
        PREDEFINED_PLACEHOLDERS = [
            {"name": "Course", "bbox": (51.0, 188.16798400878906, 507.70001220703125, 201.4519805908203)},
            {"name": "Time",   "bbox": (51.0, 188.16798400878906, 507.70001220703125, 201.4519805908203)},
            {"name": "Marks",  "bbox": (375.07000732421875, 211.0879669189453, 523.0599975585938, 224.37196350097656)},
            {"name": "CO1",    "bbox": (51.0, 255.63795471191406, 126.26000213623047, 268.9219665527344)},
            {"name": "CO2",    "bbox": (51.0, 278.5580139160156, 126.26000213623047, 291.8420104980469)},
            {"name": "CO3",    "bbox": (51.0, 301.4779968261719, 126.26000213623047, 314.7619934082031)},
            {"name": "CO4",    "bbox": (51.0, 324.2779846191406, 126.26000213623047, 337.5619812011719)},
            {"name": "CO5",    "bbox": (51.0, 347.197998046875, 126.26000213623047, 360.48199462890625)},
            {"name": "marks",  "bbox": (452.9800109863281, 530.3379516601562, 547.2999877929688, 543.6219482421875)},
        ]

        # mapping placeholders to session keys
        PLACEHOLDER_SESSION_MAP = {
            "Course": "selected_course",
            "Time": "exam_duration",
            "Marks": "max_marks_limit",
            "marks": "max_marks_limit",
        }

        template_path = Path("sample_question_paper.pdf")
        ttf_path = Path("pdf_generation") / "Times New Roman.ttf"

        # prepare font
        try:
            if ttf_path.exists():
                tfont = fitz.Font(str(ttf_path))
                font_name = getattr(tfont, "name", "Times-Roman")
            else:
                font_name = "Times-Roman"
        except Exception:
            font_name = "Times-Roman"

        doc = fitz.open()
        cover_last_y = None

        if template_path.exists():
            # copy template first page into new doc
            tpl = fitz.open(str(template_path))
            doc.insert_pdf(tpl, from_page=0, to_page=0)

            # prepare course title and total marks from session state
            course_title_val = str(st.session_state.get('selected_course', ''))
            # prefer explicit selected_questions in session; fallback to passed selection
            sess_sel = st.session_state.get('selected_questions')
            if sess_sel:
                total_marks_val = sum(_safe_int(q.get('marks')) for q in sess_sel)
            else:
                total_marks_val = sum(_safe_int(q.get('marks')) for q in selection)

            # overlay predefined placeholders
            for ph in PREDEFINED_PLACEHOLDERS:
                name = ph["name"]
                bbox = ph["bbox"]
                val = ""
                # explicit substitution for Course and Marks
                if name == 'Course':
                    val = course_title_val
                elif name in ('Marks', 'marks'):
                    val = str(total_marks_val)
                elif name == 'Time':
                    val = str(st.session_state.get('exam_duration', ''))
                else:
                    if name in PLACEHOLDER_SESSION_MAP:
                        key = PLACEHOLDER_SESSION_MAP[name]
                        val = str(st.session_state.get(key, "")) if key else ""
                # COs handled below
                if name.startswith("CO"):
                    continue
                try:
                    rect = fitz.Rect(bbox)
                    # cover existing placeholder text by drawing a white rectangle
                    try:
                        doc[0].draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))
                    except Exception:
                        pass
                    # insert replacement text (may be empty)
                    doc[0].insert_textbox(rect, val, fontsize=11, fontname=font_name, align=0)
                except Exception:
                    try:
                        # fallback: draw at the rect origin
                        doc[0].insert_text((rect.x0, rect.y0), val, fontsize=11, fontname=font_name)
                    except Exception:
                        pass

            # insert COs under course area
            selected_course_name = st.session_state.get('selected_course') or ''
            cos = CO_MAP.get(selected_course_name, {})
            # choose a start y from CO1 bbox
            co1_bbox = next((p['bbox'] for p in PREDEFINED_PLACEHOLDERS if p['name'] == 'CO1'), None)
            if co1_bbox:
                sx = co1_bbox[0]
                sy = co1_bbox[1]
            else:
                sx = LEFT_MARGIN
                sy = TOP_MARGIN + 120
            cy = sy
            for i in range(1, 6):
                key = f"CO{i}"
                line = cos.get(key, '')
                try:
                    doc[0].insert_text((sx, cy), f"{key}: {line}", fontsize=12, fontname=font_name)
                except Exception:
                    try:
                        rect = fitz.Rect(sx, cy - 2, PAGE_WIDTH - RIGHT_MARGIN, cy + 12)
                        doc[0].insert_textbox(rect, f"{key}: {line}", fontsize=12, fontname=font_name, align=0)
                    except Exception:
                        pass
                cy += 14
            cover_last_y = cy + 6
            tpl.close()
        else:
            # programmatic cover
            cover = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
            header_lines = [
                "POSSESSION OF MOBILES IN EXAM IS UFM PRACTICE",
                "Jaypee Institute of Information Technology, Noida",
                f"{title}",
            ]
            try:
                y = TOP_MARGIN - 40
                for i, hl in enumerate(header_lines):
                    cover.insert_text((PAGE_WIDTH / 2, y + i * 18), hl, fontsize=14 - i, fontname=font_name, align=1)
            except Exception:
                pass
            # course/marks
            try:
                cover.insert_text((LEFT_MARGIN, TOP_MARGIN + 30), f"Course Title: {st.session_state.get('selected_course','')}", fontsize=11, fontname=font_name)
                cover.insert_text((PAGE_WIDTH - RIGHT_MARGIN - 180, TOP_MARGIN + 30), f"Maximum Time: {st.session_state.get('exam_duration','')}", fontsize=11, fontname=font_name)
                cover.insert_text((PAGE_WIDTH - RIGHT_MARGIN - 180, TOP_MARGIN + 50), f"Maximum Marks: {st.session_state.get('max_marks_limit','')}", fontsize=11, fontname=font_name)
            except Exception:
                pass
            # COs
            selected_course_name = st.session_state.get('selected_course') or ''
            cos = CO_MAP.get(selected_course_name, {})
            cy = TOP_MARGIN + 80
            for i in range(1, 6):
                key = f"CO{i}"
                line = cos.get(key, '')
                try:
                    cover.insert_text((LEFT_MARGIN, cy), f"{key}: {line}", fontsize=12, fontname=font_name)
                except Exception:
                    pass
                cy += 14
            cover_last_y = cy + 6

        # Questions: try to continue on cover page if room
        question_font = 12
        wrap_chars = 90
        if cover_last_y and doc.page_count > 0:
            page = doc[0]
            y = cover_last_y
            # if not enough room, start fresh
            if y > PAGE_HEIGHT - BOTTOM_MARGIN - 80:
                page = None
                y = TOP_MARGIN
        else:
            page = None
            y = TOP_MARGIN

        for idx, q in enumerate(selection, 1):
            if page is None or y > PAGE_HEIGHT - BOTTOM_MARGIN - 60:
                page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
                y = TOP_MARGIN

            header = f"Q{idx}. ({_safe_int(q.get('marks'))} marks)"
            try:
                page.insert_text((LEFT_MARGIN, y), header, fontsize=11, fontname=font_name)
            except Exception:
                try:
                    page.insert_textbox(fitz.Rect(LEFT_MARGIN, y, PAGE_WIDTH - RIGHT_MARGIN, y + 14), header, fontsize=11, fontname=font_name)
                except Exception:
                    pass
            y += 16

            # sanitize question body: remove any appended metadata (Topic/Subtopic/Marks/Time)
            raw_body = q.get('question', '') or ''
            # remove any trailing metadata label like 'Topic: ...', 'Subtopic: ...', 'Marks: ...', 'Time: ...'
            # we remove from the first occurrence of such a label to the end of string to avoid leftover metadata
            body = re.sub(r"(?is)\b(?:Topic|Subtopic|Marks?|Time):.*$", "", raw_body).strip()
            if not body:
                body = '[No question text]'
            wrapped = textwrap.wrap(body, width=wrap_chars)
            for line in wrapped:
                if y > PAGE_HEIGHT - BOTTOM_MARGIN - 30:
                    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
                    y = TOP_MARGIN
                try:
                    page.insert_text((LEFT_MARGIN, y), line, fontsize=question_font, fontname=font_name)
                except Exception:
                    try:
                        page.insert_textbox(fitz.Rect(LEFT_MARGIN, y, PAGE_WIDTH - RIGHT_MARGIN, y + 14), line, fontsize=question_font, fontname=font_name)
                    except Exception:
                        pass
                y += 14
            y += 12

        # page numbers
        total_pages = doc.page_count
        for pnum in range(total_pages):
            try:
                doc[pnum].insert_text((PAGE_WIDTH/2, PAGE_HEIGHT - 30), f"Page {pnum+1} / {total_pages}", fontsize=9, fontname=font_name, align=1)
            except Exception:
                pass

        # output bytes
        try:
            pdf_bytes = doc.write()
            doc.close()
            if pdf_bytes:
                return True, pdf_bytes
        except Exception:
            pass

        # disk fallback
    except Exception as e:
        return False, f"Failed to create PDF: {e}"


def create_pdf_with_latex(selection, title: str = "Question Paper", course_title: str = "", exam_time: str = "", CO_MAP: dict | None = None, ttf_path: str | None = None):
    """Create PDF bytes via LaTeX (XeLaTeX). Returns (success, bytes_or_error).
    Requires `xelatex` on PATH. If not available the function returns a helpful message.
    """
    import re
    import tempfile
    import subprocess
    from pathlib import Path
    from shutil import copyfile

    def _escape_latex(s: str) -> str:
        if s is None:
            return ""
        s = str(s)
        s = s.replace("\\", r"\textbackslash{}")
        # for a, b in [('\\', r'\\textbackslash{}'), ('&', r'\\&'), ('%', r'\\%'), ('$', r'\\$'), ('#', r'\\#'), ('_', r'\\_'), ('{', r'\\{'), ('}', r'\\}'), ('~', r'\\textasciitilde{}'), ('^', r'\\textasciicircum{}')]:
        #     s = s.replace(a, b)
        replacements = {
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        for a, b in replacements.items():
            s = s.replace(a, b)
        return s

    def _sanitize_body(text: str) -> str:
        if not text:
            return ''
        return re.sub(r"(?is)\\b(?:Topic|Subtopic|Marks?|Time):.*$", "", text).strip()

    try:
        if CO_MAP is None:
            CO_MAP = {}

        if ttf_path is None:
            ttf_path = str(Path(__file__).parent.parent / "pdf_generation" / "Times New Roman.ttf")
        ttf_src = Path(ttf_path)
        font_basename = ttf_src.name if ttf_src.exists() else "Times New Roman"

        total_marks = 0
        q_blocks = []
        for i, q in enumerate(selection, 1):
            marks = int(q.get('marks') or 0)
            total_marks += marks
            body = _sanitize_body(q.get('question', ''))
            body = _escape_latex(body).replace('\n', '\\\\\n')
            q_blocks.append(f"\\noindent \\textbf{{Q{i}. ({marks} marks)}}\\\\\n{body}\\\\[10pt]")

        # build document as lines
        lines = []
        lines.append(r"\documentclass[12pt]{article}")
        lines.append(r"\usepackage[a4paper,margin=1in]{geometry}")
        lines.append(r"\usepackage{fontspec}")
        # Use the font file name (copied into the build dir) or fall back to system font name
        lines.append("\\setmainfont{" + font_basename + "}")
        lines.append(r"\usepackage{setspace}")
        lines.append(r"\usepackage{parskip}")
        lines.append(r"\pagestyle{plain}")
        lines.append(r"\begin{document}")
        lines.append(r"\begin{center}")
        lines.append(r"\textit{\textbf{\large POSSESSION OF MOBILES IN EXAM IS UFM PRACTICE}}\\[6pt]")
        lines.append(r"\textbf{\large Jaypee Institute of Information Technology, Noida}\\[10pt]")
        lines.append(r"\textbf{\normalsize " + _escape_latex(title) + r"}\\[12pt]")
        lines.append(r"\end{center}")
        lines.append(r"\noindent Name: \rule{6cm}{0.4pt} \hfill Enrolment No.: \rule{4.5cm}{0.4pt}")
        lines.append(r"\vspace{16pt}\\")
        lines.append(r"\noindent \textbf{Course Title:} " + _escape_latex(course_title) + r" \hfill")
        lines.append(r"\noindent \textbf{Maximum Marks:} " + str(total_marks) + r"\\\\")
        lines.append(r"\vspace{14pt}")

        for i in range(1, 6):
            lines.append(r"\noindent \textbf{CO" + str(i) + r"}: " + _escape_latex(CO_MAP.get(f"CO{i}", "")) + r"\\[6pt]")

        lines.append(r"\vspace{10pt}\noindent \textit{Note: All questions are compulsory.}\\\\\\\vspace{14pt}")

        lines.extend(q_blocks)
        lines.append(r"\end{document}")

        latex = "\n".join(lines)

        with tempfile.TemporaryDirectory() as d:
            dpath = Path(d)
            tex_file = dpath / "paper.tex"
            tex_file.write_text(latex, encoding='utf-8')
            # copy ttf into build dir if available (so fontspec can load the local file)
            if ttf_src.exists():
                try:
                    copyfile(str(ttf_src), str(dpath / ttf_src.name))
                except Exception:
                    pass
            # compile
            try:
                subprocess.run(["xelatex", "-interaction=nonstopmode", str(tex_file)], cwd=str(dpath), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(["xelatex", "-interaction=nonstopmode", str(tex_file)], cwd=str(dpath), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except FileNotFoundError:
                return False, "xelatex not found on system. Install TeX Live / MiKTeX with xelatex to enable LaTeX PDF generation."
            except subprocess.CalledProcessError:
                log = (dpath / "paper.log").read_text(encoding='utf-8') if (dpath / "paper.log").exists() else ""
                return False, f"xelatex failed. Log:\n{log}"

            pdf_file = dpath / "paper.pdf"
            if pdf_file.exists():
                return True, pdf_file.read_bytes()
            return False, "xelatex did not produce a PDF"

    except Exception as e:
        return False, str(e)
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            tmp_path = tmp.name
            tmp.close()
            doc.save(tmp_path)
            doc.close()
            with open(tmp_path, 'rb') as f:
                pdf_bytes = f.read()
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            return True, pdf_bytes
        except Exception as e:
            try:
                doc.close()
            except Exception:
                pass
            return False, str(e)
    except Exception as e:
        return False, str(e)


st.markdown("---")


def nav():
    c1, c2, c3 = st.columns([1, 11, 1])
    back = c1.button("Back")
    # disable Next on the first step until a course is selected
    disable_next = False
    try:
        if st.session_state.wizard_step == 0 and not st.session_state.get("selected_course"):
            disable_next = True
    except Exception:
        disable_next = False
    # Quick Guide placed in the center column between Back and Next
    try:
        with c2:
            with st.expander("Quick Guide", expanded=False):
                st.write(
                    "This wizard guides you through the essential steps. Use Next/Back to navigate. Advanced diagnostics are available at the bottom."
                )
    except Exception:
        pass

    # Next button on the right
    nxt = c3.button("Next", disabled=disable_next)

    if back:
        st.session_state.wizard_step = max(0, st.session_state.wizard_step - 1)
        st.session_state["_ui_trigger"] = not st.session_state.get("_ui_trigger", False)
    if nxt:
        st.session_state.wizard_step = min(5, st.session_state.wizard_step + 1)
        st.session_state["_ui_trigger"] = not st.session_state.get("_ui_trigger", False)

# render navbar (buttons + quick guide)
nav()
if st.session_state.wizard_step == 0:
    step_card("Select Course", 0)
    st.markdown("#### Select course for this question paper")
    cols = st.columns([3, 2])
    with cols[0]:
        course_options = list(st.session_state.get("available_courses", [])) + ["-- Add new course --"]
        # compute default index so the selectbox doesn't reset to the first option on reruns
        current = st.session_state.get("selected_course")
        try:
            # if a course was just added and we want to select it, prefer that
            select_after = st.session_state.get("_select_after_add")
            if select_after and select_after in course_options:
                default_index = course_options.index(select_after)
            else:
                default_index = course_options.index(current) if current in course_options else 0
        except Exception:
            default_index = 0
        chosen = st.selectbox("Choose existing course or add new", options=course_options, index=default_index, key="selected_course_widget", on_change=_on_course_selected)
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
                # mark this course to be selected on the next render (safe; widget will pick it up)
                st.session_state["_select_after_add"] = new_name
                st.success(f"Selected course: {new_name}")
                # ensure UI updates
                st.session_state["_ui_trigger"] = not st.session_state.get("_ui_trigger", False)

        # If we requested a post-add selection, propagate it into canonical selected_course now
        if st.session_state.get("_select_after_add"):
            try:
                sel_after = st.session_state.pop("_select_after_add")
                st.session_state["selected_course"] = sel_after
                st.session_state["course_asset_detect"] = detect_course_assets(sel_after)
            except Exception:
                st.session_state["course_asset_detect"] = None
            finally:
                st.session_state["_ui_trigger"] = not st.session_state.get("_ui_trigger", False)
    else:
        # selecting an existing course — asset detection is handled by the selectbox callback
        # If detection found assets, show a small banner + controls
        det = st.session_state.get('course_asset_detect')
        if det and (det.get('has_cleaned') or det.get('has_index')):
            # st.markdown(f"**Prebuilt assets detected for '{chosen}'**")
            # # preview suppressed by user preference

            # # auto-apply prebuilt index setting is defaulted in session (hidden control)
            # mf = det.get('manifest')
            # if mf:
            #     st.markdown(f"- Source: `{mf.get('source')}`  ")
            #     st.markdown(f"- Questions: **{mf.get('count_questions', '?')}**")
            #     grp = mf.get('groups') or {}
            #     st.markdown(f"- Groups: {grp}")

            action = st.radio("What would you like to do with prebuilt assets?", ["Load & Replace", "Load & Merge", "Load Index only"], index=0, key="prebuilt_action")

            if st.button("Apply prebuilt assets", key="apply_prebuilt"):
                if action == "Load & Replace":
                    # load cleaned and index, replacing any existing cleaned_questions
                    success = apply_course_assets(chosen, load_cleaned=True, load_index=True)
                    if success:
                        st.success("Prebuilt assets loaded (replace).")
                elif action == "Load & Merge":
                    # merge prebuilt cleaned into existing cleaned_questions (or set if none)
                    base = Path(det.get('base'))
                    cleaned_path = base / 'cleaned_questions.json'
                    if cleaned_path.exists():
                        try:
                            pre = json.loads(cleaned_path.read_text(encoding='utf-8'))
                            if not st.session_state.get('cleaned_questions'):
                                st.session_state.cleaned_questions = pre
                            else:
                                # merge by group (append)
                                for g, items in pre.items():
                                    st.session_state.cleaned_questions.setdefault(g, [])
                                    st.session_state.cleaned_questions[g].extend(items)
                            # optionally copy index too
                            if det.get('has_index'):
                                apply_course_assets(chosen, load_cleaned=False, load_index=True)
                            st.success("Prebuilt assets merged into current cleaned questions.")
                            st.session_state["_ui_trigger"] = not st.session_state.get("_ui_trigger", False)
                        except Exception as e:
                            st.error(f"Failed to merge prebuilt questions: {e}")
                    else:
                        st.info("No cleaned questions to merge.")
                elif action == "Load Index only":
                    if det.get('has_index'):
                        apply_course_assets(chosen, load_cleaned=False, load_index=True)
                        st.success("Prebuilt index copied into working index.")
                    else:
                        st.info("No prebuilt index available for this course.")
                else:
                    st.info("Preview-only action does not apply assets.")

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
                st.session_state["_ui_trigger"] = not st.session_state.get("_ui_trigger", False)
    with colp2:
        if st.button("Skip upload — Use existing questions", key="goto_existing"):
            # if prebuilt assets detected and user opted in, apply them automatically
            det = st.session_state.get('course_asset_detect')
            try:
                if det:
                    base = det.get('base')
                    has_cleaned = det.get('has_cleaned')
                    has_index = det.get('has_index')
                    if st.session_state.get('auto_use_prebuilt_index') and has_index:
                        # apply both cleaned (if available) and index
                        apply_course_assets(chosen, load_cleaned=bool(has_cleaned), load_index=True)
                        st.success('Applied prebuilt assets.')
            except Exception:
                pass

            # jump to Search (step index 4 after shift)
            st.session_state.wizard_step = 4
            st.session_state["_ui_trigger"] = not st.session_state.get("_ui_trigger", False)


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
                        # ensure UI updates after generator output
                        st.session_state["_ui_trigger"] = not st.session_state.get("_ui_trigger", False)
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
                    st.session_state["_ui_trigger"] = not st.session_state.get("_ui_trigger", False)
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
                st.session_state["_ui_trigger"] = not st.session_state.get("_ui_trigger", False)

        if st.session_state.cleaned_questions:
            st.write("Preview of cleaned data (first group):")
            first_key = next(iter(st.session_state.cleaned_questions.keys()))
            st.json({first_key: st.session_state.cleaned_questions[first_key][:5]})
            st.download_button("Download cleaned questions.json", data=json.dumps(st.session_state.cleaned_questions, indent=2, ensure_ascii=False).encode("utf-8"), file_name="questions.cleaned.json")


# Step 4: Index & Search
elif st.session_state.wizard_step == 4:
    step_card("Search and Select Questions for Question Paper", 4)
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

        # build_idx = st.button("Build Index (if available)", key="wizard_build_idx")
        # if build_idx:
        #     if FAISS is None or OllamaEmbeddings is None:
        #         st.error("Indexing requires FAISS and Ollama embeddings to be installed and available.")
        #     elif not st.session_state.cleaned_questions:
        #         st.warning("No cleaned questions available to index.")
        #     else:
        #         # show progress during manual build
        #         with st.spinner("Building index — this may take some time"):
        #             success, msg = build_index_from_cleaned(st.session_state.cleaned_questions, show_progress=True)
        #             if success:
        #                 st.success(msg)
        #             else:
        #                 st.error(msg)

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

                        # create results with normalized difficulty; if not present, infer from marks
                        def infer_difficulty_from_marks(m):
                            try:
                                m = int(m)
                            except Exception:
                                return ""
                            mapping = {1: "easy", 2: "medium", 3: "hard", 5: "hard"}
                            return mapping.get(m, "")

                        results = []
                        for d in filtered:
                            diff = d.metadata.get("difficulty") or d.metadata.get("difficulty_level") or ""
                            marks_meta = d.metadata.get("marks")
                            if not diff:
                                diff = infer_difficulty_from_marks(marks_meta)
                            results.append({
                                "question": d.page_content,
                                "topic": d.metadata.get("topic"),
                                "subtopic": d.metadata.get("subtopic"),
                                # ensure time is an integer minute value in the UI
                                "time": d.metadata.get("time"),
                                "marks": marks_meta,
                                "difficulty": diff,
                                "cognitive_level": d.metadata.get("cognitive_level"),
                            })

                        st.session_state.search_results = results

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
                    
                    # Show complete question text
                    st.write(f"**Question**: {item['question']}")
                    st.write("---")
                    
                    # Show metadata
                    st.write(f"**Topic**: {item['topic']}  \n"
                             f"**Subtopic**: {item['subtopic']}  \n"
                             f"**Marks**: {item['marks']}  \n"
                             f"**Time**: {display_time_str}  \n"
                             f"**Difficulty**: {item['difficulty']}  \n"
                             f"**Cognitive**: {item['cognitive_level']}")
                    # add selection control
                    q_marks = _safe_int(item.get("marks"))
                    add_key = f"add_{i}"
                    remove_key = f"rem_{i}"
                    # compute current total and remaining marks (used to disable Add button)
                    current_total = sum(_safe_int(s.get("marks")) for s in st.session_state.selected_questions)
                    limit = int(st.session_state.get("max_marks_limit", 20) or 0)
                    remaining = limit - current_total

                    # compare using normalized marks to avoid type mismatches
                    if any((s.get("question") == item.get("question") and _safe_int(s.get("marks")) == q_marks) for s in st.session_state.selected_questions):
                        # use callback to remove first matching item to ensure Streamlit reruns
                        if st.button("Remove from selection", key=remove_key, on_click=_remove_first_matching, args=(item.get("question"), q_marks)):
                            st.success("Removed from selection")
                    else:
                        # show remaining marks to the user
                        # st.write(f"Remaining marks: {remaining} / {limit}")

                        # disable Add when not enough remaining marks
                        disabled_add = (q_marks > remaining)
                        if st.button("Add to selection", key=add_key, disabled=disabled_add, on_click=_add_to_selection, args=(item, q_marks)):
                            st.success("Added to selection")
                        else:
                            if disabled_add:
                                st.info(f"Not enough remaining marks to add this question ({q_marks} required). Remove some selected items first.")

    with sel_col:
        st.markdown("### Selected Questions")
        # control to set max marks limit — fixed options persisted in session
        options = [20, 35, 100]
        # determine default index from current session value if present
        try:
            current_val = int(st.session_state.get("max_marks_limit", 20) or 20)
        except Exception:
            current_val = 20
        try:
            default_index = options.index(current_val) if current_val in options else 0
        except Exception:
            default_index = 0
        sel = st.selectbox("Max total marks (limit)", options, index=default_index, key="max_marks_limit")

        sel = st.session_state.get("selected_questions") or []
        total_selected = sum(_safe_int(s.get("marks")) for s in sel)
        limit = int(st.session_state.get("max_marks_limit", 20) or 0)
        remaining_global = max(0, limit - total_selected)
        # Bolder visual: show as metrics for quick scanning
        mcol1, mcol2 = st.columns(2)
        try:
            mcol1.metric("Total Selected", f"{total_selected} / {limit}")
            mcol2.metric("Remaining", f"{remaining_global} / {limit}")
        except Exception:
            # fallback to plain text if metric is unavailable
            st.write(f"Total selected marks: {total_selected} / {limit}")
            st.write(f"Remaining marks: {remaining_global} / {limit}")

        # list selected questions with remove option (use callbacks so Streamlit updates immediately)
        for j, s in enumerate(sel, 1):
            st.write(f"{j}. ({s.get('marks')} m) {s.get('question')[:120]}...")
            if st.button(f"Remove #{j}", key=f"sel_rem_{j}", on_click=_remove_from_selection_by_index, args=(j-1,)):
                st.success("Removed from selection")

        if sel:
            finalize_disabled = len(sel) == 0
            if st.button("Finalize & Generate PDF", disabled=finalize_disabled):
                # open the review dashboard instead of immediate PDF generation
                st.session_state["dashboard_mode"] = True
                st.session_state["wizard_step"] = 5
                st.session_state["_ui_trigger"] = not st.session_state.get("_ui_trigger", False)


# Step 5: Export / Dashboard
elif st.session_state.wizard_step == 5:
    step_card("Export / Review Dashboard", 5)

    # If dashboard_mode is set, render an interactive review dashboard
    if st.session_state.get("dashboard_mode"):
        sel = st.session_state.get("selected_questions") or []

        # Top: Paper summary split into two columns
        summary_left, summary_right = st.columns([1, 1])

        total_marks = sum(_safe_int(q.get('marks')) for q in sel)
        diffs = [ (q.get('difficulty') or q.get('difficulty_level') or 'unknown').lower() for q in sel ]
        diff_counts = {}
        for d in diffs:
            diff_counts[d] = diff_counts.get(d, 0) + 1

        topics = [ (q.get('topic') or 'General') for q in sel ]
        topic_counts = {}
        for t in topics:
            topic_counts[t] = topic_counts.get(t, 0) + 1

        with summary_left:
            st.header("Paper Summary")
            st.metric("Total Marks", f"{total_marks}")
            st.metric("Questions Selected", f"{len(sel)}")
            # Estimated completion time (sum of question 'time' fields, expected in minutes)
            try:
                total_time_mins = sum(_safe_int(q.get('time')) for q in sel)
                hrs = total_time_mins // 60
                mins = total_time_mins % 60
                if hrs:
                    time_str = f"{hrs}h {mins}m"
                else:
                    time_str = f"{mins} min"

                # Determine allowed time based on max marks limit mapping
                limit = int(st.session_state.get("max_marks_limit", 20) or 20)
                # explicit mapping: 20 -> 1h, 35 -> 2h, 100 -> 3h; otherwise choose nearest bracket
                if limit <= 20:
                    allowed_hours = 1
                elif limit <= 35:
                    allowed_hours = 2
                else:
                    allowed_hours = 3

                allowed_mins = allowed_hours * 60

                # Display estimated time and a warning if it exceeds allowed time
                st.metric("Estimated Time", time_str)
                if total_time_mins > allowed_mins:
                    st.error(f"Estimated time ({time_str}) exceeds the recommended limit of {allowed_hours}h for a {limit}-mark paper.")
                else:
                    st.success(f"Estimated time ({time_str}) is within the recommended limit of {allowed_hours}h for a {limit}-mark paper.")
            except Exception:
                st.info("Estimated time unavailable")

            st.subheader("Difficulty Breakdown")
            try:
                import plotly.graph_objects as go
                if diff_counts:
                    labels = list(diff_counts.keys())
                    values = [diff_counts.get(l, 0) for l in labels]
                    # explicit hex color mapping for difficulty
                    hex_map = {"easy": "#2ecc71", "medium": "#f1c40f", "hard": "#e74c3c"}
                    colors = [hex_map.get(lbl.lower().strip(), "#95a5a6") for lbl in labels]
                    fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors))])
                    fig.update_layout(title_text="By Difficulty")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No difficulty info available.")
            except Exception:
                st.write(diff_counts or "No difficulty info available.")

        with summary_right:
            st.header("Topics Covered")
            if topic_counts:
                for t, c in topic_counts.items():
                    st.write(f"- {t}: {c} question(s)")
            else:
                st.info("No topic metadata available.")

            # Bloom's Taxonomy / Cognitive level distribution (placed above Actions)
            try:
                st.subheader("Bloom's Taxonomy")
                levels = ["remembering", "understanding", "applying", "analyzing", "evaluating", "creating"]
                cog_counts = {lvl: 0 for lvl in levels}
                for q in sel:
                    cl = (q.get("cognitive_level") or q.get("cognitive") or "").strip().lower()
                    if not cl:
                        continue
                    if cl.startswith("remember"):
                        key = "remembering"
                    elif cl.startswith("understand"):
                        key = "understanding"
                    elif cl.startswith("apply"):
                        key = "applying"
                    elif cl.startswith("analy"):
                        key = "analyzing"
                    elif cl.startswith("evalu"):
                        key = "evaluating"
                    elif cl.startswith("creat"):
                        key = "creating"
                    else:
                        key = cl
                    cog_counts.setdefault(key, 0)
                    cog_counts[key] += 1

                labels = list(cog_counts.keys())
                values = [cog_counts.get(l, 0) for l in labels]
                color_map = {
                    "remembering": "#2ecc71",
                    "understanding": "#f1c40f",
                    "applying": "#e67e22",
                    "analyzing": "#e74c3c",
                    "evaluating": "#c0392b",
                    "creating": "#9b59b6",
                }
                colors = [color_map.get(l, "#95a5a6") for l in labels]
                import plotly.graph_objects as go
                bar = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=colors)])
                bar.update_layout(title_text="Cognitive Level Distribution (Bloom's Taxonomy)", xaxis_title="Cognitive Level", yaxis_title="Count")
                st.plotly_chart(bar, use_container_width=True)
            except Exception:
                st.write("Bloom's taxonomy unavailable")

            st.subheader("Actions")
            pdf_col1, pdf_col2 = st.columns([1, 1])
            with pdf_col1:
                if sel:
                    # prepare CO map (same mapping used by the PDF generator)
                    CO_MAP = {
                        "Data Structures": {
                            "CO1": "Recall basic data structures and their fundamental operations.",
                            "CO2": "Explain the concepts, representations, and use-cases of data structures.",
                            "CO3": "Apply suitable data structures to solve programming problems.",
                            "CO4": "Analyze the performance of operations across different data structures.",
                            "CO5": "Evaluate and justify the choice of data structures for specific problems.",
                        },
                        "Algorithms and Problem Solving": {
                            "CO1": "Recall standard algorithms and core algorithmic concepts.",
                            "CO2": "Explain various algorithmic paradigms and their applications.",
                            "CO3": "Apply algorithmic techniques to develop solutions to problems.",
                            "CO4": "Analyze algorithm correctness and computational complexity.",
                            "CO5": "Evaluate different algorithmic solutions to select the most optimal one.",
                        },
                        "Computer Organisation Architecture": {
                            "CO1": "Recall basic components and structures of computer systems.",
                            "CO2": "Explain the functioning of CPU, memory hierarchy, and instruction execution.",
                            "CO3": "Apply architectural concepts to design basic circuits and instruction formats.",
                            "CO4": "Analyze system performance based on pipelining, memory, and throughput factors.",
                            "CO5": "Evaluate architectural design trade-offs in terms of performance and cost.",
                        },
                        "Database System and Web": {
                            "CO1": "Recall fundamental database concepts, SQL basics, and web technologies.",
                            "CO2": "Explain relational models, normalization, and web architecture principles.",
                            "CO3": "Apply SQL and web development concepts to build simple applications.",
                            "CO4": "Analyze database schemas, queries, and web workflows for performance issues.",
                            "CO5": "Evaluate database and web design alternatives for scalability and security.",
                        }
                    }
                    # call LaTeX generator (will return bytes if xelatex is available)
                    # pass only the CO mapping for the selected course (inner dict)
                    selected_course_name = st.session_state.get('selected_course', '')
                    course_co_map = CO_MAP.get(selected_course_name, {}) if isinstance(CO_MAP, dict) else {}
                    success, payload = create_pdf_with_latex(
                        sel,
                        title="Question Paper",
                        course_title=selected_course_name,
                        exam_time=str(st.session_state.get('exam_duration','')),
                        CO_MAP=course_co_map,
                    )
                    if success:
                        st.download_button("Download PDF", data=payload, file_name="question_paper.pdf")
                    else:
                        if isinstance(payload, bytes):
                            st.download_button("Download (txt)", data=payload, file_name="question_paper.txt")
                        else:
                            st.error(f"Failed to create PDF: {payload}")
                else:
                    st.caption("No questions selected to generate PDF.")

            with pdf_col2:
                if st.button("Back to Selection"):
                    st.session_state["dashboard_mode"] = False
                    st.session_state["wizard_step"] = 4
                    st.session_state["_ui_trigger"] = not st.session_state.get("_ui_trigger", False)

        # Below summaries: full-width selected questions list for review and removal
        st.markdown("---")
        st.header("Selected Questions")
        st.markdown("Use the Remove buttons to edit selection. Changes update instantly.")
        if not sel:
            st.info("No questions selected.")
        for idx, q in enumerate(sel):
            with st.expander(f"Q{idx+1}: ({q.get('marks')} m) {q.get('topic') or ''} - {q.get('difficulty') or ''}"):
                st.write(q.get('question'))
                st.write(f"**Topic**: {q.get('topic')}  •  **Subtopic**: {q.get('subtopic')}")
                st.write(f"**Marks**: {q.get('marks')}  •  **Difficulty**: {q.get('difficulty')}  •  **Time**: {q.get('time')}")
                if st.button("Remove from selection", key=f"dash_rem_{idx}", on_click=_remove_from_selection_by_index, args=(idx,)):
                    st.success("Removed")

    else:
        # default Export UI (legacy)
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
    
