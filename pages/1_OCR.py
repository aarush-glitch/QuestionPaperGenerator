import streamlit as st
import tempfile
import os
from pathlib import Path
import requests
from PIL import Image

def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# --- PAGE CONFIG ---
st.set_page_config(page_title="Question Paper Generator - OCR & Refinement", layout="wide", initial_sidebar_state="expanded")

# --- STYLES ---
st.markdown("""
<style>
.solid-header {
    background: #181c24;
    padding: 2rem 0 1rem 0;
    border-radius: 1.5rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 2px 8px #0003;
}
.soft-card {
    background: #23272f;
    border-radius: 1rem;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    color: #f5f6fa;
    box-shadow: 0 2px 8px #0002;
}
.step-title {
    font-size: 1.3rem;
    color: #43cea2;
    font-weight: bold;
    margin-bottom: 0.5em;
}
</style>
""", unsafe_allow_html=True)

# Match main container width/padding with other pages
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
            max-width: 1100px !important;
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

# --- HEADER ---
with st.container():
    st.markdown(
        """
        <div class='solid-header'>
            <img src='https://img.icons8.com/ios-filled/250/ffffff/idea.png' width='120' style='margin-bottom: 1rem;' />
            <h2 style='color: #e0f7fa; font-weight: 800; letter-spacing: 1px; margin-bottom: 0.3em;'>OCR & Text Refinement</h2>
            <p style='color: #e0f7fa; font-size: 1.1rem; opacity: 0.85;'>Seamlessly extract and refine text from your PDFs and images.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<hr style='border:1px solid #43cea2;margin:2em 0;'>", unsafe_allow_html=True)

# --- LOTTIE ANIMATION ---
lottie_ocr = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_FYx0Ph.json")
with st.container():
    col1, col2 = st.columns([1, 2])
    with col1:
        if lottie_ocr:
            try:
                from streamlit_lottie import st_lottie
                st_lottie(lottie_ocr, height=180, key="ocr_anim")
            except Exception:
                st.image("https://img.icons8.com/ios-filled/250/43cea2/idea.png", width=120)
        else:
            st.image("https://img.icons8.com/ios-filled/250/43cea2/idea.png", width=120)
    with col2:
        st.markdown("""
        <div class='soft-card'>
        <span class='step-title'>How it works:</span><br>
        <b>Step 1:</b> Upload your PDF or image file.<br>
        <b>Step 2:</b> The app will extract text using OCR.<br>
        <b>Step 3:</b> The extracted text will be automatically refined.<br>
        <b>Step 4:</b> View and download the final refined text.
        </div>
        """, unsafe_allow_html=True)

# --- MAIN FUNCTIONALITY ---
with st.container():
    # st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload PDF or Image", 
        type=["pdf", "png", "jpg", "jpeg"], 
        help="Supported: PDF, PNG, JPG, JPEG",
        key="ocr_file_uploader"
    )
    st.markdown("<hr style='border:1px solid #43cea2;margin:2em 0;'>", unsafe_allow_html=True)

    if uploaded_file:
        if 'last_uploaded_filename' not in st.session_state or st.session_state['last_uploaded_filename'] != uploaded_file.name:
            suffix = Path(uploaded_file.name).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name

            try:
                from Pytesseract.refinement import handle_image, handle_pdf
            except Exception as e:
                handle_image = None
                handle_pdf = None
                _refine_import_error = str(e)

            with st.spinner("Processing file with OCR and refinement..."):
                output_text = None
                if suffix in [".png", ".jpg", ".jpeg"]:
                    if handle_image is None:
                        st.error("`handle_image` could not be imported from refinement.py")
                    else:
                        output_text = handle_image(str(temp_path))
                elif suffix == ".pdf":
                    if handle_pdf is None:
                        st.error("`handle_pdf` could not be imported from refinement.py")
                    else:
                        output_text = handle_pdf(str(temp_path))
                else:
                    st.error("Unsupported file type.")
                os.unlink(temp_path)

            st.session_state['output_text'] = output_text
            st.session_state['last_uploaded_filename'] = uploaded_file.name
        else:
            output_text = st.session_state.get('output_text', None)

        if output_text:
            st.success("OCR & refinement complete!")
            st.markdown("<b>Final Refined Text:</b>", unsafe_allow_html=True)
            unique_id = f"refined-pre-{hash(uploaded_file.name)}"
            btn_id = f"copy-btn-{hash(uploaded_file.name)}"
            st.markdown(f"""
            <div style='background:#181c24; border-radius:0.7rem; padding:1.2rem 1rem; margin-bottom:1.2rem; color:#e0f7fa; font-size:1.08rem; font-family:monospace; max-height:320px; overflow-y:auto; box-shadow:0 2px 8px #0002;'>
                <pre id='{unique_id}' style='white-space:pre-wrap; word-break:break-word; background:none; border:none; margin:0; padding:0;'>
{output_text}
                </pre>
            </div>
            <button id='{btn_id}' style='background:#43cea2; color:#181c24; border:none; border-radius:0.4rem; padding:0.5em 1.2em; font-weight:600; cursor:pointer; margin-bottom:1.2rem;'>Copy to Clipboard</button>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <script>
            const btn = document.getElementById('{btn_id}');
            if(btn){{
                btn.onclick = function(){{
                    const text = document.getElementById('{unique_id}').innerText;
                    navigator.clipboard.writeText(text);
                    btn.innerText = 'Copied!';
                    setTimeout(()=>{{btn.innerText='Copy to Clipboard';}}, 1200);
                }}
            }}
            </script>
            """, unsafe_allow_html=True)
            st.download_button(
                label="Download Refined Text",
                data=output_text,
                file_name="refined_text.txt",
                mime="text/plain",
                key=f"download-btn-{hash(uploaded_file.name)}"
            )
            st.info("You can now use this refined text for further processing or question generation.")
        else:
            st.error("No output generated. Please check your file or backend modules.")
    st.markdown("</div>", unsafe_allow_html=True)
