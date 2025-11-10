import streamlit as st
import requests

def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

st.set_page_config(page_title="Question Paper Generator - Home", layout="wide", initial_sidebar_state="expanded")

# --- STYLES ---
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
    font-size: 3.2rem;
    font-weight: bold;
    color: #f5f6fa;
    text-align: center;
    margin-bottom: 0.2em;
    font-family: 'Segoe UI', 'Arial', sans-serif;
    letter-spacing: 2px;
    text-shadow: 0 2px 8px #185a9d44;
}
.subtitle {
    font-size: 1.5rem;
    color: #e0f7fa;
    text-align: center;
    margin-bottom: 1.5em;
    font-family: 'Segoe UI', 'Arial', sans-serif;
    letter-spacing: 1px;
}
.desc-box {
    background: linear-gradient(120deg, #185a9d 60%, #43cea2 100%);
    border-radius: 1rem;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    color: #f5f6fa;
    box-shadow: 0 2px 8px #0002;
    font-size: 1.15rem;
    text-align: center;
}
.feature-badge {
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
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
with st.container():
    st.markdown(
        """
        <div style='padding: 2.5rem 0 1.5rem 0; border-radius: 1.5rem; text-align: center; margin-bottom: 2rem; box-shadow: 0 4px 24px rgba(0,0,0,0.15); background: #23272f;'>
            <!--<img src='https://img.icons8.com/ios-filled/250/ffffff/idea.png' width='120' style='margin-bottom: 1rem;' />-->
            <div class='big-title'>JIIT <br>Question Paper Generator</div>
            <div class='subtitle'>Empowering Educators with Seamless AI-driven Content Creation</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- LOTTIE ANIMATIONS ---
lottie_hero = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_2ks3pjua.json")
lottie_rocket = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_FYx0Ph.json")
lottie_celebrate = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_ikk4jhps.json")

st.markdown("<hr style='border:1px solid #43cea2;margin:2em 0;'>", unsafe_allow_html=True)

with st.container():
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("""
        <div style='text-align:center; padding-top:1.5rem;'>
            <span style='font-size:2.3rem; font-weight:700; color:#43cea2; letter-spacing:1px;'>
                Unlock the Power of Effortless Question Paper Creation
            </span><br>
            <span style='font-size:1.15rem; color:#e0f7fa; opacity:0.85;'>
                AI-driven, fast, and designed for educators like you.
            </span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='desc-box'>
        <span style='font-size:1.3rem;font-weight:600;'>Welcome to the <b>Question Paper Generator</b>!</span><br><br>
        <span style='font-size:1.1rem;'>Revolutionize your workflow with AI-powered OCR, smart text refinement, and effortless content export.<br><br></span>
        <span class='feature-badge'>PDF & Image Upload</span>
        <span class='feature-badge'>Automatic OCR</span>
        <span class='feature-badge'>Text Refinement</span>
        <span class='feature-badge'>Download Results</span>
        <br><br>
        <i>Switch to the next page to get started!</i>
        </div>
        """, unsafe_allow_html=True)

# --- PROJECTS / HIGHLIGHTS ---
with st.container():
    st.write("")
    st.markdown("<hr style='border:1px solid #43cea2;margin:2em 0;'>", unsafe_allow_html=True)
    st.header("🚀 Why use our platform?")
    left, right = st.columns([2, 1])
    with left:
        st.markdown("""
        <div class='soft-card'>
        <b>✨ Fast, Accurate, and User-Friendly</b><br>
        Our platform is designed for educators and students who want to save time and get high-quality results. Upload your study material, let our AI do the heavy lifting, and download ready-to-use content in seconds.<br><br>
        <b>🎯 Use Cases:</b><br>
        • Generate question banks from textbooks<br>
        • Digitize handwritten notes<br>
        • Prepare quizzes and assignments<br>
        • And much more!<br>
        </div>
        """, unsafe_allow_html=True)
    with right:
        if lottie_rocket:
            try:
                from streamlit_lottie import st_lottie
                st_lottie(lottie_rocket, height=180, key="rocket_anim")
            except Exception:
                st.image("https://img.icons8.com/ios-filled/250/43cea2/idea.png", width=120)
        else:
            st.image("https://img.icons8.com/ios-filled/250/43cea2/idea.png", width=120)

# --- CELEBRATE / CTA ---
with st.container():
    st.markdown("<hr style='border:1px solid #185a9d;margin:2em 0;'>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    with c1:
        if lottie_celebrate:
            try:
                from streamlit_lottie import st_lottie
                st_lottie(lottie_celebrate, height=140, key="celebrate_anim")
            except Exception:
                st.image("https://img.icons8.com/ios-filled/250/43cea2/idea.png", width=120)
        else:
            st.image("https://img.icons8.com/ios-filled/250/43cea2/idea.png", width=120)
    with c2:
        st.markdown("""
        <div class='soft-card'>
        <span style='font-size:1.15rem;'>Ready to experience the future of educational content creation?</span><br>
        <b>Head to the next page and try it out now!</b>
        </div>
        """, unsafe_allow_html=True)
