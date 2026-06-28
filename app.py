"""
Aplikasi Deteksi Spam SMS - Streamlit
Dataset: SMS Spam Collection (5572 SMS)
"""

import streamlit as st
import joblib
import re
import string

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="SpamShield", page_icon="🛡️", layout="centered")

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); }
    #MainMenu, footer, header { visibility: hidden; }
    .hero { text-align: center; padding: 30px 20px 10px 20px; }
    .hero-title {
        font-size: 40px; font-weight: 800;
        background: linear-gradient(90deg, #00d4ff, #7b2ff7);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero-sub { color: #8892b0; font-size: 15px; margin-bottom: 20px; }
    .result-spam {
        background: rgba(255,75,75,0.1); border: 1px solid rgba(255,75,75,0.4);
        border-radius: 16px; padding: 25px; text-align: center; margin-top: 20px;
    }
    .result-ham {
        background: rgba(0,212,100,0.1); border: 1px solid rgba(0,212,100,0.4);
        border-radius: 16px; padding: 25px; text-align: center; margin-top: 20px;
    }
    .result-title { font-size: 26px; font-weight: 800; margin-bottom: 8px; }
    .result-prob { font-size: 44px; font-weight: 900; margin: 8px 0; }
    .stats-container { display: flex; gap: 12px; margin-bottom: 20px; }
    .stat-card {
        flex: 1; background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px; padding: 12px; text-align: center;
    }
    .stat-value { font-size: 20px; font-weight: 800; color: #00d4ff; }
    .stat-label { font-size: 11px; color: #8892b0; margin-top: 4px; }
    .stTextArea textarea {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important; color: white !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #00d4ff, #7b2ff7) !important;
        color: white !important; border: none !important;
        border-radius: 12px !important; font-weight: 700 !important;
        font-size: 15px !important; width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

# ─── Preprocessing ───────────────────────────────────────────────────────────
def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div style="font-size:56px">🛡️</div>
    <div class="hero-title">SpamShield</div>
    <div class="hero-sub">Deteksi spam SMS menggunakan Machine Learning & NLP</div>
</div>
""", unsafe_allow_html=True)

# ─── Stats ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stats-container">
    <div class="stat-card"><div class="stat-value">5.572</div><div class="stat-label">Data Training</div></div>
    <div class="stat-card"><div class="stat-value">97%</div><div class="stat-label">Akurasi</div></div>
    <div class="stat-card"><div class="stat-value">TF-IDF</div><div class="stat-label">Representasi</div></div>
    <div class="stat-card"><div class="stat-value">LR</div><div class="stat-label">Model</div></div>
</div>
""", unsafe_allow_html=True)

# ─── Contoh Cepat ────────────────────────────────────────────────────────────
if 'teks_input' not in st.session_state:
    st.session_state['teks_input'] = ""

st.markdown("**💡 Coba contoh:**")
col1, col2 = st.columns(2)
with col1:
    if st.button("📛 Contoh Spam"):
        st.session_state['teks_input'] = "WINNER! You have been selected to receive a $1000 prize. Call now to claim your reward!"
with col2:
    if st.button("✅ Contoh Normal"):
        st.session_state['teks_input'] = "Hey, are you coming to the meeting tomorrow at 9am?"

# ─── Input ───────────────────────────────────────────────────────────────────
st.markdown("#### 📨 Masukkan Teks SMS")
teks_input = st.text_area(
    label="",
    placeholder="Ketik atau paste teks SMS di sini...",
    height=120,
    value=st.session_state['teks_input']
)

# ─── Prediksi ────────────────────────────────────────────────────────────────
if st.button("🔍 DETEKSI SEKARANG"):
    if not teks_input.strip():
        st.warning("Masukkan teks SMS terlebih dahulu!")
    else:
        teks_bersih = preprocess(teks_input)
        teks_tfidf = vectorizer.transform([teks_bersih])
        prediksi = model.predict(teks_tfidf)[0]
        probabilitas = model.predict_proba(teks_tfidf)[0]

        if prediksi == 'spam':
            prob_pct = f"{probabilitas[1]*100:.1f}%"
            st.markdown(f"""
            <div class="result-spam">
                <div class="result-title">⚠️ TERDETEKSI SPAM!</div>
                <div class="result-prob" style="color:#ff4b4b">{prob_pct}</div>
                <div style="color:#8892b0;font-size:13px">Probabilitas pesan ini adalah spam</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            prob_pct = f"{probabilitas[0]*100:.1f}%"
            st.markdown(f"""
            <div class="result-ham">
                <div class="result-title">✅ PESAN NORMAL</div>
                <div class="result-prob" style="color:#00d464">{prob_pct}</div>
                <div style="color:#8892b0;font-size:13px">Probabilitas pesan ini adalah normal</div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("🔬 Detail Preprocessing"):
            st.write(f"**Teks asli:** {teks_input}")
            st.write(f"**Setelah preprocessing:** {teks_bersih}")

# ─── Info ─────────────────────────────────────────────────────────────────────
with st.expander("ℹ️ Tentang Model"):
    st.markdown("""
    **Task:** Klasifikasi teks — Spam vs Ham  
    **Dataset:** SMS Spam Collection — 5.572 SMS real  
    **Preprocessing:** Case folding, hapus URL, hapus angka, hapus tanda baca  
    **Representasi fitur:** TF-IDF dengan bigram (ngram 1-2), max 5000 fitur  
    **Model:** Logistic Regression  
    **Akurasi:** 97.13% | **F1 Score (Spam):** 0.88  
    """)
