"""
Aplikasi Deteksi Spam SMS Bahasa Indonesia - Streamlit
"""

import streamlit as st
import joblib
import re
import string

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="SpamShield ID", page_icon="🛡️", layout="centered")

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
    <div class="hero-title">SpamShield ID</div>
    <div class="hero-sub">Deteksi spam SMS Bahasa Indonesia menggunakan Machine Learning & NLP</div>
</div>
""", unsafe_allow_html=True)

# ─── Input ───────────────────────────────────────────────────────────────────
st.markdown("#### 📨 Masukkan Teks SMS")
teks_input = st.text_area(
    label="",
    placeholder="Contoh: Selamat! Anda memenangkan hadiah Rp 50 juta...",
    height=120
)

# ─── Contoh Cepat ────────────────────────────────────────────────────────────
st.markdown("**💡 Coba contoh:**")
col1, col2 = st.columns(2)
with col1:
    if st.button("📛 Contoh Spam"):
        teks_input = "Selamat! Anda memenangkan hadiah Rp 50 juta! Hubungi kami segera dan klaim hadiah Anda sekarang!"
        st.session_state['teks'] = teks_input
with col2:
    if st.button("✅ Contoh Normal"):
        teks_input = "Hei, kamu jadi ikut rapat besok pagi kan? Jam 9 di kantor ya"
        st.session_state['teks'] = teks_input

if 'teks' in st.session_state and not teks_input:
    teks_input = st.session_state['teks']

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
    **Task:** Klasifikasi teks — Spam vs Ham (Normal)  
    **Dataset:** 40 SMS Bahasa Indonesia (20 spam, 20 ham)  
    **Preprocessing:** Case folding, hapus URL, hapus angka, hapus tanda baca  
    **Representasi fitur:** TF-IDF dengan bigram (ngram 1-2)  
    **Model:** Logistic Regression  
    **Akurasi:** 100% | **F1 Score:** 1.00  
    """)
