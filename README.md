# 🛡️ SpamShield ID - Deteksi Spam SMS Bahasa Indonesia

Aplikasi deteksi spam SMS berbahasa Indonesia menggunakan **Natural Language Processing** dan **Logistic Regression**.

## 📁 Struktur Proyek

```
nlp-spam-detection/
├── train_model.py     # Kode pelatihan model NLP
├── app.py             # Aplikasi web Streamlit
├── model.pkl          # Model Logistic Regression tersimpan
├── vectorizer.pkl     # TF-IDF Vectorizer tersimpan
├── requirements.txt   # Dependensi Python
└── README.md          # Dokumentasi proyek
```

## 📊 Dataset

- **Jumlah data:** 40 SMS (20 spam, 20 ham/normal)
- **Bahasa:** Indonesia
- **Label:** `spam` dan `ham`

## ⚙️ Cara Menjalankan

### 1. Clone repository
```bash
git clone https://github.com/USERNAME/nlp-spam-detection.git
cd nlp-spam-detection
```

### 2. Install dependensi
```bash
pip install -r requirements.txt
```

### 3. Latih model (opsional, model.pkl sudah tersedia)
```bash
python train_model.py
```

### 4. Jalankan aplikasi
```bash
streamlit run app.py
```

## 🔄 Pipeline NLP

1. **Case folding** — semua teks diubah ke huruf kecil
2. **Hapus URL** — link website dihapus
3. **Hapus angka** — karakter numerik dihapus
4. **Hapus tanda baca** — punctuation dihapus
5. **TF-IDF** — teks diubah ke representasi numerik dengan bigram
6. **Logistic Regression** — klasifikasi spam/ham

## 📈 Performa Model

| Model | Accuracy | F1 Score |
|---|---|---|
| Naive Bayes | 100% | 1.00 |
| Logistic Regression | 100% | 1.00 |

## 🚀 Deployment

Deploy menggunakan **Streamlit Community Cloud**.
