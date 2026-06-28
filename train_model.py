"""
Train Model - Deteksi Spam SMS Bahasa Indonesia
Menggunakan TF-IDF + Logistic Regression
"""

import pandas as pd
import numpy as np
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ─── 1. Dataset ───────────────────────────────────────────────────────────────
data = {
    'teks': [
        # SPAM
        "Selamat! Anda memenangkan hadiah uang tunai Rp 50.000.000! Hubungi kami segera di 0812-XXXX",
        "GRATIS pulsa Rp 100.000 untuk 100 pengguna pertama! Daftar sekarang klik link ini",
        "Anda terpilih sebagai pemenang undian berhadiah! Kirim data diri ke nomor ini",
        "Pinjaman cepat tanpa jaminan! Cair dalam 1 jam. Hubungi WA 0813-XXXX sekarang",
        "Promo spesial! Beli 1 gratis 1 hanya hari ini. Klik www.promo-palsu.com",
        "URGENT: Akun Anda akan diblokir! Verifikasi sekarang di link berikut",
        "Dapatkan iPhone 15 gratis! Anda adalah pengunjung ke-1.000.000. Klaim sekarang",
        "Transfer dana BRI Anda telah gagal. Konfirmasi data di sini untuk mengaktifkan",
        "Investasi crypto dijamin untung 200% per bulan! Modal minimal Rp 100.000",
        "Anda menang cashback Rp 500.000! Masukkan kode OTP Anda untuk pencairan",
        "Halo, saya dari bank BCA. Kartu kredit Anda bermasalah. Berikan nomor kartu Anda",
        "Diskon 90% untuk semua produk! Belanja sekarang sebelum kehabisan. Link di bio",
        "SELAMAT anda mendapatkan bonus dari telkomsel sebesar Rp200rb klik link berikut",
        "Pinjaman online langsung cair, bunga 0%, tanpa survei. Hubungi admin kami",
        "Anda dipilih untuk program loyalitas kami. Hadiah Rp 10 juta menanti Anda!",
        "Flash sale 12.12! Semua produk diskon gila-gilaan. Daftar member gratis sekarang",
        "Nomor Anda terdaftar sebagai pemenang. Hubungi customer service kami segera",
        "Raih penghasilan Rp 5 juta per hari dari rumah! Tanpa modal, tanpa risiko",
        "Verifikasi akun GoPay Anda segera atau akan dinonaktifkan dalam 24 jam",
        "Obat kuat pria, terbukti ampuh! Pesan sekarang gratis ongkir ke seluruh Indonesia",
        # HAM (normal)
        "Hei, kamu jadi ikut rapat besok pagi kan? Jam 9 di kantor",
        "Tolong belikan saya makan siang nasi padang ya, thanks",
        "Selamat ulang tahun! Semoga panjang umur dan sehat selalu",
        "Besok libur nasional, jadi kita tidak ada kuliah ya",
        "Kamu sudah mengerjakan PR matematika belum? Susah banget soalnya",
        "Ibu, saya pulang malam ya. Ada kegiatan organisasi dulu",
        "Ayah, transfer uang bulanan sudah masuk belum? Mau bayar kos",
        "Meeting jam 3 sore dibatalkan, nanti dijadwal ulang",
        "Makasih ya sudah bantu kemarin, kamu baik banget",
        "Kita jadi nonton film bareng sabtu malam? Pilih bioskop mana?",
        "Laporan sudah saya kirim ke email bapak tadi pagi",
        "Tolong jemput saya di stasiun jam 7 malam ya",
        "Buku yang kamu pinjam minggu lalu bisa dikembalikan besok?",
        "Gimana kabarnya? Lama tidak ketemu, kangen nih",
        "Sudah sampai rumah dengan selamat, makasih sudah antar",
        "Besok ada ulangan kimia, belajar bareng yuk malam ini",
        "Kamu tahu tidak resto baru yang enak di dekat kampus?",
        "Tugas kelompok dikumpul hari jumat, jangan lupa ya",
        "Saya sedang di jalan, bentar lagi sampai. Tunggu sebentar",
        "Selamat atas kelulusanmu! Bangga banget sama kamu",
    ],
    'label': ['spam']*20 + ['ham']*20
}

df = pd.DataFrame(data)
print(f"Dataset shape: {df.shape}")
print(f"Distribusi label:\n{df['label'].value_counts()}")

# ─── 2. Preprocessing ─────────────────────────────────────────────────────────
def preprocess(text):
    text = text.lower()                                         # Case folding
    text = re.sub(r'http\S+|www\S+', '', text)                 # Hapus URL
    text = re.sub(r'[0-9]+', '', text)                         # Hapus angka
    text = text.translate(str.maketrans('', '', string.punctuation))  # Hapus tanda baca
    text = ' '.join(text.split())                              # Hapus spasi berlebih
    return text

df['teks_bersih'] = df['teks'].apply(preprocess)
print("\nContoh preprocessing:")
print(df[['teks', 'teks_bersih']].head(3).to_string())

# ─── 3. Split Data ────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df['teks_bersih'], df['label'],
    test_size=0.2, random_state=42, stratify=df['label']
)
print(f"\nData latih: {len(X_train)}, Data uji: {len(X_test)}")

# ─── 4. Ekstraksi Fitur TF-IDF ───────────────────────────────────────────────
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)
print(f"Ukuran vocabulary TF-IDF: {len(vectorizer.vocabulary_)}")

# ─── 5. Training & Evaluasi ───────────────────────────────────────────────────
# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
nb_pred = nb.predict(X_test_tfidf)
print(f"\nNaive Bayes     → Accuracy: {accuracy_score(y_test, nb_pred):.4f} | F1: {f1_score(y_test, nb_pred, pos_label='spam'):.4f}")

# Logistic Regression
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_tfidf, y_train)
lr_pred = lr.predict(X_test_tfidf)
print(f"Logistic Regression → Accuracy: {accuracy_score(y_test, lr_pred):.4f} | F1: {f1_score(y_test, lr_pred, pos_label='spam'):.4f}")

# Pilih Logistic Regression sebagai model final
print("\n=== Evaluasi Model Final (Logistic Regression) ===")
print(classification_report(y_test, lr_pred, target_names=['Ham (Normal)', 'Spam']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, lr_pred))

# ─── 6. Simpan Model & Vectorizer ────────────────────────────────────────────
joblib.dump(lr, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("\nModel disimpan ke model.pkl")
print("Vectorizer disimpan ke vectorizer.pkl")
