# 🍷 Sistem Deteksi Kualitas White Wine Berbasis SVM Optimized

Proyek ini dibangun untuk memenuhi kriteria penilaian UAS mata kuliah Machine Learning. Sistem mengimplementasikan algoritma **Support Vector Machine (SVM)** untuk memprediksi apakah sampel White Wine masuk ke dalam kategori *Premium Quality* (skor $\ge 6$) atau *Normal Quality* (skor $< 6$) berlandaskan parameter fisikokimiawi lab.

## 👤 Identitas Mahasiswa
- **Nama:** Ahmad Naufal Zakirin
- **NIM:** A11.2024.15758
- **Program Studi:** TI - Universitas Dian Nuswantoro (UDINUS)

## 📁 Cakupan Struktur Repositori
- `data/`: Berisi dataset mentah CSV, berkas repositori asal, dan kamus data fitur.
- `notebooks/`: Proses eksplorasi (EDA), penanganan data penyeimbang, dan proses hyperparameter tuning model.
- `src/`: Berkas skrip pemrosesan Python modular (`train.py`, `predict.py`, dsb).
- `models/`: Penyimpanan berkas biner `.joblib` (Model SVM & Pipeline Scaler).
- `app_streamlit.py`: Aplikasi dasbor berbasis web interaktif utama.
- `app_gradio.py`: Dokumentasi GUI alternatif berbasis komponen web Gradio.

----------------------------------------------------------------------------------------------------------
## 🚀 Cara Menjalankan Aplikasi

### 1. Prasyarat (Prerequisites)
Pastikan Anda sudah menginstal **Python (versi 3.9 atau yang lebih baru)** di sistem Anda.

### 2. Instalasi Paket Dependensi
Buka terminal atau command prompt pada direktori utama proyek (`UAS-ML-Wine/`), kemudian jalankan perintah berikut untuk menginstal seluruh pustaka yang diperlukan:
```bash

3. Ekstraksi / Pelatihan Ulang Model (Opsional)
Jika berkas model .joblib belum tersedia di folder models/ atau Anda ingin melatih ulang model menggunakan skrip modular, jalankan perintah:
Bash
python src/train.py
4. Menjalankan Aplikasi Dasbor Utama (Streamlit)
Untuk membuka antarmuka GUI interaktif utama yang berbasis Web Streamlit, jalankan perintah berikut di terminal:
Bash
streamlit run app_streamlit.py


-------------------------------------------------------------------------------------------
### Cara Menjalankannya api_fastapi.py
Pastikan paket fastapi dan uvicorn sudah terinstal (bisa tambahkan ke requirements.txt):

Bash
pip install fastapi uvicorn
Jalankan server FastAPI dengan perintah berikut di terminal:

Bash
uvicorn api_fastapi:app --reload
Buka browser dan akses http://127.0.0.1:8000/docs. Kamu akan langsung melihat halaman Swagger UI interaktif untuk menguji API-mu secara langsung.

Link Drive video presentasi : https://drive.google.com/drive/folders/1JkCoeRLYXB-ILBstupIQpDMD2zQH9bi-?usp=sharing