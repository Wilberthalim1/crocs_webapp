# Crocs Analytics Dashboard

Aplikasi web Flask untuk **UAS Manajemen Data Besar** & **Analitika Perilaku & Tren** —
analisis perilaku pengguna dan pola tren rating Crocs Official Store di Shopee.

> **Penyusun:** Wilbert Halim · NIM 03081230042 · Kelas 23SI2
> **Mata Kuliah:** Manajemen Data Besar — UPH Medan, Genap 2025/2026
> **Dosen:** Frans Mikael Sinaga, M.Kom.

---

## 🎯 Apa yang dibangun?

Aplikasi web ini **meng-embed seluruh algoritma yang dibangun di Google Colab**
(K-Means Clustering, Pearson Correlation, Min-Max Normalization, Silhouette Score,
EDA) ke dalam dashboard interaktif berbasis Flask yang bisa diakses lewat browser.

Project menggabungkan dua project Colab terdahulu:
- `Management_Big_Data.py` (UTS Manajemen Data Besar)
- `analitika_perilaku___tren.py` (UTS Analitika Perilaku & Tren)

Kedua project menggunakan dataset yang sama (`shopee.xlsx`) dan algoritma yang sama,
sehingga digabungkan menjadi satu aplikasi web yang lebih komprehensif.

## 📂 Struktur Project

```
crocs_webapp/
├── app.py                  # Flask main app + routes
├── analytics.py            # Semua algoritma (K-Means, Pearson, dll)
├── requirements.txt
├── README.md
├── data/
│   ├── shopee.xlsx         # Dataset mentah hasil web scraping
│   └── (output bersih dihasilkan saat run)
├── static/
│   └── img/                # Visualisasi matplotlib (PNG)
└── templates/
    ├── base.html           # Layout shared (navigation, footer)
    ├── dashboard.html      # Halaman utama
    ├── data.html           # Tabel dataset bersih + pipeline cleaning
    ├── eda.html            # Eksplorasi data
    ├── clustering.html     # K-Means + Elbow + Silhouette
    ├── correlation.html    # Pearson correlation
    └── about.html          # Identitas + daftar pustaka
```

## 🚀 Cara Menjalankan

### 1. Setup environment

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Run aplikasi

```bash
python app.py
```

Buka browser ke: **http://127.0.0.1:5000**

Saat startup, aplikasi akan otomatis:
- Memuat `data/shopee.xlsx`
- Menjalankan pipeline cleaning + K-Means + korelasi
- Menghasilkan visualisasi PNG ke `static/img/`
- Menyimpan output bersih ke `data/`

## 🌐 Deploy ke Public URL (untuk Dosen)

Beberapa opsi gratis:

### Opsi A — PythonAnywhere (paling cocok untuk Flask)
1. Daftar di pythonanywhere.com (free tier)
2. Upload semua file project ke `/home/USERNAME/crocs_webapp`
3. Buka tab "Web" → Add a new web app → Manual config → Python 3.11
4. Set source code path & WSGI file mengarah ke `app.py`
5. URL publik: `https://USERNAME.pythonanywhere.com`

### Opsi B — Render.com
1. Push project ke GitHub (public repo)
2. Daftar di render.com → New Web Service → connect repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `python app.py`
5. URL publik otomatis: `https://NAMAPROJECT.onrender.com`

### Opsi C — Quick demo dengan ngrok (sementara)
```bash
# Terminal 1
python app.py
# Terminal 2
ngrok http 5000
# Dapat URL public sementara, e.g. https://xxx.ngrok-free.app
```

## 🧠 Algoritma yang Diimplementasikan

| Tahap | Algoritma / Teknik | Library |
|-------|---------------------|---------|
| 1. Pengumpulan | Web Scraping | (manual via Shopee) |
| 2. Cleaning | String parsing, regex, type conversion | pandas |
| 3. Storage | Multi-format (CSV, XLSX, multi-sheet) | pandas, openpyxl |
| 4. Preprocessing | Min-Max Normalization | scikit-learn |
| 5. Cluster Selection | Elbow Method, Silhouette Score | scikit-learn |
| 6. Clustering | K-Means (K=3, K-Means++) | scikit-learn |
| 7. Correlation | Pearson Correlation Test | scipy |
| 8. EDA | Descriptive Stats, Segmentation | pandas |
| 9. Visualisasi | matplotlib + seaborn (static) | matplotlib, seaborn |
| 10. Frontend | Chart.js (interactive) | Chart.js 4 |

## 📊 Halaman Dashboard

- **/** — Dashboard overview dengan KPI dan visualisasi utama
- **/data** — Tabel 30 produk + dokumentasi pipeline cleaning
- **/eda** — Eksplorasi data: deskriptif, kategori, segmentasi
- **/clustering** — K-Means dengan elbow method, silhouette, scatter interaktif
- **/correlation** — Heatmap Pearson + analisis 4 pasangan variabel
- **/about** — Identitas project + daftar pustaka

## 📡 API Endpoints (JSON)

- `GET /api/summary` — KPI ringkasan
- `GET /api/products` — 30 produk lengkap dengan label cluster
- `GET /api/eda` — Hasil EDA
- `GET /api/kmeans` — Hasil K-Means + elbow
- `GET /api/correlation` — Matriks + pair-test korelasi

## 📥 Download File Output

Tersedia tombol download di halaman **/data** untuk:
- `shopee_crocs_clean.csv` — dataset bersih
- `shopee_crocs_processed.xlsx` — Excel multi-sheet
- `statistik_deskriptif.csv` — summary statistik

---

## 🔑 Hasil Kunci

- **30 produk** dianalisis, total **30,392 unit terjual**
- **Rata-rata rating 4.91/5** — kepuasan konsumen sangat tinggi
- **K-Means K=3** dengan Silhouette Score **0.542**
- **Distribusi cluster:** Regular (17), Premium (10), Bestseller (3)
- **Insight utama:** harga TIDAK signifikan menentukan terjual (r=0.009),
  brand & model klasik yang menggerakkan penjualan
