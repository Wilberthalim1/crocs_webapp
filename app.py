"""
============================================================
CROCS ANALYTICS DASHBOARD — Flask Web Application
Mata Kuliah : Manajemen Data Besar & Analitika Perilaku/Tren
Penyusun    : Wilbert Halim (03081230042)
============================================================

Aplikasi web ini meng-embed seluruh algoritma yang sudah
dibangun di Google Colab (K-Means, Pearson, Min-Max, EDA)
ke dalam dashboard interaktif berbasis Flask.

Cara jalan:
    pip install -r requirements.txt
    python app.py
    Buka http://127.0.0.1:5000 di browser
"""

import os
from flask import Flask, render_template, jsonify, send_from_directory

import analytics

app = Flask(__name__)

# ---------- HOT CACHE ----------
# Pipeline dijalankan sekali saat startup (data hanya 30 baris,
# jadi tidak perlu re-run setiap request).
RESULT = None


def _load():
    global RESULT
    if RESULT is None:
        RESULT = analytics.run_full_pipeline()
    return RESULT


# ---------- ROUTES UI ----------
@app.route("/")
def page_dashboard():
    r = _load()
    return render_template("dashboard.html", r=r, active="dashboard")


@app.route("/data")
def page_data():
    r = _load()
    return render_template("data.html", r=r, active="data")


@app.route("/eda")
def page_eda():
    r = _load()
    return render_template("eda.html", r=r, active="eda")


@app.route("/clustering")
def page_clustering():
    r = _load()
    return render_template("clustering.html", r=r, active="clustering")


@app.route("/correlation")
def page_correlation():
    r = _load()
    return render_template("correlation.html", r=r, active="correlation")


@app.route("/about")
def page_about():
    r = _load()
    return render_template("about.html", r=r, active="about")


# ---------- ROUTES API (untuk Chart.js) ----------
@app.route("/api/summary")
def api_summary():
    return jsonify(_load()["eda"]["summary"])


@app.route("/api/products")
def api_products():
    return jsonify(_load()["products"])


@app.route("/api/eda")
def api_eda():
    return jsonify(_load()["eda"])


@app.route("/api/kmeans")
def api_kmeans():
    return jsonify(_load()["kmeans"])


@app.route("/api/correlation")
def api_correlation():
    return jsonify(_load()["corr"])


# ---------- DOWNLOAD OUTPUT ----------
@app.route("/download/<path:filename>")
def download_file(filename):
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    return send_from_directory(data_dir, filename, as_attachment=True)


if __name__ == "__main__":
    print("=" * 60)
    print(" CROCS ANALYTICS DASHBOARD")
    print(" Manajemen Data Besar & Analitika Perilaku/Tren")
    print("=" * 60)
    print(" Memuat data + menjalankan pipeline...")
    _load()
    port = int(os.environ.get("PORT", 5000))
    print(f" Pipeline siap. Buka http://127.0.0.1:{port}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False)
