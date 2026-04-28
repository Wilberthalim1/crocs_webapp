"""
============================================================
ANALYTICS MODULE - CROCS OFFICIAL STORE SHOPEE
Mata Kuliah: Manajemen Data Besar & Analitika Perilaku/Tren
============================================================

Modul ini meng-embed seluruh algoritma yang sudah dibangun di
Google Colab (K-Means Clustering, Korelasi Pearson, Min-Max
Normalization, Silhouette Score, EDA) menjadi fungsi-fungsi
yang bisa dipanggil dari Flask web app.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend non-GUI untuk Flask
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")


# ---------- KONFIGURASI ----------
DATA_PATH   = os.path.join(os.path.dirname(__file__), "data", "shopee.xlsx")
IMG_DIR     = os.path.join(os.path.dirname(__file__), "static", "img")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "data")

# Palette Crocs-ish (tealish/charcoal) tapi professional
PALETTE = {
    "primary":   "#0E7C7B",   # crocs teal
    "secondary": "#F26A1F",   # accent oranye
    "accent":    "#1B4965",   # navy
    "warn":      "#E63946",
    "muted":     "#6C757D",
    "bg":        "#FFFFFF",
    "ink":       "#0B1B2B",
}
CHART_COLORS = ["#0E7C7B", "#F26A1F", "#1B4965", "#E63946", "#9C6644", "#5C677D"]


# ================================================================
# TAHAP 1 + 2: LOADING + CLEANING
# ================================================================
def load_and_clean():
    """Baca shopee.xlsx, lakukan cleaning sesuai pipeline Colab."""
    df_raw = pd.read_excel(DATA_PATH)

    df = df_raw.copy()
    df.columns = ["url_produk", "url_gambar", "nama_produk",
                  "harga", "diskon", "promo", "rating", "jumlah_terjual"]

    # Cleaning harga: hapus titik ribuan -> numerik
    df["harga_clean"] = (df["harga"].astype(str)
                         .str.replace(".", "", regex=False)
                         .str.replace(",", "", regex=False))
    df["harga_clean"] = pd.to_numeric(df["harga_clean"], errors="coerce")

    # Cleaning diskon
    df["diskon_clean"] = (df["diskon"].astype(str)
                          .str.replace("%", "")
                          .str.replace("-", "")
                          .str.strip())
    df["diskon_clean"] = pd.to_numeric(df["diskon_clean"], errors="coerce").fillna(0)

    # Cleaning rating
    df["rating_clean"] = pd.to_numeric(df["rating"], errors="coerce")

    # Cleaning jumlah terjual: "5RB+" -> 5000
    def parse_terjual(val):
        if pd.isna(val) or str(val).strip() == "nan":
            return 0
        v = str(val).strip().lower().replace(" terjual", "").replace("+", "")
        if "rb" in v:
            try:
                return int(float(v.replace("rb", "").strip()) * 1000)
            except ValueError:
                return 0
        try:
            return int(float(v))
        except ValueError:
            return 0
    df["terjual_clean"] = df["jumlah_terjual"].apply(parse_terjual)

    # Ekstraksi kategori dari nama produk
    def ekstrak_kategori(nama):
        n = str(nama).lower()
        if "kids" in n or "children" in n: return "Kids"
        if "women" in n:                   return "Women"
        if "men slide" in n:               return "Men Slide"
        if "jibbitz" in n:                 return "Accessories"
        return "Unisex"
    df["kategori"] = df["nama_produk"].apply(ekstrak_kategori)

    # Ekstraksi warna
    def ekstrak_warna(nama):
        s = str(nama)
        return s.split(" - ")[-1].strip() if " - " in s else "Unknown"
    df["warna"] = df["nama_produk"].apply(ekstrak_warna)

    df["harga_setelah_diskon"] = df["harga_clean"] * (1 - df["diskon_clean"] / 100)
    df["ada_promo"] = df["promo"].apply(
        lambda x: 0 if pd.isna(x) or str(x).strip() == "nan" else 1)

    df_clean = df[["nama_produk", "kategori", "warna", "harga_clean", "diskon_clean",
                   "harga_setelah_diskon", "ada_promo", "rating_clean",
                   "terjual_clean", "url_gambar", "url_produk"]].copy()
    df_clean.columns = ["nama_produk", "kategori", "warna", "harga", "diskon_pct",
                        "harga_diskon", "ada_promo", "rating", "terjual",
                        "url_gambar", "url_produk"]

    return df_raw, df_clean


# ================================================================
# TAHAP 3: PENYIMPANAN
# ================================================================
def save_clean_outputs(df_raw, df_clean):
    """Simpan output bersih ke CSV + Excel multi-sheet."""
    csv_path  = os.path.join(OUTPUT_DIR, "shopee_crocs_clean.csv")
    xlsx_path = os.path.join(OUTPUT_DIR, "shopee_crocs_processed.xlsx")
    stat_path = os.path.join(OUTPUT_DIR, "statistik_deskriptif.csv")

    df_clean.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        df_raw.to_excel(w, sheet_name="Data_Mentah", index=False)
        df_clean.to_excel(w, sheet_name="Data_Bersih", index=False)
    df_clean[["harga", "diskon_pct", "harga_diskon", "rating", "terjual"]] \
        .describe().to_csv(stat_path)

    return {"csv": csv_path, "xlsx": xlsx_path, "stat": stat_path}


# ================================================================
# TAHAP 4: PENGOLAHAN (NORMALISASI + ELBOW + K-MEANS)
# ================================================================
def run_kmeans_pipeline(df_clean):
    """
    1) Min-Max normalize fitur [harga, rating, terjual]
    2) Elbow method K=2..7 -> inertia + silhouette
    3) K-Means dengan K=3
    4) Beri label cluster: Bestseller / Premium / Regular
    """
    fitur = ["harga", "rating", "terjual"]
    scaler = MinMaxScaler()
    df_norm = pd.DataFrame(scaler.fit_transform(df_clean[fitur]),
                           columns=[f"{f}_norm" for f in fitur])

    # Elbow method
    K_range = list(range(2, 8))
    inertia, silhouette = [], []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(df_norm)
        inertia.append(round(km.inertia_, 4))
        silhouette.append(round(silhouette_score(df_norm, km.labels_), 4))

    # Final K=3
    kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_clean = df_clean.copy()
    df_clean["cluster"] = kmeans_final.fit_predict(df_norm)

    cluster_stats = df_clean.groupby("cluster")[["harga", "rating", "terjual"]].mean()

    def label_cluster(c):
        avg_terjual = cluster_stats.loc[c, "terjual"]
        avg_harga   = cluster_stats.loc[c, "harga"]
        if avg_terjual == cluster_stats["terjual"].max(): return "Bestseller"
        if avg_harga   == cluster_stats["harga"].max():   return "Premium"
        return "Regular"
    df_clean["cluster_label"] = df_clean["cluster"].apply(label_cluster)

    return df_clean, {
        "K_range": K_range,
        "inertia": inertia,
        "silhouette": silhouette,
        "K_optimal": 3,
        "silhouette_K3": silhouette[K_range.index(3)],
        "cluster_stats": cluster_stats.round(2).reset_index().to_dict(orient="records"),
        "distribusi": df_clean["cluster_label"].value_counts().to_dict(),
    }


# ================================================================
# TAHAP 5: KORELASI PEARSON + EDA
# ================================================================
def run_correlation_analysis(df_clean):
    corr_mat = df_clean[["harga", "diskon_pct", "rating", "terjual"]].corr()

    pairs = [
        ("harga",      "terjual"),
        ("rating",     "terjual"),
        ("diskon_pct", "terjual"),
        ("harga",      "rating"),
    ]
    pair_results = []
    for a, b in pairs:
        r, p = stats.pearsonr(df_clean[a], df_clean[b])
        pair_results.append({
            "var_a": a, "var_b": b,
            "r": round(r, 3), "p_value": round(p, 4),
            "interpretasi": _interpret_corr(r, p),
        })
    return {
        "matrix": corr_mat.round(3).to_dict(),
        "pairs":  pair_results,
    }


def _interpret_corr(r, p):
    arah = "positif" if r > 0 else "negatif"
    kekuatan = ("sangat lemah" if abs(r) < 0.2 else
                "lemah"        if abs(r) < 0.4 else
                "sedang"       if abs(r) < 0.6 else
                "kuat"         if abs(r) < 0.8 else "sangat kuat")
    sig = "signifikan" if p < 0.05 else "tidak signifikan"
    return f"Korelasi {kekuatan} {arah}, secara statistik {sig}"


def run_eda(df_clean):
    """Eksplorasi data lengkap untuk dashboard."""
    # Statistik deskriptif
    desc = df_clean[["harga", "diskon_pct", "rating", "terjual"]].describe().round(2)

    # Distribusi rating
    rating_dist = df_clean["rating"].value_counts().sort_index()
    rating_dist = {str(k): int(v) for k, v in rating_dist.items()}

    # Per kategori
    per_kategori = (df_clean.groupby("kategori")
                    .agg(jumlah_produk=("nama_produk", "count"),
                         rata_harga=("harga", "mean"),
                         rata_rating=("rating", "mean"),
                         total_terjual=("terjual", "sum"),
                         rata_terjual=("terjual", "mean"))
                    .round(2)
                    .sort_values("total_terjual", ascending=False)
                    .reset_index())

    # Segmentasi harga
    df_clean = df_clean.copy()
    df_clean["segmen_harga"] = pd.cut(
        df_clean["harga"],
        bins=[0, 500_000, 1_000_000, 1_500_000, float("inf")],
        labels=["Budget (<500K)", "Mid (500K-1Jt)",
                "Upper-Mid (1-1.5Jt)", "Premium (>1.5Jt)"])
    seg_harga = (df_clean.groupby("segmen_harga", observed=True)
                 .agg(jumlah_produk=("nama_produk", "count"),
                      rata_terjual=("terjual", "mean"),
                      total_terjual=("terjual", "sum"))
                 .round(2).reset_index())

    # Segmentasi popularitas
    df_clean["popularitas"] = pd.cut(
        df_clean["terjual"],
        bins=[-1, 100, 500, 1000, float("inf")],
        labels=["Rendah (<100)", "Sedang (100-500)",
                "Tinggi (500-1K)", "Sangat Tinggi (>1K)"])
    pop = df_clean["popularitas"].value_counts().reindex(
        ["Rendah (<100)", "Sedang (100-500)",
         "Tinggi (500-1K)", "Sangat Tinggi (>1K)"], fill_value=0)

    # Promo
    promo_grp = df_clean.groupby("ada_promo")["terjual"].agg(["mean", "sum", "count"]).round(2)
    promo_dict = {}
    for k in promo_grp.index:
        label = "Ada Promo" if k == 1 else "Tanpa Promo"
        promo_dict[label] = {
            "rata_terjual": float(promo_grp.loc[k, "mean"]),
            "total_terjual": int(promo_grp.loc[k, "sum"]),
            "jumlah_produk": int(promo_grp.loc[k, "count"]),
        }

    return {
        "deskriptif": desc.to_dict(),
        "rating_dist": rating_dist,
        "per_kategori": per_kategori.to_dict(orient="records"),
        "segmen_harga": seg_harga.to_dict(orient="records"),
        "popularitas": {str(k): int(v) for k, v in pop.items()},
        "promo": promo_dict,
        "summary": {
            "total_produk": int(len(df_clean)),
            "total_terjual": int(df_clean["terjual"].sum()),
            "rata_harga": round(float(df_clean["harga"].mean()), 0),
            "rata_rating": round(float(df_clean["rating"].mean()), 3),
            "produk_promo": int(df_clean["ada_promo"].sum()),
            "produk_tertinggi_terjual": str(
                df_clean.loc[df_clean["terjual"].idxmax(), "nama_produk"]),
            "produk_termahal": str(
                df_clean.loc[df_clean["harga"].idxmax(), "nama_produk"]),
        },
    }


# ================================================================
# TAHAP 6: VISUALISASI (matplotlib -> PNG di static/img)
# ================================================================
def generate_all_charts(df_clean, kmeans_meta):
    """Hasilkan PNG untuk dashboard. File disimpan di static/img/."""
    os.makedirs(IMG_DIR, exist_ok=True)
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["font.family"] = "DejaVu Sans"

    _chart_dashboard(df_clean, kmeans_meta)
    _chart_clustering(df_clean)
    _chart_popularitas(df_clean)
    _chart_eda(df_clean)


def _chart_dashboard(df_clean, kmeans_meta):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Dashboard Analitika Shopee — Crocs Official Store",
                 fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    rc = df_clean["rating"].value_counts().sort_index()
    bars = ax.bar(rc.index.astype(str), rc.values,
                  color=CHART_COLORS[0], edgecolor="white")
    ax.set_title("Distribusi Rating Produk", fontweight="bold")
    ax.set_xlabel("Rating"); ax.set_ylabel("Jumlah Produk")
    for b, v in zip(bars, rc.values):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.1, str(v),
                ha="center", fontweight="bold")

    ax = axes[0, 1]
    kc = df_clean["kategori"].value_counts()
    ax.pie(kc.values, labels=kc.index, autopct="%1.1f%%",
           colors=CHART_COLORS[:len(kc)], startangle=90)
    ax.set_title("Proporsi Produk per Kategori", fontweight="bold")

    ax = axes[0, 2]
    top10 = df_clean.nlargest(10, "terjual")
    nama_pendek = [n[:28]+"..." if len(n) > 28 else n for n in top10["nama_produk"]]
    bars = ax.barh(range(len(top10)), top10["terjual"].values,
                   color=CHART_COLORS[1])
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(nama_pendek, fontsize=7)
    ax.set_title("Top 10 Produk Terlaris", fontweight="bold")
    ax.set_xlabel("Jumlah Terjual")
    ax.invert_yaxis()

    ax = axes[1, 0]
    scatter_colors = {"Bestseller": CHART_COLORS[1],
                      "Premium": CHART_COLORS[3],
                      "Regular": CHART_COLORS[0]}
    for label, grp in df_clean.groupby("cluster_label"):
        ax.scatter(grp["harga"], grp["terjual"], label=label,
                   c=scatter_colors[label], s=80, alpha=0.85,
                   edgecolors="white")
    ax.set_title("Scatter: Harga vs Jumlah Terjual", fontweight="bold")
    ax.set_xlabel("Harga (Rp)"); ax.set_ylabel("Jumlah Terjual")
    ax.legend()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}Jt"))

    ax = axes[1, 1]
    kt = df_clean.groupby("kategori")["terjual"].mean().sort_values(ascending=False)
    bars = ax.bar(kt.index, kt.values,
                  color=CHART_COLORS[:len(kt)], edgecolor="white")
    ax.set_title("Rata-rata Terjual per Kategori", fontweight="bold")
    ax.set_xlabel("Kategori"); ax.set_ylabel("Rata-rata Terjual")
    ax.tick_params(axis="x", rotation=15)
    for b, v in zip(bars, kt.values):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+30,
                f"{v:,.0f}", ha="center", fontweight="bold", fontsize=8)

    ax = axes[1, 2]
    ax2 = ax.twinx()
    K = kmeans_meta["K_range"]
    ax.plot(K, kmeans_meta["inertia"], "o-", color=CHART_COLORS[2],
            label="Inertia", linewidth=2)
    ax2.plot(K, kmeans_meta["silhouette"], "s-", color=CHART_COLORS[3],
             label="Silhouette", linewidth=2)
    ax.set_title("Elbow Method — Penentuan K Optimal", fontweight="bold")
    ax.set_xlabel("Jumlah Cluster (K)")
    ax.set_ylabel("Inertia", color=CHART_COLORS[2])
    ax2.set_ylabel("Silhouette Score", color=CHART_COLORS[3])
    ax.axvline(x=3, color="green", linestyle="--", alpha=0.7)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "viz_1_dashboard.png"),
                dpi=130, bbox_inches="tight")
    plt.close()


def _chart_clustering(df_clean):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Analisis Clustering & Korelasi — Crocs Official Store",
                 fontsize=14, fontweight="bold")

    ax = axes[0]
    cmap = {0: CHART_COLORS[0], 1: CHART_COLORS[1], 2: CHART_COLORS[2]}
    label_map = dict(zip(df_clean["cluster"], df_clean["cluster_label"]))
    for c_id in df_clean["cluster"].unique():
        mask = df_clean["cluster"] == c_id
        ax.scatter(df_clean[mask]["harga"], df_clean[mask]["terjual"],
                   c=cmap[c_id], s=110,
                   label=f"Cluster {c_id}: {label_map[c_id]}",
                   alpha=0.85, edgecolors="black", linewidths=0.5)
    ax.set_title("K-Means Clustering (K=3)\nHarga vs Jumlah Terjual",
                 fontweight="bold")
    ax.set_xlabel("Harga (Rp)"); ax.set_ylabel("Jumlah Terjual")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}Jt"))

    ax = axes[1]
    corr = df_clean[["harga", "diskon_pct", "rating", "terjual"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                vmin=-1, vmax=1, ax=ax, square=True, linewidths=0.5,
                xticklabels=["Harga", "Diskon%", "Rating", "Terjual"],
                yticklabels=["Harga", "Diskon%", "Rating", "Terjual"])
    ax.set_title("Heatmap Korelasi Pearson\nantar Variabel Numerik",
                 fontweight="bold")

    ax = axes[2]
    cluster_data = []
    for label in ["Regular", "Bestseller", "Premium"]:
        d = df_clean[df_clean["cluster_label"] == label]["rating"].values
        cluster_data.append(d if len(d) > 0 else [df_clean["rating"].mean()])
    bp = ax.boxplot(cluster_data, patch_artist=True,
                    labels=["Regular", "Bestseller", "Premium"])
    for patch, c in zip(bp["boxes"],
                        [CHART_COLORS[0], CHART_COLORS[1], CHART_COLORS[3]]):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_title("Distribusi Rating\nper Segmen Produk", fontweight="bold")
    ax.set_xlabel("Segmen"); ax.set_ylabel("Rating")
    ax.set_ylim(4.6, 5.1)

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "viz_2_clustering.png"),
                dpi=130, bbox_inches="tight")
    plt.close()


def _chart_popularitas(df_clean):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Analisis Popularitas & Pengaruh Promo — Crocs Official Store",
                 fontsize=14, fontweight="bold")

    df = df_clean.copy()
    df["popularitas"] = pd.cut(
        df["terjual"], bins=[-1, 100, 500, 1000, float("inf")],
        labels=["Rendah (<100)", "Sedang (100-500)",
                "Tinggi (500-1K)", "Sangat Tinggi (>1K)"])

    ax = axes[0]
    pc = df["popularitas"].value_counts().reindex(
        ["Rendah (<100)", "Sedang (100-500)",
         "Tinggi (500-1K)", "Sangat Tinggi (>1K)"], fill_value=0)
    bars = ax.bar(pc.index, pc.values,
                  color=CHART_COLORS[:len(pc)], edgecolor="white")
    ax.set_title("Segmentasi Popularitas Produk\n(berdasarkan jumlah terjual)",
                 fontweight="bold")
    ax.set_xlabel("Kategori Popularitas"); ax.set_ylabel("Jumlah Produk")
    ax.tick_params(axis="x", rotation=15)
    for b, v in zip(bars, pc.values):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.1, str(v),
                ha="center", fontweight="bold")

    ax = axes[1]
    pt = df.groupby("ada_promo")["terjual"].mean()
    label_map = {0: "Tanpa Promo", 1: "Ada Promo"}
    keys = sorted(pt.index)
    bar_labels = [label_map[k] for k in keys]
    bar_vals = [pt[k] for k in keys]
    bar_colors = [CHART_COLORS[3] if k == 0 else CHART_COLORS[1] for k in keys]
    bars = ax.bar(bar_labels, bar_vals, color=bar_colors,
                  edgecolor="white", width=0.5)
    ax.set_title("Rata-rata Penjualan:\nProduk dengan vs Tanpa Promo",
                 fontweight="bold")
    ax.set_ylabel("Rata-rata Terjual")
    for b, v in zip(bars, bar_vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+20,
                f"{v:,.0f}", ha="center", fontweight="bold", fontsize=11)

    ax = axes[2]
    sc = ax.scatter(df["rating"], df["terjual"],
                    c=df["harga"], cmap="YlOrRd", s=110,
                    alpha=0.85, edgecolors="gray")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Harga (Rp)", fontsize=8)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}Jt"))
    top3 = df.nlargest(3, "terjual")
    for _, row in top3.iterrows():
        ax.annotate(row["nama_produk"].split(" - ")[0][:20],
                    xy=(row["rating"], row["terjual"]),
                    xytext=(10, 5), textcoords="offset points", fontsize=6,
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.5))
    ax.set_title("Rating vs Jumlah Terjual\n(warna = harga)", fontweight="bold")
    ax.set_xlabel("Rating"); ax.set_ylabel("Jumlah Terjual")

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "viz_3_popularitas.png"),
                dpi=130, bbox_inches="tight")
    plt.close()


def _chart_eda(df_clean):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("EDA — Eksplorasi Distribusi Variabel Numerik",
                 fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.hist(df_clean["harga"], bins=8, color=CHART_COLORS[0],
            edgecolor="white", alpha=0.85)
    ax.set_title("Distribusi Harga", fontweight="bold")
    ax.set_xlabel("Harga (Rp)"); ax.set_ylabel("Frekuensi")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}Jt"))

    ax = axes[1]
    ax.hist(df_clean["terjual"], bins=10, color=CHART_COLORS[1],
            edgecolor="white", alpha=0.85)
    ax.set_title("Distribusi Jumlah Terjual", fontweight="bold")
    ax.set_xlabel("Terjual"); ax.set_ylabel("Frekuensi")

    ax = axes[2]
    ax.boxplot([df_clean["harga"]/1e6], labels=["Harga (Jt)"], patch_artist=True,
               boxprops=dict(facecolor=CHART_COLORS[2], alpha=0.7))
    ax2 = ax.twinx()
    ax2.boxplot([df_clean["terjual"]], labels=["Terjual"], patch_artist=True,
                positions=[2],
                boxprops=dict(facecolor=CHART_COLORS[1], alpha=0.7))
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Harga (Jt)", "Terjual"])
    ax.set_title("Boxplot — Outlier Detection", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "viz_0_eda.png"),
                dpi=130, bbox_inches="tight")
    plt.close()


# ================================================================
# ORCHESTRATOR — dipanggil sekali oleh app.py
# ================================================================
def run_full_pipeline():
    """Pipeline lengkap. Return semua hasil dalam dict besar."""
    df_raw, df_clean = load_and_clean()
    save_clean_outputs(df_raw, df_clean)
    df_clean, kmeans_meta = run_kmeans_pipeline(df_clean)
    corr = run_correlation_analysis(df_clean)
    eda  = run_eda(df_clean)
    generate_all_charts(df_clean, kmeans_meta)

    products = df_clean[
        ["nama_produk", "kategori", "warna", "harga", "diskon_pct",
         "harga_diskon", "ada_promo", "rating", "terjual",
         "cluster", "cluster_label", "url_gambar", "url_produk"]
    ].copy()
    products["harga"] = products["harga"].astype(int)
    products["harga_diskon"] = products["harga_diskon"].astype(int)
    products["terjual"] = products["terjual"].astype(int)
    products["cluster"] = products["cluster"].astype(int)

    return {
        "kmeans": kmeans_meta,
        "corr":   corr,
        "eda":    eda,
        "products": products.to_dict(orient="records"),
        "rows_raw":   int(len(df_raw)),
        "rows_clean": int(len(df_clean)),
    }


if __name__ == "__main__":
    result = run_full_pipeline()
    print("Total produk:", result["rows_clean"])
    print("K-Means:", result["kmeans"]["distribusi"])
    print("Korelasi pairs:", len(result["corr"]["pairs"]))
    print("Charts saved at:", IMG_DIR)
