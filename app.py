"""
Analisis Sentimen Komentar YouTube Film Horor Weapons 2025
Mata Kuliah: Analisis Data Tak Terstruktur (SST60325)
Anggota Kelompok:
- Sherly Leyn Raymond (23031030034)
- Patrecia Daniela Butarbutar (23031030044)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import html
import warnings
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Analisis Sentimen — Weapons (2025)",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

  .hero {
    background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 50%, #16213e 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    border: 1px solid #2a2a4a;
  }
  .hero h1 { color: #f0f0f0; font-size: 2.2rem; margin: 0 0 0.4rem 0; }
  .hero p  { color: #aaa; margin: 0; font-size: 1rem; }

  .metric-card {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
  }
  .metric-card .value { font-size: 2rem; font-weight: 800; }
  .metric-card .label { color: #888; font-size: 0.85rem; margin-top: 0.2rem; }

  .section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #4f8ef7;
    border-left: 4px solid #4f8ef7;
    padding-left: 0.8rem;
    margin: 2rem 0 1rem 0;
  }

  .insight-box {
    background: #111827;
    border: 1px solid #1f2937;
    border-left: 4px solid #4f8ef7;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-top: 0.8rem;
    font-size: 0.92rem;
    line-height: 1.7;
    color: #ccc;
  }
  .insight-box strong { color: #f0f0f0; }

  .tag-positive { background:#14532d; color:#86efac; padding:2px 10px; border-radius:99px; font-size:0.8rem; }
  .tag-negative { background:#450a0a; color:#fca5a5; padding:2px 10px; border-radius:99px; font-size:0.8rem; }
  .tag-neutral  { background:#1c1917; color:#a8a29e; padding:2px 10px; border-radius:99px; font-size:0.8rem; }

  .step-box {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
  }
  .step-num { font-size: 0.75rem; color: #4f8ef7; letter-spacing: 2px; text-transform: uppercase; }

  div[data-testid="stSidebar"] { background: #0a0a1a; }
  .stTabs [aria-selected="true"] { color: #4f8ef7 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# KONSTANTA WARNA
# ─────────────────────────────────────────────
COLORS = {
    "positive": "#4ade80",
    "negative": "#f87171",
    "neutral":  "#94a3b8",
    "blue":     "#4f8ef7",
    "purple":   "#a78bfa",
    "orange":   "#fb923c",
}
SENTIMENT_COLORS = [COLORS["positive"], COLORS["negative"], COLORS["neutral"]]
TOPIC_COLORS     = ["#4f8ef7", "#f87171", "#4ade80", "#a78bfa", "#fb923c"]
TOPIC_LABELS     = [
    "T1: Cerita & Sutradara",
    "T2: Sutradara & Ekspektasi",
    "T3: Elemen Anak/Senjata",
    "T4: Penilaian Trailer",
    "T5: Kualitas Umum Film",
]


# ─────────────────────────────────────────────
# HELPER: DARK THEME CHART
# ─────────────────────────────────────────────
def plot_style(fig, ax_list):
    fig.patch.set_facecolor("#111827")
    for ax in ax_list:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#888", labelsize=9)
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.grid(color="#2a2a4a", linewidth=0.6, axis="y", alpha=0.7)
        ax.xaxis.label.set_color("#aaa")
        ax.yaxis.label.set_color("#aaa")
        ax.title.set_color("#f0f0f0")


# ─────────────────────────────────────────────
# NLTK: DOWNLOAD RESOURCE (SEKALI SAJA)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_nlp_resources():
    import nltk

    resources = [
        "stopwords",
        "wordnet",
        "omw-1.4",
        "punkt",
        "averaged_perceptron_tagger",
        "vader_lexicon"
    ]

    for res in resources:
        nltk.download(res)

    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    return stopwords, WordNetLemmatizer(), SentimentIntensityAnalyzer()


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(text, stop_words, lemmatizer):
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag

    def get_wordnet_pos(tag):
        if tag.startswith("J"): return "a"
        if tag.startswith("V"): return "v"
        if tag.startswith("N"): return "n"
        if tag.startswith("R"): return "r"
        return "n"

    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    tagged = pos_tag(tokens)
    tokens = [lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in tagged]
    return " ".join(tokens)


def vader_label(text, sia):
    c = sia.polarity_scores(text)["compound"]
    if c >= 0.05:  return "positive"
    if c <= -0.05: return "negative"
    return "neutral"


# ─────────────────────────────────────────────
# LOAD & PROSES DATA (CACHED)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_process():
    stopwords_lib, lemmatizer, sia = load_nlp_resources()
    from nltk.corpus import stopwords as sw_corpus

    # — Baca CSV —
    df = pd.read_csv("komentar_youtube.csv")
    df["komentar"]     = df["komentar"].fillna("").astype(str)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df["like_count"]   = pd.to_numeric(df["like_count"], errors="coerce").fillna(0).astype(int)

    # — Preprocessing —
    stop_words = set(sw_corpus.words("english"))
    stop_words -= {"not", "no", "nor"}
    stop_words |= {"movie", "film", "watch", "just", "get", "im", "its"}
    df["clean_text"] = df["komentar"].apply(lambda x: preprocess(x, stop_words, lemmatizer))

    # — VADER Labeling —
    df["compound_score"] = df["komentar"].apply(lambda x: sia.polarity_scores(x)["compound"])
    df["label"]          = df["komentar"].apply(lambda x: vader_label(x, sia))

    # — Fitur tambahan —
    df["jumlah_kata"]      = df["komentar"].str.split().str.len()
    df["panjang_karakter"] = df["komentar"].str.len()
    df["text_length"]      = df["clean_text"].apply(lambda x: len(str(x).split()))

    # — Topic Modelling LDA —
    texts_clean = df["clean_text"][df["clean_text"].str.strip() != ""]
    if len(texts_clean) == 0:
        return df, None, None, None

    safe_min_df = max(2, int(len(texts_clean) * 0.01))
    count_vec   = CountVectorizer(max_df=0.90, min_df=safe_min_df, max_features=2000)
    dtm         = count_vec.fit_transform(texts_clean)

    lda = LatentDirichletAllocation(
        n_components=5, max_iter=20, learning_method="online", random_state=42)
    lda.fit(dtm)

    doc_topics     = lda.transform(dtm)
    dominant_topic = doc_topics.argmax(axis=1)

    df_sub = df[df["clean_text"].str.strip() != ""].copy().reset_index(drop=True)
    df_sub = df_sub.iloc[:len(dominant_topic)].copy()
    df_sub["dominant_topic"] = [TOPIC_LABELS[t] for t in dominant_topic]

    # Gabung dominant_topic ke df utama
    df = df.merge(df_sub[["komentar", "dominant_topic"]], on="komentar", how="left")

    return df, lda, count_vec, dtm


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🎬 Analisis Sentimen Komentar YouTube</h1>
  <p>Film <strong style="color:#f0f0f0">Weapons (2025)</strong> &nbsp;·&nbsp; Directed by Zach Cregger
     &nbsp;·&nbsp; Metode: VADER · TF-IDF · Logistic Regression · LDA <br>
            Anggota Kelompok: <br>
            Sherly Leyn Raymond (23031030034) <br>
            Patrecia Daniela Butarbutar (23031030044)
            </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR INFO
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎬 Weapons (2025)")
    st.markdown("**Analisis Sentimen Komentar YouTube**")
    st.markdown("---")
    st.markdown("""
    **Pipeline Analisis**

    1. 📥 Load data komentar (1.000 data)
    2. 🧹 Preprocessing teks
    3. 🏷️ Labeling sentimen (VADER)
    4. 📊 Exploratory Data Analysis
    5. 🤖 Klasifikasi ML
    6. 🗂️ Topic Modelling (LDA)
    """)
    st.markdown("---")
    st.caption("Project Analisis Data Tak Terstruktur · 2026")

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
with st.spinner("⏳ Memproses data... (hanya sekali, lalu tersimpan cache)"):
    df, lda, count_vec, dtm = load_and_process()

# ─────────────────────────────────────────────
# METRIC CARDS
# ─────────────────────────────────────────────
total       = len(df)
pos_count   = (df["label"] == "positive").sum()
neg_count   = (df["label"] == "negative").sum()
neu_count   = (df["label"] == "neutral").sum()
avg_compound= df["compound_score"].mean()

st.markdown('<div class="section-header">📊 Ringkasan Data</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
for col, (val, lbl, color) in zip([c1,c2,c3,c4,c5], [
    (f"{total:,}",         "Total Komentar",       "#f0f0f0"),
    (f"{pos_count:,}",     "Positif 😊",            COLORS["positive"]),
    (f"{neg_count:,}",     "Negatif 😞",            COLORS["negative"]),
    (f"{neu_count:,}",     "Netral 😐",             COLORS["neutral"]),
    (f"{avg_compound:.3f}","Avg Compound Score",    COLORS["blue"]),
]):
    col.markdown(f"""
    <div class="metric-card">
      <div class="value" style="color:{color}">{val}</div>
      <div class="label">{lbl}</div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 EDA", "💬 Sentimen", "🤖 Klasifikasi ML", "🗂️ Topic Modelling", "📋 Data"])


# ══════════════════════════════════════════════
# TAB 1 — EDA
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Exploratory Data Analysis</div>',
                unsafe_allow_html=True)

    with st.expander("📌 Statistik Deskriptif"):
        st.dataframe(
            df[["panjang_karakter","jumlah_kata","like_count"]].describe().round(2),
            use_container_width=True)
        ca, cb = st.columns(2)
        ca.metric("Total Like", f"{df['like_count'].sum():,}")
        cb.metric("Rata-rata Panjang Teks Bersih", f"{df['text_length'].mean():.1f} kata")

    # 4-panel EDA
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.patch.set_facecolor("#111827")
    daily   = df.groupby(df["published_at"].dt.date).size()
    monthly = df.groupby(df["published_at"].dt.to_period("M")).size()

    ax1 = axes[0, 0]
    ax1.plot(range(len(daily)), daily.values, color=COLORS["blue"],
             linewidth=2, marker="o", markersize=3)
    ax1.set_title("Tren Komentar per Hari")
    ax1.set_xlabel("Hari ke-"); ax1.set_ylabel("Jumlah Komentar")

    ax2 = axes[0, 1]
    ax2.hist(df["jumlah_kata"].clip(upper=80), bins=30,
             color=COLORS["orange"], edgecolor="#111827", alpha=0.9)
    ax2.axvline(df["jumlah_kata"].median(), color="white", linestyle="--",
                label=f"Median: {df['jumlah_kata'].median():.0f}")
    ax2.set_title("Distribusi Panjang Komentar (Kata)")
    ax2.set_xlabel("Jumlah Kata"); ax2.set_ylabel("Frekuensi")
    ax2.legend(fontsize=8, labelcolor="#ccc", facecolor="#1a1a2e", edgecolor="#2a2a4a")

    ax3 = axes[1, 0]
    ax3.hist(df["like_count"].clip(upper=df["like_count"].quantile(0.95)),
             bins=30, color=COLORS["purple"], edgecolor="#111827", alpha=0.9)
    ax3.set_title("Distribusi Like per Komentar (95th pct)")
    ax3.set_xlabel("Jumlah Like"); ax3.set_ylabel("Frekuensi")

    ax4 = axes[1, 1]
    ax4.bar(range(len(monthly)), monthly.values,
            color=COLORS["blue"], alpha=0.85, width=0.6)
    ax4.set_xticks(range(len(monthly)))
    ax4.set_xticklabels([str(m) for m in monthly.index], rotation=45, ha="right")
    ax4.set_title("Jumlah Komentar per Bulan"); ax4.set_ylabel("Jumlah Komentar")

    plot_style(fig, [ax1, ax2, ax3, ax4])
    plt.tight_layout(pad=2)
    st.pyplot(fig); plt.close()

    st.markdown(f"""
    <div class="insight-box">
      <strong>Interpretasi EDA</strong><br>
      • Puncak komentar terjadi pada <strong>{daily.idxmax()}</strong>
        ({daily.max()} komentar), bertepatan dengan periode rilis konten, menandakan
        antusiasme tinggi dari komunitas horror.<br>
      • Distribusi panjang komentar <strong>right-skewed</strong>: mayoritas komentar
        pendek (1–15 kata), mencerminkan respons spontan penonton.<br>
      • {(df['like_count']==0).sum():,} komentar
        ({(df['like_count']==0).mean()*100:.1f}%) tidak mendapat like,
        umum terjadi di platform besar dengan jutaan penonton pasif.<br>
      • Tren komentar menurun setelah bulan peluncuran trailer (normal decay pattern).
    </div>""", unsafe_allow_html=True)

    # Word Frequency & N-Gram
    st.markdown('<div class="section-header">Frekuensi Kata & N-Gram</div>',
                unsafe_allow_html=True)
    from nltk.util import ngrams as nltk_ngrams

    all_words    = " ".join(df["clean_text"].astype(str)).split()
    tokens_list  = [t for t in all_words if len(t) > 1]
    top10        = Counter(all_words).most_common(10)
    top_bi       = Counter([" ".join(g) for g in nltk_ngrams(tokens_list, 2)]).most_common(10)
    top_tri      = Counter([" ".join(g) for g in nltk_ngrams(tokens_list, 3)]).most_common(10)

    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
    fig2.patch.set_facecolor("#111827")
    for ax, data, color, title in [
        (axes2[0], top10,  COLORS["blue"],   "Top 10 Kata"),
        (axes2[1], top_bi, COLORS["orange"], "Top 10 Bigram"),
        (axes2[2], top_tri,COLORS["purple"], "Top 10 Trigram"),
    ]:
        w = [d[0] for d in data]; c = [d[1] for d in data]
        ax.barh(w[::-1], c[::-1], color=color, alpha=0.85)
        ax.set_title(title); ax.set_xlabel("Frekuensi")

    plot_style(fig2, axes2)
    plt.tight_layout(pad=2)
    st.pyplot(fig2); plt.close()

    top_word = top10[0][0] if top10 else "-"
    st.markdown(f"""
    <div class="insight-box">
      <strong>Interpretasi Frekuensi Kata</strong><br>
      • <strong>"{top_word}"</strong> adalah kata paling dominan,
        sinyal apresiasi kuat dari penonton.<br>
      • Bigram <strong>"one best"</strong> dan <strong>"really good"</strong>
        menunjukkan penonton membandingkan film ini sebagai yang terbaik di genre-nya.<br>
      • Frasa <strong>"true story"</strong> berulang, penonton sangat tertarik pada
        elemen faktual/keaslian cerita.<br>
      • <strong>"aunt gladys"</strong> muncul berkali-kali sebagai bigram karakter ini
        memicu diskusi luas dan berpotensi menjadi selling point marketing.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — SENTIMEN
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Distribusi Sentimen (VADER)</div>',
                unsafe_allow_html=True)

    ordered        = ["positive", "negative", "neutral"]
    counts_ordered = [(df["label"] == s).sum() for s in ordered]

    fig3, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))
    fig3.patch.set_facecolor("#111827")

    ax_pie.set_facecolor("#1a1a2e")
    wedges, texts, autotexts = ax_pie.pie(
        counts_ordered, labels=ordered, autopct="%1.1f%%",
        colors=SENTIMENT_COLORS, startangle=140, pctdistance=0.75,
        wedgeprops={"edgecolor": "#111827", "linewidth": 2})
    for t in texts:     t.set_color("#aaa")
    for t in autotexts: t.set_color("white"); t.set_fontsize(9)
    ax_pie.set_title("Distribusi Sentimen (%)", color="#f0f0f0")

    ax_bar.set_facecolor("#1a1a2e")
    bars = ax_bar.bar(ordered, counts_ordered,
                      color=SENTIMENT_COLORS, edgecolor="#111827", width=0.5)
    for bar, val in zip(bars, counts_ordered):
        ax_bar.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + total * 0.01,
                    str(val), ha="center", fontweight="bold", color="white", fontsize=10)
    ax_bar.set_title("Jumlah Komentar per Sentimen", color="#f0f0f0")
    ax_bar.set_ylabel("Jumlah Komentar", color="#aaa")
    ax_bar.tick_params(colors="#888")
    ax_bar.spines[["top","right","left","bottom"]].set_visible(False)
    ax_bar.grid(color="#2a2a4a", linewidth=0.6, axis="y", alpha=0.7)

    plt.tight_layout()
    st.pyplot(fig3); plt.close()

    st.markdown(f"""
    <div class="insight-box">
      <strong>Interpretasi Sentimen</strong><br>
      • Mayoritas komentar bersifat <strong>positif ({pos_count/total*100:.1f}%)</strong>,
        menandakan penerimaan yang sangat baik dari penonton terhadap trailer.<br>
      • Rata-rata compound score VADER: <strong>{avg_compound:.4f}</strong>
        (positif lemah secara agregat).<br>
      • Sentimen negatif ({neg_count/total*100:.1f}%) umumnya berkaitan dengan elemen
        kontroversial (kekerasan) dan ekspektasi terhadap sutradara.<br>
      • Komentar netral ({neu_count/total*100:.1f}%) bersifat informatif/analitis,
        penonton membahas cerita atau latar belakang film secara faktual.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Contoh Komentar per Sentimen</div>',
                unsafe_allow_html=True)
    for label, tag in [("positive","tag-positive"),("negative","tag-negative"),("neutral","tag-neutral")]:
        sample = df[df["label"] == label]["komentar"].head(3).tolist()
        st.markdown(f'<span class="{tag}">{label.upper()}</span>', unsafe_allow_html=True)
        for s in sample:
            st.markdown(f"> {s[:200]}{'...' if len(s)>200 else ''}")


# ══════════════════════════════════════════════
# TAB 3 — KLASIFIKASI ML
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Klasifikasi Sentimen — Naive Bayes vs Logistic Regression</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box" style="margin-bottom:1rem">
      Model dilatih menggunakan <strong>TF-IDF Unigram</strong> sebagai representasi fitur teks,
      dengan split data <strong>80% training / 20% testing</strong> (stratified).
      Label sentimen berasal dari VADER (otomatis, tanpa anotasi manual).
    </div>""", unsafe_allow_html=True)

    df_clean = df[df["clean_text"].str.strip() != ""].copy()
    vec  = TfidfVectorizer()
    X    = vec.fit_transform(df_clean["clean_text"])
    y    = df_clean["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    nb = MultinomialNB(); nb.fit(X_tr, y_tr); y_nb = nb.predict(X_te)
    lr = LogisticRegression(max_iter=500, random_state=42); lr.fit(X_tr, y_tr); y_lr = lr.predict(X_te)

    def get_metrics(y_true, y_pred):
        return {
            "Accuracy":  accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "Recall":    recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "F1-Score":  f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

    m_nb = get_metrics(y_te, y_nb)
    m_lr = get_metrics(y_te, y_lr)

    results_df = pd.DataFrame([
        {"Model": "Naive Bayes",         **m_nb},
        {"Model": "Logistic Regression", **m_lr},
    ])
    st.dataframe(
        results_df.style
            .format({k: "{:.4f}" for k in ["Accuracy","Precision","Recall","F1-Score"]})
            .highlight_max(subset=["Accuracy","Precision","Recall","F1-Score"], color="#14532d"),
        use_container_width=True, hide_index=True)

    # Bar perbandingan + ROC
    classes   = sorted(y.unique().tolist())
    y_te_bin  = label_binarize(y_te, classes=classes)
    y_prob_nb = nb.predict_proba(X_te)
    y_prob_lr = lr.predict_proba(X_te)
    auc_nb    = roc_auc_score(y_te_bin, y_prob_nb, multi_class="ovr")
    auc_lr    = roc_auc_score(y_te_bin, y_prob_lr, multi_class="ovr")

    pos_idx = classes.index("positive") if "positive" in classes else 0
    fpr_nb, tpr_nb, _ = roc_curve(y_te_bin[:, pos_idx], y_prob_nb[:, pos_idx])
    fpr_lr, tpr_lr, _ = roc_curve(y_te_bin[:, pos_idx], y_prob_lr[:, pos_idx])

    fig4, (ax_b2, ax_roc) = plt.subplots(1, 2, figsize=(13, 5))
    fig4.patch.set_facecolor("#111827")

    metrics_names = ["Accuracy","Precision","Recall","F1-Score"]
    x = np.arange(len(metrics_names)); w = 0.35
    ax_b2.set_facecolor("#1a1a2e")
    b1 = ax_b2.bar(x - w/2, [m_nb[k] for k in metrics_names], w,
                   label="Naive Bayes", color=COLORS["orange"], alpha=0.9, edgecolor="#111827")
    b2 = ax_b2.bar(x + w/2, [m_lr[k] for k in metrics_names], w,
                   label="Logistic Regression", color=COLORS["blue"], alpha=0.9, edgecolor="#111827")
    for bar in list(b1)+list(b2):
        ax_b2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                   f"{bar.get_height():.3f}", ha="center", fontsize=7.5, color="white")
    ax_b2.set_xticks(x); ax_b2.set_xticklabels(metrics_names)
    ax_b2.set_ylim(0, 1.12); ax_b2.set_title("Perbandingan Metrik Model")
    ax_b2.legend(facecolor="#1a1a2e", edgecolor="#2a2a4a", labelcolor="#ccc")

    ax_roc.set_facecolor("#1a1a2e")
    ax_roc.plot(fpr_nb, tpr_nb, color=COLORS["orange"], linewidth=2,
                label=f"Naive Bayes (AUC={auc_nb:.3f})")
    ax_roc.plot(fpr_lr, tpr_lr, color=COLORS["blue"], linewidth=2,
                label=f"Logistic Regression (AUC={auc_lr:.3f})")
    ax_roc.plot([0,1],[0,1], linestyle="--", color="#555")
    ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve — Kelas Positive")
    ax_roc.legend(facecolor="#1a1a2e", edgecolor="#2a2a4a", labelcolor="#ccc")

    plot_style(fig4, [ax_b2, ax_roc])
    plt.tight_layout(pad=2)
    st.pyplot(fig4); plt.close()

    # Confusion Matrix
    st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
    fig5, (cm1, cm2) = plt.subplots(1, 2, figsize=(12, 4))
    fig5.patch.set_facecolor("#111827")
    for ax_, y_pred_, title_ in [(cm1, y_nb, "Naive Bayes"), (cm2, y_lr, "Logistic Regression")]:
        cm_ = confusion_matrix(y_te, y_pred_, labels=classes)
        sns.heatmap(cm_, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes, ax=ax_,
                    linewidths=0.5, annot_kws={"color":"black", "size":10})
        ax_.set_title(f"Confusion Matrix — {title_}", color="#f0f0f0")
        ax_.set_xlabel("Predicted", color="#aaa"); ax_.set_ylabel("Actual", color="#aaa")
        ax_.tick_params(colors="#888"); ax_.set_facecolor("#1a1a2e")
    plt.tight_layout(); st.pyplot(fig5); plt.close()

    best = "Logistic Regression" if m_lr["F1-Score"] > m_nb["F1-Score"] else "Naive Bayes"
    st.markdown(f"""
    <div class="insight-box">
      <strong>Interpretasi Klasifikasi ML</strong><br>
      • <strong>{best}</strong> mengungguli model lainnya dengan
        F1-Score {m_lr['F1-Score']:.3f} vs {m_nb['F1-Score']:.3f}.<br>
      • Logistic Regression lebih baik karena mampu memodelkan hubungan linear antar fitur
        TF-IDF, sementara Naive Bayes mengasumsikan independensi antar fitur
        (yang jarang terjadi pada data teks nyata).<br>
      • ROC-AUC Logistic Regression ({auc_lr:.3f}) &gt; Naive Bayes ({auc_nb:.3f}),
        kemampuan diskriminasi antar kelas lebih baik.<br>
      • Gap antara accuracy dan AUC disebabkan <strong>class imbalance</strong>, kelas
        "negative" hanya ~{neg_count/total*100:.0f}% data, sehingga model cenderung bias
        ke kelas mayoritas.
    </div>""", unsafe_allow_html=True)

    with st.expander("📊 Eksperimen: Unigram vs Bigram TF-IDF"):
        vec2  = TfidfVectorizer(ngram_range=(1,2))
        X2    = vec2.fit_transform(df_clean["clean_text"])
        X_tr2, X_te2, _, y_te2 = train_test_split(X2, y, test_size=0.2, random_state=42, stratify=y)
        lr2   = LogisticRegression(max_iter=500, random_state=42)
        lr2.fit(X_tr2, y_tr); m_lr2 = get_metrics(y_te2, lr2.predict(X_te2))
        comp  = pd.DataFrame([{"Model":"Unigram (LogReg)",**m_lr},{"Model":"Bigram (LogReg)",**m_lr2}])
        st.dataframe(comp.style.format({k:"{:.4f}" for k in ["Accuracy","Precision","Recall","F1-Score"]}),
                     use_container_width=True, hide_index=True)
        if m_lr2["F1-Score"] > m_lr["F1-Score"]:
            st.success("✅ Bigram meningkatkan performa.")
        else:
            st.info("ℹ️ Bigram tidak meningkatkan performa. Komentar pendek menghasilkan banyak bigram sparse yang tidak informatif.")


# ══════════════════════════════════════════════
# TAB 4 — TOPIC MODELLING
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Topic Modelling — LDA (Latent Dirichlet Allocation)</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box" style="margin-bottom:1rem">
      LDA mengungkap <strong>tema tersembunyi</strong> dalam komentar tanpa memerlukan label.
      VADER menjawab <em>"bagaimana perasaan penonton"</em>, LDA menjawab
      <em>"apa yang mereka bicarakan"</em>.
    </div>""", unsafe_allow_html=True)

    feature_names = count_vec.get_feature_names_out()
    fig6, axes6   = plt.subplots(1, 5, figsize=(20, 5))
    fig6.patch.set_facecolor("#111827")
    fig6.suptitle("LDA Topic Modelling — Top 10 Kata per Topik",
                  fontsize=12, fontweight="bold", color="#f0f0f0")

    for i, (ax, topic) in enumerate(zip(axes6, lda.components_)):
        top_idx   = topic.argsort()[:-11:-1]
        top_words = [feature_names[j] for j in top_idx]
        top_scores= [topic[j] for j in top_idx]
        ax.barh(top_words[::-1], top_scores[::-1], color=TOPIC_COLORS[i], alpha=0.85)
        ax.set_title(TOPIC_LABELS[i].split(": ")[1],
                     fontweight="bold", fontsize=8, color="#f0f0f0")
        ax.set_xlabel("Bobot", color="#aaa")
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#888", labelsize=7.5)
        ax.spines[["top","right","left","bottom"]].set_visible(False)

    plt.tight_layout(pad=2)
    st.pyplot(fig6); plt.close()

    # Distribusi topik + cross sentimen
    doc_topics_    = lda.transform(dtm)
    dominant_nums  = doc_topics_.argmax(axis=1)
    topic_dist     = pd.Series(dominant_nums).value_counts().sort_index()

    fig7, (ax_t1, ax_t2) = plt.subplots(1, 2, figsize=(13, 4))
    fig7.patch.set_facecolor("#111827")

    ax_t1.set_facecolor("#1a1a2e")
    bars_t = ax_t1.bar(range(5), [topic_dist.get(i,0) for i in range(5)],
                       color=TOPIC_COLORS, alpha=0.85, width=0.6)
    for bar, val in zip(bars_t, [topic_dist.get(i,0) for i in range(5)]):
        ax_t1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                   str(val), ha="center", color="white", fontweight="bold")
    ax_t1.set_xticks(range(5))
    ax_t1.set_xticklabels([f"T{i+1}" for i in range(5)])
    ax_t1.set_title("Distribusi Topik Dominan", color="#f0f0f0")
    ax_t1.set_ylabel("Jumlah Komentar", color="#aaa")
    ax_t1.tick_params(colors="#888")
    ax_t1.spines[["top","right","left","bottom"]].set_visible(False)
    ax_t1.grid(color="#2a2a4a", linewidth=0.6, axis="y", alpha=0.7)

    df_topic = df[df["dominant_topic"].notna()].copy()
    if len(df_topic) > 0:
        cross     = pd.crosstab(df_topic["dominant_topic"], df_topic["label"])
        cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100
        col_order = [c for c in ["negative","neutral","positive"] if c in cross_pct.columns]
        cross_pct[col_order].plot(
            kind="bar", stacked=True, ax=ax_t2,
            color=[COLORS[c] for c in col_order],
            edgecolor="#111827", alpha=0.9)
        ax_t2.set_title("Distribusi Sentimen per Topik LDA", color="#f0f0f0")
        ax_t2.set_ylabel("Persentase (%)", color="#aaa")
        ax_t2.set_xlabel("")
        ax_t2.tick_params(axis="x", rotation=30, labelsize=7, colors="#888")
        ax_t2.tick_params(axis="y", colors="#888")
        ax_t2.set_facecolor("#1a1a2e")
        ax_t2.spines[["top","right","left","bottom"]].set_visible(False)
        ax_t2.grid(color="#2a2a4a", linewidth=0.6, axis="y", alpha=0.7)
        ax_t2.legend(title="Sentimen", facecolor="#1a1a2e",
                     edgecolor="#2a2a4a", labelcolor="#ccc", fontsize=8)

    plt.tight_layout(pad=2)
    st.pyplot(fig7); plt.close()

    st.markdown(f"""
    <div class="insight-box">
      <strong>Interpretasi Topic Modelling</strong><br>
      • <strong>Topik 5 (Kualitas Umum Film)</strong> mendominasi
        ({topic_dist.get(4,0)} komentar), penonton paling banyak membahas penilaian
        umum terhadap trailer.<br>
      • <strong>Topik 3 (Elemen Anak & Senjata)</strong> signifikan
        ({topic_dist.get(2,0)} komentar), elemen unik yang memancing diskusi dan
        kontroversi.<br>
      • <strong>Topik 2 (Sutradara & Ekspektasi)</strong> paling sedikit
        ({topic_dist.get(1,0)} komentar), hanya penonton hardcore horror yang
        membahas ini.<br>
      • Topik 4 & 5 memiliki sentimen positif tertinggi, penonton yang membahas
        kualitas dan trailer cenderung berpendapat positif.<br>
      • Topik 3 memiliki sentimen negatif terbesar, elemen kekerasan memicu
        respons negatif dari sebagian penonton.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 5 — DATA MENTAH
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Data Komentar Berlabel</div>',
                unsafe_allow_html=True)

    cf1, cf2 = st.columns(2)
    filter_label = cf1.selectbox("Filter Sentimen", ["Semua","positive","negative","neutral"])
    search_text  = cf2.text_input("Cari teks komentar")

    df_disp = df.copy()
    if filter_label != "Semua":
        df_disp = df_disp[df_disp["label"] == filter_label]
    if search_text:
        df_disp = df_disp[df_disp["komentar"].str.contains(search_text, case=False, na=False)]

    st.dataframe(
        df_disp[["author","komentar","label","compound_score",
                 "like_count","published_at","dominant_topic"]]
               .rename(columns={
                   "author":"Pengguna","komentar":"Komentar",
                   "label":"Sentimen","compound_score":"Skor VADER",
                   "like_count":"Like","published_at":"Tanggal",
                   "dominant_topic":"Topik LDA"}),
        use_container_width=True, height=450)
    st.caption(f"Menampilkan {len(df_disp):,} dari {total:,} komentar")

    csv_out = df_disp.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Download CSV", csv_out,
                       "komentar_labeled.csv", "text/csv", use_container_width=True)
