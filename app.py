# ============================================================
# APLIKASI PREDIKSI DROPOUT MAHASISWA
# Skripsi - CRISP-DM Methodology
# Deployment: Streamlit - Premium Dark UI
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="EduRisk · Prediksi Dropout",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS PREMIUM DARK THEME
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #0a0e1a; color: #e8eaf0; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0d1224 0%,#111827 100%);
    border-right: 1px solid rgba(99,102,241,0.2);
}
section[data-testid="stSidebar"] * { color: #c9cde0 !important; }
#MainMenu, footer, header { visibility: hidden; }

.hero-header {
    background: linear-gradient(135deg,#1e1b4b 0%,#312e81 40%,#1e3a5f 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content:''; position:absolute; top:-50%; right:-10%;
    width:400px; height:400px;
    background:radial-gradient(circle,rgba(99,102,241,0.15) 0%,transparent 70%);
    border-radius:50%;
}
.hero-badge {
    display:inline-block;
    background:rgba(99,102,241,0.2);
    border:1px solid rgba(99,102,241,0.4);
    color:#a5b4fc; padding:0.25rem 0.75rem;
    border-radius:20px; font-size:0.75rem;
    font-weight:500; letter-spacing:0.08em;
    text-transform:uppercase; margin-bottom:0.75rem;
}
.hero-title {
    font-family:'Syne',sans-serif; font-size:2.8rem;
    font-weight:800;
    background:linear-gradient(135deg,#a5b4fc,#818cf8,#34d399);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; margin:0; line-height:1.1;
}
.hero-subtitle {
    font-size:1rem; color:#94a3b8;
    margin-top:0.5rem; font-weight:300; letter-spacing:0.03em;
}

.section-title {
    font-family:'Syne',sans-serif; font-size:1.3rem;
    font-weight:700; color:#e2e8f0;
    margin:1.5rem 0 1rem 0;
    display:flex; align-items:center; gap:0.5rem;
}
.section-title::after {
    content:''; flex:1; height:1px;
    background:linear-gradient(90deg,rgba(99,102,241,0.4),transparent);
    margin-left:0.5rem;
}

.stat-grid {
    display:grid; grid-template-columns:repeat(4,1fr);
    gap:1rem; margin:1.5rem 0;
}
.stat-card {
    background:linear-gradient(135deg,#1e293b,#1a2035);
    border:1px solid rgba(99,102,241,0.2);
    border-radius:16px; padding:1.25rem 1.5rem;
    position:relative; overflow:hidden;
    transition:transform 0.2s,border-color 0.2s;
}
.stat-card:hover { transform:translateY(-2px); border-color:rgba(99,102,241,0.5); }
.stat-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg,#6366f1,#34d399);
}
.stat-value {
    font-family:'Syne',sans-serif; font-size:1.8rem;
    font-weight:800; color:#a5b4fc; line-height:1;
}
.stat-label {
    font-size:0.78rem; color:#64748b;
    margin-top:0.3rem; text-transform:uppercase; letter-spacing:0.08em;
}
.stat-icon { font-size:1.4rem; margin-bottom:0.4rem; }

.info-box {
    background:linear-gradient(135deg,rgba(99,102,241,0.08),rgba(52,211,153,0.05));
    border:1px solid rgba(99,102,241,0.25);
    border-left:3px solid #6366f1;
    border-radius:12px; padding:1rem 1.25rem;
    margin:1rem 0; color:#cbd5e1;
    font-size:0.9rem; line-height:1.6;
}

.alert-dropout {
    background:linear-gradient(135deg,rgba(239,68,68,0.12),rgba(220,38,38,0.06));
    border:1px solid rgba(239,68,68,0.35);
    border-left:4px solid #ef4444;
    border-radius:16px; padding:1.5rem; margin:1rem 0;
}
.alert-dropout h3 { font-family:'Syne',sans-serif; color:#fca5a5; font-size:1.3rem; margin:0 0 0.5rem 0; }
.alert-dropout p  { color:#fecaca; margin:0.25rem 0; }

.alert-safe {
    background:linear-gradient(135deg,rgba(52,211,153,0.12),rgba(16,185,129,0.06));
    border:1px solid rgba(52,211,153,0.35);
    border-left:4px solid #34d399;
    border-radius:16px; padding:1.5rem; margin:1rem 0;
}
.alert-safe h3 { font-family:'Syne',sans-serif; color:#6ee7b7; font-size:1.3rem; margin:0 0 0.5rem 0; }
.alert-safe p  { color:#a7f3d0; margin:0.25rem 0; }

.phase-card {
    background:#111827; border:1px solid rgba(99,102,241,0.2);
    border-radius:14px; padding:1.25rem; height:100%;
}
.phase-card h4 {
    font-family:'Syne',sans-serif; color:#a5b4fc;
    font-size:0.82rem; text-transform:uppercase;
    letter-spacing:0.1em; margin:0 0 0.75rem 0;
    padding-bottom:0.5rem;
    border-bottom:1px solid rgba(99,102,241,0.2);
}
.phase-card p { color:#94a3b8; font-size:0.85rem; line-height:1.9; margin:0; }

.form-section-title {
    font-family:'Syne',sans-serif; font-size:0.82rem;
    font-weight:700; color:#6366f1;
    text-transform:uppercase; letter-spacing:0.1em;
    margin-bottom:0.75rem; padding-bottom:0.5rem;
    border-bottom:1px solid rgba(99,102,241,0.15);
}

div[data-testid="stMetric"] {
    background:#111827; border:1px solid rgba(99,102,241,0.2);
    border-radius:12px; padding:1rem;
}
div[data-testid="stMetric"] label {
    color:#64748b !important; font-size:0.78rem !important;
    text-transform:uppercase; letter-spacing:0.06em;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color:#a5b4fc !important;
    font-family:'Syne',sans-serif !important;
    font-size:1.6rem !important;
}

.stButton > button {
    background:linear-gradient(135deg,#6366f1,#4f46e5) !important;
    color:white !important; border:none !important;
    border-radius:12px !important;
    padding:0.75rem 2rem !important;
    font-family:'Syne',sans-serif !important;
    font-weight:700 !important; font-size:1rem !important;
    letter-spacing:0.05em !important;
    box-shadow:0 4px 20px rgba(99,102,241,0.3) !important;
    transition:all 0.2s !important;
}
.stButton > button:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 8px 30px rgba(99,102,241,0.5) !important;
}

.stTabs [data-baseweb="tab-list"] {
    background:#111827; border-radius:12px; padding:4px; gap:4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius:8px !important; color:#64748b !important;
    font-family:'DM Sans',sans-serif !important; font-weight:500 !important;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#6366f1,#4f46e5) !important;
    color:white !important;
}
.stDataFrame { border:1px solid rgba(99,102,241,0.2) !important; border-radius:12px !important; }

.stProgress > div > div > div {
    background:linear-gradient(90deg,#6366f1,#34d399) !important;
    border-radius:10px !important;
}
.stProgress > div > div { background:#1e293b !important; border-radius:10px !important; }
</style>
""", unsafe_allow_html=True)

# Matplotlib dark theme
plt.rcParams.update({
    'figure.facecolor':'#111827','axes.facecolor':'#111827',
    'axes.edgecolor':'#1e293b','axes.labelcolor':'#94a3b8',
    'xtick.color':'#64748b','ytick.color':'#64748b',
    'text.color':'#e2e8f0','grid.color':'#1e293b',
    'grid.linewidth':0.8,'figure.dpi':130,
})

# ============================================================
# LOAD ARTIFACTS
# ============================================================
@st.cache_resource
def load_models():
    return {
        'Logistic Regression': joblib.load('model_lr.pkl'),
        'Decision Tree'      : joblib.load('model_dt.pkl'),
        'Random Forest'      : joblib.load('model_rf.pkl'),
        'Gradient Boosting'  : joblib.load('model_gb.pkl'),
    }

@st.cache_resource
def load_artifacts():
    return (joblib.load('scaler.pkl'), joblib.load('feature_names.pkl'),
            joblib.load('results.pkl'), joblib.load('df_encoded.pkl'),
            joblib.load('feature_importance.pkl'))

try:
    models = load_models()
    scaler, feature_names, results, df_enc, feat_imp = load_artifacts()
    load_success = True
except Exception as e:
    load_success = False; error_msg = str(e)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1.5rem 0 1rem 0;'>
        <div style='font-size:2.5rem'>🎓</div>
        <div style='font-family:Syne,sans-serif;font-size:1.2rem;
                    font-weight:800;color:#a5b4fc;margin-top:0.5rem;'>EduRisk</div>
        <div style='font-size:0.7rem;color:#475569;letter-spacing:0.1em;
                    text-transform:uppercase;margin-top:0.2rem;'>
                    Dropout Prediction System</div>
    </div>
    <hr style='border:none;border-top:1px solid rgba(99,102,241,0.2);margin:0.5rem 0 1rem 0;'>
    """, unsafe_allow_html=True)

    menu = st.radio("", [
        "🏠  Beranda","📊  Eksplorasi Data","🔮  Prediksi Dropout",
        "📈  Perbandingan Model","🔍  Feature Importance","ℹ️   Tentang Aplikasi"
    ], label_visibility="collapsed")

    st.markdown("""
    <hr style='border:none;border-top:1px solid rgba(99,102,241,0.15);margin:1rem 0;'>
    <div style='font-size:0.7rem;color:#334155;text-transform:uppercase;
                letter-spacing:0.1em;margin-bottom:0.75rem;'>Dataset Info</div>
    """, unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1: st.metric("Records","10K"); st.metric("Features","26")
    with c2: st.metric("Dropout","23.5%"); st.metric("AUC","0.821")

    st.markdown("""
    <hr style='border:none;border-top:1px solid rgba(99,102,241,0.15);margin:1rem 0;'>
    <div style='font-size:0.72rem;color:#475569;line-height:1.9;'>
    🐍 Python 3.x · sklearn 1.6.1<br>
    📊 Pandas · NumPy · Seaborn<br>
    🚀 CRISP-DM Methodology
    </div>""", unsafe_allow_html=True)


# ============================================================
# PAGE 1: BERANDA
# ============================================================
if menu == "🏠  Beranda":
    st.markdown("""
    <div class='hero-header'>
        <div class='hero-badge'>CRISP-DM · Machine Learning · Skripsi 2024</div>
        <div class='hero-title'>Sistem Prediksi<br>Dropout Mahasiswa</div>
        <div class='hero-subtitle'>
        Deteksi dini risiko putus sekolah menggunakan 4 algoritma Machine Learning
        berbasis data akademik, demografis, dan perilaku mahasiswa.
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='stat-grid'>
        <div class='stat-card'><div class='stat-icon'>📁</div>
            <div class='stat-value'>10K</div><div class='stat-label'>Total Record</div></div>
        <div class='stat-card'><div class='stat-icon'>📋</div>
            <div class='stat-value'>26</div><div class='stat-label'>Fitur Dataset</div></div>
        <div class='stat-card'><div class='stat-icon'>⚠️</div>
            <div class='stat-value'>23.5%</div><div class='stat-label'>Tingkat Dropout</div></div>
        <div class='stat-card'><div class='stat-icon'>🏆</div>
            <div class='stat-value'>82.1%</div><div class='stat-label'>Best AUC-ROC</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>🔄 Alur Metodologi CRISP-DM</div>",
                unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("""<div class='phase-card'><h4>📌 Phase 1–2 · Understanding</h4>
        <p>✅ Business Understanding<br>✅ Data Understanding (EDA)<br>
        ✅ Analisis Korelasi<br>✅ Distribusi Fitur & Target</p></div>""",
        unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='phase-card'><h4>🛠️ Phase 3 · Preparation</h4>
        <p>✅ Handling Missing Values<br>✅ Deteksi Outlier (IQR)<br>
        ✅ Encoding Kategorikal<br>✅ Feature Scaling<br>✅ Oversampling</p></div>""",
        unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='phase-card'><h4>🤖 Phase 4–6 · Model & Deploy</h4>
        <p>✅ Logistic Regression<br>✅ Decision Tree<br>
        ✅ Random Forest<br>✅ Gradient Boosting<br>✅ Streamlit Deployment</p></div>""",
        unsafe_allow_html=True)

    st.markdown("<div class='section-title'>🏆 Ringkasan Performa Model</div>",
                unsafe_allow_html=True)
    if load_success:
        rdf = pd.DataFrame(results).T.sort_values('F1-Score', ascending=False)
        fig, ax = plt.subplots(figsize=(12,4))
        metrics = ['Accuracy','Precision','Recall','F1-Score','AUC-ROC']
        x = np.arange(len(metrics)); w = 0.18
        pal = ['#6366f1','#34d399','#f59e0b','#ef4444']
        for i,(name,color) in enumerate(zip(rdf.index,pal)):
            vals = [rdf.loc[name,m] for m in metrics]
            bars = ax.bar(x+i*w, vals, w, label=name, color=color, alpha=0.85, edgecolor='none', zorder=3)
            for bar,v in zip(bars,vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008,
                        f'{v:.2f}', ha='center', va='bottom', fontsize=6.5,
                        color='#94a3b8', fontweight='bold')
        ax.set_xticks(x+w*1.5); ax.set_xticklabels(metrics, fontsize=10)
        ax.set_ylim(0,1.1); ax.set_ylabel('Score', fontsize=10)
        ax.legend(fontsize=8, framealpha=0.15, facecolor='#1e293b',
                  edgecolor='#334155', labelcolor='#94a3b8')
        ax.grid(axis='y', alpha=0.2, zorder=0)
        ax.spines[['top','right','left','bottom']].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()


# ============================================================
# PAGE 2: EDA
# ============================================================
elif menu == "📊  Eksplorasi Data":
    st.markdown("""
    <div class='hero-header'>
        <div class='hero-badge'>Phase 2 · Data Understanding</div>
        <div class='hero-title'>Eksplorasi Data</div>
        <div class='hero-subtitle'>Visualisasi distribusi, korelasi, dan pola dalam dataset</div>
    </div>""", unsafe_allow_html=True)

    if not load_success: st.error(f"❌ {error_msg}"); st.stop()

    tab1,tab2,tab3 = st.tabs(["🎯 Target Variable","📈 Distribusi Fitur","🔗 Korelasi"])

    with tab1:
        dc = df_enc['Dropout'].value_counts()
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("Total Data", f"{len(df_enc):,}")
        with c2: st.metric("Tidak Dropout", f"{dc[0]:,}")
        with c3: st.metric("Dropout", f"{dc[1]:,}")
        with c4: st.metric("Dropout Rate", f"{dc[1]/len(df_enc)*100:.1f}%")

        fig, axes = plt.subplots(1,2,figsize=(11,4.5))
        colors = ['#6366f1','#ef4444']
        wedges,texts,autotexts = axes[0].pie(
            dc, labels=['Tidak Dropout','Dropout'], autopct='%1.1f%%',
            colors=colors, startangle=90, explode=(0.04,0.04),
            textprops={'color':'#94a3b8','fontsize':10},
            wedgeprops={'edgecolor':'#111827','linewidth':2})
        for at in autotexts: at.set_color('#e2e8f0'); at.set_fontweight('bold')
        axes[0].set_title('Proporsi Dropout', color='#e2e8f0', fontsize=12, fontweight='bold', pad=15)
        bars = axes[1].bar(['Tidak Dropout','Dropout'], dc.values, color=colors, edgecolor='none', width=0.5, zorder=3)
        for bar,val in zip(bars,dc.values):
            axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+80,
                         f'{val:,}', ha='center', fontsize=12, fontweight='bold', color='#e2e8f0')
        axes[1].set_title('Jumlah Mahasiswa', color='#e2e8f0', fontsize=12, fontweight='bold', pad=15)
        axes[1].set_ylim(0,9800)
        axes[1].spines[['top','right','left','bottom']].set_visible(False)
        axes[1].grid(axis='y', alpha=0.2, zorder=0)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        num_feat = st.selectbox("Pilih Fitur:", ['Age','Family_Income','Study_Hours_per_Day',
            'Attendance_Rate','Travel_Time_Minutes','Stress_Index','GPA','Semester_GPA'])
        fig, axes = plt.subplots(1,2,figsize=(12,4.5))
        colors = ['#6366f1','#ef4444']
        for label,color,lname in zip([0,1],colors,['Tidak Dropout','Dropout']):
            axes[0].hist(df_enc[df_enc['Dropout']==label][num_feat], bins=30,
                         alpha=0.65, color=color, label=lname, edgecolor='none')
        axes[0].set_title(f'Distribusi {num_feat}', color='#e2e8f0', fontweight='bold')
        axes[0].set_xlabel(num_feat); axes[0].set_ylabel('Frekuensi')
        axes[0].legend(framealpha=0.2, facecolor='#1e293b', edgecolor='#334155', labelcolor='#94a3b8')
        axes[0].spines[['top','right','left','bottom']].set_visible(False)
        d0 = df_enc[df_enc['Dropout']==0][num_feat].dropna()
        d1 = df_enc[df_enc['Dropout']==1][num_feat].dropna()
        bp = axes[1].boxplot([d0,d1], labels=['Tidak Dropout','Dropout'], patch_artist=True,
                              medianprops=dict(color='#f8fafc',linewidth=2.5),
                              boxprops=dict(linewidth=0),
                              whiskerprops=dict(color='#475569',linewidth=1.5),
                              capprops=dict(color='#475569',linewidth=1.5),
                              flierprops=dict(marker='o',markersize=3,markerfacecolor='#64748b',alpha=0.5))
        bp['boxes'][0].set_facecolor('#6366f1'); bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor('#ef4444'); bp['boxes'][1].set_alpha(0.7)
        axes[1].set_title(f'Boxplot {num_feat}', color='#e2e8f0', fontweight='bold')
        axes[1].spines[['top','right','left','bottom']].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        num_cols = ['Age','Family_Income','Study_Hours_per_Day','Attendance_Rate',
                    'Travel_Time_Minutes','Stress_Index','GPA','Semester_GPA',
                    'Assignment_Delay_Days','Dropout']
        corr = df_enc[num_cols].corr()
        fig, ax = plt.subplots(figsize=(12,8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                    cmap=sns.diverging_palette(240,10,as_cmap=True),
                    center=0, linewidths=0.5, linecolor='#0a0e1a',
                    square=True, ax=ax, annot_kws={'size':9,'color':'#e2e8f0'},
                    cbar_kws={'shrink':0.8})
        ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold', color='#e2e8f0', pad=15)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        cd = corr['Dropout'].drop('Dropout').sort_values(key=abs, ascending=False)
        st.markdown("<div class='section-title'>Korelasi dengan Dropout</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            'Fitur':cd.index, 'Korelasi':cd.values.round(4),
            'Kekuatan':['🔴 Kuat' if abs(v)>0.3 else '🟡 Sedang' if abs(v)>0.1 else '⚪ Lemah' for v in cd.values]
        }), use_container_width=True, hide_index=True)


# ============================================================
# PAGE 3: PREDIKSI
# ============================================================
elif menu == "🔮  Prediksi Dropout":
    st.markdown("""
    <div class='hero-header'>
        <div class='hero-badge'>Phase 6 · Real-time Prediction</div>
        <div class='hero-title'>Prediksi Dropout</div>
        <div class='hero-subtitle'>Masukkan data mahasiswa untuk memprediksi risiko dropout secara real-time</div>
    </div>""", unsafe_allow_html=True)

    if not load_success: st.error(f"❌ {error_msg}"); st.stop()

    st.markdown("""<div class='info-box'>
    🏆 Model: <b>Logistic Regression</b> — F1-Score: 0.5847 · AUC-ROC: 0.8213 · Recall: 76.22%
    </div>""", unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("<div class='form-section-title'>👤 Data Demografis</div>", unsafe_allow_html=True)
        age           = st.slider("Usia", 17, 30, 21)
        gender        = st.selectbox("Jenis Kelamin", ["Male","Female"])
        family_income = st.number_input("Pendapatan Keluarga", min_value=25000, max_value=200000, value=40000, step=1000)
        parental_edu  = st.selectbox("Pendidikan Orang Tua", ["Bachelor","High School","Master","PhD"])
        internet      = st.selectbox("Akses Internet", ["Yes","No"])
    with c2:
        st.markdown("<div class='form-section-title'>📚 Data Akademik</div>", unsafe_allow_html=True)
        semester     = st.selectbox("Tahun Semester", ["Year 1","Year 2","Year 3","Year 4"])
        department   = st.selectbox("Jurusan", ["Arts","Business","CS","Engineering","Science"])
        gpa          = st.slider("GPA (IPK)", 0.0, 4.0, 2.5, 0.01)
        semester_gpa = st.slider("Semester GPA", 0.0, 4.0, 2.5, 0.01)
        attendance   = st.slider("Tingkat Kehadiran (%)", 38.0, 100.0, 80.0, 0.1)
    with c3:
        st.markdown("<div class='form-section-title'>⚡ Data Perilaku</div>", unsafe_allow_html=True)
        study_hours = st.slider("Jam Belajar/Hari", 0.5, 9.0, 4.0, 0.1)
        stress      = st.slider("Indeks Stres (1–10)", 1.0, 10.0, 5.5, 0.1)
        travel_time = st.slider("Waktu Perjalanan (menit)", 5.0, 75.0, 30.0, 0.5)
        delay_days  = st.slider("Keterlambatan Tugas (hari)", 0, 8, 2)
        part_time   = st.selectbox("Kerja Paruh Waktu", ["No","Yes"])
        scholarship = st.selectbox("Beasiswa", ["No","Yes"])

    st.markdown("---")
    _,mid,_ = st.columns([1,2,1])
    with mid:
        predict_btn = st.button("🔮  PREDIKSI SEKARANG", use_container_width=True)

    if predict_btn:
        inp = {
            'Age':age, 'Gender':1 if gender=="Male" else 0,
            'Family_Income':family_income,
            'Internet_Access':1 if internet=="Yes" else 0,
            'Study_Hours_per_Day':study_hours, 'Attendance_Rate':attendance,
            'Assignment_Delay_Days':delay_days, 'Travel_Time_Minutes':travel_time,
            'Part_Time_Job':1 if part_time=="Yes" else 0,
            'Scholarship':1 if scholarship=="Yes" else 0,
            'Stress_Index':stress, 'GPA':gpa, 'Semester_GPA':semester_gpa,
            'Semester_Year 1':1 if semester=="Year 1" else 0,
            'Semester_Year 2':1 if semester=="Year 2" else 0,
            'Semester_Year 3':1 if semester=="Year 3" else 0,
            'Semester_Year 4':1 if semester=="Year 4" else 0,
            'Department_Arts':1 if department=="Arts" else 0,
            'Department_Business':1 if department=="Business" else 0,
            'Department_CS':1 if department=="CS" else 0,
            'Department_Engineering':1 if department=="Engineering" else 0,
            'Department_Science':1 if department=="Science" else 0,
            'Parental_Education_Bachelor':1 if parental_edu=="Bachelor" else 0,
            'Parental_Education_High School':1 if parental_edu=="High School" else 0,
            'Parental_Education_Master':1 if parental_edu=="Master" else 0,
            'Parental_Education_PhD':1 if parental_edu=="PhD" else 0,
        }
        input_df     = pd.DataFrame([inp])[feature_names]
        input_scaled = scaler.transform(input_df)
        lr           = models['Logistic Regression']
        pred         = lr.predict(input_scaled)[0]
        prob         = lr.predict_proba(input_scaled)[0]
        p_drop       = prob[1]*100; p_safe = prob[0]*100

        st.markdown("<div class='section-title'>🎯 Hasil Prediksi</div>", unsafe_allow_html=True)

        if pred==1:
            st.markdown(f"""<div class='alert-dropout'>
            <h3>⚠️ BERISIKO DROPOUT</h3>
            <p>Model memprediksi mahasiswa ini <b>berisiko mengalami dropout</b>.</p>
            <p>Probabilitas Dropout: <b>{p_drop:.2f}%</b></p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class='alert-safe'>
            <h3>✅ TIDAK BERISIKO DROPOUT</h3>
            <p>Model memprediksi mahasiswa ini <b>aman dari risiko dropout</b>.</p>
            <p>Probabilitas Tidak Dropout: <b>{p_safe:.2f}%</b></p>
            </div>""", unsafe_allow_html=True)

        c1,c2 = st.columns(2)
        with c1: st.metric("✅ Prob. Tidak Dropout", f"{p_safe:.2f}%")
        with c2: st.metric("⚠️ Prob. Dropout",       f"{p_drop:.2f}%")
        st.markdown("**Tingkat Risiko Dropout:**"); st.progress(int(p_drop))

        st.markdown("<div class='section-title'>📊 Prediksi Semua Model</div>", unsafe_allow_html=True)
        rows = []
        for name,model in models.items():
            p  = model.predict(input_scaled)[0]
            pr = model.predict_proba(input_scaled)[0]
            rows.append({'Model':name,
                         'Prediksi':'⚠️ Dropout' if p==1 else '✅ Tidak Dropout',
                         'Prob. Aman':f"{pr[0]*100:.2f}%",
                         'Prob. Dropout':f"{pr[1]*100:.2f}%"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if pred==1:
            st.markdown("<div class='section-title'>💡 Rekomendasi Intervensi</div>", unsafe_allow_html=True)
            recs = []
            if gpa<2.0:        recs.append(("🔴","GPA Rendah","Bimbingan akademik intensif & program remedial"))
            if attendance<75:  recs.append(("🟠","Kehadiran Rendah","Konseling motivasi & monitoring kehadiran"))
            if stress>7:       recs.append(("🟣","Stres Tinggi","Rujuk konseling psikologis kampus"))
            if study_hours<2:  recs.append(("🟡","Belajar Minim","Program mentoring & jadwal belajar terstruktur"))
            if part_time=="Yes": recs.append(("🔵","Kerja Paruh Waktu","Pelatihan manajemen waktu & beasiswa"))
            if not recs:       recs.append(("⚪","Monitoring","Pemantauan berkala oleh wali akademik"))
            for icon,title,desc in recs:
                st.warning(f"{icon} **{title}** — {desc}")
        else:
            st.success("✅ Mahasiswa dalam kondisi baik. Pertahankan dan tingkatkan prestasi!")


# ============================================================
# PAGE 4: PERBANDINGAN MODEL
# ============================================================
elif menu == "📈  Perbandingan Model":
    st.markdown("""
    <div class='hero-header'>
        <div class='hero-badge'>Phase 5 · Evaluation</div>
        <div class='hero-title'>Perbandingan Model</div>
        <div class='hero-subtitle'>Evaluasi performa 4 algoritma Machine Learning secara komprehensif</div>
    </div>""", unsafe_allow_html=True)

    if not load_success: st.error(f"❌ {error_msg}"); st.stop()

    rdf = pd.DataFrame(results).T.sort_values('F1-Score', ascending=False)

    def highlight_best(s):
        return ['background-color:rgba(99,102,241,0.2);color:#a5b4fc;font-weight:700'
                if v==s.max() else '' for v in s]

    st.markdown("<div class='section-title'>📊 Tabel Metrik</div>", unsafe_allow_html=True)
    st.dataframe(rdf.style.apply(highlight_best).format("{:.4f}"), use_container_width=True)

    st.markdown("""<div class='info-box'>
    🏆 <b>Logistic Regression</b> — F1-Score terbaik <b>(0.5847)</b>,
    AUC-ROC terbaik <b>(0.8213)</b>, Recall terbaik <b>(76.22%)</b>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>📊 Visualisasi Perbandingan</div>", unsafe_allow_html=True)
    metrics = ['Accuracy','Precision','Recall','F1-Score','AUC-ROC']
    x = np.arange(len(metrics)); w = 0.18
    pal = ['#6366f1','#f59e0b','#34d399','#ef4444']
    fig, ax = plt.subplots(figsize=(13,5.5))
    for i,(name,color) in enumerate(zip(rdf.index,pal)):
        vals = [rdf.loc[name,m] for m in metrics]
        bars = ax.bar(x+i*w, vals, w, label=name, color=color, alpha=0.85, edgecolor='none', zorder=3)
        for bar,v in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=6.5, color='#94a3b8', fontweight='bold')
    ax.set_xticks(x+w*1.5); ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0,1.1); ax.set_ylabel('Score',fontsize=10)
    ax.legend(fontsize=9, framealpha=0.15, facecolor='#1e293b', edgecolor='#334155', labelcolor='#94a3b8')
    ax.grid(axis='y', alpha=0.2, zorder=0)
    ax.spines[['top','right','left','bottom']].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("<div class='section-title'>📖 Penjelasan Metrik</div>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("""| Metrik | Penjelasan |
|---|---|
| **Accuracy** | % total prediksi yang benar |
| **Precision** | % prediksi dropout yang tepat |
| **Recall** | % dropout yang berhasil terdeteksi |""")
    with c2:
        st.markdown("""| Metrik | Penjelasan |
|---|---|
| **F1-Score** | Harmonic mean Precision & Recall |
| **AUC-ROC** | Kemampuan diskriminasi model |
| **Prioritas** | Recall & F1 terpenting untuk dropout |""")


# ============================================================
# PAGE 5: FEATURE IMPORTANCE
# ============================================================
elif menu == "🔍  Feature Importance":
    st.markdown("""
    <div class='hero-header'>
        <div class='hero-badge'>Phase 5 · Model Interpretation</div>
        <div class='hero-title'>Feature Importance</div>
        <div class='hero-subtitle'>Faktor paling berpengaruh terhadap prediksi dropout — Random Forest</div>
    </div>""", unsafe_allow_html=True)

    if not load_success: st.error(f"❌ {error_msg}"); st.stop()

    top_n    = st.slider("Tampilkan Top N Fitur:", 5, 26, 15)
    top_feat = feat_imp.head(top_n).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, max(5,top_n*0.5)))
    colors_bar = ['#ef4444' if i==0 else '#f97316' if i==1 else '#f59e0b' if i==2
                  else '#6366f1' if i<6 else '#334155' for i in range(len(top_feat))]
    bars = ax.barh(top_feat['Feature'][::-1], top_feat['Importance'][::-1],
                   color=colors_bar[::-1], edgecolor='none', alpha=0.9)
    for bar,val in zip(bars, top_feat['Importance'][::-1]):
        ax.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9, color='#94a3b8', fontweight='bold')
    ax.set_xlabel('Importance Score', fontsize=10)
    ax.set_title(f'Top {top_n} Feature Importance — Random Forest',
                 fontsize=13, fontweight='bold', color='#e2e8f0', pad=15)
    ax.spines[['top','right','left','bottom']].set_visible(False)
    ax.grid(axis='x', alpha=0.2, zorder=0)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("<div class='section-title'>📋 Tabel Detail</div>", unsafe_allow_html=True)
    fs = top_feat.copy()
    fs.insert(0,'Rank',range(1,len(fs)+1))
    fs['Importance (%)'] = (fs['Importance']*100).round(2)
    fs['Kategori'] = fs['Rank'].apply(lambda r:'🥇 Sangat Tinggi' if r<=3 else
                                       '🥈 Tinggi' if r<=6 else '🥉 Sedang' if r<=10 else '⚪ Rendah')
    st.dataframe(fs[['Rank','Feature','Importance','Importance (%)','Kategori']],
                 use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title'>💡 Insight Utama</div>", unsafe_allow_html=True)
    interp = ["IPK kumulatif adalah penentu utama dropout — makin rendah GPA makin tinggi risiko",
              "Performa semester terkini sangat mempengaruhi kemungkinan dropout mahasiswa",
              "Tingkat stres & kehadiran berperan besar dalam keputusan mahasiswa untuk bertahan"]
    for i,((_,row),desc,medal) in enumerate(zip(feat_imp.head(3).iterrows(), interp, ['🥇','🥈','🥉'])):
        st.markdown(f"""<div class='info-box'>
        <b>{medal} #{i+1} — {row['Feature']}</b> &nbsp;|&nbsp;
        Importance: <b>{row['Importance']:.4f}</b> ({row['Importance']*100:.1f}%)<br>
        <span style='color:#64748b;font-size:0.85rem;'>{desc}</span>
        </div>""", unsafe_allow_html=True)


# ============================================================
# PAGE 6: TENTANG
# ============================================================
elif menu == "ℹ️   Tentang Aplikasi":
    st.markdown("""
    <div class='hero-header'>
        <div class='hero-badge'>Documentation</div>
        <div class='hero-title'>Tentang Aplikasi</div>
        <div class='hero-subtitle'>Dokumentasi lengkap metodologi, model, dan teknologi yang digunakan</div>
    </div>""", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("""<div class='phase-card'><h4>📖 Deskripsi Penelitian</h4>
        <p>Aplikasi ini merupakan hasil penelitian skripsi yang membangun sistem prediksi
        dropout mahasiswa menggunakan metodologi <b style='color:#a5b4fc'>CRISP-DM</b>.<br><br>
        Dataset: <b style='color:#a5b4fc'>10.000 record</b> dengan 19 fitur asli mencakup
        aspek akademik, demografis, dan perilaku mahasiswa.</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='phase-card'><h4>🎯 Tujuan Penelitian</h4>
        <p>✅ Membangun model prediksi dropout akurat<br>
        ✅ Identifikasi faktor utama penyebab dropout<br>
        ✅ Membandingkan 4 algoritma ML<br>
        ✅ Deploy sistem prediksi berbasis web<br>
        ✅ Mendukung intervensi dini kampus</p></div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>🔄 Tahapan CRISP-DM</div>", unsafe_allow_html=True)
    phases = [
        ("1","Business Understanding","Identifikasi masalah, tujuan bisnis & kriteria keberhasilan"),
        ("2","Data Understanding","EDA, statistik deskriptif, analisis korelasi & visualisasi"),
        ("3","Data Preparation","Missing values, outlier IQR, encoding, scaling, oversampling"),
        ("4","Modeling","Training 4 algoritma dengan train-test split 80:20"),
        ("5","Evaluation","Accuracy, Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix"),
        ("6","Deployment","Aplikasi Streamlit 6 halaman interaktif"),
    ]
    for num,title,desc in phases:
        st.markdown(f"""<div class='info-box' style='margin:0.4rem 0;'>
        <b style='color:#a5b4fc'>Phase {num} · {title}</b><br>
        <span style='color:#64748b;font-size:0.85rem;'>{desc}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>🤖 Model & Performa</div>", unsafe_allow_html=True)
    st.markdown("""
| No | Algoritma | Peran | AUC-ROC | F1-Score |
|---|---|---|---|---|
| 1 | **Logistic Regression** | 🏆 Model Terbaik | 0.8213 | 0.5847 |
| 2 | Gradient Boosting | Boosting Model | 0.8107 | 0.5845 |
| 3 | Random Forest | Ensemble Model | 0.7983 | 0.5336 |
| 4 | Decision Tree | Interpretable | 0.6399 | 0.4498 |
    """)

    st.markdown("<div class='section-title'>🛠️ Teknologi</div>", unsafe_allow_html=True)
    tc1,tc2,tc3 = st.columns(3)
    with tc1:
        st.markdown("""<div class='phase-card'><h4>🐍 Backend</h4>
        <p>Python 3.x<br>Scikit-learn 1.6.1<br>Pandas 2.2.2<br>NumPy 2.0.2<br>Joblib</p></div>""",
        unsafe_allow_html=True)
    with tc2:
        st.markdown("""<div class='phase-card'><h4>📊 Visualisasi</h4>
        <p>Matplotlib<br>Seaborn<br>Custom Dark Theme<br>Interactive Charts</p></div>""",
        unsafe_allow_html=True)
    with tc3:
        st.markdown("""<div class='phase-card'><h4>🚀 Deployment</h4>
        <p>Streamlit<br>Streamlit Cloud<br>GitHub Integration<br>Real-time Prediction</p></div>""",
        unsafe_allow_html=True)