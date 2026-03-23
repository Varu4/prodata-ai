# ==================================================
# PRODATA AI — Ultimate Edition v15.0
# TWO MODES: One-Click (auto everything) + Manual (full control)
# Built by Varun Walekar
# ==================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile, os, re
import plotly.io as pio
pio.defaults.default_format = "png"

from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, LabelEncoder)
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                               GradientBoostingRegressor, GradientBoostingClassifier)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                              accuracy_score, mean_absolute_percentage_error,
                              confusion_matrix)
from fpdf import FPDF
import anthropic

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ProData AI | Ultimate Edition",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* ── Main background — deep navy with subtle grid ── */
.main {
    background-color: #070b14;
    background-image:
        linear-gradient(rgba(0,212,170,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,170,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
}
.block-container {
    padding-top: 1.8rem !important;
    padding-bottom: 2rem !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0e1a 0%, #0d1526 100%);
    border-right: 1px solid rgba(0,212,170,0.15);
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stCaption { color: #8892a4 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #e2e8f0 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(0,212,170,0.12) !important; }
[data-testid="stSidebar"] .stTextInput input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(0,212,170,0.2) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(0,212,170,0.2) !important;
    color: #e2e8f0 !important;
}

/* ── All text in main area ── */
.stMarkdown, p, span, label, .stText { color: #cbd5e1; }
h1, h2, h3 { color: #f1f5f9 !important; font-weight: 600; letter-spacing: -0.02em; }

/* ── Metrics ── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(0,212,170,0.06), rgba(0,212,170,0.02));
    border: 1px solid rgba(0,212,170,0.18);
    border-radius: 14px;
    padding: 1.1rem 1.2rem;
    position: relative;
    overflow: hidden;
}
div[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00d4aa, transparent);
}
div[data-testid="stMetricValue"] {
    font-size: 1.85rem !important;
    font-weight: 700 !important;
    color: #00d4aa !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: -0.03em;
}
div[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
div[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-bottom: 1px solid rgba(0,212,170,0.12);
    gap: 0;
    padding: 0 4px;
    border-radius: 10px 10px 0 0;
}
.stTabs [data-baseweb="tab"] {
    color: #64748b !important;
    font-size: 0.82rem !important;
    font-weight: 500;
    padding: 10px 18px;
    border-bottom: 2px solid transparent;
    font-family: 'Sora', sans-serif;
    letter-spacing: 0.01em;
}
.stTabs [aria-selected="true"] {
    color: #00d4aa !important;
    border-bottom: 2px solid #00d4aa !important;
    background: rgba(0,212,170,0.06) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-top: none;
    border-radius: 0 0 12px 12px;
    padding: 1.5rem 1.25rem;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00d4aa, #00b894);
    color: #070b14 !important;
    border: none;
    border-radius: 9px;
    font-weight: 600;
    font-family: 'Sora', sans-serif;
    font-size: 0.82rem;
    letter-spacing: 0.02em;
    padding: 0.5rem 1.2rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,212,170,0.4);
    background: linear-gradient(135deg, #00e6ba, #00d4aa);
}
.stButton > button:active { transform: translateY(0); }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #4f46e5);
    color: white !important;
    box-shadow: 0 0 20px rgba(99,102,241,0.25);
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(99,102,241,0.45);
}
.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.05);
    color: #94a3b8 !important;
    border: 1px solid rgba(255,255,255,0.1);
}
.stButton > button[kind="secondary"]:hover {
    background: rgba(255,255,255,0.09);
    border-color: rgba(0,212,170,0.3);
    color: #e2e8f0 !important;
    box-shadow: none;
}

/* ── Dataframes ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }
.stDataFrame table { background: rgba(255,255,255,0.02) !important; }
.stDataFrame th { background: rgba(0,212,170,0.08) !important; color: #00d4aa !important; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; }
.stDataFrame td { color: #cbd5e1 !important; font-size: 0.83rem; border-color: rgba(255,255,255,0.04) !important; }

/* ── Inputs ── */
.stTextInput input, .stSelectbox > div > div, .stMultiSelect > div > div, .stSlider {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #e2e8f0 !important;
    border-radius: 9px !important;
}
.stTextInput input:focus, .stSelectbox > div > div:focus {
    border-color: rgba(0,212,170,0.5) !important;
    box-shadow: 0 0 0 2px rgba(0,212,170,0.12) !important;
}
.stRadio label, .stCheckbox label { color: #94a3b8 !important; font-size: 0.85rem; }

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #00d4aa, #6366f1) !important;
    border-radius: 99px;
}
.stProgress > div { background: rgba(255,255,255,0.06) !important; border-radius: 99px; }

/* ── Alerts ── */
.stSuccess { background: rgba(0,212,170,0.08) !important; border: 1px solid rgba(0,212,170,0.3) !important; color: #6ee7b7 !important; border-radius: 10px; }
.stWarning { background: rgba(251,191,36,0.08) !important; border: 1px solid rgba(251,191,36,0.25) !important; color: #fbbf24 !important; border-radius: 10px; }
.stError { background: rgba(239,68,68,0.08) !important; border: 1px solid rgba(239,68,68,0.25) !important; color: #f87171 !important; border-radius: 10px; }
.stInfo { background: rgba(99,102,241,0.08) !important; border: 1px solid rgba(99,102,241,0.25) !important; color: #a5b4fc !important; border-radius: 10px; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 9px !important;
    color: #94a3b8 !important;
}
.streamlit-expanderContent {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-top: none !important;
    border-radius: 0 0 9px 9px !important;
}

/* ── Badges ── */
.badge-done {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(0,212,170,0.12);
    color: #00d4aa;
    border: 1px solid rgba(0,212,170,0.3);
    padding: 4px 13px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.04em;
}
.badge-done::before { content: '✓ '; }
.badge-running {
    display: inline-flex; align-items: center;
    background: rgba(99,102,241,0.12);
    color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.3);
    padding: 4px 13px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600;
    animation: pulse 1.5s ease-in-out infinite;
}
.badge-wait {
    display: inline-flex; align-items: center;
    background: rgba(255,255,255,0.04);
    color: #475569;
    border: 1px solid rgba(255,255,255,0.07);
    padding: 4px 13px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 500;
}
.badge-manual {
    display: inline-flex; align-items: center;
    background: rgba(139,92,246,0.12);
    color: #c4b5fd;
    border: 1px solid rgba(139,92,246,0.3);
    padding: 4px 13px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.6} }

/* ── Chat bubbles ── */
.chat-ai {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(0,212,170,0.15);
    border-left: 3px solid #00d4aa;
    border-radius: 0 14px 14px 14px;
    padding: 1rem 1.25rem;
    margin: 0.6rem 0;
    font-size: 0.87rem;
    line-height: 1.75;
    color: #cbd5e1;
    backdrop-filter: blur(4px);
}
.chat-user {
    background: rgba(99,102,241,0.07);
    border: 1px solid rgba(99,102,241,0.2);
    border-right: 3px solid #6366f1;
    border-radius: 14px 0 14px 14px;
    padding: 1rem 1.25rem;
    margin: 0.6rem 0;
    font-size: 0.87rem;
    color: #c7d2fe;
    text-align: right;
}
.lbl-ai {
    font-size: 0.66rem; font-weight: 700;
    color: #00d4aa; text-transform: uppercase;
    letter-spacing: 0.1em; margin-bottom: 4px;
}
.lbl-user {
    font-size: 0.66rem; font-weight: 700;
    color: #6366f1; text-transform: uppercase;
    letter-spacing: 0.1em; margin-bottom: 4px; text-align: right;
}
.mem-pill {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(0,212,170,0.1);
    color: #00d4aa;
    border: 1px solid rgba(0,212,170,0.25);
    padding: 3px 12px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600; margin-bottom: 10px;
}
.mem-pill::before {
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    background: #00d4aa;
    border-radius: 50%;
    animation: pulse 1.5s ease-in-out infinite;
}

/* ── Insight / warn boxes ── */
.insight-box {
    background: rgba(0,212,170,0.05);
    border: 1px solid rgba(0,212,170,0.2);
    border-left: 3px solid #00d4aa;
    border-radius: 0 12px 12px 12px;
    padding: 1rem 1.25rem;
    font-size: 0.87rem;
    color: #a7f3d0;
    line-height: 1.75;
    margin: 0.75rem 0;
}
.warn-box {
    background: rgba(251,191,36,0.06);
    border: 1px solid rgba(251,191,36,0.2);
    border-left: 3px solid #fbbf24;
    border-radius: 0 12px 12px 12px;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: #fde68a;
    margin: 0.5rem 0;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1.5px dashed rgba(0,212,170,0.25) !important;
    border-radius: 12px !important;
    padding: 0.5rem;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,212,170,0.5) !important;
    background: rgba(0,212,170,0.03) !important;
}

/* ── Plotly chart backgrounds ── */
.js-plotly-plot .plotly .bg { fill: rgba(7,11,20,0) !important; }

/* ── Link button ── */
.stLinkButton a {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(0,212,170,0.2) !important;
    color: #00d4aa !important;
    border-radius: 9px !important;
    font-size: 0.82rem !important;
    font-family: 'Sora', sans-serif !important;
    transition: all 0.2s;
}
.stLinkButton a:hover {
    background: rgba(0,212,170,0.08) !important;
    box-shadow: 0 4px 14px rgba(0,212,170,0.2) !important;
}

/* ── Caption / small text ── */
.stCaption { color: #475569 !important; font-size: 0.77rem !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
::-webkit-scrollbar-thumb { background: rgba(0,212,170,0.25); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,212,170,0.45); }
</style>
""", unsafe_allow_html=True)


# ── Plotly dark theme ────────────────────────────────────────────────────────
DARK_LAYOUT = dict(
    paper_bgcolor='rgba(7,11,20,0)',
    plot_bgcolor='rgba(255,255,255,0.02)',
    font=dict(family='Sora, sans-serif', color='#94a3b8', size=11),
    title_font=dict(family='Sora, sans-serif', color='#e2e8f0', size=13),
    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.08)',
               tickfont=dict(color='#64748b'), title_font=dict(color='#94a3b8')),
    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.08)',
               tickfont=dict(color='#64748b'), title_font=dict(color='#94a3b8')),
    legend=dict(bgcolor='rgba(255,255,255,0.03)', bordercolor='rgba(255,255,255,0.08)',
                borderwidth=1, font=dict(color='#94a3b8')),
    margin=dict(l=16, r=16, t=44, b=16),
)
CHART_COLORS = ['#00d4aa', '#6366f1', '#f59e0b', '#ec4899', '#3b82f6', '#10b981', '#f97316']

def dark_fig(fig, height=320):
    fig.update_layout(height=height, **DARK_LAYOUT)
    return fig

# ── PDF ───────────────────────────────────────────────────────────────────────
class EnterpriseReport(FPDF):
    def __init__(self, client, project, agent, mode=""):
        super().__init__()
        self.client = client
        self.project = project
        self.agent = agent
        self.mode = mode

    def header(self):
        self.set_fill_color(15, 17, 23)
        self.rect(0, 0, 210, 38, 'F')
        self.set_text_color(255, 255, 255)
        self.set_font('Arial', 'B', 18)
        self.set_y(12)
        self.cell(0, 8, self.project.upper(), 0, 1, 'C')
        self.set_font('Arial', '', 9)
        self.set_text_color(180, 180, 180)
        self.cell(0, 7, f"Client: {self.client}  |  ProData AI v15.0  |  Mode: {self.mode}", 0, 1, 'C')
        self.ln(18)

    def footer(self):
        self.set_y(-14)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Generated by {self.agent} | ProData AI | Page {self.page_no()}", 0, 0, 'C')

    def section(self, title):
        self.ln(5)
        self.set_fill_color(37, 99, 235)
        self.set_text_color(255, 255, 255)
        self.set_font('Arial', 'B', 11)
        self.cell(0, 8, f"  {title}", 0, 1, 'L', fill=True)
        self.ln(3)

    def body(self, text):
        self.set_font('Arial', '', 10)
        self.set_text_color(50, 50, 50)
        clean = str(text).encode('latin-1', 'ignore').decode('latin-1')
        clean = re.sub(r'\*\*(.+?)\*\*', r'\1', clean)
        clean = re.sub(r'\*(.+?)\*', r'\1', clean)
        clean = re.sub(r'#+\s*', '', clean)
        self.multi_cell(0, 5.5, clean)
        self.ln(1)

    def mono(self, text):
        self.set_font('Courier', '', 8)
        self.set_text_color(30, 30, 30)
        clean = str(text).encode('latin-1', 'ignore').decode('latin-1')
        self.multi_cell(0, 4.5, clean)
        self.ln(1)

    def add_img(self, path, title=''):
        if not os.path.exists(path):
            return
        try:
            self.add_page()
            if title:
                self.set_font('Arial', 'B', 12)
                self.set_text_color(30, 30, 30)
                self.cell(0, 8, title, 0, 1, 'C')
                self.ln(3)
            self.image(path, x=10, w=185)
        except Exception as e:
            self.body(f"[Image error: {e}]")


# ── Shared helpers ────────────────────────────────────────────────────────────
def get_types(df):
    return (df.select_dtypes(include=np.number).columns.tolist(),
            df.select_dtypes(exclude=np.number).columns.tolist())

def save_fig(fig, title, key='pdf_imgs'):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    try:
        fig.write_image(tmp.name, format="png", scale=2)
        if os.path.exists(tmp.name):
            st.session_state[key].append({"title": title, "path": tmp.name})
    except Exception:
        pass

def call_claude(prompt, history, system, api_key):
    client = anthropic.Anthropic(api_key=api_key)
    msgs = history + [{"role": "user", "content": prompt}]
    r = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=system,
        messages=msgs
    )
    return r.content[0].text

def build_ctx(df, results):
    num_cols, cat_cols = get_types(df)
    stats = []
    for c in num_cols[:10]:
        s = df[c].dropna()
        if len(s):
            stats.append(f"{c}: avg={s.mean():.1f}, min={s.min()}, max={s.max()}, nulls={df[c].isna().sum()}")
    ctx = (f"Dataset: {df.shape[0]} rows x {df.shape[1]} cols\n"
           f"Numeric: {', '.join(num_cols)}\nCategorical: {', '.join(cat_cols)}\n"
           f"Missing total: {df.isna().sum().sum()}\n\nStats:\n" + "\n".join(stats))
    if results.get('ml'):
        ml = results['ml']
        ctx += f"\n\nML: target={ml['target']}"
        ctx += f", accuracy={ml['acc']:.2%}" if ml.get('is_class') else f", R2={ml.get('r2','N/A'):.3f}, MAE={ml.get('mae','N/A'):.2f}"
    if results.get('forecast'):
        fc = results['forecast']
        ctx += f"\n\nForecast: {fc['col']} for {fc['horizon']} days, latest={fc['latest']:.2f}, projected={fc['projected']:.2f}"
    if results.get('drivers'):
        dr = results['drivers']
        ctx += f"\n\nTop KPI driver for {dr['kpi']}: {dr['top']}"
    ctx += f"\n\nSample:\n{df.head(30).to_csv(index=False)}"
    return ctx

def build_pdf(results, df, audit, chat_msgs, client_name, project, agent, mode):
    pdf = EnterpriseReport(client_name, project, agent, mode)
    pdf.add_page()

    pdf.section("1. Executive Summary")
    pdf.body(f"Analysis of: {df.shape[0]} rows x {df.shape[1]} columns. Mode: {mode}.")
    if results.get('ml'):
        ml = results['ml']
        if ml.get('is_class'):
            pdf.body(f"ML predicted '{ml['target']}' with {ml['acc']:.2%} accuracy.")
        elif ml.get('r2') is not None:
            pdf.body(f"ML predicted '{ml['target']}' with R2={ml['r2']:.3f}, MAE={ml['mae']:.2f}.")
    if results.get('drivers'):
        pdf.body(f"Top driver for '{results['drivers']['kpi']}': {results['drivers']['top']}.")
    if results.get('forecast'):
        fc = results['forecast']
        pdf.body(f"30-day forecast: {fc['col']} from {fc['latest']:.1f} to {fc['projected']:.1f}.")

    if results.get('ai_insights'):
        pdf.section("2. AI Analyst Insights")
        pdf.body(results['ai_insights'])

    if chat_msgs:
        pdf.section("3. AI Chat Conversation")
        for msg in chat_msgs:
            role = "AI Analyst" if msg['role'] == 'assistant' else "You"
            pdf.body(f"[{role}]")
            content = msg['content'][:600] + ("..." if len(msg['content']) > 600 else "")
            pdf.body(content)
            pdf.ln(2)

    pdf.section("4. Data Cleaning Log")
    for line in results.get('clean', ['No issues found.']):
        pdf.body(f"- {line}")

    pdf.section("5. Dataset Profile")
    for col, dtype in df.dtypes.items():
        pdf.body(f"  {col}: {dtype}  |  {df[col].isna().sum()} nulls")
    pdf.ln(2)
    pdf.body("Statistics:")
    pdf.mono(df.describe().round(2).to_string())

    if results.get('ml'):
        ml = results['ml']
        pdf.section("6. ML Model Results")
        pdf.body(f"Target: {ml['target']}  |  Model: {ml.get('model_name', 'Random Forest')}")
        if ml.get('is_class'):
            pdf.body(f"Accuracy: {ml['acc']:.2%}")
        else:
            pdf.body(f"R2: {ml['r2']:.3f}   MAE: {ml['mae']:.2f}   RMSE: {ml['rmse']:.2f}")
        if ml.get('importances'):
            pdf.body("Top features:")
            for k, v in list(ml['importances'].items())[:5]:
                pdf.body(f"  - {k}: {v:.4f}")

    if results.get('forecast'):
        fc = results['forecast']
        pdf.section("7. Forecast Results")
        pdf.body(f"Metric: {fc['col']}   Horizon: {fc['horizon']} days")
        pdf.body(f"Latest: {fc['latest']:.2f}   Projected: {fc['projected']:.2f}")
        if fc.get('mape'):
            pdf.body(f"MAPE: {fc['mape']*100:.1f}%   RMSE: {fc['rmse']:.2f}")

    if results.get('drivers'):
        dr = results['drivers']
        pdf.section("8. Business Drivers")
        pdf.body(f"KPI: {dr['kpi']}   Top driver: {dr['top']}")
        for k, v in dr['top5'].items():
            pdf.body(f"  - {k}: {v:.4f}")

    pdf.section("9. Audit Log")
    for line in audit:
        pdf.body(f"- {line}")

    pdf.section("10. Strategic Recommendations")
    for rec in [
        "Focus on the top identified driver for maximum KPI impact.",
        "Review columns with high null counts before high-stakes decisions.",
        "Use the forecast to plan inventory, staffing, or budget proactively.",
        "Re-run analysis monthly to detect trend changes early.",
        "Share this report with stakeholders to align on data-driven priorities.",
    ]:
        pdf.body(f"- {rec}")

    for img in st.session_state.get('pdf_imgs', []):
        pdf.add_img(img['path'], img.get('title', ''))
        try:
            os.remove(img['path'])
        except Exception:
            pass
    st.session_state['pdf_imgs'] = []

    result = pdf.output()
    return bytes(result) if not isinstance(result, bytes) else result


# ── Session state ─────────────────────────────────────────────────────────────
def init():
    defaults = {
        'df': None, 'fname': '', 'mode': 'oneclick',
        # one-click state
        'oc_ran': False, 'oc_results': {}, 'oc_audit': [],
        # manual state
        'man_results': {}, 'man_audit': [],
        # shared
        'chat_msgs': [], 'chat_hist': [],
        'pdf_imgs': [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
init()

ID_KW = ['id', 'index', 'serial', 'customerid', 'orderid']


# ══════════════════════════════════════════════════════════════════════════════
# AUTOML ENGINE — trains 6 models, ranks them, picks the best automatically
# ══════════════════════════════════════════════════════════════════════════════
def run_automl(df, target, feats, split=0.2, auto_target=False):
    """
    Trains 6 models on the data, ranks them by key metric,
    returns dict with winner stats, leaderboard, and charts.
    """
    X = pd.get_dummies(df[feats], drop_first=True).fillna(0)
    y = df[target].fillna(df[target].median())
    is_class = (y.nunique() < 15) and (y.nunique() > 1)
    y_enc = LabelEncoder().fit_transform(y) if is_class else y.values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_enc, test_size=split, random_state=42)

    if is_class:
        candidates = [
            ("Random Forest",     RandomForestClassifier(100, random_state=42)),
            ("Gradient Boosting", GradientBoostingClassifier(random_state=42)),
            ("Logistic Reg.",     LogisticRegression(max_iter=1000, random_state=42)),
            ("Decision Tree",     DecisionTreeClassifier(max_depth=8, random_state=42)),
            ("K-Nearest Neigh.", KNeighborsClassifier(n_neighbors=5)),
            ("Extra Trees",       RandomForestClassifier(100, random_state=42, bootstrap=False)),
        ]
    else:
        candidates = [
            ("Random Forest",     RandomForestRegressor(100, random_state=42)),
            ("Gradient Boosting", GradientBoostingRegressor(random_state=42)),
            ("Ridge Regression",  Ridge()),
            ("Lasso Regression",  Lasso(max_iter=5000)),
            ("Decision Tree",     DecisionTreeRegressor(max_depth=8, random_state=42)),
            ("K-Nearest Neigh.", KNeighborsRegressor(n_neighbors=5)),
        ]

    leaderboard = []
    trained_models = {}
    for name, mdl in candidates:
        try:
            mdl.fit(X_tr, y_tr)
            preds = mdl.predict(X_te)
            if is_class:
                score = accuracy_score(y_te, preds)
                row = {"Model": name, "Accuracy": round(score, 4),
                       "Score": score, "MAE": "-", "RMSE": "-", "R²": "-"}
            else:
                r2  = r2_score(y_te, preds)
                mae = mean_absolute_error(y_te, preds)
                rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
                score = r2 if r2 >= 0 else -999
                row = {"Model": name, "R²": round(r2, 4),
                       "MAE": round(mae, 4), "RMSE": round(rmse, 4),
                       "Score": score, "Accuracy": "-"}
            row["Status"] = "✓"
            leaderboard.append(row)
            trained_models[name] = (mdl, preds)
        except Exception as e:
            leaderboard.append({"Model": name, "Score": -999,
                                 "Status": "✗", "Error": str(e)})

    leaderboard.sort(key=lambda x: x["Score"], reverse=True)
    winner_name = leaderboard[0]["Model"]
    winner_model, winner_preds = trained_models[winner_name]
    winner_row = leaderboard[0]

    # Build leaderboard chart
    lb_df = pd.DataFrame(leaderboard)
    metric_col = "Accuracy" if is_class else "R²"
    lb_df_plot = lb_df[lb_df[metric_col] != "-"].copy()
    lb_df_plot[metric_col] = pd.to_numeric(lb_df_plot[metric_col], errors='coerce')
    bar_colors = ['#00d4aa' if r["Model"] == winner_name else '#334155'
                  for r in leaderboard if r.get(metric_col, "-") != "-"]
    fig_lb = go.Figure(go.Bar(
        x=lb_df_plot[metric_col],
        y=lb_df_plot["Model"],
        orientation='h',
        marker_color=bar_colors,
        text=[f"{v:.3f}" for v in lb_df_plot[metric_col]],
        textposition='outside',
        textfont=dict(color='#94a3b8', size=11),
    ))
    dark_fig(fig_lb, 300)
    fig_lb.update_layout(
        title=dict(text=f"Model Leaderboard — {metric_col}", font=dict(color='#e2e8f0', size=13)),
        xaxis_title=metric_col,
        yaxis=dict(categoryorder='total ascending', tickfont=dict(color='#94a3b8')),
    )

    # Feature importance from winner
    fig_imp = None
    importances = {}
    if hasattr(winner_model, 'feature_importances_'):
        imp = pd.Series(winner_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
        importances = imp.to_dict()
        fig_imp = px.bar(imp, orientation='h',
                         title=f"Feature Importances — {winner_name}",
                         color_discrete_sequence=['#6366f1'])
        fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
        dark_fig(fig_imp, 320)

    # Actual vs predicted (regression only)
    fig_sc = None
    if not is_class:
        fig_sc = px.scatter(
            x=list(y_te), y=list(winner_preds),
            labels={'x': 'Actual', 'y': 'Predicted'},
            color_discrete_sequence=['#00d4aa']
        )
        fig_sc.add_shape(type="line", line=dict(dash="dash", color="#f59e0b"),
                         x0=float(y_enc.min()), y0=float(y_enc.min()),
                         x1=float(y_enc.max()), y1=float(y_enc.max()))
        dark_fig(fig_sc, 300)
        fig_sc.update_layout(title=dict(text=f"Actual vs Predicted — {winner_name}",
                                        font=dict(color='#e2e8f0', size=13)))

    return {
        'target': target,
        'is_class': is_class,
        'model_name': winner_name,
        'leaderboard': leaderboard,
        'fig_lb': fig_lb,
        'fig_imp': fig_imp,
        'fig_sc': fig_sc,
        'importances': importances,
        'acc': winner_row.get('Accuracy') if is_class else None,
        'r2': winner_row.get('R²') if not is_class else None,
        'mae': winner_row.get('MAE') if not is_class else None,
        'rmse': winner_row.get('RMSE') if not is_class else None,
        'n_models': len([r for r in leaderboard if r.get('Status') == '✓']),
    }


def render_automl_tab(df, results, audit, mode_key):
    """Shared AutoML tab UI for both manual and results views."""
    st.markdown("""
    <div style='background:rgba(0,212,170,0.05);border:1px solid rgba(0,212,170,0.2);
    border-left:3px solid #00d4aa;border-radius:0 12px 12px 12px;padding:1rem 1.25rem;margin-bottom:1.25rem;'>
    <strong style='color:#00d4aa;'>AutoML Engine</strong>
    <span style='color:#94a3b8;font-size:0.85rem;'> — trains 6 models simultaneously, ranks them by performance, and picks the winner automatically.</span>
    </div>
    """, unsafe_allow_html=True)

    filtered_cols = [c for c in df.columns if not any(k in c.lower() for k in ID_KW)]
    num_cols, _ = get_types(df)
    c1, c2, c3 = st.columns(3)
    with c1:
        target = st.selectbox("Target variable", filtered_cols, key=f"aml_target_{mode_key}")
    with c2:
        feat_options = [c for c in df.columns if c != target]
        feats = st.multiselect("Features (leave blank = use all numeric)",
                               feat_options, key=f"aml_feats_{mode_key}")
    with c3:
        split_pct = st.slider("Test split %", 10, 40, 20, 5, key=f"aml_split_{mode_key}")

    if st.button("🚀 Run AutoML — Compare All Models", type="primary", key=f"aml_run_{mode_key}"):
        # auto-select all numeric features if none chosen
        use_feats = feats if feats else [c for c in num_cols if c != target and
                                          not any(k in c.lower() for k in ID_KW)]
        if not use_feats:
            st.warning("No numeric features found. Select features manually.")
        else:
            with st.spinner(f"Training 6 models on '{target}'..."):
                try:
                    ml_res = run_automl(df, target, use_feats, split=split_pct/100)
                    results['ml'] = ml_res
                    if mode_key == "manual":
                        st.session_state['man_results'] = results
                    audit.append(f"AutoML: {ml_res['n_models']} models tested, winner = {ml_res['model_name']}")
                    st.session_state[f'aml_done_{mode_key}'] = True
                    st.rerun()
                except Exception as e:
                    st.error(f"AutoML error: {e}")

    # Show results if available
    ml = results.get('ml')
    if ml and ml.get('leaderboard'):
        _render_automl_results(ml)


def _render_automl_results(ml):
    """Render AutoML results: winner banner, leaderboard, charts."""
    metric_label = "Accuracy" if ml['is_class'] else "R²"
    metric_val = f"{ml['acc']:.2%}" if ml['is_class'] else f"{ml['r2']:.4f}"
    n = ml.get('n_models', 6)

    # Winner banner
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,rgba(0,212,170,0.1),rgba(99,102,241,0.08));
    border:1px solid rgba(0,212,170,0.3);border-radius:14px;padding:1.25rem 1.5rem;margin:0.75rem 0 1.25rem;
    display:flex;align-items:center;gap:1rem;'>
        <div style='font-size:2rem;'>🏆</div>
        <div>
            <div style='color:#e2e8f0;font-size:1rem;font-weight:600;margin-bottom:3px;'>
                Best Model: {ml['model_name']}
            </div>
            <div style='color:#64748b;font-size:0.8rem;'>
                Tested {n} models automatically · {metric_label}: 
                <span style='color:#00d4aa;font-weight:600;'>{metric_val}</span>
                · Target: <code style='color:#a5b4fc;'>{ml['target']}</code>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Winner", ml['model_name'].split()[0])
    m2.metric(metric_label, metric_val)
    if not ml['is_class']:
        m3.metric("MAE", f"{ml['mae']:.3f}" if ml['mae'] != '-' else '—')
        m4.metric("RMSE", f"{ml['rmse']:.3f}" if ml['rmse'] != '-' else '—')
    else:
        m3.metric("Models tested", n)
        m4.metric("Task type", "Classification")

    st.divider()

    # Leaderboard table + chart side by side
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.markdown("**Model leaderboard**")
        lb = ml['leaderboard']
        rows = []
        for i, r in enumerate(lb):
            medal = ["🥇", "🥈", "🥉"] [i] if i < 3 else f"#{i+1}"
            name = r['Model']
            status = r.get('Status', '✗')
            if ml['is_class']:
                score = f"{r['Accuracy']:.4f}" if r.get('Accuracy', '-') != '-' else '—'
                rows.append({"": medal, "Model": name, "Accuracy": score, "": status})
            else:
                r2_v = f"{r['R²']:.4f}" if r.get('R²', '-') != '-' else '—'
                mae_v = f"{r['MAE']:.4f}" if r.get('MAE', '-') != '-' else '—'
                rows.append({"": medal, "Model": name, "R²": r2_v, "MAE": mae_v, "": status})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with c2:
        if ml.get('fig_lb'):
            st.plotly_chart(ml['fig_lb'], use_container_width=True)

    # Feature importance + scatter
    if ml.get('fig_imp') or ml.get('fig_sc'):
        c1, c2 = st.columns(2)
        with c1:
            if ml.get('fig_imp'):
                st.plotly_chart(ml['fig_imp'], use_container_width=True)
        with c2:
            if ml.get('fig_sc'):
                st.plotly_chart(ml['fig_sc'], use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ONE-CLICK PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_oneclick(df, api_key, agent, client_name, project):
    results = {}
    st.session_state['pdf_imgs'] = []
    num_cols, cat_cols = get_types(df)
    usable_num = [c for c in num_cols if not any(k in c.lower() for k in ID_KW)]
    audit = []

    prog = st.progress(0, text="Starting one-click analysis...")
    status = st.empty()

    # 1 — Clean
    status.markdown('<div class="badge-running">Step 1/5 — Cleaning data...</div>', unsafe_allow_html=True)
    df_clean = df.copy()
    cleaned = []
    dupes = df_clean.duplicated().sum()
    if dupes:
        df_clean = df_clean.drop_duplicates()
        cleaned.append(f"Removed {dupes} duplicate rows")
        audit.append(f"Removed {dupes} duplicates")
    for c in num_cols:
        n = df_clean[c].isna().sum()
        if n:
            df_clean[c] = df_clean[c].fillna(df_clean[c].median())
            cleaned.append(f"{c}: filled {n} nulls (median)")
            audit.append(f"Filled {n} nulls in '{c}' with median")
    for c in cat_cols:
        n = df_clean[c].isna().sum()
        if n:
            mode_val = df_clean[c].mode()
            df_clean[c] = df_clean[c].fillna(mode_val[0] if len(mode_val) else "Unknown")
            cleaned.append(f"{c}: filled {n} nulls (mode)")
            audit.append(f"Filled {n} nulls in '{c}' with mode")
    results['clean'] = cleaned if cleaned else ['Data was already clean.']
    prog.progress(18)

    # 2 — AutoML
    status.markdown('<div class="badge-running">Step 2/5 — Running AutoML (6 models)...</div>', unsafe_allow_html=True)
    if len(usable_num) >= 2:
        target = usable_num[-1]
        feats = usable_num[:-1]
        try:
            ml_res = run_automl(df_clean, target, feats, auto_target=True)
            results['ml'] = ml_res
            score_str = f"acc={ml_res['acc']:.3f}" if ml_res['is_class'] else f"R2={ml_res['r2']:.3f}"
            audit.append(f"AutoML winner: {ml_res['model_name']} on '{target}' — {score_str}")
        except Exception as e:
            results['ml_error'] = str(e)
    else:
        results['ml_error'] = "Need 2+ numeric columns."
    prog.progress(36)

    # 3 — Forecast
    status.markdown('<div class="badge-running">Step 3/5 — Forecasting...</div>', unsafe_allow_html=True)
    date_cols = [c for c in df_clean.columns if pd.api.types.is_datetime64_any_dtype(df_clean[c])]
    if not date_cols:
        for c in df_clean.columns:
            if any(k in c.lower() for k in ['date', 'time', 'month', 'year', 'day']):
                try:
                    parsed = pd.to_datetime(df_clean[c], errors='coerce')
                    if parsed.notna().sum() > 10:
                        df_clean[c] = parsed
                        date_cols.append(c)
                        break
                except Exception:
                    pass
    if date_cols and usable_num:
        try:
            df_ts = df_clean[[date_cols[0], usable_num[0]]].dropna().sort_values(date_cols[0])
            if len(df_ts) >= 20:
                prophet_df = df_ts.rename(columns={date_cols[0]: 'ds', usable_num[0]: 'y'})
                m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True, interval_width=0.95)
                m.fit(prophet_df.iloc[:int(len(prophet_df)*0.8)])
                freq = pd.infer_freq(prophet_df['ds']) or 'D'
                future = m.make_future_dataframe(periods=30, freq=freq)
                forecast = m.predict(future)
                test_df = prophet_df.iloc[int(len(prophet_df)*0.8):]
                fc_test = forecast[['ds', 'yhat']].merge(test_df[['ds', 'y']], on='ds', how='inner')
                mape = mean_absolute_percentage_error(fc_test['y'], fc_test['yhat']) if len(fc_test) > 2 else None
                rmse_fc = float(np.sqrt(mean_squared_error(fc_test['y'], fc_test['yhat']))) if len(fc_test) > 2 else None
                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], mode='lines', name='Historical', line=dict(color='#00d4aa')))
                fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='#6366f1')))
                fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False))
                fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines',
                                            fill='tonexty', fillcolor='rgba(99,102,241,0.1)', line=dict(width=0), name='95% CI'))
                fig_fc.update_layout(hovermode='x unified'); dark_fig(fig_fc, 340); fig_fc.update_layout(title=dict(text=f'30-Day Forecast — {usable_num[0]}', font=dict(color='#e2e8f0', size=13)))
                results['forecast'] = {'col': usable_num[0], 'horizon': 30,
                                       'latest': float(prophet_df['y'].iloc[-1]),
                                       'projected': float(forecast['yhat'].iloc[-1]),
                                       'mape': mape, 'rmse': rmse_fc, 'fig': fig_fc}
                save_fig(fig_fc, f"30-Day Forecast — {usable_num[0]}")
                audit.append(f"Forecast: {usable_num[0]} for 30 days")
            else:
                results['forecast_skip'] = f"Need 20+ rows with dates (found {len(df_ts)})."
        except Exception as e:
            results['forecast_error'] = str(e)
    else:
        results['forecast_skip'] = "No date column detected."
    prog.progress(55)

    # 4 — Drivers
    status.markdown('<div class="badge-running">Step 4/5 — Business drivers...</div>', unsafe_allow_html=True)
    if len(usable_num) >= 2:
        kpi = usable_num[-1]
        try:
            X_dr = df_clean[[c for c in usable_num if c != kpi]].fillna(0)
            y_dr = df_clean[kpi].fillna(0)
            m_dr = RandomForestRegressor(100, random_state=42)
            m_dr.fit(X_dr, y_dr)
            imp_dr = pd.Series(m_dr.feature_importances_, index=X_dr.columns).sort_values(ascending=False).head(8)
            fig_dr = px.bar(imp_dr, orientation='h', title=f"What drives {kpi}?",
                            color_discrete_sequence=['#6366f1'])
            fig_dr.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False); dark_fig(fig_dr, 320); dark_fig(fig_dr, 300)
            results['drivers'] = {'kpi': kpi, 'top': imp_dr.index[0], 'top5': imp_dr.head(5).to_dict(), 'fig': fig_dr}
            save_fig(fig_dr, f"Business Drivers — {kpi}")
            audit.append(f"Top driver for '{kpi}': '{imp_dr.index[0]}'")
        except Exception as e:
            results['driver_error'] = str(e)
    prog.progress(72)

    # 5 — AI
    status.markdown('<div class="badge-running">Step 5/5 — AI insights...</div>', unsafe_allow_html=True)
    if api_key:
        try:
            ctx = build_ctx(df_clean, results)
            summary = []
            if results.get('ml'):
                ml = results['ml']
                summary.append(f"ML {'acc='+str(round(ml['acc'],3)) if ml['is_class'] else 'R2='+str(round(ml['r2'],3))} on '{ml['target']}'")
            if results.get('drivers'):
                summary.append(f"Top driver: {results['drivers']['top']} -> {results['drivers']['kpi']}")
            if results.get('forecast'):
                fc = results['forecast']
                summary.append(f"Forecast: {fc['col']} {fc['latest']:.1f} -> {fc['projected']:.1f}")
            prompt = (f"Results:\n{chr(10).join(summary)}\n\nContext:\n{ctx}\n\n"
                      "Give 5 specific actionable business insights with numbers. Then 3 concrete recommendations.")
            ai_insights = call_claude(prompt, [], "Expert business data analyst. Specific, numbers-driven, actionable.", api_key)
            results['ai_insights'] = ai_insights
            audit.append("AI insights generated")
        except Exception as e:
            results['ai_insights'] = f"AI unavailable: {e}"
    else:
        results['ai_insights'] = "Add API key in sidebar for AI insights."
    prog.progress(88, text="Generating PDF...")

    # PDF
    try:
        pdf_bytes = build_pdf(results, df_clean, audit, [], client_name, project, agent, "One-Click")
        results['pdf'] = pdf_bytes
        audit.append("PDF auto-generated")
    except Exception as e:
        results['pdf_error'] = str(e)

    prog.progress(100, text="Done!")
    status.markdown('<div class="badge-done">Analysis complete!</div>', unsafe_allow_html=True)

    st.session_state['oc_results'] = results
    st.session_state['oc_audit'] = audit
    st.session_state['df'] = df_clean
    st.session_state['oc_ran'] = True
    st.session_state['chat_msgs'] = []
    st.session_state['chat_hist'] = []

    if results.get('ai_insights') and api_key and not results['ai_insights'].startswith("Add"):
        st.session_state['chat_msgs'] = [{"role": "assistant",
                                          "content": f"**Analysis complete:**\n\n{results['ai_insights']}"}]
        st.session_state['chat_hist'] = [{"role": "assistant", "content": results['ai_insights']}]


# ══════════════════════════════════════════════════════════════════════════════
# SHARED CHAT WIDGET
# ══════════════════════════════════════════════════════════════════════════════
def render_chat(df, results, api_key):
    if not api_key:
        st.warning("Add your Anthropic API key in the sidebar to enable AI Chat.")
        return

    n = len(st.session_state['chat_hist'])
    st.markdown(f'<div class="mem-pill">Memory: {n} message{"s" if n!=1 else ""} in context</div>', unsafe_allow_html=True)

    col_chat, col_sug = st.columns([2, 1])
    with col_chat:
        for msg in st.session_state['chat_msgs']:
            if msg['role'] == 'assistant':
                st.markdown('<div class="lbl-ai">ProData AI</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-ai">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="lbl-user">You</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)

        user_q = st.chat_input("Ask anything about your data...")
        pending = st.session_state.pop('_chat_pending', None)
        question = pending or user_q

        if question:
            st.session_state['chat_msgs'].append({"role": "user", "content": question})
            st.session_state['chat_hist'].append({"role": "user", "content": question})
            ctx = build_ctx(df, results)
            system = (f"You are ProData AI — expert data analyst. Full context:\n{ctx}\n"
                      "Be specific, reference numbers and column names. Concise and actionable.")
            with st.spinner("Analyzing..."):
                try:
                    ans = call_claude(question, st.session_state['chat_hist'][:-1], system, api_key)
                except Exception as e:
                    ans = f"Error: {e}"
            st.session_state['chat_msgs'].append({"role": "assistant", "content": ans})
            st.session_state['chat_hist'].append({"role": "assistant", "content": ans})
            st.rerun()

    with col_sug:
        st.markdown("**Quick questions:**")
        sugs = ["Summarize key insights", "Top 3 recommendations?",
                "Any data quality concerns?", "Explain ML results simply",
                "What does the forecast mean?"]
        if results.get('drivers'):
            sugs.insert(0, f"Why is {results['drivers']['top']} the top driver?")
        for s in sugs[:6]:
            if st.button(s, key=f"chat_sug_{s[:20]}", use_container_width=True):
                st.session_state['_chat_pending'] = s
                st.rerun()
        st.divider()
        if st.button("Clear chat", use_container_width=True):
            st.session_state['chat_msgs'] = []
            st.session_state['chat_hist'] = []
            st.rerun()
        if st.session_state['chat_msgs']:
            chat_txt = "\n\n".join(
                [f"{'AI' if m['role']=='assistant' else 'You'}: {m['content']}"
                 for m in st.session_state['chat_msgs']]
            )
            st.download_button("Export chat (.txt)", chat_txt,
                               file_name="prodata_chat.txt", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ONE-CLICK RESULTS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def render_oneclick_dashboard(df, api_key, agent, client_name, project):
    results = st.session_state['oc_results']
    num_cols, _ = get_types(df)
    audit = st.session_state['oc_audit']

    # Status strip
    steps = [("Cleaned", bool(results.get('clean'))), ("ML", bool(results.get('ml'))),
             ("Forecast", bool(results.get('forecast'))), ("Drivers", bool(results.get('drivers'))),
             ("AI insights", bool(results.get('ai_insights'))), ("PDF", bool(results.get('pdf')))]
    cols = st.columns(len(steps))
    for col, (label, done) in zip(cols, steps):
        with col:
            cls = "badge-done" if done else "badge-wait"
            st.markdown(f'<div class="{cls}">{"✓ " if done else ""}{label}</div>', unsafe_allow_html=True)
    st.divider()

    tabs = st.tabs(["Summary", "ML Model", "Forecast", "Drivers", "AI Chat", "Data", "PDF Report"])

    with tabs[0]:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", f"{df.shape[0]:,}")
        m2.metric("Columns", df.shape[1])
        m3.metric("Issues fixed", len([x for x in results.get('clean', []) if 'clean' not in x.lower()]))
        m4.metric("Missing left", df.isna().sum().sum())
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Cleaning log**")
            for line in results.get('clean', ['No issues.']):
                st.markdown(f"✓ {line}")
        with c2:
            st.markdown("**Results**")
            if results.get('ml'):
                ml = results['ml']
                val = f"{ml['acc']:.2%}" if ml['is_class'] else f"R²={ml['r2']:.3f}"
                st.markdown(f"✓ ML — `{ml['target']}` — {val}")
            if results.get('forecast'):
                fc = results['forecast']
                st.markdown(f"✓ Forecast — `{fc['col']}`: {fc['latest']:.1f} → **{fc['projected']:.1f}**")
            if results.get('drivers'):
                dr = results['drivers']
                st.markdown(f"✓ Top driver for `{dr['kpi']}`: **{dr['top']}**")
            if results.get('pdf'):
                st.markdown("✓ PDF ready to download")
        if results.get('ai_insights'):
            st.divider()
            st.markdown(f'<div class="insight-box">{results["ai_insights"]}</div>', unsafe_allow_html=True)

    with tabs[1]:
        if results.get('ml_error'):
            st.markdown(f'<div class="warn-box">AutoML skipped: {results["ml_error"]}</div>', unsafe_allow_html=True)
        elif results.get('ml'):
            _render_automl_results(results['ml'])

    with tabs[2]:
        if results.get('forecast_skip'):
            st.markdown(f'<div class="warn-box">{results["forecast_skip"]}</div>', unsafe_allow_html=True)
        elif results.get('forecast_error'):
            st.markdown(f'<div class="warn-box">Forecast error: {results["forecast_error"]}</div>', unsafe_allow_html=True)
        elif results.get('forecast'):
            fc = results['forecast']
            m1, m2, m3 = st.columns(3)
            m1.metric("Latest value", f"{fc['latest']:.2f}")
            m2.metric("Projected (30d)", f"{fc['projected']:.2f}")
            if fc.get('mape'):
                m3.metric("MAPE", f"{fc['mape']*100:.1f}%")
            st.plotly_chart(fc['fig'], use_container_width=True)

    with tabs[3]:
        if results.get('driver_error'):
            st.markdown(f'<div class="warn-box">Driver error: {results["driver_error"]}</div>', unsafe_allow_html=True)
        elif results.get('drivers'):
            dr = results['drivers']
            c1, c2 = st.columns([2, 1])
            with c1:
                st.plotly_chart(dr['fig'], use_container_width=True)
            with c2:
                st.markdown(f'<div class="insight-box">🏆 <strong>{dr["top"]}</strong> is the top driver of <strong>{dr["kpi"]}</strong>.</div>', unsafe_allow_html=True)
                for k, v in dr['top5'].items():
                    st.markdown(f"• `{k}`: {v:.4f}")

    with tabs[4]:
        render_chat(df, results, api_key)

    with tabs[5]:
        sub = st.tabs(["Preview", "Statistics", "Correlation"])
        with sub[0]:
            st.dataframe(df.head(20), use_container_width=True)
        with sub[1]:
            st.dataframe(df.describe().round(2), use_container_width=True)
        with sub[2]:
            if len(num_cols) > 1:
                fig_corr = px.imshow(df[num_cols].corr(), text_auto=True, color_continuous_scale='RdBu_r'); dark_fig(fig_corr, 400); fig_corr.update_layout(title=dict(text='Correlation Heatmap', font=dict(color='#e2e8f0', size=13)))
                st.plotly_chart(fig_corr, use_container_width=True)

    with tabs[6]:
        if results.get('pdf_error'):
            st.error(f"PDF error: {results['pdf_error']}")
        elif results.get('pdf'):
            st.success("PDF auto-generated and ready.")
            st.download_button("⬇ Download PDF Report", data=results['pdf'],
                               file_name=f"{project.replace(' ','_')}_OneClick.pdf",
                               mime="application/pdf", use_container_width=True)
            st.divider()
            for line in audit:
                st.markdown(f"✓ {line}")
        else:
            st.info("PDF will appear after analysis.")


# ══════════════════════════════════════════════════════════════════════════════
# MANUAL DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def render_manual_dashboard(df, api_key, agent, client_name, project):
    num_cols, cat_cols = get_types(df)
    results = st.session_state['man_results']
    audit = st.session_state['man_audit']

    tabs = st.tabs(["Data Overview", "Visuals", "Data Cleaning", "Preparation",
                    "ML Models", "Forecasting", "Business Drivers", "AI Chat", "PDF Report"])

    # Data Overview
    with tabs[0]:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", f"{df.shape[0]:,}")
        m2.metric("Columns", df.shape[1])
        m3.metric("Numeric", len(num_cols))
        m4.metric("Missing", df.isna().sum().sum())
        sub = st.tabs(["Head", "Tail", "Schema", "Duplicates", "Statistics"])
        with sub[0]: st.dataframe(df.head(), use_container_width=True)
        with sub[1]: st.dataframe(df.tail(), use_container_width=True)
        with sub[2]:
            schema = pd.DataFrame({'Type': df.dtypes.astype(str), 'Non-Null': df.notna().sum(),
                                   'Null': df.isna().sum(), 'Unique': df.nunique()})
            st.dataframe(schema, use_container_width=True)
        with sub[3]:
            dupes = df.duplicated().sum()
            st.write(f"Duplicates: **{dupes}**")
            if dupes > 0 and st.button("Remove Duplicates"):
                st.session_state['df'] = df.drop_duplicates()
                audit.append(f"Removed {dupes} duplicates")
                st.rerun()
        with sub[4]:
            st.dataframe(df.describe().round(2), use_container_width=True)
        st.divider()
        if len(num_cols) > 1 and st.button("Generate Correlation Heatmap"):
            fig_corr = px.imshow(df[num_cols].corr(), text_auto=True,
                                  color_continuous_scale='RdBu_r', title="Correlation Heatmap")
            st.plotly_chart(fig_corr, use_container_width=True)
            save_fig(fig_corr, "Correlation Heatmap")

    # Visuals
    with tabs[1]:
        st.subheader("Visual Explorer")
        viz_type = st.radio("Chart type", ["Histogram", "Boxplot", "Scatter", "Bar", "Line"], horizontal=True)
        c1, c2 = st.columns([1, 3])
        with c1:
            x_ax = st.selectbox("X-Axis", df.columns, key="v_x")
            y_ax = st.selectbox("Y-Axis", num_cols, key="v_y") if viz_type in ["Scatter", "Bar", "Line"] else None
            color = st.selectbox("Color by", [None] + list(df.columns), key="v_c")
        with c2:
            try:
                if viz_type == "Histogram":
                    fig = px.histogram(df, x=x_ax, color=color, title=f"Distribution of {x_ax}")
                elif viz_type == "Boxplot":
                    if x_ax in num_cols:
                        fig = px.box(df, y=x_ax, points="outliers", title=f"Outliers: {x_ax}")
                    else:
                        st.warning("Select a numeric column for boxplot.")
                        st.stop()
                elif viz_type == "Scatter":
                    fig = px.scatter(df, x=x_ax, y=y_ax, color=color, title=f"{x_ax} vs {y_ax}")
                elif viz_type == "Bar":
                    fig = px.bar(df, x=x_ax, y=y_ax, color=color, title=f"{y_ax} by {x_ax}")
                elif viz_type == "Line":
                    fig = px.line(df, x=x_ax, y=y_ax, color=color, title=f"{y_ax} over {x_ax}")
                st.plotly_chart(fig, use_container_width=True)
                if st.button("Add to PDF"):
                    plt.figure(figsize=(10, 6))
                    if viz_type == "Scatter" and y_ax:
                        sns.scatterplot(data=df, x=x_ax, y=y_ax, hue=color)
                    elif viz_type == "Histogram":
                        sns.histplot(data=df, x=x_ax, hue=color)
                    elif viz_type == "Boxplot":
                        sns.boxplot(data=df, y=x_ax)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                        plt.savefig(tmp.name, bbox_inches='tight')
                        st.session_state['pdf_imgs'].append({"title": f"{viz_type}: {x_ax}", "path": tmp.name})
                    st.success("Added to PDF!")
                    plt.close()
            except Exception as e:
                st.error(f"Chart error: {e}")

    # Cleaning
    with tabs[2]:
        st.subheader("Data Cleaning")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Numeric missing values**")
            missing_num = df[num_cols].isna().sum()
            st.dataframe(missing_num[missing_num > 0], use_container_width=True)
            if st.button("Fill Numeric (Median)"):
                for c in num_cols:
                    df[c] = df[c].fillna(df[c].median())
                st.session_state['df'] = df
                audit.append("Filled numeric NaNs with median")
                st.rerun()
        with c2:
            st.write("**Categorical missing values**")
            missing_cat = df[cat_cols].isna().sum()
            st.dataframe(missing_cat[missing_cat > 0], use_container_width=True)
            if st.button("Fill Categorical (Mode)"):
                for c in cat_cols:
                    if df[c].isna().sum() > 0:
                        df[c] = df[c].fillna(df[c].mode()[0])
                st.session_state['df'] = df
                audit.append("Filled categorical NaNs with mode")
                st.rerun()
        st.divider()
        st.subheader("Outlier Management")
        col_out = st.selectbox("Column to scan", num_cols, key="out_col")
        Q1, Q3 = df[col_out].quantile(0.25), df[col_out].quantile(0.75)
        IQR = Q3 - Q1
        lb, ub = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        outliers = df[(df[col_out] < lb) | (df[col_out] > ub)]
        co1, co2, co3 = st.columns(3)
        co1.metric("Outliers", len(outliers))
        co2.metric("Lower limit", f"{lb:.2f}")
        co3.metric("Upper limit", f"{ub:.2f}")
        if len(outliers) > 0:
            action = st.radio("Action", ["Cap Values", "Remove Rows"], horizontal=True)
            if st.button("Apply"):
                if action == "Remove Rows":
                    df = df[(df[col_out] >= lb) & (df[col_out] <= ub)]
                    msg = f"Removed {len(outliers)} outliers from {col_out}"
                else:
                    df[col_out] = np.clip(df[col_out], lb, ub)
                    msg = f"Capped outliers in {col_out}"
                st.session_state['df'] = df
                audit.append(msg)
                st.success(msg)
                st.rerun()
        else:
            st.success("No outliers detected.")

    # Preparation
    with tabs[3]:
        st.subheader("Feature Scaling")
        to_scale = st.multiselect("Columns to scale", num_cols)
        method = st.radio("Method", ["MinMax (0-1)", "StandardScaler (z-score)"])
        if st.button("Apply Scaling") and to_scale:
            scaler = MinMaxScaler() if "MinMax" in method else StandardScaler()
            df[to_scale] = scaler.fit_transform(df[to_scale])
            st.session_state['df'] = df
            audit.append(f"Scaled {len(to_scale)} cols — {method}")
            st.success(f"Scaled {len(to_scale)} columns.")
            st.rerun()



    # AutoML Models
    with tabs[4]:
        render_automl_tab(df, results, audit, "manual")

    # Forecasting
    with tabs[5]:
        st.subheader("Enterprise Forecasting (Prophet)")
        date_candidates = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if not date_candidates:
            st.warning("No datetime column detected. Convert a date column first.")
            st.code("df['Date'] = pd.to_datetime(df['Date'])")
        else:
            valid_num = [c for c in num_cols if not any(k in c.lower() for k in ID_KW)]
            c1, c2, c3 = st.columns(3)
            with c1: date_col = st.selectbox("Date column", date_candidates)
            with c2: val_col = st.selectbox("KPI to forecast", valid_num)
            with c3: horizon = st.selectbox("Horizon (days)", [30, 60, 90, 180, 365])
            if st.button("Generate Forecast", type="primary"):
                try:
                    df_ts = df[[date_col, val_col]].dropna().sort_values(date_col)
                    prophet_df = df_ts.rename(columns={date_col: 'ds', val_col: 'y'})
                    split_idx = int(len(prophet_df) * 0.8)
                    m = Prophet(daily_seasonality=False, weekly_seasonality=True,
                                yearly_seasonality=True, interval_width=0.95)
                    m.fit(prophet_df.iloc[:split_idx])
                    freq = pd.infer_freq(prophet_df['ds']) or 'D'
                    future = m.make_future_dataframe(periods=horizon, freq=freq)
                    forecast = m.predict(future)
                    test_df = prophet_df.iloc[split_idx:]
                    fc_test = forecast[['ds','yhat']].merge(test_df[['ds','y']], on='ds', how='inner')
                    mape = mean_absolute_percentage_error(fc_test['y'], fc_test['yhat']) if len(fc_test) > 2 else None
                    rmse_fc = float(np.sqrt(mean_squared_error(fc_test['y'], fc_test['yhat']))) if len(fc_test) > 2 else None
                    fig_fc = go.Figure()
                    fig_fc.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], mode='lines', name='Historical', line=dict(color='#00d4aa')))
                    fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='#6366f1')))
                    fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False))
                    fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines',
                                                fill='tonexty', fillcolor='rgba(99,102,241,0.1)',
                                                line=dict(width=0), name='95% CI'))
                    fig_fc.update_layout(hovermode='x unified'); dark_fig(fig_fc, 340); fig_fc.update_layout(title=dict(text=f'{horizon}-Day Forecast — {val_col}', font=dict(color='#e2e8f0', size=13)))
                    st.plotly_chart(fig_fc, use_container_width=True)
                    save_fig(fig_fc, f"{horizon}-Day Forecast — {val_col}")
                    if mape:
                        fm1, fm2 = st.columns(2)
                        fm1.metric("MAPE", f"{mape*100:.1f}%")
                        fm2.metric("RMSE", f"{rmse_fc:.2f}")
                    results['forecast'] = {'col': val_col, 'horizon': horizon,
                                           'latest': float(prophet_df['y'].iloc[-1]),
                                           'projected': float(forecast['yhat'].iloc[-1]),
                                           'mape': mape, 'rmse': rmse_fc, 'fig': fig_fc}
                    st.session_state['man_results'] = results
                    audit.append(f"Forecast: {val_col} for {horizon} days")
                    st.success("Forecast complete! Ask the AI Chat to interpret results.")
                except Exception as e:
                    st.error(f"Forecast error: {e}")

    # Business Drivers
    with tabs[6]:
        st.subheader("Business Driver Analysis (XAI)")
        kpi = st.selectbox("KPI to analyze", num_cols, key="dr_kpi")
        if st.button("Run Driver Analysis", type="primary"):
            X_dr = pd.get_dummies(df.drop(columns=[kpi]).select_dtypes(include=np.number)).fillna(0)
            y_dr = df[kpi].fillna(0)
            m_dr = RandomForestRegressor(100, random_state=42)
            m_dr.fit(X_dr, y_dr)
            imp_dr = pd.Series(m_dr.feature_importances_, index=X_dr.columns).sort_values(ascending=False).head(10)
            c1, c2 = st.columns([2, 1])
            with c1:
                fig_dr = px.bar(imp_dr, orientation='h', title=f"Factors driving {kpi}",
                                color_discrete_sequence=['#6366f1'])
                fig_dr.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False); dark_fig(fig_dr, 320)
                st.plotly_chart(fig_dr, use_container_width=True)
                save_fig(fig_dr, f"Business Drivers — {kpi}")
            with c2:
                st.markdown(f'<div class="insight-box">🏆 <strong>{imp_dr.index[0]}</strong> is the top driver of <strong>{kpi}</strong>.</div>', unsafe_allow_html=True)
                for k, v in imp_dr.head(5).items():
                    st.markdown(f"• `{k}`: {v:.4f}")
            results['drivers'] = {'kpi': kpi, 'top': imp_dr.index[0], 'top5': imp_dr.head(5).to_dict(), 'fig': fig_dr}
            st.session_state['man_results'] = results
            audit.append(f"Driver analysis on '{kpi}'")
            st.success("Done! Ask AI Chat to explain what this means for the business.")

    # AI Chat
    with tabs[7]:
        render_chat(df, results, api_key)

    # PDF Report
    with tabs[8]:
        st.subheader("Generate PDF Report")
        st.write("Compiles all manual analysis, charts, ML results, forecasts, and AI chat into a PDF.")
        if st.button("Generate PDF", type="primary"):
            try:
                pdf_bytes = build_pdf(
                    results, df, audit,
                    st.session_state['chat_msgs'],
                    client_name, project, agent, "Manual"
                )
                results['pdf'] = pdf_bytes
                st.session_state['man_results'] = results
                st.download_button("⬇ Download PDF Report", data=pdf_bytes,
                                   file_name=f"{project.replace(' ','_')}_Manual.pdf",
                                   mime="application/pdf", use_container_width=True)
                st.success("PDF ready!")
            except Exception as e:
                st.error(f"PDF error: {e}")
        st.divider()
        st.markdown("**Audit log**")
        for line in audit:
            st.markdown(f"✓ {line}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    with st.sidebar:
        st.markdown("""
<div style='padding: 0.5rem 0 0.75rem;'>
    <div style='
        width: 60px; height: 60px;
        border-radius: 16px;
        background: rgba(0,212,170,0.08);
        border: 1px solid rgba(0,212,170,0.2);
        display: flex; align-items: center; justify-content: center;
        margin-bottom: 14px;
        box-shadow: 0 0 18px rgba(0,212,170,0.35), 0 0 40px rgba(0,212,170,0.12);
    '>
        <img src='https://cdn-icons-png.flaticon.com/512/2103/2103633.png'
             width='40' style='filter: drop-shadow(0 0 6px rgba(0,212,170,0.6));'/>
    </div>
    <div style='font-family: Sora, sans-serif; font-size: 1.18rem; font-weight: 700;
                color: #ffffff; letter-spacing: -0.02em; margin-bottom: 5px;
                text-shadow: 0 0 20px rgba(255,255,255,0.1);'>
        ProData AI
    </div>
    <div style='font-family: Sora, sans-serif; font-size: 0.78rem; font-weight: 400;
                color: #475569; letter-spacing: 0.01em;'>
        Professional Automated Data Analyst
    </div>
    <div style='height:1px; background:linear-gradient(90deg,rgba(0,212,170,0.4),transparent);
                margin-top:12px;'></div>
</div>
""", unsafe_allow_html=True)
        api_key = st.text_input("Anthropic API Key", type="password",
                                 placeholder="sk-ant-...",
                                 help="For AI insights & chat. console.anthropic.com")
        st.divider()
        st.markdown("### Upload Dataset")
        uploaded = st.file_uploader("CSV or Excel", type=['csv', 'xlsx'])
        if st.button("Load Demo (Titanic)", use_container_width=True):
            import urllib.request
            try:
                url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
                with urllib.request.urlopen(url) as r:
                    df_demo = pd.read_csv(r)
                st.session_state.update({
                    'df': df_demo, 'fname': 'titanic.csv',
                    'oc_ran': False, 'oc_results': {}, 'oc_audit': [],
                    'man_results': {}, 'man_audit': [],
                    'chat_msgs': [], 'chat_hist': [], 'pdf_imgs': []
                })
                st.rerun()
            except Exception as e:
                st.error(f"Demo error: {e}")
        st.divider()
        with st.expander("Settings"):
            agent = st.text_input("Agent Name", "Varun Walekar")
            client_name = st.text_input("Client", "Acme Corp")
            project = st.text_input("Project", "Data Analysis")
        if st.button("Reset App", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        st.divider()
        st.link_button("Hire the Developer",
                       "mailto:varunanalyzes.data@gmail.com?subject=Custom AI Project",
                       use_container_width=True)
        st.markdown("<div style='color:#334155;font-size:0.7rem;text-align:center;padding-top:4px;'>v16.0 · AutoML · Forecasting · AI Chat</div>", unsafe_allow_html=True)

    # Handle upload
    if uploaded:
        fname = uploaded.name
        if st.session_state.get('fname') != fname:
            try:
                df = pd.read_csv(uploaded) if fname.endswith('.csv') else pd.read_excel(uploaded)
                st.session_state.update({
                    'df': df, 'fname': fname,
                    'oc_ran': False, 'oc_results': {}, 'oc_audit': [],
                    'man_results': {}, 'man_audit': [],
                    'chat_msgs': [], 'chat_hist': [], 'pdf_imgs': []
                })
                st.rerun()
            except Exception as e:
                st.error(f"File error: {e}")

    df = st.session_state.get('df')

    # Landing
    if df is None:
        st.markdown("""
<div style='display:flex; align-items:center; gap:16px; padding: 2rem 0 1.25rem;'>
    <div style='position:relative; flex-shrink:0;'>
        <img src='https://cdn-icons-png.flaticon.com/512/2103/2103633.png'
             width='60'
             style='display:block; filter: drop-shadow(0 0 14px rgba(0,212,170,0.7));'/>
    </div>
    <div>
        <div style='font-family: Sora, sans-serif; font-size: 1.75rem; font-weight: 700;
                    color: #ffffff; letter-spacing: -0.03em; line-height: 1.15;'>
            ProData AI
        </div>
        <div style='font-family: Sora, sans-serif; font-size: 0.92rem; font-weight: 400;
                    color: #64748b; letter-spacing: 0.01em; margin-top: 3px;'>
            Automated Data Scientist
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
        st.markdown("#### One app. Two modes. Full control.")
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### ⚡ One-Click Mode")
            st.caption("Upload a file and everything runs automatically — clean, ML, forecast, drivers, AI insights, PDF. Done in 30 seconds.")
            for step in ["Auto clean data", "Auto train ML model", "Auto forecast trends",
                         "Auto driver analysis", "Auto AI insights", "Auto PDF report"]:
                st.markdown(f"✓ {step}")
        with c2:
            st.markdown("##### 🔧 Manual Mode")
            st.caption("Full control over every step. Choose your model, pick features, tune parameters, add charts to PDF.")
            for step in ["Custom ML model selection", "Choose target & features",
                         "Manual outlier handling", "Custom forecast horizon",
                         "Add charts to report", "Generate PDF when ready"]:
                st.markdown(f"✓ {step}")
        st.divider()
        st.info("Upload a CSV or Excel file in the sidebar — or load the Titanic demo to get started.")
        return

    # Mode toggle
    st.markdown(f"## {project}")
    st.caption(f"File: **{st.session_state['fname']}**  |  {df.shape[0]:,} rows × {df.shape[1]} cols  |  Client: {client_name}")

    col_a, col_b, col_c = st.columns([1, 1, 4])
    with col_a:
        if st.button("⚡ One-Click Mode",
                     type="primary" if st.session_state['mode'] == 'oneclick' else "secondary",
                     use_container_width=True):
            st.session_state['mode'] = 'oneclick'
            st.rerun()
    with col_b:
        if st.button("🔧 Manual Mode",
                     type="primary" if st.session_state['mode'] == 'manual' else "secondary",
                     use_container_width=True):
            st.session_state['mode'] = 'manual'
            # Reset chat when switching to manual so it uses manual results
            st.session_state['chat_msgs'] = []
            st.session_state['chat_hist'] = []
            st.rerun()

    mode_label = "⚡ One-Click Mode — everything runs automatically" if st.session_state['mode'] == 'oneclick' else "🔧 Manual Mode — you control every step"
    cls = "badge-done" if st.session_state['mode'] == 'oneclick' else "badge-manual"
    st.markdown(f'<div class="{cls}">{mode_label}</div>', unsafe_allow_html=True)
    st.divider()

    # Render the right mode
    if st.session_state['mode'] == 'oneclick':
        if not st.session_state['oc_ran']:
            run_oneclick(df, api_key, agent, client_name, project)
            st.rerun()
        else:
            render_oneclick_dashboard(df, api_key, agent, client_name, project)
    else:
        render_manual_dashboard(df, api_key, agent, client_name, project)

    st.divider()
    st.markdown("**Built by Varun Walekar** — varunanalyzes.data@gmail.com")


if __name__ == "__main__":
    main()
