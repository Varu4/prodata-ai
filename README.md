# 🧠 ProData AI — Enterprise Edition v16.0

<div align="center">

![ProData AI](https://img.shields.io/badge/ProData_AI-Enterprise_v15.0-6C63FF?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Powered_by-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Claude](https://img.shields.io/badge/AI_Layer-Claude_Sonnet-7C3AED?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Professional ML, Forecasting & AutoML — now supercharged with a Claude AI Chat layer.**

[🎮 Try Live Demo](https://varu4-prodata-ai-app-d2bocc.streamlit.app) · [🛒 Get on Gumroad](https://varunanalyze.gumroad.com/l/dgluuk) · [📺 Watch Demo](https://www.youtube.com/watch?v=RLVS7EOylAo) · [🤖 MCP Server](https://github.com/Varu4/prodata-ai-mcp)

</div>

---

## ✨ Overview

ProData AI Enterprise Edition is a **data science intelligence platform** built with Streamlit. It combines a battle-tested ML and forecasting engine with a Claude-powered AI Chat layer — giving non-technical users the ability to ask plain-English questions about their data, models, and predictions.

> Original ML/Forecasting engine by **Varun Walekar** · Upgraded with **Claude AI Chat Layer**

---

## 🆕 What's New in v16.0

| Feature | Description |
|---------|-------------|
| 💬 **AI Chat Tab** | Powered by Claude Sonnet with full conversation memory |
| 🔍 **Auto-insights** | Claude automatically analyzes your dataset on load |
| 🧠 **Context-aware Chat** | Claude knows your ML results, forecasts, and driver analysis |
| 📥 **Chat Export** | Download the full AI conversation as a text file |
| 📄 **AI Insights in PDF** | Conversation included in the final generated report |
| 🎨 **Refreshed UI** | DM Sans typography with a clean blue accent theme |

<details>
<summary>📋 Previous release highlights (v13.0)</summary>

- AI Chat tab introduced — powered by Claude
- Auto-insights on dataset load
- Context-aware chat with ML + forecast results
- Chat export as text file
- AI insights embedded in PDF report

</details>

---

## 🛠️ Feature Breakdown

### 📊 ML & AutoML Engine
- Auto-trains and compares multiple models (Gradient Boosting, Random Forest, Linear Regression, etc.)
- Returns best model with R² / accuracy scores
- Feature importance analysis — identifies top KPI drivers
- Handles both classification and regression tasks

### 📈 Forecasting
- Time series forecasting powered by **Prophet**
- MAPE scoring and projected value visualization
- Works on any date + value column pair

### 💬 AI Chat (Claude-Powered)
Claude has full context of your session including:
- Dataset stats, sample rows, and column types
- ML model results (R², accuracy, feature importances)
- Forecast results (MAPE, projected values)
- Driver analysis (top KPI drivers)
- Full conversation memory — Claude remembers every message

**Example questions users ask:**
```
"Why is Survived driven by Pclass?"
"Explain the R² score in plain English"
"What should we do about the missing values?"
"Give me a business recommendation based on the forecast"
"Which model should I trust more and why?"
```

### 📄 PDF Report
- Full stats, charts, ML results, and forecast
- AI conversation included as an insights section
- Exportable in one click

---

## ⚡ Run Locally

```bash
git clone https://github.com/Varu4/prodata-ai-app.git
cd prodata-ai
pip install -r requirements.txt
streamlit run app.py
```

### Add your Anthropic API Key

**Option A — secrets file (recommended):**
```toml
# .streamlit/secrets.toml
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```

**Option B — paste in the sidebar at runtime** (no setup needed)

---

## ☁️ Deploy Free on Streamlit Cloud

1. Push this folder to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, set file: `app.py`
4. Under **Settings → Secrets**, add:
```toml
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```
5. Hit **Deploy** — live URL in ~2 minutes ✅

---

## 🏗️ Architecture

```
User (Browser)
      ↓
Streamlit App (app.py)
      ↓              ↓
ML Engine          AI Chat Tab
(scikit-learn,     (Claude Sonnet API)
 Prophet)               ↓
      ↓          Full context injection
   Results    (dataset + ML + forecast)
      ↓              ↓
         PDF Report + Chat Export
```

---

## 🧱 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Python + Streamlit |
| AI Chat | Anthropic Claude API (`claude-sonnet-4`) |
| ML | scikit-learn |
| Forecasting | Prophet |
| Charts | Plotly + Seaborn |
| PDF Export | fpdf2 |
| Typography | DM Sans |

---

## 💼 Pricing (Freelance / White-Label)

Want this deployed for your business or clients? Here are the available packages:

| Package | Scope | Price |
|---------|-------|-------|
| **Starter** | Deploy + branding | $100 |
| **Professional** | Custom prompts + domain tuning | $500 |
| **Enterprise** | Auth + white-label + support | $1000+ |
| **Retainer** | Monthly updates & maintenance | $400/mo |

📩 Reach out via [Gumroad](https://varunanalyze.gumroad.com/l/dgluuk) or connect on [GitHub](https://github.com/Varu4)

---

## 📊 Performance Benchmarks

Tested on a 120-row retail sales dataset:

| Model | R² Score |
|-------|----------|
| **Gradient Boosting** | **0.9866** ✅ Best |
| Random Forest | 0.9741 |
| Linear Regression | 0.8923 |

**Top Feature:** Marketing Spend → Revenue

---

## 🗺️ Roadmap

- [x] AutoML training & model comparison
- [x] Time series forecasting (Prophet)
- [x] Feature importance & driver analysis
- [x] Claude AI Chat with full session context
- [x] Auto-insights on dataset load
- [x] PDF report with AI conversation
- [x] Chat export as text file
- [ ] Multi-dataset comparison
- [ ] Scheduled report emails
- [ ] User auth + multi-tenant support

---

## 🔗 Related Projects

| Project | Description |
|---------|-------------|
| [prodata-ai-mcp](https://github.com/Varu4/prodata-ai-mcp) | MCP Server — use ProData AI tools inside Claude, Cursor, VS Code |
| [Live Demo](https://varu4-prodata-ai-app-d2bocc.streamlit.app) | Try it live in your browser |

---

## 👨‍💻 Author

**Varun Walekar** — Data Analyst & AI Developer, Bengaluru

[![GitHub](https://img.shields.io/badge/GitHub-Varu4-181717?style=flat&logo=github)](https://github.com/Varu4)
[![Gumroad](https://img.shields.io/badge/Gumroad-Shop-FF90E8?style=flat)](https://varunanalyze.gumroad.com/l/dgluuk)
[![Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?style=flat&logo=streamlit)](https://varu4-prodata-ai-app-d2bocc.streamlit.app)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Found this useful? Drop a ⭐ — it helps others discover ProData AI!**

</div>
