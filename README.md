# ProData AI — Enterprise Edition v13.0

**Original ML/Forecasting engine by Varun Walekar**
**Upgraded with Claude AI Chat Layer**

---

## What's new in v13.0

- **AI Chat tab** — powered by Claude, with full conversation memory
- **Auto-insights** — Claude analyzes your dataset automatically on load
- **Context-aware chat** — Claude knows your ML results, forecasts, and driver analysis
- **Chat export** — download the full AI conversation as a text file
- **AI insights in PDF** — the conversation is included in the final report
- Refreshed UI with DM Sans typography and blue accent theme

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Add your Anthropic API key to `.streamlit/secrets.toml` — or paste it in the sidebar at runtime.

---

## Deploy free on Streamlit Cloud

1. Push this folder to a GitHub repo
2. Go to share.streamlit.io → New app → select repo, set file: `app.py`
3. Under Settings → Secrets, add:
   ```
   ANTHROPIC_API_KEY = "sk-ant-your-key-here"
   ```
4. Deploy — live URL in ~2 minutes

---

## How the AI Chat works

The AI Chat tab uses Claude with:
- **Full dataset context** (stats, sample rows, column types)
- **ML model results** (R², accuracy, feature importances)
- **Forecast results** (MAPE, projected values)
- **Driver analysis** (top KPI drivers)
- **Full conversation memory** — Claude remembers every message in the session

Users can ask follow-up questions like:
- "Why is Survived driven by Pclass?"
- "Explain the R² score in plain English"
- "What should we do about the missing values?"
- "Give me a business recommendation based on the forecast"

---

## Pricing your service

| Package | Scope | Price |
|---------|-------|-------|
| Starter | Deploy + branding | $500 |
| Professional | Custom prompts + domain tuning | $1,000 |
| Enterprise | Auth + white-label + support | $2,000+ |
| Retainer | Monthly updates | $400/mo |

---

## Tech stack

- Python + Streamlit
- Anthropic Claude API (claude-sonnet-4)
- Prophet (forecasting)
- scikit-learn (ML)
- Plotly + Seaborn (charts)
- fpdf2 (PDF export)
