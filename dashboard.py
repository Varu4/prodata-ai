# ==================================================
# dashboard.py
# Interactive Power BI-Style Dashboard for ProData AI
# Built with Plotly — no Power BI license needed
# ==================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Color palette ──────────────────────────────────────────────────────────────
TEAL   = '#00d4aa'
INDIGO = '#6366f1'
AMBER  = '#f59e0b'
PINK   = '#ec4899'
BLUE   = '#3b82f6'
GREEN  = '#10b981'
RED    = '#ef4444'
PURPLE = '#8b5cf6'
DARK   = '#070b14'
CARD   = 'rgba(13,31,26,0.8)'

CHART_COLORS = [TEAL, INDIGO, AMBER, PINK, BLUE, GREEN, RED, PURPLE]

DARK_LAYOUT = dict(
    paper_bgcolor='rgba(7,11,20,0)',
    plot_bgcolor='rgba(255,255,255,0.02)',
    font=dict(family='Sora, Arial, sans-serif', color='#94a3b8', size=11),
    title_font=dict(family='Sora, Arial, sans-serif', color='#e2e8f0', size=13),
    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.08)',
               tickfont=dict(color='#64748b')),
    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.08)',
               tickfont=dict(color='#64748b')),
    legend=dict(bgcolor='rgba(255,255,255,0.03)', bordercolor='rgba(255,255,255,0.08)',
                borderwidth=1, font=dict(color='#94a3b8')),
    margin=dict(l=16, r=16, t=44, b=16),
)


def dark_fig(fig, height=300):
    fig.update_layout(height=height, **DARK_LAYOUT)
    return fig


def kpi_card(label, value, delta=None, color=TEAL):
    delta_html = ''
    if delta is not None:
        arrow = '▲' if delta >= 0 else '▼'
        dcol = GREEN if delta >= 0 else RED
        delta_html = f"<div style='font-size:0.75rem;color:{dcol};margin-top:2px;'>{arrow} {abs(delta):.1f}%</div>"
    return f"""
    <div style='background:rgba(13,31,26,0.6);border:1px solid {color}33;
    border-left:3px solid {color};border-radius:10px;padding:14px 16px;
    min-height:80px;'>
        <div style='font-size:0.72rem;color:#64748b;text-transform:uppercase;
        letter-spacing:0.08em;margin-bottom:6px;'>{label}</div>
        <div style='font-size:1.6rem;font-weight:700;color:{color};
        font-family:monospace;line-height:1.1;'>{value}</div>
        {delta_html}
    </div>
    """


def render_dashboard(df, results):
    """Main dashboard renderer — call inside a Streamlit tab."""

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    ml       = results.get('ml')
    forecast = results.get('forecast')
    drivers  = results.get('drivers')
    insights = results.get('ai_insights', '')

    # ── Dashboard header ───────────────────────────────────────────────────────
    st.markdown("""
    <div style='background:linear-gradient(135deg,rgba(0,212,170,0.08),rgba(99,102,241,0.06));
    border:1px solid rgba(0,212,170,0.2);border-radius:14px;
    padding:1.2rem 1.5rem;margin-bottom:1.5rem;'>
        <div style='font-size:1.25rem;font-weight:700;color:#ffffff;'>
            📊 Interactive Analytics Dashboard
        </div>
        <div style='font-size:0.83rem;color:#64748b;margin-top:4px;'>
            ProData AI — Live business intelligence powered by ML + AI
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Global filters sidebar ─────────────────────────────────────────────────
    with st.expander("🎛️ Dashboard Filters", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            filter_col = st.selectbox("Filter by column", ['None'] + cat_cols, key='db_filter_col')
        with fc2:
            if filter_col != 'None':
                filter_vals = st.multiselect(
                    f"Select {filter_col} values",
                    df[filter_col].dropna().unique().tolist(),
                    default=df[filter_col].dropna().unique().tolist()[:5],
                    key='db_filter_vals'
                )
                if filter_vals:
                    df = df[df[filter_col].isin(filter_vals)]
            else:
                st.markdown("Select a column to filter")
        with fc3:
            if num_cols:
                sample_size = st.slider("Max rows to display", 100, min(len(df), 10000),
                                        min(len(df), 5000), 100, key='db_sample')
                if len(df) > sample_size:
                    df = df.sample(sample_size, random_state=42)

    # ── Page selector ──────────────────────────────────────────────────────────
    pages = ["📌 Overview", "🤖 ML Results", "📈 Forecast", "🎯 Business Drivers",
             "🔍 Data Explorer", "💬 AI Insights"]
    page = st.radio("Dashboard page", pages, horizontal=True, key='db_page',
                    label_visibility='collapsed')

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 1 — OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    if page == "📌 Overview":

        # KPI cards row 1
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(kpi_card("Total Records", f"{len(df):,}", color=TEAL), unsafe_allow_html=True)
        with k2:
            st.markdown(kpi_card("Columns", str(df.shape[1]), color=INDIGO), unsafe_allow_html=True)
        with k3:
            missing = int(df.isna().sum().sum())
            st.markdown(kpi_card("Missing Values", f"{missing:,}", color=AMBER if missing > 0 else GREEN), unsafe_allow_html=True)
        with k4:
            completeness = round((1 - df.isna().sum().sum() / (len(df) * len(df.columns))) * 100, 1)
            st.markdown(kpi_card("Data Completeness", f"{completeness}%", color=GREEN if completeness > 90 else AMBER), unsafe_allow_html=True)

        # KPI cards row 2 — ML metrics if available
        if ml:
            st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(kpi_card("Best ML Model", ml.get('model_name','').split()[0], color=TEAL), unsafe_allow_html=True)
            with m2:
                if not ml.get('is_class') and ml.get('r2') not in (None, '-'):
                    st.markdown(kpi_card("R² Score", str(ml.get('r2','')), color=GREEN), unsafe_allow_html=True)
                elif ml.get('acc') not in (None, '-'):
                    st.markdown(kpi_card("Accuracy", f"{ml.get('acc',0):.2%}", color=GREEN), unsafe_allow_html=True)
            with m3:
                if forecast:
                    st.markdown(kpi_card("30-Day Forecast", f"{forecast.get('projected',0):.1f}", color=INDIGO), unsafe_allow_html=True)
            with m4:
                if drivers:
                    st.markdown(kpi_card("Top Driver", drivers.get('top',''), color=AMBER), unsafe_allow_html=True)

        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

        # Charts row 1
        if num_cols and cat_cols:
            c1, c2 = st.columns([1.4, 1])
            with c1:
                metric = st.selectbox("Metric", num_cols, key='ov_metric')
                dimension = st.selectbox("Dimension", cat_cols, key='ov_dim')
                agg_df = df.groupby(dimension)[metric].mean().reset_index().sort_values(metric, ascending=False).head(10)
                fig = px.bar(agg_df, x=dimension, y=metric, title=f"Average {metric} by {dimension}",
                             color_discrete_sequence=[TEAL])
                fig.update_traces(marker_line_width=0)
                dark_fig(fig, 320)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                cat_col = st.selectbox("Category", cat_cols, key='ov_pie_col')
                pie_df = df[cat_col].value_counts().head(8).reset_index()
                pie_df.columns = [cat_col, 'Count']
                fig2 = px.pie(pie_df, values='Count', names=cat_col,
                              title=f"Distribution of {cat_col}",
                              color_discrete_sequence=CHART_COLORS, hole=0.4)
                dark_fig(fig2, 320)
                st.plotly_chart(fig2, use_container_width=True)

        elif num_cols:
            c1, c2 = st.columns(2)
            with c1:
                col1 = st.selectbox("Column 1", num_cols, key='ov_c1')
                fig = px.histogram(df, x=col1, title=f"Distribution of {col1}",
                                   color_discrete_sequence=[TEAL])
                dark_fig(fig, 300)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                col2 = st.selectbox("Column 2", num_cols, index=min(1, len(num_cols)-1), key='ov_c2')
                fig2 = px.box(df, y=col2, title=f"Box Plot — {col2}",
                              color_discrete_sequence=[INDIGO], points='outliers')
                dark_fig(fig2, 300)
                st.plotly_chart(fig2, use_container_width=True)

        # Correlation heatmap
        if len(num_cols) > 2:
            corr_cols = num_cols[:12]
            corr = df[corr_cols].corr()
            fig_corr = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                                  title='Correlation Heatmap', zmin=-1, zmax=1)
            dark_fig(fig_corr, 380)
            st.plotly_chart(fig_corr, use_container_width=True)

        # Data summary table
        with st.expander("📋 Data Summary", expanded=False):
            st.dataframe(df.describe().round(2), use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 2 — ML RESULTS
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "🤖 ML Results":
        if not ml:
            st.info("Run One-Click analysis or Manual → ML Models to see results here.")
        else:
            is_class = ml.get('is_class', False)
            metric_col = 'Accuracy' if is_class else 'R²'
            winner = ml.get('model_name', '')

            # Winner banner
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,rgba(0,212,170,0.12),rgba(99,102,241,0.08));
            border:1px solid rgba(0,212,170,0.4);border-radius:12px;padding:1.2rem 1.5rem;
            margin-bottom:1.25rem;display:flex;align-items:center;gap:16px;'>
                <div style='font-size:2.5rem;'>🏆</div>
                <div>
                    <div style='color:#e2e8f0;font-size:1.1rem;font-weight:700;'>
                        Winner: {winner}
                    </div>
                    <div style='color:#64748b;font-size:0.83rem;margin-top:3px;'>
                        Target: <code style="color:#a5b4fc;">{ml.get("target","")}</code> ·
                        Task: {"Classification" if is_class else "Regression"} ·
                        Models tested: {ml.get("n_models", 6)}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Metric cards
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(kpi_card("Winner", winner.split()[0], color=TEAL), unsafe_allow_html=True)
            with k2:
                if is_class:
                    st.markdown(kpi_card("Accuracy", f"{ml.get('acc',0):.2%}", color=GREEN), unsafe_allow_html=True)
                else:
                    st.markdown(kpi_card("R² Score", str(ml.get('r2','')), color=GREEN), unsafe_allow_html=True)
            with k3:
                if not is_class:
                    st.markdown(kpi_card("MAE", str(ml.get('mae','')), color=AMBER), unsafe_allow_html=True)
            with k4:
                if not is_class:
                    st.markdown(kpi_card("RMSE", str(ml.get('rmse','')), color=PINK), unsafe_allow_html=True)

            st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

            # Leaderboard chart
            leaderboard = ml.get('leaderboard', [])
            if leaderboard:
                c1, c2 = st.columns([1.5, 1])
                with c1:
                    lb_data = []
                    for r in leaderboard:
                        score = r.get(metric_col, '-')
                        if score != '-':
                            try:
                                lb_data.append({'Model': r['Model'], metric_col: float(score)})
                            except Exception:
                                pass
                    if lb_data:
                        lb_df = pd.DataFrame(lb_data).sort_values(metric_col, ascending=True)
                        colors = [TEAL if m == winner else '#334155' for m in lb_df['Model']]
                        fig_lb = go.Figure(go.Bar(
                            x=lb_df[metric_col], y=lb_df['Model'], orientation='h',
                            marker_color=colors,
                            text=[f"{v:.4f}" for v in lb_df[metric_col]],
                            textposition='outside',
                            textfont=dict(color='#94a3b8', size=11),
                        ))
                        fig_lb.update_layout(
                            title=dict(text=f'Model Leaderboard — {metric_col}',
                                      font=dict(color='#e2e8f0', size=13)),
                            yaxis=dict(categoryorder='total ascending')
                        )
                        dark_fig(fig_lb, 340)
                        st.plotly_chart(fig_lb, use_container_width=True)

                with c2:
                    st.markdown("**Full Leaderboard**")
                    medals = ['🥇', '🥈', '🥉']
                    rows = []
                    for i, r in enumerate(leaderboard):
                        medal = medals[i] if i < 3 else f'#{i+1}'
                        if is_class:
                            rows.append({'': medal, 'Model': r.get('Model',''), 'Accuracy': r.get('Accuracy','-')})
                        else:
                            rows.append({'': medal, 'Model': r.get('Model',''), 'R²': r.get('R²','-'), 'MAE': r.get('MAE','-')})
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Feature importance + scatter
            fi_col, sc_col = st.columns(2)
            with fi_col:
                if ml.get('fig_imp'):
                    st.plotly_chart(ml['fig_imp'], use_container_width=True)
            with sc_col:
                if ml.get('fig_sc'):
                    st.plotly_chart(ml['fig_sc'], use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 3 — FORECAST
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "📈 Forecast":
        if not forecast:
            st.info("Run One-Click analysis or Manual → Forecasting to see results here.")
        else:
            col = forecast.get('col', '')
            latest = forecast.get('latest', 0)
            projected = forecast.get('projected', 0)
            horizon = forecast.get('horizon', 30)
            change = ((projected - latest) / latest * 100) if latest != 0 else 0

            # KPI cards
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(kpi_card("Forecast Column", col, color=TEAL), unsafe_allow_html=True)
            with k2:
                st.markdown(kpi_card("Current Value", f"{latest:.2f}", color=INDIGO), unsafe_allow_html=True)
            with k3:
                st.markdown(kpi_card(f"{horizon}-Day Forecast", f"{projected:.2f}",
                                     delta=change, color=GREEN if change >= 0 else RED), unsafe_allow_html=True)
            with k4:
                direction = "↑ Upward" if change >= 0 else "↓ Downward"
                st.markdown(kpi_card("Trend Direction", direction,
                                     color=GREEN if change >= 0 else RED), unsafe_allow_html=True)

            st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

            # Main forecast chart
            if forecast.get('fig'):
                st.plotly_chart(forecast['fig'], use_container_width=True)

            # Trend analysis
            if num_cols:
                st.markdown("**Trend Analysis — Select a metric:**")
                trend_col = st.selectbox("", [col] + [c for c in num_cols if c != col], key='fc_trend')
                if pd.api.types.is_numeric_dtype(df[trend_col]):
                    # Rolling average
                    df_trend = df[[trend_col]].dropna().reset_index(drop=True)
                    df_trend['Rolling Avg (10)'] = df_trend[trend_col].rolling(10).mean()
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(
                        y=df_trend[trend_col], mode='lines', name='Actual',
                        line=dict(color=TEAL, width=1.5)
                    ))
                    fig_trend.add_trace(go.Scatter(
                        y=df_trend['Rolling Avg (10)'], mode='lines', name='Rolling Avg',
                        line=dict(color=AMBER, width=2, dash='dot')
                    ))
                    fig_trend.update_layout(title=dict(text=f'{trend_col} Trend', font=dict(color='#e2e8f0')))
                    dark_fig(fig_trend, 300)
                    st.plotly_chart(fig_trend, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 4 — BUSINESS DRIVERS
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "🎯 Business Drivers":
        if not drivers and (not ml or not ml.get('importances')):
            st.info("Run One-Click analysis or Manual → Business Drivers to see results here.")
        else:
            imp = drivers.get('top5') if drivers else (ml.get('importances') if ml else {})
            kpi = drivers.get('kpi', ml.get('target', '')) if drivers else ml.get('target', '') if ml else ''
            top = drivers.get('top', list(imp.keys())[0] if imp else '') if drivers else ''

            # Top driver banner
            if top:
                st.markdown(f"""
                <div style='background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.3);
                border-radius:12px;padding:1rem 1.5rem;margin-bottom:1rem;'>
                    <div style='color:#fbbf24;font-size:0.72rem;text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:4px;'>Top Business Driver</div>
                    <div style='color:#ffffff;font-size:1.4rem;font-weight:700;'>{top}</div>
                    <div style='color:#64748b;font-size:0.83rem;margin-top:3px;'>
                    This factor has the highest impact on <strong style="color:#fbbf24;">{kpi}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            c1, c2 = st.columns([1.5, 1])
            with c1:
                if imp:
                    sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
                    imp_df = pd.DataFrame(sorted_imp, columns=['Driver', 'Importance'])
                    colors = [AMBER if i == 0 else TEAL if i < 3 else '#334155'
                              for i in range(len(imp_df))]
                    fig_imp = go.Figure(go.Bar(
                        x=imp_df['Importance'], y=imp_df['Driver'], orientation='h',
                        marker_color=colors,
                        text=[f"{v:.4f}" for v in imp_df['Importance']],
                        textposition='outside',
                        textfont=dict(color='#94a3b8', size=11),
                    ))
                    fig_imp.update_layout(
                        title=dict(text=f'What drives {kpi}?', font=dict(color='#e2e8f0', size=13)),
                        yaxis=dict(categoryorder='total ascending')
                    )
                    dark_fig(fig_imp, 380)
                    st.plotly_chart(fig_imp, use_container_width=True)

            with c2:
                if imp:
                    st.markdown("**Driver Rankings:**")
                    sorted_full = sorted(imp.items(), key=lambda x: x[1], reverse=True)
                    medals = ['🥇', '🥈', '🥉']
                    for i, (feat, score) in enumerate(sorted_full):
                        medal = medals[i] if i < 3 else f'#{i+1}'
                        bar_width = int(score / max(imp.values()) * 100) if max(imp.values()) > 0 else 0
                        st.markdown(f"""
                        <div style='margin-bottom:10px;'>
                            <div style='display:flex;justify-content:space-between;
                            font-size:0.83rem;color:#cbd5e1;margin-bottom:3px;'>
                                <span>{medal} {feat}</span>
                                <span style='color:#64748b;'>{score:.4f}</span>
                            </div>
                            <div style='background:rgba(255,255,255,0.05);border-radius:4px;height:6px;'>
                                <div style='background:{"#f59e0b" if i==0 else "#00d4aa" if i<3 else "#334155"};
                                width:{bar_width}%;height:6px;border-radius:4px;'></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            # Driver vs KPI scatter
            if imp and num_cols and kpi in num_cols:
                st.markdown("**Driver Relationships:**")
                top_drivers = [d for d, _ in sorted(imp.items(), key=lambda x: x[1], reverse=True)[:4]
                               if d in df.columns]
                if len(top_drivers) >= 2:
                    sc1, sc2 = st.columns(2)
                    for i, driver in enumerate(top_drivers[:4]):
                        col_idx = sc1 if i % 2 == 0 else sc2
                        with col_idx:
                            if driver in df.columns and kpi in df.columns:
                                fig_sc = px.scatter(df.sample(min(500, len(df))),
                                                    x=driver, y=kpi,
                                                    title=f"{driver} vs {kpi}",
                                                    color_discrete_sequence=[CHART_COLORS[i]],
                                                    trendline="ols")
                                dark_fig(fig_sc, 260)
                                st.plotly_chart(fig_sc, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 5 — DATA EXPLORER
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "🔍 Data Explorer":
        st.markdown("**Interactive Data Explorer — build any chart on the fly**")

        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            chart_type = st.selectbox("Chart type",
                ["Bar", "Line", "Scatter", "Histogram", "Box", "Heatmap", "Bubble", "Treemap"],
                key='exp_chart')
        with ec2:
            x_col = st.selectbox("X axis", df.columns.tolist(), key='exp_x')
        with ec3:
            y_col = st.selectbox("Y axis", ['None'] + num_cols, key='exp_y')

        ec4, ec5, ec6 = st.columns(3)
        with ec4:
            color_col = st.selectbox("Color by", ['None'] + list(df.columns), key='exp_color')
        with ec5:
            size_col = st.selectbox("Size by (bubble)", ['None'] + num_cols, key='exp_size') if chart_type == "Bubble" else None
        with ec6:
            agg_func = st.selectbox("Aggregation", ['mean', 'sum', 'count', 'max', 'min'], key='exp_agg')

        try:
            color = color_col if color_col != 'None' else None
            y = y_col if y_col != 'None' else None

            if chart_type == "Bar":
                if y and color:
                    plot_df = df.groupby([x_col, color])[y].agg(agg_func).reset_index()
                elif y:
                    plot_df = df.groupby(x_col)[y].agg(agg_func).reset_index()
                else:
                    plot_df = df[x_col].value_counts().reset_index()
                    plot_df.columns = [x_col, 'count']
                    y = 'count'
                fig = px.bar(plot_df.head(30), x=x_col, y=y, color=color,
                             color_discrete_sequence=CHART_COLORS, barmode='group')

            elif chart_type == "Line":
                plot_df = df.groupby(x_col)[y].agg(agg_func).reset_index() if y else df
                fig = px.line(plot_df, x=x_col, y=y, color=color,
                              color_discrete_sequence=CHART_COLORS)

            elif chart_type == "Scatter":
                fig = px.scatter(df.sample(min(1000, len(df))), x=x_col, y=y, color=color,
                                 color_discrete_sequence=CHART_COLORS, trendline="ols")

            elif chart_type == "Histogram":
                fig = px.histogram(df, x=x_col, color=color,
                                   color_discrete_sequence=CHART_COLORS, nbins=30)

            elif chart_type == "Box":
                fig = px.box(df, x=color if color else None, y=x_col if x_col in num_cols else y,
                             color=color, color_discrete_sequence=CHART_COLORS, points="outliers")

            elif chart_type == "Heatmap":
                if len(num_cols) > 1:
                    corr = df[num_cols[:15]].corr()
                    fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
                else:
                    fig = px.histogram(df, x=x_col, color_discrete_sequence=CHART_COLORS)

            elif chart_type == "Bubble":
                size = size_col if size_col and size_col != 'None' else None
                fig = px.scatter(df.sample(min(500, len(df))), x=x_col, y=y,
                                 size=size, color=color, color_discrete_sequence=CHART_COLORS)

            elif chart_type == "Treemap":
                path_cols = [x_col] + ([color] if color else [])
                fig = px.treemap(df, path=path_cols, values=y if y else None,
                                 color_discrete_sequence=CHART_COLORS)

            dark_fig(fig, 420)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Chart error: {e}. Try different column combinations.")

        # Data table with search
        st.markdown("**Data Table**")
        search = st.text_input("Search in data", placeholder="Type to filter...", key='exp_search')
        display_df = df.copy()
        if search:
            mask = display_df.astype(str).apply(lambda row: row.str.contains(search, case=False)).any(axis=1)
            display_df = display_df[mask]
        st.dataframe(display_df.head(500), use_container_width=True)
        st.caption(f"Showing {min(500, len(display_df)):,} of {len(display_df):,} rows")

        # Download filtered data
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Download filtered data as CSV", data=csv,
                           file_name="prodata_ai_filtered.csv", mime="text/csv")

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 6 — AI INSIGHTS
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "💬 AI Insights":
        if not insights:
            st.info("Run One-Click analysis with your Anthropic API key to generate AI insights here.")
        else:
            st.markdown(f"""
            <div style='background:rgba(0,212,170,0.05);
            border:1px solid rgba(0,212,170,0.2);border-left:3px solid #00d4aa;
            border-radius:0 14px 14px 14px;padding:1.5rem 1.75rem;
            font-size:0.9rem;color:#a7f3d0;line-height:1.85;'>
            {insights.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

        # Summary stats cards
        if num_cols:
            st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
            st.markdown("**Key Statistics at a Glance:**")
            stat_cols = num_cols[:4]
            cols = st.columns(len(stat_cols))
            for i, col_name in enumerate(stat_cols):
                with cols[i]:
                    val = df[col_name].mean()
                    formatted = f"{val:,.2f}" if abs(val) < 1e6 else f"{val/1e6:.2f}M"
                    st.markdown(kpi_card(f"Avg {col_name}", formatted,
                                        color=CHART_COLORS[i % len(CHART_COLORS)]),
                                unsafe_allow_html=True)
