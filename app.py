# app.py
import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import joblib
from sklearn.decomposition import PCA
from ai.ai_agents import (
    ask_question,
    generate_insight,
    generate_country_report,
    ask_economic_interpreter
)

# ---------- Page config ----------
st.set_page_config(page_title="EconsGPT – Economic Intelligence", layout="wide")
st.title("🚀 EconsGPT: Economic Forecasting & AI Assistant")
st.markdown("G20 economic data – visualise, ask AI, predict the future.")

# ---------- Load data (cached) ----------
@st.cache_data
def load_sqlite_data():
    conn = sqlite3.connect("data/economics.db")
    df = pd.read_sql("SELECT * FROM economic_cleaned", conn)
    conn.close()
    return df

@st.cache_data
def load_ml_data():
    df = pd.read_csv("data/featured_economic.csv")
    df.columns = df.columns.str.strip()
    return df

@st.cache_resource
def load_models():
    models = {
        'gdp': joblib.load("models/gdp_forecast_model.pkl"),
        'inflation': joblib.load("models/inflation_forecast_model.pkl"),
        'unemployment': joblib.load("models/unemployment_forecast_model.pkl"),
        'recession': joblib.load("models/recession_classifier.pkl"),
        'kmeans': joblib.load("models/kmeans_model.pkl"),
        'scaler': joblib.load("models/kmeans_scaler.pkl")
    }
    return models

df_sql = load_sqlite_data()
df_ml = load_ml_data()
models = load_models()

# ---------- Latest year for ML (2024) ----------
df_ml_latest = df_ml[df_ml['year'] == 2024].copy()

# ---------- Sidebar: country filter ----------
st.sidebar.header("🌍 Country Filter")
all_countries = sorted(df_sql["country_code"].unique())
selected_countries = st.sidebar.multiselect(
    "Select countries",
    options=all_countries,
    default=all_countries
)

# Filter data
filtered_sql = df_sql[df_sql["country_code"].isin(selected_countries)]
filtered_ml = df_ml_latest[df_ml_latest["country_code"].isin(selected_countries)]

if filtered_ml.empty:
    st.warning("No data for selected countries. Adjust filter.")
    st.stop()

# ---------- Compute ML forecasts for filtered countries ----------
all_cols = df_ml.columns.tolist()

# GDP features
exclude_gdp = ['country_code', 'year', 'gdp_per_capita', 'gdp_next_year', 'recession',
               'inflation_next_year', 'unemployment_next_year']
feat_gdp = [c for c in all_cols if c not in exclude_gdp]
X_gdp = filtered_ml[feat_gdp].fillna(0)

# Inflation features
exclude_inf = ['country_code', 'year', 'gdp_per_capita', 'inflation', 'inflation_next_year',
               'gdp_next_year', 'recession', 'unemployment_next_year']
feat_inf = [c for c in all_cols if c not in exclude_inf]
X_inf = filtered_ml[feat_inf].fillna(0)

# Unemployment features
exclude_unemp = ['country_code', 'year', 'gdp_per_capita', 'unemployment', 'unemployment_next_year',
                 'gdp_next_year', 'recession', 'inflation_next_year']
feat_unemp = [c for c in all_cols if c not in exclude_unemp]
X_unemp = filtered_ml[feat_unemp].fillna(0)

# Predict
filtered_ml['gdp_forecast_2025'] = models['gdp'].predict(X_gdp)
filtered_ml['inflation_forecast_2025'] = models['inflation'].predict(X_inf)
filtered_ml['unemployment_forecast_2025'] = models['unemployment'].predict(X_unemp)
filtered_ml['recession_prob'] = models['recession'].predict_proba(X_gdp)[:, 1]

# Percentage changes
filtered_ml['gdp_pct_change'] = ((filtered_ml['gdp_forecast_2025'] - filtered_ml['gdp_per_capita']) / filtered_ml['gdp_per_capita']) * 100
filtered_ml['inf_pct_change'] = ((filtered_ml['inflation_forecast_2025'] - filtered_ml['inflation']) / filtered_ml['inflation']) * 100
filtered_ml['unemp_pct_change'] = ((filtered_ml['unemployment_forecast_2025'] - filtered_ml['unemployment']) / filtered_ml['unemployment']) * 100

# ---------- Clustering for filtered countries ----------
X_cluster = filtered_ml[['gdp_per_capita', 'inflation', 'unemployment']].values
if len(X_cluster) >= 3:
    X_scaled = models['scaler'].transform(X_cluster)
    filtered_ml['cluster'] = models['kmeans'].predict(X_scaled)
else:
    filtered_ml['cluster'] = -1

# ---------- Tabs ----------
tab_vis, tab_ai, tab_fc = st.tabs(["📊 Visualise", "🤖 AI Assistant", "🔮 Forecast"])

# ========================== TAB 1: VISUALISE ==========================
with tab_vis:
    st.header("Interactive Economic Dashboard")
    st.markdown("*Charts update based on selected countries.*")

    # Category counts
    st.subheader("Number of Countries by Category")
    col1, col2, col3 = st.columns(3)
    with col1:
        gdp_cat = filtered_sql["gdp_category"].value_counts().reset_index()
        gdp_cat.columns = ["GDP Category", "Count"]
        fig = px.bar(gdp_cat, x="GDP Category", y="Count", color="GDP Category", text="Count", title="GDP")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        inf_cat = filtered_sql["inflation_category"].value_counts().reset_index()
        inf_cat.columns = ["Inflation Category", "Count"]
        fig = px.bar(inf_cat, x="Inflation Category", y="Count", color="Inflation Category", text="Count", title="Inflation")
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        unemp_cat = filtered_sql["unemployment_category"].value_counts().reset_index()
        unemp_cat.columns = ["Unemployment Category", "Count"]
        fig = px.bar(unemp_cat, x="Unemployment Category", y="Count", color="Unemployment Category", text="Count", title="Unemployment")
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.subheader("Correlation between Indicators")
    corr = filtered_sql[["gdp_per_capita", "inflation", "unemployment"]].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
    st.plotly_chart(fig, use_container_width=True)

    # Scatter plot: inflation vs unemployment
    st.subheader("Inflation vs Unemployment (bubble size = GDP per capita)")
    fig = px.scatter(
        filtered_sql, x="inflation", y="unemployment", size="gdp_per_capita",
        color="country_code", hover_name="country_code", size_max=60
    )
    st.plotly_chart(fig, use_container_width=True)

    # Box plot: GDP by category
    st.subheader("GDP per capita Distribution by Income Group")
    fig = px.box(filtered_sql, x="gdp_category", y="gdp_per_capita", color="gdp_category")
    st.plotly_chart(fig, use_container_width=True)

# ========================== TAB 2: AI ASSISTANT ==========================
with tab_ai:
    st.header("AI Economic Assistant")
    st.markdown("Ask natural language questions, get SQL + insights, or generate country reports.")

    ai_sub1, ai_sub2, ai_sub3 = st.tabs(["📝 Text-to-SQL", "💬 Economic Interpreter", "📄 Country Report"])

    with ai_sub1:
        q = st.text_input("Your question:", placeholder="e.g., Which country has the highest inflation?")
        if q:
            with st.spinner("Generating SQL..."):
                sql, df_res = ask_question(q)
            with st.expander("📝 Generated SQL Query", expanded=True):
                st.code(sql, language="sql")
            if "error" in df_res.columns:
                st.error(df_res["error"][0])
            elif df_res.empty:
                st.warning("No results.")
            else:
                st.dataframe(df_res, use_container_width=True)
                if st.button("💡 Generate Insight"):
                    insight = generate_insight(q, df_res)
                    st.info(f"📈 **Insight:** {insight}")

    with ai_sub2:
        econ_q = st.text_input("Ask an economic question:", placeholder="Why does Argentina have high inflation?")
        if econ_q:
            with st.spinner("Thinking..."):
                answer = ask_economic_interpreter(econ_q)
            st.markdown("### Answer")
            st.write(answer)

    with ai_sub3:
        country = st.selectbox("Select a country:", all_countries)
        if st.button("📄 Generate Report"):
            with st.spinner("Generating report..."):
                conn = sqlite3.connect("data/economics.db")
                row = pd.read_sql(f"SELECT * FROM economic_cleaned WHERE country_code = '{country}'", conn).iloc[0].to_dict()
                conn.close()
                report = generate_country_report(country, row)
            st.markdown(report)
            st.download_button("⬇️ Download Report", report, f"{country}_report.txt")

# ========================== TAB 3: FORECAST ==========================
with tab_fc:
    st.header("Economic Forecasts for 2025")
    st.markdown(f"Based on data from 2024 – showing predictions for selected countries.")

    # GDP forecast
    st.subheader("GDP per capita (USD)")
    gdp_df = filtered_ml[['country_code', 'gdp_per_capita', 'gdp_forecast_2025', 'gdp_pct_change']]
    fig = px.bar(gdp_df, x='country_code', y='gdp_forecast_2025', color='country_code',
                 title="GDP Forecast 2025", hover_data={'gdp_per_capita': ':.0f', 'gdp_pct_change': ':.1f'})
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(gdp_df, use_container_width=True)

    # Inflation forecast
    st.subheader("Inflation Rate (%)")
    inf_df = filtered_ml[['country_code', 'inflation', 'inflation_forecast_2025', 'inf_pct_change']]
    fig = px.bar(inf_df, x='country_code', y='inflation_forecast_2025', color='country_code',
                 title="Inflation Forecast 2025", color_continuous_scale='Reds')
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(inf_df, use_container_width=True)

    # Unemployment forecast
    st.subheader("Unemployment Rate (%)")
    unemp_df = filtered_ml[['country_code', 'unemployment', 'unemployment_forecast_2025', 'unemp_pct_change']]
    fig = px.bar(unemp_df, x='country_code', y='unemployment_forecast_2025', color='country_code',
                 title="Unemployment Forecast 2025")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(unemp_df, use_container_width=True)

    # Recession risk
    st.subheader("Recession Risk Probability for 2025")
    risk_df = filtered_ml[['country_code', 'recession_prob']]
    fig = px.bar(risk_df, x='country_code', y='recession_prob', color='recession_prob',
                 color_continuous_scale='Reds', title="Probability of Recession")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(risk_df, use_container_width=True)

    # Clustering
    st.subheader("Country Clusters (k=3)")
    if len(filtered_ml) < 3:
        st.warning("Need at least 3 countries to display clusters.")
    else:
        pca = PCA(n_components=2)
        X_clust = filtered_ml[['gdp_per_capita', 'inflation', 'unemployment']].values
        X_scaled = models['scaler'].transform(X_clust)
        pca_coords = pca.fit_transform(X_scaled)
        filtered_ml['pca_x'] = pca_coords[:, 0]
        filtered_ml['pca_y'] = pca_coords[:, 1]
        fig = px.scatter(filtered_ml, x='pca_x', y='pca_y', color='cluster', text='country_code',
                         title="Clusters (PCA projection)")
        fig.update_traces(textposition='top center', marker=dict(size=12))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(filtered_ml[['country_code', 'cluster', 'gdp_per_capita', 'inflation', 'unemployment']])

st.markdown("---")
st.caption("Data: World Bank API | Models: XGBoost, Random Forest, K‑Means | AI: Ollama (CodeGemma)")