# 🧠 EconsGPT – Economic Intelligence MVP

> **Economic data visualisation + AI assistant + machine learning forecasts – unified in one Streamlit dashboard.**

---

## 🎯 What is EconsGPT?

EconsGPT brings together **three portfolio projects** into a single, polished application:

- **Visualise** – interactive charts, maps, and filters for G20 economic data
- **AI Assistant** – text-to-SQL, economic Q&A, and country reports using a local LLM (Ollama)
- **Forecast** – machine learning predictions for GDP, inflation, unemployment, recession risk, and clustering

---

## 📦 Data & Model Sources

This repository consolidates work from three previous projects:
- **`economic-analysis`** – provides the SQLite database (`economics.db`) and AI assistant logic.
- **`economic-machine`** – supplies the pre‑processed feature CSV and all trained ML models (`.pkl` files).
- **`economics-dashboard`** – contributes the visualisation components and chart designs.

All data originates from the **World Bank Open Data API** (2010–2024). The models are pre‑trained and included for immediate use; no re‑training is required.

---

## 🚀 Features

### 📊 Visualise
- GDP, inflation, unemployment dashboards
- Bar charts, scatter plots, maps, box plots, correlation heatmap
- Global sidebar country filter (updates all tabs)

---

### 🤖 AI Assistant (requires Ollama locally)
- **Text-to-SQL** → ask questions like *"Which country has highest inflation?"*
- **Economic Interpreter** → explain economic patterns (e.g. Argentina hyperinflation)
- **Country Report Generator** → structured 2–3 paragraph reports per country

---

### 🔮 Forecast (Machine Learning)
- GDP per capita prediction (2025)
- Inflation prediction (2025)
- Unemployment prediction (2025)
- Recession risk probability (Random Forest)
- Country clustering (K-Means + PCA visualization)
- Percentage change vs 2024 values

Models are trained using:
- XGBoost
- Random Forest
- Scikit-learn clustering
- Feature engineering (lags, rolling averages, growth rates)

---

## 🧪 Setup & Installation

### Prerequisites
- Python 3.10+
- Git
- (Optional) Ollama for AI features

---

### 1. Clone repository
```bash
git clone https://github.com/Syawaluddin-Adzran/econs-gpt.git
cd econs-gpt
```

---

### 2. Create virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

---

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

### 4. Run app
```bash
streamlit run app.py
```

Open:
```
http://localhost:8501
```

---

### 5. (Optional) Enable AI features

Install Ollama:
https://ollama.com

Pull model:
```bash
ollama pull phi3:3.8b-mini-instruct-q4_K_S
```

Run Ollama in background before launching app.

---

## 📂 Repository Structure

```
econs-gpt/
├── data/
│   ├── economics.db
│   └── featured_economic.csv
│
├── models/
│   ├── gdp_forecast_model.pkl
│   ├── inflation_forecast_model.pkl
│   ├── unemployment_forecast_model.pkl
│   ├── recession_classifier.pkl
│   ├── kmeans_model.pkl
│   └── kmeans_scaler.pkl
│
├── ai/
│   └── ai_agents.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 🧠 Model Performance (Test 2022–2024)

| Model | Metric | Value |
|------|--------|-------|
| GDP Forecast (XGBoost) | R² | 0.945 |
| GDP Forecast | RMSE | ~5,300 USD |
| Recession Classifier | Accuracy | 91% |
| Recession Classifier | Precision / Recall | 95% / 84% |
| Clustering (K-Means) | k | 3 optimal clusters |

### Clusters:
- **Cluster 0:** Argentina (hyperinflation >200%)
- **Cluster 1:** Most G20 countries (stable economies)
- **Cluster 2:** South Africa (high unemployment)

---

## 🌐 Deployment

- Cloud version: Visualisation + Forecast only
- AI features require local Ollama setup

👉 Streamlit Cloud:
https://econs-gpt.streamlit.app

---

## 📚 Data Source

- World Bank Open Data API (2010–2024)

---

## 🤝 Acknowledgements

- World Bank
- Streamlit
- XGBoost
- Scikit-learn
- Ollama

---

## 👤 Author

**Muhammad Syawaluddin Bin Adzran**

- GitHub: https://github.com/Syawaluddin-Adzran  
- LinkedIn: https://www.linkedin.com/in/muhammad-syawaluddin-bin-adzran/

---

## 📄 License

MIT License – free to use, modify, and distribute.

---

## 🏁 Final Note

This project combines:
- Data engineering
- Business intelligence
- AI (LLM integration)
- Machine learning forecasting

into one end-to-end **economic intelligence system**.
