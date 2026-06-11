# ai/ai_agents.py
import sqlite3
import pandas as pd
import ollama
import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, 'data', 'economics.db')
MODEL_NAME = "codegemma:7b-instruct-q4_K_S"   # or "phi3:3.8b-mini-instruct-q4_K_S"

# ---------- Helper to get database schema ----------
def get_db_schema():
    if not os.path.exists(DB_PATH):
        return "Error: Database not found."
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='economic_cleaned'")
    row = cursor.fetchone()
    conn.close()
    if row and row[0]:
        return row[0]
    return """CREATE TABLE economic_cleaned (
        country_code TEXT,
        gdp_per_capita REAL,
        inflation REAL,
        unemployment REAL,
        year TEXT,
        gdp_category TEXT,
        inflation_category TEXT,
        unemployment_category TEXT
    )"""

# ---------- Text-to-SQL ----------
def ask_question(question):
    schema = get_db_schema()
    if schema.startswith("Error"):
        return "", pd.DataFrame({"error": [schema]})

    prompt = f"""
You are an expert SQLite assistant. The table schema is:
{schema}

**CRITICAL RULES:**
- Always include `country_code` in SELECT unless only a single value is asked.
- Use exact category strings: gdp_category: 'High Income', 'Upper Middle', 'Lower Middle', 'Low Income'
- inflation_category: 'Hyperinflation', 'Very High', 'High', 'Moderate', 'Low', 'Unknown'
- unemployment_category: 'Crisis Level (>20%)', 'High Unemployment (>10%)', 'Elevated (5-10%)', 'Low (≤5%)', 'Unknown'

Examples:
User: "Show me countries with Upper Middle GDP" → SELECT country_code, gdp_category, gdp_per_capita FROM economic_cleaned WHERE gdp_category = 'Upper Middle';
User: "Which country has the highest inflation?" → SELECT country_code, inflation FROM economic_cleaned ORDER BY inflation DESC LIMIT 1;

Now convert: {question}
SQL query:
"""
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': prompt}])
        sql_query = response['message']['content'].strip()
    except Exception as e:
        return "", pd.DataFrame({"error": [f"Ollama error: {str(e)}"]})

    # Clean markdown
    if sql_query.startswith("```sql"):
        sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
    elif sql_query.startswith("```"):
        sql_query = sql_query.split("```")[1].split("```")[0].strip()

    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(sql_query, conn)
    except Exception as e:
        conn.close()
        return sql_query, pd.DataFrame({"error": [str(e)]})
    conn.close()
    return sql_query, df

# ---------- Insight generator ----------
def generate_insight(question: str, df: pd.DataFrame) -> str:
    if df.empty:
        return "No data available."
    data_str = df.to_string(index=False, max_rows=10)
    if len(df) > 10:
        data_str += f"\n... and {len(df)-10} more rows."
    prompt = f"""
You are an economic analyst. Based on the user's question and the data below, write ONE sentence that provides an insightful economic observation.

User question: {question}
Data:
{data_str}

Insightful sentence:
"""
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content'].strip()
    except Exception as e:
        return f"Insight generation failed: {str(e)}"

# ---------- Country report generator ----------
COUNTRY_NAMES = {
    "USA": "United States", "CHN": "China", "JPN": "Japan", "DEU": "Germany",
    "GBR": "United Kingdom", "FRA": "France", "IND": "India", "BRA": "Brazil",
    "ITA": "Italy", "CAN": "Canada", "RUS": "Russia", "KOR": "South Korea",
    "AUS": "Australia", "MEX": "Mexico", "IDN": "Indonesia", "TUR": "Turkey",
    "SAU": "Saudi Arabia", "ARG": "Argentina", "ZAF": "South Africa", "EUU": "European Union"
}

def generate_country_report(country_code: str, data: dict) -> str:
    full_name = COUNTRY_NAMES.get(country_code, country_code)
    summary = f"""
Country: {full_name} ({country_code})
Year: {data.get('year', 'N/A')}
GDP per capita: {data.get('gdp_per_capita', 'N/A')} USD
Inflation rate: {data.get('inflation', 'N/A')}%
Unemployment rate: {data.get('unemployment', 'N/A')}%
GDP category: {data.get('gdp_category', 'N/A')}
Inflation category: {data.get('inflation_category', 'N/A')}
Unemployment category: {data.get('unemployment_category', 'N/A')}
"""
    prompt = f"""
You are an expert economic analyst. Write a short, professional country report (2‑3 paragraphs) for {full_name} based on the data below. Include an overview, interpretation of categories, policy recommendations if needed, and a one‑sentence outlook.

Data:
{summary}

Report:
"""
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content'].strip()
    except Exception as e:
        return f"Report generation failed: {str(e)}"

# ---------- Economic interpreter (general Q&A) ----------
def ask_economic_interpreter(question: str) -> str:
    prompt = f"""
You are an expert economist. When answering, note that country codes like USA, ZAF, CHN, etc. are standard ISO3 codes (e.g., ZAF = South Africa, SAU = Saudi Arabia). Answer the following question clearly and concisely.

Question: {question}
Answer:
"""
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content'].strip()
    except Exception as e:
        return f"Error: {str(e)}"