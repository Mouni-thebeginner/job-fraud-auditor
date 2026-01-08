import streamlit as st
import pandas as pd
import re
import os
import requests

# ================= ML ================= #
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ================= CONFIG ================= #

DATASET_PATH = "fake_job_postings.csv"

# GROQ (FREE LLM)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# ================= TEXT CLEAN ================= #

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# ================= ML MODEL ================= #

@st.cache_resource
def load_ml_model():
    df = pd.read_csv(DATASET_PATH).fillna("")
    df["text"] = df["description"] + " " + df["requirements"]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=3000
    )

    X = vectorizer.fit_transform(df["text"].apply(clean_text))
    y = df["fraudulent"]

    model = LogisticRegression(class_weight="balanced")
    model.fit(X, y)

    return model, vectorizer

# ================= LLM ANALYSIS ================= #

def analyze_with_llm(job_text, prediction):
    verdict = "FRAUDULENT" if prediction == 1 else "REAL"

    prompt = f"""
You are a cybersecurity analyst.

MODEL VERDICT: {verdict}

NEW JOB TEXT:
{job_text[:1000]}

TASK:
Give 3 clear reasons explaining whether this job is fraudulent or legitimate.
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }

    response = requests.post(GROQ_URL, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

# ================= STREAMLIT UI ================= #

st.set_page_config(page_title="AI Job Fraud Auditor", layout="wide")
st.title("🛡️ AI Job Fraud Auditor (ML + LLM)")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY missing. Add it in Manage App → Secrets.")
    st.stop()

with st.spinner("Loading ML model..."):
    ml_model, tfidf = load_ml_model()

st.subheader("Paste Job Description")

job_text = st.text_area(
    "Paste the job description here",
    height=300
)

if st.button("🚀 Run Analysis", type="primary"):
    if job_text.strip():
        cleaned = clean_text(job_text)
        vec = tfidf.transform([cleaned])
        prediction = ml_model.predict(vec)[0]

        if prediction == 1:
            st.error("### ❌ VERDICT: FRAUDULENT")
        else:
            st.success("### ✅ VERDICT: PROBABLY REAL")

        with st.spinner("AI Analyst reasoning..."):
            reasoning = analyze_with_llm(job_text, prediction)
            st.markdown(reasoning)
    else:
        st.warning("Please paste a job description.")
