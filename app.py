import streamlit as st
import re
import os
import requests

# ================= CONFIG ================= #

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# ================= SIMPLE RULE-BASED CLASSIFIER ================= #

SCAM_KEYWORDS = [
    "pay upfront",
    "registration fee",
    "processing fee",
    "whatsapp only",
    "telegram",
    "earn daily",
    "quick money",
    "no interview",
    "limited slots",
    "act fast",
    "work from home and earn",
    "investment required",
    "click the link",
    "guaranteed income"
]

def rule_based_prediction(text):
    text = text.lower()
    score = sum(1 for kw in SCAM_KEYWORDS if kw in text)
    return 1 if score >= 2 else 0   # 1 = Fraudulent, 0 = Probably Real

# ================= LLM ANALYSIS ================= #

def analyze_with_llm(job_text, prediction):
    verdict = "FRAUDULENT" if prediction == 1 else "PROBABLY REAL"

    prompt = f"""
You are a cybersecurity analyst.

RULE-BASED MODEL VERDICT: {verdict}

JOB DESCRIPTION:
{job_text[:1200]}

TASK:
Explain in 3 clear bullet points whether this job posting shows scam patterns.
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
st.title("🛡️ AI Job Fraud Auditor (Cloud-Safe Demo)")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY missing. Add it in Manage App → Settings → Secrets")
    st.stop()

st.subheader("Paste Job Description")

job_text = st.text_area(
    "Paste the job description here",
    height=300
)

if st.button("🚀 Run Analysis", type="primary"):
    if job_text.strip():
        prediction = rule_based_prediction(job_text)

        if prediction == 1:
            st.error("### ❌ VERDICT: LIKELY FRAUDULENT")
        else:
            st.success("### ✅ VERDICT: PROBABLY REAL")

        with st.spinner("AI Analyst reasoning..."):
            reasoning = analyze_with_llm(job_text, prediction)
            st.markdown(reasoning)
    else:
        st.warning("Please paste a job description.")
