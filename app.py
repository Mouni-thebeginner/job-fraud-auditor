import streamlit as st
import re
import os
import requests

# ================= CONFIG ================= #

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# ================= RULE-BASED FRAUD DETECTION ================= #

SCAM_KEYWORDS = [
    "registration fee",
    "processing fee",
    "pay upfront",
    "investment required",
    "no interview",
    "earn daily",
    "quick money",
    "guaranteed income",
    "limited slots",
    "act fast",
    "telegram",
    "whatsapp only",
    "click the link",
    "work from home and earn",
    "refund after payment"
]

def rule_based_prediction(text: str) -> int:
    text = text.lower()
    matches = sum(1 for kw in SCAM_KEYWORDS if kw in text)
    return 1 if matches >= 1 else 0   # 1 = Fraud, 0 = Probably Real

# ================= LLM ANALYSIS (SAFE) ================= #

def analyze_with_llm(job_text: str, prediction: int) -> str:
    verdict = "FRAUDULENT" if prediction == 1 else "PROBABLY REAL"

    prompt = f"""
You are a cybersecurity analyst.

RULE-BASED VERDICT: {verdict}

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
    "model": "llama-3.1-8b-instant",
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.3
    }

    try:
        response = requests.post(
            GROQ_URL,
            headers=headers,
            json=payload,
            timeout=30
        )

        data = response.json()

        # ✅ SUCCESS RESPONSE
        if isinstance(data, dict) and "choices" in data:
            return data["choices"][0]["message"]["content"]

        # ❌ API ERROR RESPONSE
        if isinstance(data, dict) and "error" in data:
            return f"⚠️ AI Error: {data['error'].get('message', 'Unknown error')}"

        # ❌ UNKNOWN RESPONSE
        return "⚠️ AI returned an unexpected response. Please try again later."

    except Exception as e:
        return f"⚠️ AI request failed: {e}"

# ================= STREAMLIT UI ================= #

st.set_page_config(
    page_title="AI Job Fraud Auditor",
    layout="wide"
)

st.title("🛡️ AI Job Fraud Auditor (Cloud-Safe)")

if not GROQ_API_KEY:
    st.error(
        "❌ GROQ_API_KEY not found.\n\n"
        "Go to **Manage App → Settings → Secrets** and add:\n"
        "`GROQ_API_KEY = \"gsk_...\"`"
    )
    st.stop()

st.subheader("Paste Job Description")

job_text = st.text_area(
    "Enter the job description below",
    height=300
)

if st.button("🚀 Run Analysis", type="primary"):
    if job_text.strip():
        prediction = rule_based_prediction(job_text)

        if prediction == 1:
            st.error("### ❌ VERDICT: LIKELY FRAUDULENT")
        else:
            st.success("### ✅ VERDICT: PROBABLY REAL")

        with st.spinner("AI analyst reasoning..."):
            reasoning = analyze_with_llm(job_text, prediction)
            st.markdown(reasoning)
    else:
        st.warning("Please paste a job description to analyze.")



