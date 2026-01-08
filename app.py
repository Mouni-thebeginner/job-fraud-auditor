import streamlit as st
import pandas as pd
import re
import os
import requests

# ================= OPTIONAL CHROMADB ================= #
try:
    from chromadb import PersistentClient
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except Exception:
    CHROMA_AVAILABLE = False

# ================= OPTIONAL PDF OCR ================= #
try:
    from pdf2image import convert_from_bytes
    PDF_OCR_AVAILABLE = True
except Exception:
    PDF_OCR_AVAILABLE = False

# ================= IMAGE OCR ================= #
from PIL import Image
import pytesseract

# ================= ML ================= #
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ================= CONFIG ================= #

DATASET_PATH = "fake_job_postings.csv"

# GROQ (FREE LLM)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# ================= VECTOR DB (SAFE) ================= #

@st.cache_resource
def init_vector_db():
    if not CHROMA_AVAILABLE:
        return None

    client = PersistentClient(path="./chroma_db")
    embedding = embedding_functions.DefaultEmbeddingFunction()

    collection = client.get_or_create_collection(
        name="job_scams",
        embedding_function=embedding
    )

    if collection.count() == 0 and os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH).fillna("")
        scams = df[df["fraudulent"] == 1].head(150)

        documents = (scams["description"] + " " + scams["requirements"]).tolist()
        ids = [f"scam_{i}" for i in range(len(documents))]

        collection.add(documents=documents, ids=ids)

    return collection

# ================= OCR ================= #

def extract_text_from_file(uploaded_file):
    text = ""
    try:
        if uploaded_file.type.startswith("image"):
            image = Image.open(uploaded_file)
            text = pytesseract.image_to_string(image)

        elif uploaded_file.type == "application/pdf":
            if not PDF_OCR_AVAILABLE:
                st.warning("⚠️ PDF OCR is not supported in cloud. Please upload an image instead.")
                return ""

            images = convert_from_bytes(uploaded_file.read())
            for img in images:
                text += pytesseract.image_to_string(img) + "\n"

    except Exception as e:
        st.error(f"OCR Error: {e}")

    return text.strip()

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

def analyze_with_llm(job_text, prediction, collection):
    verdict = "FRAUDULENT" if prediction == 1 else "REAL"

    retrieved_context = "Vector database unavailable."

    if collection is not None:
        results = collection.query(query_texts=[job_text], n_results=2)
        retrieved_context = "\n---\n".join(results["documents"][0])

    prompt = f"""
You are a cybersecurity analyst.

MODEL VERDICT: {verdict}

REFERENCE CONTEXT:
{retrieved_context}

NEW JOB TEXT:
{job_text[:1000]}

TASK:
Give 3 clear reasons explaining whether this job follows scam patterns.
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
st.title("🛡️ AI Job Fraud Auditor (ML + LLM + OCR)")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY missing. Add it in Manage App → Secrets.")
    st.stop()

with st.spinner("Initializing systems..."):
    vector_db = init_vector_db()
    ml_model, tfidf = load_ml_model()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Job Data")

    mode = st.radio("Choose input method", ["Paste Text", "Upload Image / PDF"])
    final_text = ""

    if mode == "Paste Text":
        final_text = st.text_area("Paste job description", height=250)
    else:
        file = st.file_uploader(
            "Upload Image or PDF",
            type=["png", "jpg", "jpeg", "pdf"]
        )
        if file:
            final_text = extract_text_from_file(file)
            if final_text:
                st.success("Text extracted successfully!")

if st.button("🚀 Run Analysis", type="primary"):
    if final_text.strip():
        cleaned = clean_text(final_text)
        vec = tfidf.transform([cleaned])
        prediction = ml_model.predict(vec)[0]

        with col2:
            st.subheader("Results")

            if prediction == 1:
                st.error("### ❌ VERDICT: FRAUDULENT")
            else:
                st.success("### ✅ VERDICT: PROBABLY REAL")

            with st.spinner("AI Analyst reasoning..."):
                reasoning = analyze_with_llm(
                    final_text,
                    prediction,
                    vector_db
                )
                st.markdown(reasoning)
    else:
        st.warning("Please provide job text or upload an image.")
