import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define the path for the feedback file
FEEDBACK_FILE = "user_feedback.csv"

# --- Helper functions ---
def clean_text(text):
    """Clean the text data."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove punctuation and numbers
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    return text

@st.cache_resource
def load_and_train_model():
    """
    Loads the base dataset, incorporates user feedback, and trains a
    Logistic Regression model.
    """
    # Load base dataset
    try:
        df = pd.read_csv("fake_job_postings.csv")
    except FileNotFoundError:
        st.error("The base dataset 'fake_job_postings.csv' was not found.")
        st.stop()
        
    df = df.drop_duplicates().fillna('')

    # Combine text columns for the base dataset
    text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    df['combined_text'] = df[text_cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    # Load and incorporate user feedback if it exists
    if os.path.exists(FEEDBACK_FILE):
        feedback_df = pd.read_csv(FEEDBACK_FILE)
        # Ensure feedback dataframe has the same structure
        feedback_df.rename(columns={'text': 'combined_text'}, inplace=True)
        df = pd.concat([df, feedback_df], ignore_index=True)

    # Clean all text data
    df['cleaned_text'] = df['combined_text'].apply(clean_text)

    # Feature and target variables
    X = df['cleaned_text']
    y = df['fraudulent']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(stop_words='english', max_features=1500, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Train Logistic Regression model
    model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_tfidf)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    return model, tfidf, metrics

# --- Streamlit UI ---
st.set_page_config(page_title="Fake Job Detector", page_icon="🕵️")
st.title("🕵️ Fake Job Postings Detector")
st.markdown("Enter a job description to determine if it's likely real or fraudulent. You can also provide feedback to help improve the model over time.")

# Load and train the model, showing a spinner
with st.spinner("Loading and training model... This might take a moment."):
    model, tfidf, metrics = load_and_train_model()

# Display model performance metrics in an expander
with st.expander("📊 View Model Performance"):
    st.write(f"**Accuracy:** {metrics['accuracy']*100:.2f}%")
    st.write(f"**Precision:** {metrics['precision']*100:.2f}%")
    st.write(f"**Recall:** {metrics['recall']*100:.2f}%")
    st.write(f"**F1 Score:** {metrics['f1']*100:.2f}%")

# Prediction input
st.subheader("🔍 Analyze a Job Posting")
user_input = st.text_area("Paste the full job posting text here:", height=250)

if st.button("Predict", type="primary"):
    if user_input.strip():
        # Clean and transform user input
        cleaned_input = clean_text(user_input)
        vectorized_input = tfidf.transform([cleaned_input])

        # Make prediction
        prediction = model.predict(vectorized_input)[0]
        proba = model.predict_proba(vectorized_input)[0]

        # Store prediction in session state for feedback
        st.session_state['last_input'] = user_input
        st.session_state['last_prediction'] = int(prediction)

        # Display prediction result
        if prediction == 1:
            st.error("#### ⚠️ This job posting is likely **FRAUDULENT**")
        else:
            st.success("#### ✅ This job posting appears to be **REAL**")

        # Display prediction probabilities
        st.subheader("📝 Prediction Analysis")
        st.write(f"**Probability of being Real:** {proba[0]*100:.2f}%")
        st.write(f"**Probability of being Fraudulent:** {proba[1]*100:.2f}%")
    else:
        st.warning("Please enter some text from a job posting.")

# --- Feedback Section ---
if 'last_prediction' in st.session_state:
    st.markdown("---")
    st.subheader("🙋 Was this prediction correct?")

    col1, col2, col3 = st.columns([1, 1, 3])

    with col1:
        if st.button("👍 Yes"):
            feedback_data = {
                'text': [st.session_state['last_input']],
                'fraudulent': [st.session_state['last_prediction']]
            }
            feedback_df = pd.DataFrame(feedback_data)
            feedback_df.to_csv(FEEDBACK_FILE, mode='a', header=not os.path.exists(FEEDBACK_FILE), index=False)
            st.success("Thanks for your feedback!")
            # Clear state after feedback
            del st.session_state['last_prediction']
            del st.session_state['last_input']


    with col2:
        if st.button("👎 No"):
            # Flip the label for incorrect predictions
            correct_label = 1 - st.session_state['last_prediction']
            feedback_data = {
                'text': [st.session_state['last_input']],
                'fraudulent': [correct_label]
            }
            feedback_df = pd.DataFrame(feedback_data)
            feedback_df.to_csv(FEEDBACK_FILE, mode='a', header=not os.path.exists(FEEDBACK_FILE), index=False)
            st.success("Thank you! Your correction has been saved to improve the model.")
            # Clear state after feedback
            del st.session_state['last_prediction']
            del st.session_state['last_input']

# --- Retraining Section ---
st.markdown("---")
st.subheader("🔄 Retrain Model with New Feedback")
st.markdown("If you've provided feedback, you can retrain the model to include the new data. This will update the model for all users.")

if st.button("Retrain Now"):
    with st.spinner("Retraining model with all feedback... Please wait."):
        # Clear the cached resource to force re-execution
        st.cache_resource.clear()
        st.success("Model retrained successfully! The performance metrics have been updated.")
        st.rerun()
