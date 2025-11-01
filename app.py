import streamlit as st
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Sentiment Analyzer | Home",
    page_icon="✈️",
    layout="centered", # 'centered' is cleaner for this page
    initial_sidebar_state="expanded"
)

# --- 2. Load All Saved Models and Objects ---
@st.cache_resource
def load_all_models():
    print("Loading models... This will run only once.")
    try:
        models = {
            "nb": joblib.load('nb_model.joblib'),
            "svm": joblib.load('svm_model.joblib'),
            "dl": load_model('dl_model.keras'),
            "tokenizer": joblib.load('tokenizer.joblib'),
            "encoder": joblib.load('label_encoder.joblib')
        }
        return models
    except FileNotFoundError:
        st.error("Model files not found! Please run train.py first to train and save the models.")
        return None

models = load_all_models()
if not models:
    st.stop()

class_labels = models["encoder"].classes_

# --- 3. Define Prediction Functions ---

def predict_ml(text_input, model):
    label = model.predict([text_input])[0]
    probs = model.predict_proba([text_input])[0]
    prob_df = pd.DataFrame(probs, index=model.classes_, columns=['Probability'])
    return label, prob_df

def predict_dl(text_input):
    MAX_LENGTH = 50 
    seq = models["tokenizer"].texts_to_sequences([text_input])
    pad = pad_sequences(seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
    pred_probs = models["dl"].predict(pad)[0]
    pred_index = np.argmax(pred_probs)
    label = class_labels[pred_index]
    prob_df = pd.DataFrame(pred_probs, index=class_labels, columns=['Probability'])
    return label, prob_df

# --- 4. Build the App Interface ---
st.title("✈️ Airline Tweet Sentiment Analyzer")
st.markdown("""
Welcome to the main analysis tool. This app uses three models to predict sentiment. 
Use the navigation sidebar on the left to explore other tools, including:
- **Batch Analyzer**: Upload a CSV file of tweets for analysis.
- **Live Simulator**: A dashboard simulating a live feed of tweets.
- **Model Explanation**: An XAI tool to see *why* a prediction was made.
""")

with st.container(border=True):
    user_text = st.text_area("Enter your tweet here:", "@AmericanAir My flight was delayed again, this service is terrible.", height=100)
    
    if st.button("Analyze Sentiment"):
        if user_text:
            st.divider()
            
            # --- Make all predictions ---
            pred_nb, probs_nb = predict_ml(user_text, models["nb"])
            pred_svm, probs_svm = predict_ml(user_text, models["svm"])
            pred_dl, probs_dl = predict_dl(user_text)
            
            def get_emoji(sentiment):
                if sentiment == 'positive': return "😊"
                if sentiment == 'negative': return "😠"
                return "😐"
            
            st.subheader("Model Predictions")
            col1, col2, col3 = st.columns(3)
            col1.metric("Naive Bayes", f"{pred_nb.capitalize()} {get_emoji(pred_nb)}")
            col2.metric("Tuned SVM", f"{pred_svm.capitalize()} {get_emoji(pred_svm)}")
            col3.metric("LSTM (DL)", f"{pred_dl.capitalize()} {get_emoji(pred_dl)}")
            
            st.divider()
            
            st.subheader("Prediction Probabilities")
            prob_col1, prob_col2, prob_col3 = st.columns(3)
            with prob_col1:
                st.write("Naive Bayes Confidence:")
                st.bar_chart(probs_nb)
            with prob_col2:
                st.write("Tuned SVM Confidence:")
                st.bar_chart(probs_svm)
            with prob_col3:
                st.write("LSTM Confidence:")
                st.bar_chart(probs_dl)
        else:
            st.warning("Please enter some text to analyze.")