import streamlit as st
import pandas as pd
import joblib
import time

# --- 1. Page Config ---
st.set_page_config(page_title="Live Simulator", page_icon="📡")

# --- 2. Load Models and Data ---
@st.cache_resource
def load_models():
    try:
        svm_model = joblib.load('svm_model.joblib')
        return svm_model
    except FileNotFoundError:
        st.error("SVM model not found! Please run train.py.")
        return None

@st.cache_data
def load_data(csv_file):
    try:
        return pd.read_csv(csv_file)
    except FileNotFoundError:
        st.error(f"Data file '{csv_file}' not found.")
        return None

svm_model = load_models()
df = load_data('Tweets.csv')

if not svm_model or df is None:
    st.stop()

# --- 3. App Interface ---
st.title("📡 Live Tweet Feed Simulator")
st.markdown("""
This dashboard simulates a live feed of incoming tweets from the dataset. 
It runs the **Tuned SVM** model on a new tweet every few seconds to
demonstrate a real-time analysis pipeline.
""")

# --- 4. Live Dashboard ---
# Create placeholders
placeholder = st.empty()

# Start/Stop button
if 'running' not in st.session_state:
    st.session_state.running = False

if st.button("Start/Stop Feed"):
    st.session_state.running = not st.session_state.running

st.write(f"Status: **{'Running' if st.session_state.running else 'Stopped'}**")

# Initialize counters
if 'counts' not in st.session_state:
    st.session_state.counts = {'negative': 0, 'neutral': 0, 'positive': 0}

while st.session_state.running:
    # Get a random tweet
    sample_tweet = df.sample(1).iloc[0]
    tweet_text = sample_tweet['text']
    airline = sample_tweet['airline']
    
    # Analyze the tweet
    prediction = svm_model.predict([tweet_text])[0]
    
    # Update counts
    st.session_state.counts[prediction] += 1
    
    # Update the dashboard
    with placeholder.container(border=True):
        st.subheader("Real-Time Sentiment Analysis")
        
        # Display the tweet
        st.markdown(f"**Airline:** `{airline}`")
        st.markdown(f"**Tweet:** *'{tweet_text}'*")
        
        # Display the prediction
        emoji = "😊" if prediction == 'positive' else "😠" if prediction == 'negative' else "😐"
        st.markdown(f"### Model Prediction: **{prediction.capitalize()}** {emoji}")
        
        st.divider()
        
        # Display running totals
        st.subheader("Running Totals")
        col1, col2, col3 = st.columns(3)
        col1.metric("Negative 😠", st.session_state.counts['negative'])
        col2.metric("Neutral 😐", st.session_state.counts['neutral'])
        col3.metric("Positive 😊", st.session_state.counts['positive'])
        
    # Wait for a few seconds
    time.sleep(3)