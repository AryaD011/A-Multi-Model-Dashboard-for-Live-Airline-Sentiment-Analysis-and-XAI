import streamlit as st
import pandas as pd
import joblib
from io import BytesIO

# --- 1. Load Models (minimal, only what's needed) ---
@st.cache_resource
def load_models():
    try:
        svm_model = joblib.load('svm_model.joblib')
        return svm_model
    except FileNotFoundError:
        st.error("SVM model not found! Please run train.py.")
        return None

svm_model = load_models()
if not svm_model:
    st.stop()

# --- 2. App Interface ---
st.set_page_config(page_title="Batch Analyzer", page_icon="🗂️")
st.title("🗂️ Batch Tweet Analyzer")
st.markdown("""
Upload a CSV file containing tweets, and this tool will predict the sentiment for each one 
using our best model (the **Tuned SVM**).
""")

# --- 3. File Upload ---
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Load the uploaded data
        df_upload = pd.read_csv(uploaded_file)
        
        # Ask user to select the text column
        st.info("Please select the column that contains the tweet text.")
        column_options = df_upload.columns.tolist()
        text_column = st.selectbox("Which column has the text?", column_options)

        if st.button("Analyze File"):
            if text_column:
                # --- 4. Run Predictions ---
                with st.spinner('Analyzing file... This may take a moment.'):
                    # Get predictions
                    predictions = svm_model.predict(df_upload[text_column])
                    # Get probabilities
                    probabilities = svm_model.predict_proba(df_upload[text_column])
                    
                    # Add results to the DataFrame
                    df_upload['predicted_sentiment'] = predictions
                    # Add probabilities for each class
                    for i, class_name in enumerate(svm_model.classes_):
                        df_upload[f'{class_name}_probability'] = probabilities[:, i]

                st.success("Analysis complete!")
                
                # --- 5. Display Results and Download ---
                st.subheader("Analysis Results")
                st.dataframe(df_upload)
                
                # Convert DataFrame to CSV for download
                @st.cache_data
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')
                
                csv = convert_df_to_csv(df_upload)
                
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name='sentiment_analysis_results.csv',
                    mime='text/csv',
                )
    except Exception as e:
        st.error(f"An error occurred: {e}")