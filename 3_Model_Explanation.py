import streamlit as st
import joblib
import lime
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components
from PIL import Image

# --- 1. Page Config ---
st.set_page_config(page_title="Model Explanation", page_icon="💡")

# --- 2. Load SVM Model (all we need for LIME) ---
@st.cache_resource
def load_svm_and_explainer():
    try:
        svm_model = joblib.load('svm_model.joblib')
        # Create the LIME explainer
        explainer = LimeTextExplainer(class_names=svm_model.classes_)
        return svm_model, explainer
    except FileNotFoundError:
        st.error("SVM model not found! Please run train.py.")
        return None, None

svm_model, lime_explainer = load_svm_and_explainer()
if not svm_model:
    st.stop()

# --- 3. LIME Predictor Function ---
def svm_lime_predictor(texts):
    return svm_model.predict_proba(texts)

# --- 4. App Interface ---
st.title("💡 Model Prediction Explanation (XAI)")
st.markdown("""
This page uses **LIME** (Local Interpretable Model-agnostic Explanations) to 
show *why* our best model (the **Tuned SVM**) made its decision.
""")

with st.container(border=True):
    user_text = st.text_area("Enter a tweet to explain:", "@AmericanAir this is the best flight I've ever had, thank you!")
    
    if st.button("Explain Prediction"):
        if user_text:
            st.divider()
            with st.spinner("Generating LIME explanation... This can take a moment."):
                # Generate the LIME explanation
                explanation = lime_explainer.explain_instance(
                    user_text,
                    svm_lime_predictor,
                    num_features=10,
                    top_labels=3
                )
                
                # Get the HTML visualization
                html_explanation = explanation.as_html(predict_proba=True, show_predicted_value=True)
                
                # Display it
                st.subheader("LIME Explanation")
                components.html(html_explanation, height=500, scrolling=True)

with st.expander("See Overall Model Performance"):
    st.markdown("This plot shows the confusion matrices from the original project, demonstrating how each model performed on the test dataset.")
    try:
        img = Image.open('all_conversation_matrices_plot.png') # Use the 3-plot image
        st.image(img, caption="Comparison of Confusion Matrices for all 3 models.")
    except FileNotFoundError:
        try:
            img = Image.open('all_confusion_matrices_plot.png')
            st.image(img, caption="Comparison of Confusion Matrices for all 3 models.")
        except FileNotFoundError:
            st.warning("Could not find 'all_confusion_matrices_plot.png' or 'all_conversation_matrices_plot.png'.")
            st.info("Run the `sentiment_analysis.py` script once to generate this plot.")