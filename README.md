Live Airline Sentiment Analysis: A Multi-Model Dashboard

Short Description

This project is a complete, end-to-end Natural Language Processing (NLP) application that classifies airline tweet sentiment. It addresses the problem that airlines have in processing high volumes of unstructured customer feedback on social media. Understanding this feedback in real-time is critical for brand management, customer service, and identifying operational issues.

To solve this, we built a multi-page interactive web application that not only predicts sentiment (positive, neutral, negative) using three different models but also explains why a model made its decision using Explainable AI (XAI). The final product is a dashboard that can be used for single-tweet analysis, batch processing of CSV files, and simulating a live data feed, demonstrating a full data-product lifecycle.

Dataset Source

Source: The project uses the "Twitter US Airline Sentiment" dataset, a popular, human-annotated corpus from Kaggle.

Data Size: We are working with 14,640 real tweets, each manually labeled as positive, neutral, or negative. The dataset is highly imbalanced, with negative tweets making up the majority of the data.

Preprocessing: The raw dataset was used directly. No filtering was performed. All "special treatment" was performed inside the model pipelines to make the process repeatable and ready for live data:

Text Cleaning: All text was converted to lowercase and punctuation was removed.

Stop Word Removal: Common English "stop words" (like 'the', 'is', 'at') were filtered out.

Vectorization (for ML): The cleaned text was converted into a numerical matrix using TfidfVectorizer.

Tokenization & Padding (for DL): For the LSTM model, the text was converted into sequences of integers using Tokenizer and padded to a uniform length of 50 tokens.

Label Encoding: The target labels ('negative', 'neutral', 'positive') were encoded into integers (0, 1, 2) for the Deep Learning model.

Methods

Our approach was to solve this as a supervised classification problem by comparing multiple models of increasing complexity. This allows us to find the best trade-off between performance, speed, and interpretability.

We considered using only a Deep Learning model but chose to include ML models as a baseline, which is a standard academic and professional practice to justify complex model choices.

ML - Naive Bayes (Baseline): We used MultinomialNB as our baseline. This model is extremely fast and works surprisingly well with text, making it a perfect benchmark.

ML - Tuned SVM (Primary Model): We chose SVC(kernel='linear') as our primary workhorse model. SVMs are highly effective in high-dimensional sparse spaces, which is exactly what text data becomes after TF-IDF vectorization. We used GridSearchCV to find the optimal hyperparameters for C (regularization) and ngram_range (word pairs), ensuring maximum performance.

DL - LSTM (Advanced Alternative): As an advanced alternative, we built a Recurrent Neural Network (RNN) using tensorflow.keras. Unlike the other models, an LSTM can understand word order and context, which can be critical for understanding nuances like sarcasm.

Explainable AI - LIME: We didn't just want a "black box" prediction. We integrated LimeTextExplainer to analyze our best ML model (the SVM). LIME explains individual predictions by showing which specific words in a tweet contributed to or against a prediction. This is our approach to adding transparency and trust to the model.

Steps to run the code

You must have Python 3.10 or 3.11 installed.

1. Clone or Download the Project

Get all the files and folders for this project onto your computer.

*// 2. Create a Virtual Environment (Recommended)

# Create a new environment in a folder named 'venv'
python -m venv venv

# Activate the environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate //*


3. Install All Required Libraries

This project needs several libraries. You can install them all at once using the requirements.txt file.

pip install -r requirements.txt


4. Train The Models (Run This Once)

You must run the train.py script to train all three models and save them to your folder.

WARNING: This step will take a long time (15-30+ minutes), mainly because of the GridSearchCV for the SVM. You only have to do this once.

python train.py


You will see several new files appear, like svm_model.joblib, nb_model.joblib, and dl_model.keras.

5. Run the Streamlit App

Once the models are trained, you can start the web application.

python -m streamlit run app.py


Your web browser should automatically open to the app.

Experiments/Results Summary

We conducted experiments to find the best-performing model. Our "experiment" involved training three distinct models and, for the SVM, conducting a hyperparameter search.

Hyperparameter Experiment

We used GridSearchCV to test various settings for our SVC model. The search concluded that the best-performing parameters were {'clf__C': 1, 'tfidf__ngram_range': (1, 2)}, which means a regularization strength of 1.0 and using both single words (1-grams) and two-word pairs (2-grams) were optimal.

Performance Comparison

The models were evaluated on a 20% test split of the data (this was done in our sentiment_analysis.py script). The results clearly show the Tuned SVM is the most balanced and accurate model, while the Naive Bayes model performed poorly.

Model

Accuracy

Weighted F1-Score

Naive Bayes (ML)

~67.8%

~59.3%

Tuned SVM (ML)

~77.9%

~77.4%

LSTM (DL)

~77.1%

~76.8%

(Note: Scores are from the test run and may vary slightly on different training runs)





Results Visualization

The "Model Performance" tab of the app shows a confusion matrix plot for all three models.

The Naive Bayes matrix shows a heavy bias, misclassifying most neutral and positive tweets as negative.

The Tuned SVM and LSTM matrices show a much stronger "diagonal" line, indicating they are far more balanced and accurate at identifying all three classes correctly.

We also used LIME to gain insight into how the SVM model works. The "Model Explanation" page of the app demonstrates this, highlighting (in green/red) the words that the model used to make its decision. This is a key visualization technique for understanding model behavior.

Conclusion

Key Result: Our comparative analysis found that the Tuned SVM is the best all-around model for this task, slightly outperforming the much more complex LSTM and significantly outperforming the Naive Bayes baseline.

What We Learned: We learned that for this text classification task, a well-tuned classic ML model (SVM) with good text features (TF-IDF n-grams) can be just as, if not more, effective than a Deep Learning model. We also learned that building a functional model is only half the battle; creating a usable, interactive, and interpretable application (with tools like Streamlit and LIME) is what turns a project into a true data product.

Project Structure

NLP-Project/
├── .streamlit/
│   └── config.toml        # The custom theme for the app
├── pages/
│   ├── 1_Batch_Analyzer.py  # Page 2 of the app
│   ├── 2_Live_Feed_Simulator.py # Page 3 of the app
│   └── 3_Model_Explanation.py # Page 4 of the app
├── app.py                 # The main Home page of the app
├── train.py               # The script to train all models
├── requirements.txt       # List of all libraries to install
├── Tweets.csv             # The raw dataset
└── ... (model files like svm_model.joblib)


References

Dataset: Kaggle: Twitter US Airline Sentiment

Libraries:

Streamlit

Scikit-learn (SVM, Naive Bayes, GridSearchCV)

TensorFlow/Keras (LSTM)

LIME (Explainable AI)

Pandas