Live Airline Sentiment Analysis: Multi-Model Dashboard.

Short Description

This project is a full end-to-end Natural Language Processing (NLP) application which is an airline sentiment in tweets. It deals with the issue that airlines face in terms of unstructured customer feedback in large quantities in social media. It is essential to manage the brand, customer service, and recognize operational problems based on the real-time feedback of this feedback.

As a solution to this, we have created a multi-page interactive web application that does not only predict the sentiment (positive, neutral, negative) using three different models but also provides the reasons why a model has made its choice within the framework of Explainable AI (XAI). The end product is a dashboard that is capable of single-tweet analysis, CSV file batch processing, simulating a live data feed, and shows a complete data-product lifecycle.

Dataset Source

Data: The project is based on the popular, manually annotated, Kaggle dataset, which is called Twitter US Airline Sentiment.

Data Size: We have 14640 real tweets, which were manually marked as positive, neutral, or negative. The data is much skewed with most of the data comprising negative tweets.

Preprocessing: The raw data was directly taken. No filtering was performed. The model pipelines were used to carry out all the special treatment in order to repeat the process and prepare it using real data:

Text Cleaning: Text was also changed to lower case and all punctuations were deleted.

Stop Word Removal: "stop words" in the English language (such as the, is, at) were removed.

Vectorization (ML): TfidfVectorizer was used to transform the cleaned text into a numerical matrix.

Tokenization & Padding (DL): In the case of the LSTM model, the text was turned into a sequence of integers with the help of the Tokenizer and padded to the same size of 50 tokens.

Label Encoding: Deep Learning model The Deep Learning model was coded using integers (0, 1, 2) representing the target labels (negative, neutral, positive).

Methods

This was solved as a supervised classification problem where we compared a set of models of increasing complexity. This will enable us to be able to find the optimal trade-off between performance, speed, and interpretability.

We thought of just using a Deep Learning model but decided to have some ML models to serve as a baseline as is a common academic and professional convention to explain the need to use complex models.

ML - Naive Bayes (Baseline): MultinomialNB was used as our baseline. This model is very quick and surprisingly effective with text, and it is an ideal benchmark.

ML - Tuned SVM (Primary Model):Our primary workhorse model is SVC(kernel=linear). The performance of SVMs is very good in high dimensional sparse spaces, and that is what happens to text data after TF-IDFvectorization. Our preferred hyperparameters are C (regularization) and ngram range (word pairs) which were optimized using GridSearchCV so as to guarantee optimal performance.

DL - LSTM (Advanced Alternative): We developed a Recurrent Neural Network (RNN) with the help of tensorflow.keras as an advanced alternative. Contrary to the other models, an LSTM is able to appreciate word sequence and context which can be essential in discerning details such as sarcasm.

Explainable AI - LIME: it was not enough to us to have a black box prediction. We have used LimeTextExplainer to analyze our most effective ML model (the SVM). From the individual predictions, LIME displays which words in a particular tweet played a part in a prediction or went against a prediction. This is how we are going to provide transparency and trust on the model.

Steps to run the code

Python 3.10 or 3.11 has to be installed.

1. Duplicate or Replicate the Project.

Put all files and folders of this project in your computer.

*// 2. Design an imaginary setting (Abstractly) (Recommended)

Design a new setting inside the folder named as venv.
python -m venv venv

# Activate the environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate //*


3. Install Install All necessities Libraries.

A number of libraries are required in this project. The requirements.txt file allows you to install all of them at the same time.

pip install requirements.txt.


4. Train The Models (Run This Once)

To train all the three models, you need to run the train.py script and save them to your folder.

DANGER: The step will consume a lot of time (15-30+ minutes), primarily due to the GridSearchCV of the SVM. You only have to do this once.

python train.py


Some new files will appear such as svm model joblib, nb model joblib and learning model in the form of keras.

5. Run the Streamlit App

After the training of the models, the web application can be launched.

python -m streamlit run app.py


This will open the web.

Experiments/Results Summary

We tested different models to determine the most performing model. We used three different models and (in the case of SVM) a hyperparameter search as part of our experiment.

Hyperparameter Experiment

GridSearchCV was used to evaluate different settings of our SVC model. The final parameters that appeared to be most effective were {'clf__C': 1, 'tfidf__ngram_range': (1, 2)'}, meaning that the strength of regularization of 1.0 and the two-word combinations (2-grams) were the most successful.

Performance Comparison

On this test split (20 percent), the models were tested (this was performed in our sentiment_analysis.py script). The findings provide evident results that Tuned SVM is the most balanced and accurate model and Naive Bayes model did not operate properly.

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

Test run scores indicate the mean between the two runs (Note: the scores are obtained during the running of the test and might differ slightly when the test is run again).





Results Visualization

The confusion matrix plot of all three models is displayed in the tab of the app named Model Performance.

The Naive Bayses matrix is highly skewed with majority of the neutral and positive tweets being classified as negative.

The Tuned SVM matrix and LSTM matrix have a significantly stronger "diagonal" line; hence, they are much more balanced and precise in recognizing all three types properly.

LIME was also the tool we applied to understand the functioning of the SVM model. This is evidenced by the "Model Explanation" page of the app, where the words that the model considered to make its decision are highlighted (in green/red). It is one of the major visualizations in the behavior of models.

