Sentiment Analysis Live Airline Multi-Model Dashboard.







Short Description







This is an end-to-end Natural Language Processing (NLP) application that is an airline sentiment on tweets. It is concerned with the challenge that airlines encounter with respect to unstructured customer feedback in bulk in the social media. This feedback is necessary to control the brand, customer service, and identify operational issues with references to the real-time feedback of this feedback.







We have developed a multi-page interactive web application, as a solution to this, which, besides predicting its sentiment (positive, neutral, negative) by three different models, it also gives the reasons why a model has made its decision in the context of Explainable AI (XAI). The final product is a dashboard, which can single-tweet, process a batch of CSV files, and simulate a live data feed, and displays a lifecycle of a data-product.







Dataset Source







Data: The project is founded on the popular, manually labeled, Kaggle dataset, which is named Twitter US Airline Sentiment.







Size of Data: There are 14640 actual tweets that were manually classified as being positive, neutral or negative. The statistics are highly skewed with majority of the statistics being negative tweets.







Preprocessing: Raw data was immediately taken. No filtering was performed. All the special treatment was run on the model pipelines to repeat the process and prepare it with real data:







Text Cleaning: The text was also converted to lower case and all punctuations were removed.







Stop Word Removal:The English language has stop words (the, is, at) which were eliminated.







ML: A numerical matrix was created by using TfidfVectorizer to convert the cleaned text into a numerical format.







Tokenization & Padding (DL): The text was converted to a sequence of integers using the Tokenizer and padded to the number of tokens of 50 in the case of the LSTM model.







Label Encoding: Deep Learning model The Deep Learning model was implemented by means of integers (0, 1, 2) denoting the target labels (negative, neutral, positive).







Methods







This was addressed as a monitored classification issue, wherein we contrasted a group of models of more and more intricacy. This will make it possible to be in a position to come up with the best trade-off between performance, speed and interpretability.







We considered simply applying aDeep Learning model but chose to have some ML models to be used as a baseline since it is a typical scholarly and professional practice to justify the need to apply complex models.







ML -Naive Bayes (Baseline): MultinomialNB served as our baseline. This model is extremely fast and surprisingly efficient when it comes to text and that is an ideal standard.







ML - Tuned SVM (Primary Model):The main model that we use is SVC(kernel=linear). SVMs perform quite well in high dimensional sparse space, which is the case with text data after TF-IDFvectorization. We have preferred hyperparameters of relevance; C (regularization) and ngram range (word pairs) that have been optimized with the help of the GridSearchCV in such a way that it ensures the optimal performance.







DL - LSTM (Advanced Alternative): As an advanced alternative, we constructed a Recurrent Neural Network (RNN) using the assistance of tensorflow.keras. Unlike the other models, an LSTM can recognize sequence and context of words that may be vital in identifying other details like sarcasm.







Explainable AI - LIME: not to us, it was not sufficient to have a black box prediction. LimeTextExplainer has been employed to examine our most successful ML model (the SVM). Based on the single predictions, LIME shows how words in a certain tweet contributed to a prediction or contradicted a prediction. It is in this way we are going to give transparency and trust on the model.







Steps to run the code







Python 3.10 or 3.11 has to be installed.







1. Clone or Replicate the Project.







Place all files and folders of this project in the computer.







*// 2. Create a fictitious environment (Abstractly) (Recommended)







Create a new environment within the folder that is called as venv.



python -m venv venv







# Activate the environment



# On Windows:



.\venv\Scripts\activate



# On macOS/Linux:



source venv/bin/activate //*











3. Install Libraries Install all requirements.







This project will need a number of libraries. They can all be installed simultaneously without the need to modify any of them.







pip install requirements.txt.











4. Train The Models (Run This Once)







In order to train all the three models you must execute the train.py code and store them in your folder.







RISK: The step will be time-consuming (15-30+ minutes), mostly because of the GridSearchCV of the SVM. You only have to do this once.







python train.py











There will be some additional files like svm model joblib, nb model joblib and learning model in the form of keras.







5. Run the Streamlit App







The web application may be released after the models have been trained.







python -m streamlit run app.py











This will open the web.







Experiments/Results Summary







We tried various models to establish the best performing model. Our experiment involved the utilization of three models, and (with SVM) a hyperparameter search.







Hyperparameter Experiment







Different settings of our SVC model were tested by using gridSearchCV. The last parameters that were seemed to be most successful were {'clf__C': 1, 'tfidf__ngram_range': (1, 2)'} which implies that the regularization strength of 1.0 and the two-word combinations (2-grams) worked most successfully.







Performance Comparison







This split of the test was done with 20 percent (this was done in our sentiment_analysis.py script). The results give clear findings that Tuned SVM is the best balanced and correct model and Naive Bayes model was not functioning properly.







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







The scores in test run means are the average of the two runs (Note: The scores will be obtained when the test is run and may vary a little when the test is run again).























Results Visualization







The tab of the app called Model Performance shows the confusion matrix plot of all the three models.





The Naive Bayses matrix is highly skewed with majority of the neutral and positive tweets being classified as negative.



The Tuned SVM matrix and LSTM matrix have a significantly stronger "diagonal" line; hence, they are much more balanced and precise in recognizing all three types properly.



LIME was also the tool we applied to understand the functioning of the SVM model. This is evidenced by the "Model Explanation" page of the app, where the words that the model considered to make its decision are highlighted (in green/red). It is one of the major visualizations in the behavior of models.









