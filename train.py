# --- 0. Import Libraries ---
import pandas as pd
import time
import joblib 

# ML Libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC  # <--- CHANGED FROM LinearSVC TO SVC
from sklearn.preprocessing import LabelEncoder

# DL Libraries
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D

print("--- Starting Model Training (A+ Version) ---")
print("NOTE: This will be much slower than before, please be patient.")

# --- 1. Load the Dataset ---
file_name = "Tweets.csv"
df = pd.read_csv(file_name)

# --- 2. Define Features (X) and Target (y) ---
X = df['text']
y_text = df['airline_sentiment'] 

label_encoder = LabelEncoder()
y_numeric = label_encoder.fit_transform(y_text)
print("Data loaded and labels encoded.")

X_train, y_text_train, y_numeric_train = (X, y_text, y_numeric)

# --- 4. Train and Save Model 1: Naive Bayes ---
print("\nTraining Model 1: Naive Bayes...")
start_time = time.time()
nb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')), 
    ('clf', MultinomialNB())                          
])
nb_pipeline.fit(X_train, y_text_train)
print(f"Training took: {time.time() - start_time:.2f} seconds")
joblib.dump(nb_pipeline, 'nb_model.joblib')
print("Saved 'nb_model.joblib'")


# --- 5. Train and Save Model 2: Tuned SVC (with Probabilities!) ---
print("\nTraining Model 2: Tuned SVM (GridSearchCV)...")
print("(This is the VERY SLOW step. Please wait...)")
start_time = time.time()

# *** THIS IS THE KEY CHANGE ***
# We are now using SVC(probability=True) so we can get probabilities.
svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', SVC(kernel='linear', probability=True, random_state=42)) 
])

param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__C': [0.1, 1, 10]
}
grid_search_svm = GridSearchCV(svm_pipeline, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
grid_search_svm.fit(X_train, y_text_train)
best_svm_model = grid_search_svm.best_estimator_

print(f"GridSearchCV training took: {time.time() - start_time:.2f} seconds")
joblib.dump(best_svm_model, 'svm_model.joblib')
print("Saved 'svm_model.joblib'")


# --- 6. Train and Save Model 3: LSTM Neural Network ---
print("\nTraining Model 3: LSTM (Advanced DL)...")
start_time = time.time()

VOCAB_SIZE = 10000
MAX_LENGTH = 50
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LENGTH, padding='post', truncating='post')

dl_model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LENGTH),
    SpatialDropout1D(0.2),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(3, activation='softmax') 
])
dl_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

EPOCHS = 4 
BATCH_SIZE = 64
history = dl_model.fit(
    X_train_pad, 
    y_numeric_train, 
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE,
    verbose=1
)
print(f"Deep Learning (LSTM) training took: {time.time() - start_time:.2f} seconds")

dl_model.save('dl_model.keras')
joblib.dump(tokenizer, 'tokenizer.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')
print("Saved DL model, Tokenizer, and Label Encoder.")

print("\n--- ALL MODELS TRAINED AND SAVED ---")