# --- 0. Import Libraries ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np

# ML Libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# DL Libraries (TensorFlow / Keras)
try:
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
    from tensorflow.keras.utils import to_categorical
except ImportError:
    print("="*50)
    print("ERROR: TensorFlow library not found.")
    print("Please install it by running: pip install tensorflow")
    print("="*50)
    exit()

# --- 1. Load the Dataset ---
print("Loading dataset 'Tweets.csv'...")
file_name = "Tweets.csv"
df = pd.read_csv(file_name)

# --- 2. Define Features (X) and Target (y) ---
X = df['text']
y_text = df['airline_sentiment'] # y_text = 'negative', 'neutral', 'positive'

# --- 3. Preprocessing for DL Model ---
# The DL model needs numerical labels (0, 1, 2)
label_encoder = LabelEncoder()
y_numeric = label_encoder.fit_transform(y_text) # y_numeric = 0, 1, 2
class_labels = label_encoder.classes_ # This stores ['negative', 'neutral', 'positive']
print(f"Data loaded: {len(X)} tweets.")

# --- 4. Split the Data ---
print("\nSplitting data into 80% train and 20% test...")
# We split all data. ML models will use y_text_train, DL will use y_numeric_train
X_train, X_test, y_text_train, y_text_test, y_numeric_train, y_numeric_test = train_test_split(
    X, y_text, y_numeric, test_size=0.2, random_state=42, stratify=y_text
)

# --- 5. Train Model 1: Naive Bayes ---
print("\n--- Training Model 1: Naive Bayes (Baseline ML) ---")
start_time = time.time()
nb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')), 
    ('clf', MultinomialNB())                          
])
nb_pipeline.fit(X_train, y_text_train)
print(f"Naive Bayes training took: {time.time() - start_time:.2f} seconds")
y_pred_nb = nb_pipeline.predict(X_test)


# --- 6. Train Model 2: Tuned Linear SVM ---
print("\n--- Training Model 2: Tuned Linear SVM (Advanced ML) ---")
print("(This may take 5-10 minutes with GridSearchCV...)")
start_time = time.time()

svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LinearSVC(random_state=42, dual=False))
])
param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],  # Test 1-word vs 1- and 2-word pairs
    'clf__C': [0.1, 1, 10]                   # Test different regularization strengths
}
grid_search_svm = GridSearchCV(svm_pipeline, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1) # cv=3 for speed
grid_search_svm.fit(X_train, y_text_train)

print(f"GridSearchCV training took: {time.time() - start_time:.2f} seconds")
print(f"Best SVM Hyperparameters: {grid_search_svm.best_params_}")
best_svm_model = grid_search_svm.best_estimator_
y_pred_svm = best_svm_model.predict(X_test)


# --- 7. Train Model 3: LSTM Neural Network ---
print("\n--- Training Model 3: LSTM (Advanced DL) ---")
print("(This may take 5-15 minutes...)")
start_time = time.time()

# 7.1 Tokenize and Pad Text Data
VOCAB_SIZE = 10000  # Max number of words to keep
MAX_LENGTH = 50    # Max length of a tweet
EMBEDDING_DIM = 100 # Dimension for word vectors

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LENGTH, padding='post', truncating='post')

# 7.2 Build the Keras LSTM Model
dl_model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LENGTH),
    SpatialDropout1D(0.2), # Dropout layer for embeddings
    LSTM(64, dropout=0.2, recurrent_dropout=0.2), # LSTM layer
    Dense(3, activation='softmax') # 3 outputs (neg, neu, pos), softmax for probabilities
])

# 7.3 Compile the Model
# We use 'sparse_categorical_crossentropy' because our y_numeric is NOT one-hot encoded
dl_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("\nDL Model Summary:")
dl_model.summary()

# 7.4 Train the Model
EPOCHS = 4 # 4-5 epochs is often enough for a good start
BATCH_SIZE = 64
history = dl_model.fit(
    X_train_pad, 
    y_numeric_train, 
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE,
    validation_data=(X_test_pad, y_numeric_test),
    verbose=1
)

print(f"Deep Learning (LSTM) training took: {time.time() - start_time:.2f} seconds")

# 7.5 Get DL Predictions
# The model outputs probabilities for each class (e.g., [0.1, 0.2, 0.7])
y_pred_dl_probs = dl_model.predict(X_test_pad)
# We use np.argmax to get the index of the highest probability (e.g., 2)
y_pred_dl_numeric = np.argmax(y_pred_dl_probs, axis=1)

# *** IMPORTANT ***
# Convert the DL model's numeric predictions (0, 1, 2) back to text labels
# so we can compare it fairly with the other models
y_pred_dl_text = label_encoder.inverse_transform(y_pred_dl_numeric)


# --- 8. Final Evaluation & Comparison ---
print("\n" + "="*50)
print("--- FINAL MODEL COMPARISON RESULTS ---")
print("="*50)

# --- Naive Bayes Metrics ---
print("\n--- Model 1: Naive Bayes ---")
accuracy_nb = accuracy_score(y_text_test, y_pred_nb)
f1_nb = f1_score(y_text_test, y_pred_nb, average='weighted')
print(f"  Accuracy: {accuracy_nb:.4f}")
print(f"  Weighted F1-Score: {f1_nb:.4f}")
print(classification_report(y_text_test, y_pred_nb, labels=class_labels))

# --- Tuned Linear SVM Metrics ---
print("\n--- Model 2: Tuned Linear SVM ---")
accuracy_svm = accuracy_score(y_text_test, y_pred_svm)
f1_svm = f1_score(y_text_test, y_pred_svm, average='weighted')
print(f"  Accuracy: {accuracy_svm:.4f}")
print(f"  Weighted F1-Score: {f1_svm:.4f}")
print(classification_report(y_text_test, y_pred_svm, labels=class_labels))

# --- LSTM DL Model Metrics ---
print("\n--- Model 3: LSTM Neural Network ---")
accuracy_dl = accuracy_score(y_text_test, y_pred_dl_text)
f1_dl = f1_score(y_text_test, y_pred_dl_text, average='weighted')
print(f"  Accuracy: {accuracy_dl:.4f}")
print(f"  Weighted F1-Score: {f1_dl:.4f}")
print(classification_report(y_text_test, y_pred_dl_text, labels=class_labels))

# --- 9. Create Comparison Table ---
results_data = {
    'Model': ['Naive Bayes (ML)', 'Tuned Linear SVM (ML)', 'LSTM (DL)'],
    'Accuracy': [accuracy_nb, accuracy_svm, accuracy_dl],
    'Weighted F1-Score': [f1_nb, f1_svm, f1_dl]
}
results_df = pd.DataFrame(results_data)

print("\n\n--- Final Comparison Table ---")
print(results_df.to_markdown(index=False, floatfmt=".4f"))

# --- 10. Plot Confusion Matrices ---
print("\nGenerating confusion matrix plots...")

# Create a figure with three subplots (1 row, 3 columns)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
fig.suptitle('Model Confusion Matrices', fontsize=20)

# Plot 1: Naive Bayes
cm_nb = confusion_matrix(y_text_test, y_pred_nb, labels=class_labels)
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', ax=ax1, 
            xticklabels=class_labels, yticklabels=class_labels)
ax1.set_title('Naive Bayes (ML)')
ax1.set_xlabel('Predicted Label')
ax1.set_ylabel('True Label')

# Plot 2: Tuned Linear SVM
cm_svm = confusion_matrix(y_text_test, y_pred_svm, labels=class_labels)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens', ax=ax2,
            xticklabels=class_labels, yticklabels=class_labels)
ax2.set_title('Tuned Linear SVM (ML)')
ax2.set_xlabel('Predicted Label')
ax2.set_ylabel('True Label')

# Plot 3: LSTM Deep Learning
cm_dl = confusion_matrix(y_text_test, y_pred_dl_text, labels=class_labels)
sns.heatmap(cm_dl, annot=True, fmt='d', cmap='Oranges', ax=ax3,
            xticklabels=class_labels, yticklabels=class_labels)
ax3.set_title('LSTM (DL)')
ax3.set_xlabel('Predicted Label')
ax3.set_ylabel('True Label')

# Save the plot to a file
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('all_confusion_matrices_plot.png')
print("Saved 'all_confusion_matrices_plot.png' to your folder.")

# Show the plot
print("Displaying plots...")
plt.show()

print("\n--- Project Complete ---")