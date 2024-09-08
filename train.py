import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import time

start_time = time.time()

print("Loading dataset...")
url_data = pd.read_csv('malicious_phish.csv')
print(f"Dataset loaded in {time.time() - start_time:.2f} seconds.")

print("Cleaning and preprocessing dataset...")
url_data.dropna(inplace=True)
label_encoder = LabelEncoder()
url_data['type'] = label_encoder.fit_transform(url_data['type'])
print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds.")

print("Splitting dataset...")
X = url_data['url']
y = url_data['type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Dataset split in {time.time() - start_time:.2f} seconds.")

print("Extracting features...")
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"Feature extraction completed in {time.time() - start_time:.2f} seconds.")

print("Training model...")
model = RandomForestClassifier(n_jobs=-1)
model.fit(X_train_tfidf, y_train)
print(f"Model trained in {time.time() - start_time:.2f} seconds.")

print("Evaluating model...")
y_pred = model.predict(X_test_tfidf)
print(f"Evaluation completed in {time.time() - start_time:.2f} seconds.")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted')}")

print("Saving model and vectorizer...")

joblib.dump(model, 'url_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print(f"Model and vectorizer saved in {time.time() - start_time:.2f} seconds.")

total_time = time.time() - start_time
print(f"Total execution time: {total_time:.2f} seconds.")