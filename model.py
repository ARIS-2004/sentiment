import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

nltk.download('stopwords')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text) 
    text = re.sub(r'[^a-z\s]', '', text)  
    tokens = text.split()
    stops = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stops]
    return ' '.join(tokens)

def train_and_save_model(csv_path='tweets.csv'):
    
    df = pd.read_csv(csv_path)
    df['clean_text'] = df['text'].apply(clean_text)

    X = df['clean_text']
    y = df['airline_sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(model, 'sentiment_model.joblib')
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    print("Model and vectorizer saved!")

if __name__ == "__main__":
    train_and_save_model()
