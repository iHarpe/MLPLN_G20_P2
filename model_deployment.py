#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    stopwords_es = list(stopwords.words('spanish'))
    stopwords_en = list(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stopwords_es = list(stopwords.words('spanish'))
    stopwords_en = list(stopwords.words('english'))

def load_model():
    model_path = os.path.dirname(__file__) + '/TF-IDF.pkl'
    vectorizer_path = os.path.dirname(__file__) + '/vector_tfidf.pkl'
    mlb_path = os.path.dirname(__file__) + '/mlb.pkl'

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    mlb = joblib.load(mlb_path)
    return model, vectorizer, mlb

def predict_genres(text, title="", top_n=3):
    model, vectorizer, mlb = load_model()
    texto_completo = title + " " + text if title else text
    df = pd.DataFrame({'texto_completo': [texto_completo]})
    X = vectorizer.transform(df['texto_completo'])
    y_pred_proba = model.predict_proba(X)
    top_indices = y_pred_proba[0].argsort()[-top_n:][::-1]
    top_probs = y_pred_proba[0][top_indices]
    result = []
    for idx, prob in zip(top_indices, top_probs):
        genre = mlb.classes_[idx]
        result.append({"genre": genre, "probability": float(prob)})

    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Please add a movie plot text')
    else:
        text = sys.argv[1]
        title = sys.argv[2] if len(sys.argv) > 2 else ""
        top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 3

        genres = predict_genres(text, title, top_n)

        print('Predicted genres:')
        for genre in genres:
            print(f"{genre['genre']}: {genre['probability']:.4f}")