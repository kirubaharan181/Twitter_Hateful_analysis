import joblib
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import getpass
from transformers import BertTokenizer, TFBertForSequenceClassification
from datetime import datetime
from flask import Flask, request, jsonify, render_template
import webbrowser
import threading
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

app = Flask(__name__)

class HateSpeechPredictor:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.load_models()

    def load_models(self):
        print(f"Loading models from {self.model_dir}...")
        
        print("Loading TF-IDF vectorizer...")
        try:
            self.tfidf = joblib.load(os.path.join(self.model_dir, 'tfidf_vectorizer.joblib'))
            print("TF-IDF vectorizer loaded successfully")
        except Exception as e:
            print(f"Error loading TF-IDF vectorizer: {str(e)}")
            self.tfidf = None
        
        self.models = {}
        print("Loading classification models...")
        model_names = ['SVM', 'Naive Bayes', 'Logistic Regression', 
                      'Random Forest', 'XGBoost', 'k-NN', 'ANN']
        
        for name in model_names:
            try:
                self.models[name] = joblib.load(os.path.join(self.model_dir, f'{name}_model.joblib'))
                print(f"Loaded {name} model successfully")
            except Exception as e:
                print(f"Warning: Could not load {name} model: {str(e)}")

        print("\nLoading LSTM model...")
        self.lstm_model = None
        self.tokenizer = None
        try:
            tokenizer_path = os.path.join(self.model_dir, 'tokenizer.pkl')
            keras_path = os.path.join(self.model_dir, 'lstm_hate_speech_model.keras')
            
            if not os.path.exists(tokenizer_path):
                print(f"Error: Tokenizer file not found at {tokenizer_path}")
            elif not os.path.exists(keras_path):
                print(f"Error: LSTM model file not found at {keras_path}")
            else:
                with open(tokenizer_path, 'rb') as handle:
                    self.tokenizer = pickle.load(handle)
                    print("Tokenizer loaded successfully")
                
                print(f"Attempting to load {keras_path}...")
                self.lstm_model = tf.keras.models.load_model(keras_path, compile=True)
                print("Successfully loaded LSTM model (.keras format)")
        except Exception as e:
            print(f"Error loading LSTM model: {str(e)}")
            self.lstm_model = None
            self.tokenizer = None

        print("\nLoading BERT model and tokenizer...")
        self.bert_model = None
        self.tokenizer_bert = None
        try:
            bert_model_path = os.path.join(self.model_dir, 'bert_hate_speech_model')
            if not os.path.exists(bert_model_path):
                print(f"Error: BERT model directory not found at {bert_model_path}")
            else:
                self.bert_model = TFBertForSequenceClassification.from_pretrained(bert_model_path)
                self.tokenizer_bert = BertTokenizer.from_pretrained(bert_model_path)
                print("BERT model and tokenizer loaded successfully")
        except Exception as e:
            print(f"Error loading BERT model: {str(e)}")
            self.bert_model = None
            self.tokenizer_bert = None

        print("\nModel loading completed!")

    def preprocess_text(self, text):
        text = re.sub(r'[^A-Za-z0-9\s]', '', str(text))
        text = text.lower()
        stop_words = set(stopwords.words('english'))
        words = text.split()
        text = ' '.join([word for word in words if word not in stop_words])
        lemmatizer = WordNetLemmatizer()
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
        return text

    def predict(self, text):
        cleaned_text = self.preprocess_text(text)
        results = {}
        
        if self.tfidf is not None:
            try:
                X_input_tfidf = self.tfidf.transform([cleaned_text])
                for name, model in self.models.items():
                    pred = model.predict(X_input_tfidf)[0]
                    results[name] = "Hate Speech" if pred == 0 else "Non-Hate Speech"
            except Exception as e:
                print(f"Error in traditional models prediction: {str(e)}")
        
        if self.lstm_model is not None and self.tokenizer is not None:
            try:
                sequence = self.tokenizer.texts_to_sequences([cleaned_text])
                padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=100, padding='post')
                lstm_prediction = self.lstm_model.predict(padded_sequence, verbose=0)[0][0]
                results['LSTM'] = "Hate Speech" if lstm_prediction < 0.5 else "Non-Hate Speech"
            except Exception as e:
                print(f"Error in LSTM prediction: {str(e)}")
        
        if self.bert_model is not None and self.tokenizer_bert is not None:
            try:
                inputs = self.tokenizer_bert(
                    cleaned_text,
                    truncation=True,
                    padding=True,
                    max_length=100,
                    return_tensors='tf'
                )
                outputs = self.bert_model(inputs)
                bert_prediction = tf.nn.softmax(outputs.logits, axis=1)
                pred_class = tf.argmax(bert_prediction, axis=1).numpy()[0]
                results['BERT'] = "Hate Speech" if pred_class == 0 else "Non-Hate Speech"
            except Exception as e:
                print(f"Error in BERT prediction: {str(e)}")
        
        return results

model_dir = "saved_models_2025-03-21_00-55-36"
predictor = HateSpeechPredictor(model_dir)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    predictions = predictor.predict(text)
    return jsonify({'predictions': predictions})

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    threading.Timer(1, open_browser).start()
    app.run(debug=False)