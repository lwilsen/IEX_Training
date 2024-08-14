from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import tensorflow as tf
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk
import sqlite3
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from log_transformer import LogTransformer


app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return ('Hello World!')

# Build Query tool (not including mnist for now)

def query_tool(query):
    try:
        conn = sqlite3.connect('/final_project/Database/final_project.db')
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [description[0] for description in cursor.description]
        data = cursor.fetchall()
    except sqlite3.Error as e:
        print(e)
    finally:
        conn.close()
    return {"Columns":columns, "Data":data}

# Prediction Functions/models

## Import Models

### Best for Titanic
svc_pipe = pickle.load(open('Models/svc_pipeline.pkl', 'rb'))

### Best for LR on Housing
ridge_regression = pickle.load(open('Models/ridge_pipe.pkl','rb'))

### NLP
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

nlp_model = pickle.load(open('Models/NLP_model.pkl','rb'))

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def nlp_preprocessing(text):
    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    processed_tokens = []
    for word, tag in pos_tags:
        lemmatized = lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        stemmed_word = stemmer.stem(lemmatized)
        processed_tokens.append(stemmed_word)
    joined_tokens = ' '.join(processed_tokens)
    return joined_tokens

### MNIST model

mnist_model = tf.keras.models.load_model('Models/mnist_model.keras')

# Functions that recieve POST requests

@app.route('/query', methods = ['GET','POST'])
def query():
    if request.method == 'GET':
        return "THIS WILL BE WHERE QUERIES ARE SENT"
    elif request.method == 'POST':
        try:
            query = request.json.get('query')
            if not query:
                return jsonify({'error':'No Query Provided'}), 400
            data = query_database(query)
            return jsonify(data)
        except Exception as e:
            return jsonify({'error':str(e)}), 500

if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug = True)