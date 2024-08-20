from flask import Flask, request, jsonify, render_template
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
import re
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return ('Hello World! This is the home page of the app!')

# Build Query tool (not including mnist for now)

def query_database(query):
    try:
        conn = sqlite3.connect('/Users/lukewilsen/Desktop/IEX/IEX_Training/final_project/Database/final_project.db')
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
with open('data.pkl','rb') as f:
    housing_df = pickle.load(f)

housing_df['Log_saleprice'] = np.log(housing_df['SalePrice'])
housing_df = housing_df.iloc[:,list(range(0,7)) + [8]]

log_target = 'Log_saleprice'
log_feats = housing_df.columns[(housing_df.columns != log_target)]
# Now housing only has feature columns and log target column

X_log = housing_df[log_feats]

y_log = housing_df[log_target]

Xl_train, Xl_test,yl_train,yl_test = train_test_split(X_log,y_log,test_size=0.3,random_state=123)

ridge = Ridge(alpha=1.0)
ridge.fit(Xl_train,yl_train)

porter = PorterStemmer()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

### NLP
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

#NLP model is the lr_tfidf
#nlp_model = pickle.load(open('Models/nlp_model.pkl','rb'))
nlp_vect = pickle.load(open('Models/nlp_vect.pkl','rb'))


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


import re
def preprocessor(text):
    text = re.sub('<[^>]*>',"",text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    
    return text

text_clf = Pipeline([('vect', nlp_vect),
                     ('clf', LogisticRegression(solver='liblinear')),])
best_params = {'clf__C': 10.0, 'clf__penalty': 'l2', 'vect__ngram_range': (1, 1), 'vect__stop_words': None, 'vect__tokenizer': tokenizer_porter}
text_clf.set_params(**best_params)

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
        

@app.route('/predict_titanic', methods = ["POST"])
def predict_titanic():
    data = request.json
    df = pd.DataFrame([data])
    prediction = svc_pipe.predict(df)
    #survival_prob = svc_pipe.predict_proba(df)[0][1]
    return jsonify({'Survived': int(prediction)})

@app.route('/predict_housing', methods = ["POST"])
def predict_housing():
    data = request.json
    df = pd.DataFrame([data])
    prediction = ridge.predict(df)
    return jsonify({"Sale_Price": float(prediction)})

@app.route('/predict_sentiment', methods = ["POST"])
def predict_sentiment():
    text = request.json.get('text','')
    text_array = [text]
    pred = text_clf.predict(text_array)
    #sentiment = 'Negative' if pred[0] == 0 else "Positive"
    return {"pred":pred, "mod_stats":text_clf.get_params()}





if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug = True)