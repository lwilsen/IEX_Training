import streamlit as st
import pandas as pd

st.header('Model Training and Evaluation')
st.write('---')
st.write('## Logistic Regression')
st.write('''The first model we're going to use for our NLP is a logistic Regression Classifier, 
         that is based on a bag of words model, that represents how common each stemmed word is in our dataset.
         The parameters for this model are as follows: 
         
         'clf__C': 10.0, 'clf__penalty': 'l2', 'vect__ngram_range': (1, 1), 'vect__stop_words': None, 'vect__tokenizer': tokenizer_porter''')
st.write('''**The accuracy for this model was 89.2% on the training data, and 89.3% on the testing dataset.**
''')
st.write('---')
st.write('## Out-of-Core Learning with a Stochastic Gradient Descent Classifier')
st.write('''The previous model took a very long time to run, mostly due to the grid search performed, and 
         one solution to this is to incrementally fit the classifier to smaller batches of the dataset. This method
         uses stochastic gradient descent to incrementally optimize the loss function we choose (log loss in this case).
         **The accuracy for this model was 86.6% on the test data.**''')

df = pd.read_csv('movie_data.csv', encoding='utf-8')
