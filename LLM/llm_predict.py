import cohere
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pyprind
import sys

cohere_api_key = os.getenv("CO_API_KEY")
co = cohere.Client(api_key=cohere_api_key)

reviews = pd.read_csv('LLM/movie_data.csv', encoding='utf-8')

reviews_sample = reviews.sample(n=100,random_state=123)

X_train, X_test, y_train, y_test = train_test_split(reviews_sample['review'], reviews_sample['sentiment'], test_size=0.2, random_state=123)

def get_predictions(reviews):
    pbar = pyprind.ProgBar(len(reviews), stream = sys.stdout)
    predictions = []
    errors = 0
    for review in reviews:
        prompt = f"Your job is to analyze the sentiment of the followng text and determine whether it is positive or negative. :\n{review}"
        response = co.chat(
            model="command-r-plus",
            message=prompt
        )
        prediction = str(response.text).strip().lower()
        if "positive" in prediction:
            predictions.append(1)
        elif "negative" in prediction:
            predictions.append(0)
        else:
            predictions.append(-1)
            print('ERROR: -1 appended') #unknown errors
            errors += 1
            print(prediction)
        pbar.update()
    return [predictions, errors]

X_test_list = X_test.tolist()

output = get_predictions(X_test_list)
llm_predictions = output[0]
errors = output[1]

if errors == 0:
    llm_accuracy = accuracy_score(y_test, llm_predictions)
elif errors > 0:
    valid_indices = [i for i, pred in enumerate(llm_predictions) if pred != -1]
    y_test_filtered = y_test.iloc[valid_indices]
    y_pred_llm_filtered = [llm_predictions[i] for i in valid_indices]
    llm_accuracy = accuracy_score(y_test_filtered, y_pred_llm_filtered)

print(f'LLM Accuracy: {llm_accuracy}')