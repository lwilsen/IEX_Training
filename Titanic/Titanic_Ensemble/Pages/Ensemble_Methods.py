import pandas as pd
import streamlit as st
import plotly.express as px
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    BaggingClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
    VotingClassifier,
)

with open("test.pkl", "rb") as f:
    test = pickle.load(f)

with open("train.pkl", "rb") as f:
    train = pickle.load(f)

gender_submission = pd.read_csv("gender_submission.csv")

x_train = train[
    ["Male", "Age", "SibSp", "Parch", "Fare", "class_1", "class_2", "class_3"]
]
y_train = train[["Survived"]]

x_test = test[
    ["Male", "Age", "SibSp", "Parch", "Fare", "class_1", "class_2", "class_3"]
]
y_test = gender_submission[["Survived"]]

# define the scaler
scaler = MinMaxScaler()

# define models
lr_mod = LogisticRegression()
svc_model = SVC()
rf_model = RandomForestClassifier()
knn_mod = KNeighborsClassifier(n_neighbors=5)

models = {}

# pipelines
pipeline_lr = Pipeline([("scaler", scaler), ("lr", lr_mod)])

pipeline_svc = Pipeline([("scaler", scaler), ("svc", svc_model)])

pipeline_rf = Pipeline([("scaler", scaler), ("rfc", rf_model)])

pipeline_knn = Pipeline([("scaler", scaler), ("knn", knn_mod)])

st.write(f"I used the following models, with the following base accuracies:")
st.write("---")

# Fit pipelines
pipeline_lr.fit(x_train, y_train)
y_pred_lr = pipeline_lr.predict(x_test)
pipeline_lr_acc = accuracy_score(y_pred_lr, y_test)
st.write(f"Logistic Regression Accuracy = {round(pipeline_lr_acc, 2)}")
models[pipeline_lr.__class__.__name__] = round(pipeline_lr_acc, 3)

pipeline_svc.fit(x_train, y_train)
y_pred_svc = pipeline_svc.predict(x_test)
pipeline_svc_acc = accuracy_score(y_pred_svc, y_test)
st.write(f"Support Vector Machine Accuracy = {round(pipeline_svc_acc, 2)}")
models[pipeline_svc.__class__.__name__] = round(pipeline_svc_acc, 3)

pipeline_rf.fit(x_train, y_train)
y_pred_rf = pipeline_rf.predict(x_test)
pipeline_rf_acc = accuracy_score(y_pred_rf, y_test)
st.write(f"Random Forest Accuracy = {round(pipeline_rf_acc, 2)}")
models[pipeline_rf.__class__.__name__] = round(pipeline_rf_acc, 3)

pipeline_knn.fit(x_train, y_train)
y_pred_knn = pipeline_knn.predict(x_test)
pipeline_knn_acc = accuracy_score(y_pred_knn, y_test)
st.write(f"K-Nearest Neighbors Accuracy = {round(pipeline_knn_acc, 2)}")
models[pipeline_knn.__class__.__name__] = round(pipeline_knn_acc, 3)
st.write("---")

st.write("Then I used the following ensemble methods and got the following accuracies:")
st.write("---")

st.write("## Bagging")
bagging_model = BaggingClassifier(
    estimator=pipeline_lr, n_estimators=10, random_state=42
)
bagging_model.fit(x_train, y_train)
y_pred = bagging_model.predict(x_test)
bag_accuracy = accuracy_score(y_test, y_pred)
st.write(f"Bagging (w/ Logistic Regression) Model: {round(bag_accuracy,2)}")
models[bagging_model.__class__.__name__] = round(bag_accuracy,3)

# Make all headers bold

st.write(
    """## Stacking: 
         
Base models: SVC, RF, KNN
         
Final Model: Logistic Regression"""
)

level1_models = [("svc", svc_model), ("rf", rf_model), ("knn", knn_mod)]
# Define the final estimator
final_estimator = lr_mod

stacking_model = StackingClassifier(
    estimators=level1_models, final_estimator=final_estimator, cv=5
)
stacking_model.fit(x_train, y_train)
y_pred = stacking_model.predict(x_test)
stack_accuracy = accuracy_score(y_test, y_pred)

st.write(f"Stacking Model Accuracy: {stack_accuracy:.2f}")
models[stacking_model.__class__.__name__] = round(stack_accuracy, 3)

st.write("## Boosting")
boosting_model = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, random_state=42
)
boosting_model.fit(x_train, y_train)
y_pred = boosting_model.predict(x_test)
boost_accuracy = accuracy_score(y_test, y_pred)

st.write(f"Boosting Model Accuracy: {boost_accuracy:.2f}")
models[boosting_model.__class__.__name__] = round(boost_accuracy,3)

st.write("## Majority Voting")
st.write("""Base models: SVC, RF, KNN""")

voting_model = VotingClassifier(
    estimators=level1_models, voting="hard"
)  # Hard voting for classification - SOFT is regression
voting_model.fit(x_train, y_train)
y_pred = voting_model.predict(x_test)
maj_accuracy = accuracy_score(y_test, y_pred)

st.write(f"Majority Voting Model Accuracy: {maj_accuracy:.2f}")
models[voting_model.__class__.__name__] = round(maj_accuracy,3)


st.write(
    f"**The overall best model was the: {max(models)} model with an accuracy of: {models[max(models)]}**"
)
