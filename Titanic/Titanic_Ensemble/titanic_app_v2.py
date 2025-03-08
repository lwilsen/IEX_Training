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

st.write("""# Titanic App Version 2 (Ensemble Methods)""")

st.write(
    """Here I'm exploring how to use ensemble methods within the 
         framework of my original titanic dataset exploration,
         as well as implementing these methods in a streamlit app."""
)

train = pd.read_csv("titanic_train.csv")
test = pd.read_csv("titanic_test.csv")
gender_submission = pd.read_csv("gender_submission.csv")

df_data = pd.concat([train, test])

df_data["Title"] = df_data.Name.str.extract(" ([A-Za-z]+)\.", expand=False)

df_data["Title"] = df_data["Title"].replace("Master", "Master")
df_data["Title"] = df_data["Title"].replace("Mlle", "Miss")
df_data["Title"] = df_data["Title"].replace(["Mme", "Dona", "Ms"], "Mrs")
df_data["Title"] = df_data["Title"].replace(["Don", "Jonkheer"], "Mr")
df_data["Title"] = df_data["Title"].replace(
    ["Capt", "Rev", "Major", "Col", "Dr"], "Military"
)
df_data["Title"] = df_data["Title"].replace(["Lady", "Countess", "Sir"], "Honor")

train["Title"] = df_data["Title"][:891]
test["Title"] = df_data["Title"][891:]

titledummies = pd.get_dummies(train[["Title"]], prefix_sep="_")  # Title
train = pd.concat([train, titledummies], axis=1)
ttitledummies = pd.get_dummies(test[["Title"]], prefix_sep="_")  # Title
test = pd.concat([test, ttitledummies], axis=1)


titles = ["Master", "Miss", "Mr", "Mrs", "Military", "Honor"]
for title in titles:
    age_to_impute = df_data.groupby("Title")["Age"].median()[title]
    df_data.loc[(df_data["Age"].isnull()) & (df_data["Title"] == title), "Age"] = (
        age_to_impute
    )
train["Age"] = df_data["Age"][:891]
test["Age"] = df_data["Age"][891:]
train = pd.get_dummies(train, columns=["Pclass"], prefix=["class"])
test = pd.get_dummies(test, columns=["Pclass"], prefix=["class"])
train[["class_1", "class_2", "class_3"]] = train[
    ["class_1", "class_2", "class_3"]
].astype(int)
test[["class_1", "class_2", "class_3"]] = test[
    ["class_1", "class_2", "class_3"]
].astype(int)
train["Sex"] = pd.get_dummies(train[["Sex"]], drop_first=True)
train.rename(columns={"Sex": "Male"}, inplace=True)
test["Sex"] = pd.get_dummies(test[["Sex"]], drop_first=True)
test.rename(columns={"Sex": "Male"}, inplace=True)
train = train.drop(["Name", "Ticket", "Embarked", "Cabin"], axis=1)
train = train.drop(["Title"], axis=1)
title_list = [
    "Title_Honor",
    "Title_Master",
    "Title_Military",
    "Title_Miss",
    "Title_Mr",
    "Title_Mrs",
]
train = train.drop(title_list, axis=1)
test = test.drop(["Name", "Ticket", "Embarked", "Cabin"], axis=1)
test = test.drop(["Title"], axis=1)
test_title_list = [
    "Title_Master",
    "Title_Military",
    "Title_Miss",
    "Title_Mr",
    "Title_Mrs",
]
test = test.drop(test_title_list, axis=1)
test["Fare"].fillna(value=round(test["Fare"].mean()), inplace=True)

with open("train.pkl", "wb") as f:
    pickle.dump(train, f)
with open("test.pkl", "wb") as f:
    pickle.dump(test, f)
