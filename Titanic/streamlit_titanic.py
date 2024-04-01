import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

# Need to remove unnecessary imports

import streamlit as st

train = pd.read_csv('titanic_train.csv')
test = pd.read_csv('titanic_test.csv')
gender_submission = pd.read_csv('gender_submission.csv')

df_data = pd.concat([train,test])

# title based age imputing

df_data["Title"] = df_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

df_data["Title"] = df_data["Title"].replace('Master', 'Master')
df_data["Title"] = df_data["Title"].replace('Mlle', 'Miss')
df_data["Title"] = df_data["Title"].replace(['Mme', 'Dona', 'Ms'], 'Mrs')
df_data["Title"] = df_data["Title"].replace(['Don','Jonkheer'],'Mr')
df_data["Title"] = df_data["Title"].replace(['Capt','Rev','Major', 'Col','Dr'], 'Military')
df_data["Title"] = df_data["Title"].replace(['Lady', 'Countess','Sir'], 'Honor')

train["Title"] = df_data['Title'][:891]
test["Title"] = df_data['Title'][891:]

# convert Title categories to Columns
titledummies=pd.get_dummies(train[['Title']], prefix_sep='_') #Title
train = pd.concat([train, titledummies], axis=1) 
ttitledummies=pd.get_dummies(test[['Title']], prefix_sep='_') #Title
test = pd.concat([test, ttitledummies], axis=1) 

#df_data[df_data['Title'] == "Master"]['Age'].describe()

titles = ['Master', 'Miss', 'Mr', 'Mrs', 'Military','Honor']
for title in titles:
    age_to_impute = df_data.groupby('Title')['Age'].median()[title]
    #print(f'Title:{title}, Median:{age_to_impute}')
    df_data.loc[(df_data['Age'].isnull()) & (df_data['Title'] == title), 'Age'] = age_to_impute
# Age in df_train and df_test:
train["Age"] = df_data['Age'][:891]
test["Age"] = df_data['Age'][891:]
train = pd.get_dummies(train, columns=['Pclass'], prefix=['class'])
test = pd.get_dummies(test, columns=['Pclass'], prefix=['class'])
train[['class_1', 'class_2', 'class_3']] = train[['class_1', 'class_2', 'class_3']].astype(int)
test[['class_1', 'class_2', 'class_3']] = test[['class_1', 'class_2', 'class_3']].astype(int)
train["Sex"] = pd.get_dummies(train[["Sex"]], drop_first = True)
train.rename(columns={'Sex': 'Male'}, inplace=True)
test["Sex"] = pd.get_dummies(test[["Sex"]], drop_first = True)
test.rename(columns={'Sex': 'Male'}, inplace=True)
train = train.drop(["Name", "Ticket", "Embarked", "Cabin"], axis = 1)
train = train.drop(['Title'], axis = 1)
title_list = ['Title_Honor','Title_Master','Title_Military','Title_Miss','Title_Mr','Title_Mrs']
train = train.drop(title_list, axis = 1)
test = test.drop(["Name", "Ticket", "Embarked", "Cabin"], axis = 1)
test = test.drop(['Title'], axis = 1)
test_title_list = ['Title_Master','Title_Military', 'Title_Miss', 'Title_Mr', 'Title_Mrs']
test = test.drop(test_title_list, axis = 1)
test['Fare'].fillna(value = round(test['Fare'].mean()), inplace = True)

x_train = train[['Male', 'Age', 'SibSp', 'Parch', 'Fare','class_1', 'class_2', 'class_3']]
y_train = train[['Survived']]
x_test = test[['Male', 'Age', 'SibSp', 'Parch', 'Fare','class_1', 'class_2', 'class_3']]
y_test = gender_submission[['Survived']]



model_accuracy_titanic_compare = {}

def roc_auc_func(mod, scaler, x_train_features, y_train_labels, x_test_features, y_test_labels):
    x_train_scaled = scaler.fit_transform(x_train_features)
    x_test_scaled = scaler.transform(x_test_features)
    
    y_train_labels = np.ravel(y_train_labels)
    mod.fit(x_train_scaled, y_train_labels)
    y_proba = mod.predict_proba(x_test_scaled)[:, 1]
    
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test_labels, y_proba)
    roc_auc = roc_auc_score(y_test_labels, y_proba)
    
    return fpr, tpr, roc_auc

def predict_func(mod, scaler,user_params, x_train_features = x_train, y_train_labels = y_train):
    x_train_scaled = scaler.fit_transform(x_train_features)
    y_train_labels = np.ravel(y_train_labels)
    mod.fit(x_train_scaled, y_train_labels)
    prediction = mod.predict(user_params)
    
    return prediction

standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
robust_scaler = RobustScaler()

def home():
    st.write("# Home Page")
    st.write("Welcome to the Home Page")
    st.write("""
    # Titanic ML Exploration Project

    Below is an overview of what I learned and how I learned it.
            
    ### Starting Dataset
            """)

    st.write(df_data.head())

    st.write("## Test and Train datasets!")
    st.write('### Test')
    st.write(test.head())
    st.write('### Train')
    st.write(train.head())

def exploration(y_train = y_train):
    st.write("# Exploration page")
    st.write("Here we'll talk about data exploration")
    st.write('## Feature Selection')
    importance_list = []
    y_train = np.ravel(y_train)
    feat_labels = x_train.columns
    forest = RandomForestClassifier(n_estimators = 500,
                                random_state = 1,
                                n_jobs = 2)
    forest.fit(x_train,y_train)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(x_train.shape[1]):
        rank = f + 1
        feature = feat_labels[indices[f]]
        importance = importances[indices[f]]
        importance_list.append({'Rank': rank, 'Feature': feature, 'Importance': importance})
    importance_df = pd.concat([pd.DataFrame([x]) for x in importance_list], ignore_index=True)
    st.write(importance_df)

def models():
    st.write("# Models page")
    st.write("Here we'll talk about a couple of models")
    st.write('''
    ## Model Creation

    The best models I created were:
    1. SVC (linear) - MinMaxScaler (100 % accuracy)
    2. LogisticRegression - MinMaxScaler (95.9% accuracy)
    3. SVC (rbf) - StandardScaler (94.3% accuracy)
            
    (I got a couple of other 100% models but it would be boring to show 5 different 100% models)
            ''')

    st.write('### Logistic Regression')
    lr_mod = LogisticRegression()
    lr_fpr,lr_tpr,lr_roc_auc = roc_auc_func(lr_mod, minmax_scaler, x_train, y_train, x_test, y_test)

    # Plot ROC curve
    st.write("#### LR ROC Curve")
    st.write(f"ROC AUC: {lr_roc_auc:.2f}")

    fig, ax = plt.subplots()
    ax.plot(lr_fpr, lr_tpr, label=f"ROC Curve (AUC = {lr_roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend()
    st.pyplot(fig)

def prediction():
    st.write('# Prediction page')
    st.write("Would you have survived on the titanic?")
    st.sidebar.header('User Input Params')


    st.subheader('User Input Parameters')

    def user_input_features():
        Male = st.sidebar.slider('Male (0 = No, 1 = Yes)', 0, 1, 0)
        Age = st.sidebar.slider('Age',0,100,20)
        SibSp = st.sidebar.slider('Number of Siblings and/or Spouses', 0,20,0)
        Parch = st.sidebar.slider('Number of Parents and/or Children', 0,20,0)
        Fare = st.sidebar.slider('Fare',0,513,20)
        class_1 = st.sidebar.slider('Class 1',0,1,1)
        class_2 = st.sidebar.slider('Class 2',0,1,0)
        class_3 = st.sidebar.slider('Class 3',0,1,0)
        data ={"Male":Male,
            "Age": Age,
            "SibSp": SibSp,
            "Parch": Parch,
            "Fare": Fare,
            "class_1":class_1,
            "class_2":class_2,
            "class_3":class_3}
        features = pd.DataFrame(data,index = [0])
        return features
    df = user_input_features()
    st.write(df)
    st.write(predict_func(LogisticRegression(),user_params=df))
    



pages = {'home':home,
         'exploration':exploration,
         'models':models,
         'prediction':prediction}

selected_page = st.sidebar.selectbox('Navigation', list(pages.keys()))
pages[selected_page]()
