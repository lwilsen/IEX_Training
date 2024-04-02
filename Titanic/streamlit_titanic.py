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
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay, precision_recall_curve, auc, PrecisionRecallDisplay

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

def scale_fit(mod, scaler, x_train, y_train):
    x_train_scaled = scaler.fit_transform(x_train)
    y_train_labels = np.ravel(y_train)
    
    return mod.fit(x_train_scaled, y_train_labels)

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
    st.write("""
    # Titanic ML Exploration Project
""")
    
    st.image("/Users/lukewilsen/Desktop/IEX/IEX_Training/Titanic/titanic.jpeg",caption = "Titanic", use_column_width= True)

    st.write("""
    This project is an exploration of the Titanic kaggle dataset and my attempt to model the survival outcome based on certain variables in the dataset.
            
    ### Original Dataset
            """)

    st.write(df_data)

    st.write("### Test and Train datasets!")
    st.write('#### Test')
    st.write(test)
    st.write('#### Train')
    st.write(train)

def exploration(y_train = y_train, x_train = x_train, train = train):
    st.write("# Exploration page")
    st.write('## Feature Selection')
    st.write("This feature selection was important to my analysis because it helped me to determine which variables I needed to pay the most attention to")
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
    st.write("## Exploring Important Variables")
    st.write("### Age")
    fig, ax = plt.subplots()
    ax.hist(x_train['Age'], bins=30, color='skyblue', edgecolor='black')
    ax.set_xlabel('Age')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Age')
    st.pyplot(fig)
    st.write("### Fare")
    fig, ax = plt.subplots()
    ax.hist(x_train['Fare'], bins=30, color='skyblue', edgecolor='black')
    ax.set_xlabel('Fare')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Fare')
    st.pyplot(fig)
    st.write("### 'Zoomed in' Fare <200)")
    fig, ax = plt.subplots()
    ax.hist(x_train['Fare'], bins=30, color='skyblue', edgecolor='black')
    ax.set_xlabel('Fare')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Fare')
    ax.set_xlim(0, 200)
    for i in range(0,500,25):
        ax.axhline(i, color='gray', linestyle='--', linewidth=0.5)
    st.pyplot(fig)
    st.write("### 'Zoomed in' Fare (>200)")
    fig, ax = plt.subplots()
    ax.hist(x_train['Fare'], bins=30, color='skyblue', edgecolor='black')
    ax.set_xlabel('Fare')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Fare')
    ax.set_xlim(200, 520)
    ax.set_ylim(0,10)
    for i in range(0,10):
        ax.axhline(i, color='gray', linestyle='--', linewidth=0.5)
    st.pyplot(fig)
    st.write('''
    ## Correlation
             
    ### Heatmap
    ''')

    corr_train = train.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_train, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    st.pyplot(fig)

    st.write('''
    Here you can see that there is a strong correlation between, Sex, Fare and Survived. 
    
    An interesting note: Age isn't highly correlated with survival, but our feature selection indicated that it was. 
        ''')
    

def models():
    st.write("# Models page")
    st.write("Here we'll talk about a couple of models")
    st.write('''
    ## Model Creation

    The best models I created were:
    1. SVC (linear) - MinMaxScaler (100 % accuracy)
    2. LogisticRegression - MinMaxScaler (95.9% accuracy)
    3. SVC (rbf) - StandardScaler (94.3% accuracy)
            
             
    ### SVC with Linear Kernel
            ''')
    
    svm_mod = SVC(kernel='linear', C=1)
    svm_mod = scale_fit(svm_mod, standard_scaler, x_train,y_train)
    x_test_standard = standard_scaler.fit_transform(x_test)
    svm_ROC_disp = RocCurveDisplay.from_estimator(svm_mod,x_test_standard,y_test)
    svm_PR_disp = PrecisionRecallDisplay.from_estimator(svm_mod,x_test_standard,y_test)

    fig, ax = plt.subplots(figsize = (12,8))
    svm_ROC_disp.plot(ax = ax)
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend()
    st.pyplot(fig)

    st.write('#### PR Curve')

    fig, ax = plt.subplots(figsize = (12,8))
    svm_PR_disp.plot(ax=ax)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision - Recall Curve")
    ax.legend()
    st.pyplot(fig)

    st.write('### Logistic Regression')

    lr_mod = LogisticRegression()
    lr_mod = scale_fit(lr_mod,standard_scaler, x_train, y_train)
    x_test_minmax = minmax_scaler.fit_transform(x_test)
    lr_ROC_disp = RocCurveDisplay.from_estimator(lr_mod,x_test_minmax,y_test)
    lr_PR_disp = PrecisionRecallDisplay.from_estimator(lr_mod,x_test_minmax,y_test)

    st.write("#### LR ROC Curve")

    fig, ax = plt.subplots(figsize = (12,8))
    lr_ROC_disp.plot(ax = ax)
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend()
    st.pyplot(fig)

    st.write('#### PR Curve')

    fig, ax = plt.subplots(figsize = (12,8))
    lr_PR_disp.plot(ax=ax)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision - Recall Curve")
    ax.legend()
    st.pyplot(fig)
    
    st.write('### SVC with RBF (Radial Basis Function) Kernel')

    svc_rbf_mod = SVC(kernel = "rbf", C = 1, random_state = 1, gamma = 0.1)
    svc_rbf_mod = scale_fit(svc_rbf_mod,standard_scaler,x_train,y_train)
    svc_ROC_disp = RocCurveDisplay.from_estimator(svc_rbf_mod, x_test_standard,y_test)
    svc_PR_disp = PrecisionRecallDisplay.from_estimator(svc_rbf_mod,x_test_standard,y_test)

    st.write("#### SVC ROC Curve")

    fig, ax = plt.subplots(figsize = (12,8))
    svc_ROC_disp.plot(ax = ax)
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend()
    st.pyplot(fig)

    st.write('#### PR Curve')

    fig,ax = plt.subplots(figsize = (12,8))
    svc_PR_disp.plot(ax = ax)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision - Recall Curve")
    st.pyplot(fig)



    
    return prediction

def prediction():
    st.write('# Prediction page')
    st.write("Would you have survived on the titanic?")
    st.sidebar.header('User Input Params')
    st.subheader('User Input Parameters')

    def user_input_features():
        Male = st.sidebar.slider('Male (0 = No, 1 = Yes)', 0, 1, 0)
        Age = st.sidebar.text_input('Enter your Age:','20')
        Sib = st.sidebar.text_input('Number of Siblings', 0,15,2)
        Sp = st.sidebar.slider('Spouse (0 = No, 1 = Yes)',0,1,0)
        SibSp = int(Sib) + Sp
        Par = st.sidebar.text_input('Number of Parents:','2')
        ch = st.sidebar.text_input('Number of Children', 0,15,0)
        Parch = int(Par) + int(ch)
        Fare = st.sidebar.text_input('Fare','20')
        Fare = int(Fare)
        Class = st.sidebar.slider('Class',1,3,1)
        if Class == 1:
            class_1 = 1
            class_2 = 0
            class_3 = 0
        elif Class == 2:
            class_1 = 0
            class_2 = 1
            class_3 = 0
        else:
            class_1 = 0
            class_2 = 0
            class_3 = 1
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
    
    lr_mod = LogisticRegression()
    lr_mod = scale_fit(lr_mod,standard_scaler,x_train,y_train)
    
    user_scaled = standard_scaler.fit_transform(df)
    if lr_mod.predict(df) == 1:
        st.write('Congratulations, you would have survived!')
    else:
        st.write('Tough luck, you would have died.')



pages = {'home':home,
         'exploration':exploration,
         'models':models,
         'prediction':prediction}

selected_page = st.sidebar.selectbox('Navigation', list(pages.keys()))
pages[selected_page]()
