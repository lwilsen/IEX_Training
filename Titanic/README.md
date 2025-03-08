# Titanic ML Exploration Project

The goal of this project was to predict the survival outcome of passengers on the titanic based on data from this Kaggle dataset: https://www.kaggle.com/c/titanic/data.

---

## Methods and Libraries

### Data Preprocessing
- astype: Converts data types for specific columns to ensure compatibility with machine learning models.
- drop: Removes unnecessary columns or rows, such as irrelevant features or missing data.
- fillna: Fills missing values with specific strategies (mean, median, mode, etc.).
- groupby: Aggregates data, often used for feature engineering or to explore relationships between variables.
- replace: Replaces certain values within the dataset, such as mapping categorical variables to numerical representations.
- Feature scaling: Used StandardScaler, MinMaxScaler, and RobustScaler to standardize or normalize features.
### Dimensionality Reduction
- PCA (Principal Component Analysis): Reduces the dimensionality of the dataset by transforming features into a set of linearly uncorrelated components.
### Model Training and Evaluation
-  model_training: Trains machine learning models using a fitting process (e.g., calling .fit()).
- piped_traineval: A custom function that likely streamlines the process of training and evaluating models through a pipeline.
- train_eval: Another custom function used to assess model performance, possibly handling cross-validation or other evaluation metrics.
- Train-test split: Uses train_test_split to divide data for training and testing.
- Pipeline-based model training and evaluation: Combines multiple steps (preprocessing, training, and evaluation) into a single pipeline.
### Libraries Used
- pandas: Data manipulation and analysis.
- numpy: Numerical operations and array handling.
- matplotlib: Data visualization.
- seaborn: Statistical data visualization.
- sklearn (scikit-learn): Machine learning models and preprocessing.
- PCA: Dimensionality reduction.
- Pipeline: Streamlined model training and evaluation.
- train_test_split: Splitting data into training and testing sets.
- StandardScaler, MinMaxScaler, RobustScaler: Feature scaling.
- LogisticRegression, RandomForestClassifier, SVC, KNeighborsClassifier, LinearSVC: Classification models.
- DBSCAN, KMeans: Clustering algorithms.
- GridSearchCV: Hyperparameter tuning.
- SequentialFeatureSelector: Feature selection.
- accuracy_score: Model performance evaluation.
- silhouette_samples: Clustering performance metric.
- variance_inflation_factor: Multicollinearity check.
---

## Streamlit App

To run the Streamlit app:
1. Copy this repository to your local device
2. Run `streamlit run streamlit_titanic.py`

## Project Overview

### Variables Used

Below is a table of which variables I used to train my models, and a description of what they are.

![Column Descriptions](https://github.com/lwilsen/IEX_Training/blob/main/Titanic/Images/Screenshot%202024-04-23%20at%208.46.46%E2%80%AFAM.png)

### Models

I used a couple of different models, and after hyper parameter tuning, I got the following models and associated scores.

![Models and scores](https://github.com/lwilsen/IEX_Training/blob/main/Titanic/Images/Screenshot%202024-05-01%20at%203.28.13%E2%80%AFPM.png)

### Model Evaluation Methods

I also wanted to highlight my use of Learning Curves and ROC graphs to evaluate the accuracy of my model, while also looking at different metrics like specificity and sensitivity.

![Learning Curve](https://github.com/lwilsen/IEX_Training/blob/main/Titanic/Images/Screenshot%202024-04-23%20at%208.48.06%E2%80%AFAM.png)

![Validation Curve](https://github.com/lwilsen/IEX_Training/blob/main/Titanic/Images/Screenshot%202024-04-23%20at%208.48.19%E2%80%AFAM.png)
