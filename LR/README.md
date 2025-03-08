# Linear Regression!

This directory houses my linear regression learning for classification and prediction, as well as a streamlit app (and dockerized version) that predicts housing prices based of the Ames Iowa housing dataset. (https://www.kaggle.com/datasets/marcopale/housing)

### Methods Used:
Linear Regression – Ordinary Least Squares (OLS) model for basic trend estimation.
RANSAC Regression – Robust model to handle outliers.
Lasso Regression – L1 regularization for feature selection.
Ridge Regression – L2 regularization to prevent overfitting.
Elastic Net Regression – Combination of L1 and L2 penalties.
Polynomial Regression – Non-linear relationships modeled with polynomial features.
Decision Tree and Random Forest Regression – Tree-based models for non-linear patterns.
Data Visualization – seaborn, matplotlib, and mlxtend.scatterplotmatrix() for feature exploration.

### Libraries Used:
scikit-learn (modeling, preprocessing, evaluation)
pandas (data manipulation)
numpy (numerical operations)
matplotlib & seaborn (visualization)
mlxtend (scatter plot matrix for data exploration)
The models are trained and evaluated using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and $R^2$ Score.