# NLP Sentiment Analysis

This directory contains a sentiment analysis project that uses NLP techniques to classify IMDB movie reviews.

### Methods used
- Text Feature Extraction:
  - TF-IDF Vectorization – TfidfVectorizer for term importance
  - Text Stemming - nltk's PorterStemmer for text simplification and improved efficiency
- Vectorization - Conversion of text into a numerical vector:
  - HashingVectorizer
  - CountVectorizer
- Logistic Regression – Supervised learning model for sentiment classification.
- Stochastic Gradient Descent Classifier - Unsupervised learning model for sentiment classification
- Latent Dirichlet Allocation - Generative Bayesian network used to identify different movie genres
- Hyperparameter Tuning – GridSearchCV for optimal model selection.
- Performance Metrics – Accuracy score & confusion matrix.
- Visualization – matplotlib & seaborn for confusion matrix plotting.
### Libraries Used:
- sklearn.feature_extraction.text, nltk, and pyprind.
