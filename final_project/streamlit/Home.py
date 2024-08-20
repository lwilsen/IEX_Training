import streamlit as st

st.title("The Cumulative Data Science App")
st.write("Luke Wilsen")

st.divider()

st.markdown("""
### Overview
This app demonstrates various data science and machine learning techniques applied to multiple datasets. Below is a brief summary of each dataset.

### Data
- **Titanic**: Passenger data from the Titanic disaster, used for binary classification (survived or not).
- **Ames Housing**: Housing data from Ames, Iowa, used with regression to predict house prices.
- **IMDB Movie Reviews**: Sentiment analysis of movie reviews, used for binary text classification (positive or negative).
- **MNIST Digits**: Handwritten digit images (0-9), used for image classification.
""")