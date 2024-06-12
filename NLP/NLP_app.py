import streamlit as st

from PIL import Image

# Set the title and introduction
st.title("Natural Language Processing with IMDB Movie Reviews")
st.markdown("""
Welcome to the **Natural Language Processing (NLP) Demo App**! This app demonstrates the basics of NLP using the classic IMDB movie review dataset. You'll learn about text preprocessing, sentiment analysis, and more.

The IMDB dataset contains 50,000 movie reviews labeled as positive or negative, making it an excellent resource for binary sentiment classification tasks.

---

## Outline
- **Text Preprocessing**: Cleaning and preparing text data for analysis.
- **Vectorization**: Converting text data into numerical representations.
- **Sentiment Analysis**: Classifying text as positive or negative.
- **Model Evaluation**: Assessing the performance of your model.

---

## How to Use This App
1. **Navigate** through the sidebar to explore different sections.

Let's get started and dive into the fascinating world of NLP!
""")

st.write("---")
# Main page content
st.header("Introduction")
st.markdown("""
Natural Language Processing (NLP) is a subfield of artificial intelligence that focuses on the interaction between computers and humans through natural language. In this demo, we will use the IMDB movie review dataset to perform basic NLP tasks.
""")