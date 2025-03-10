import streamlit as st
import cohere
import os


def main():
    st.title("Text Sentiment Analysis")
    st.write("Enter a paragraph and we will analyze the sentiment of the text.")

    text_input = st.text_area("Enter Paragraph to Analyze:", height=400)
    if st.button("Analyze Code"):
        analyze_code(text_input)


def analyze_code(text):
    cohere_api_key = os.getenv("CO_API_KEY")
    co = cohere.Client(api_key=cohere_api_key)
    prompt = f"Your job is to analyze the sentiment of the followng text and determine whether it is positive or negative. :\n{text}"
    response = co.chat(model="command-r-plus", message=prompt)

    st.write(type(response.text))


if __name__ == "__main__":
    main()
