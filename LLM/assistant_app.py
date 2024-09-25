import streamlit as st
import cohere
import os


def main():
    st.title("Python Code Assistant")
    st.write("Python Code Helper! Enter your problematic code!")

    code_input = st.text_area("Enter Python Code:", height=400)
    if st.button("Analyze Code"):
        analyze_code(code_input)


def analyze_code(code):
    cohere_api_key = os.getenv("CO_API_KEY")
    co = cohere.Client(api_key=cohere_api_key)
    prompt = f"You are a python code helper. Analyze the provided python code and suggest improvements:\n{code}"
    response = co.chat(model="command-r-plus", message=prompt)
    st.write(response.text)


if __name__ == "__main__":
    main()
