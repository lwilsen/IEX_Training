import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import numpy as np
import pandas as pd
import requests

if st.sidebar.button("Reset Page"):
    st.markdown(
        """
        <script>
        document.documentElement.scrollTop = 0;
        </script>
        """,
        unsafe_allow_html=True
    )


st.header("Interactive Predictions!")

table_option = st.radio("Select Dataset:", ("Titanic",'Housing','Movie', 'MNIST'))

st.divider()

if table_option == "Titanic":
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
        class_to_one_hot = lambda Class: [1,0,0] if Class == 1 else [0,1,0] if Class == 2 else [0,0,1]
        class_1,class_2,class_3 = class_to_one_hot(Class)
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
    user_input_features_df = user_input_features()
    st.write(user_input_features_df)

    if st.button("Predict"):
        response = requests.post("http://localhost:8000/predict_titanic", json = user_input_features_df.to_dict(orient='records')[0])
        result = response.json()
        result_df = pd.DataFrame([result])
        #survival_prob = result_df['Survival_prob'][0]
        prediction = result_df['Survived'][0]

        if prediction == 1:
            st.success("You Survived!")
            st.image('/Users/lukewilsen/Desktop/IEX/IEX_Training/Titanic/happy_sailor.jpg',use_column_width=False)

        else:
            st.error('Uh Oh.')
            st.image('/Users/lukewilsen/Desktop/IEX/IEX_Training/Titanic/you_died.png', use_column_width= False)

elif table_option == "Housing":
    st.write("""Housing Exploration and Prediction Tool""")

    # Streamlit inputs
    overall_qual = st.select_slider('Overall Quality (1-10):', options=range(1, 11), value=5)
    overall_cond = st.select_slider('Overall Condition (1-10):', options=range(1, 11), value=5)
    exter_qual = st.selectbox('Exterior Quality:', options=['Poor', 'Fair', 'Good', 'Very Good'])
    total_bsmt_sf = st.slider('Total Basement SF (sq. ft.):', 0, 3000, 1000)
    central_air = st.selectbox('Central Air Conditioning:', options=['Yes', 'No'])
    gr_liv_area = st.slider('Above Grade Living Area (sq. ft):', 0, 5000, 1500)
    fireplaces = st.number_input('Number of Fireplaces:', min_value=0, max_value=5, value=2)
    
    

    # Convert categorical data to numerical for consistency in modeling
    central_air_numeric = 1 if central_air == 'Yes' else 0
    exter_qual_mapping = {
        'Poor': 1,
        'Fair': 2,
        'Good': 3,
        'Very Good': 4
    }
    exter_qual_numeric = exter_qual_mapping[exter_qual]

    # Create DataFrame
    user_input = pd.DataFrame({
        'Overall Qual': [overall_qual],
        'Overall Cond': [overall_cond],
        'Exter Qual': [exter_qual_numeric],
        'Total Bsmt SF': [total_bsmt_sf],
        'Central Air': [central_air_numeric],
        'Gr Liv Area': [gr_liv_area],
        'Fireplaces': [fireplaces]
    }).to_dict(orient='records')[0]

    if st.button("Predict"):
        response = requests.post("http://localhost:8000/predict_housing", json = user_input)
        result = response.json()
        result_df = pd.DataFrame([result])
        st.write("Predicted Sale Price")
        exp_value = np.exp(result_df['Sale_Price'][0])

        # Format the value to avoid scientific notation
        formatted_value = f"${exp_value:,.2f}"

        # Display the formatted value in Streamlit
        st.write(formatted_value)

elif table_option == "Movie":
    st.write("Text Sentiment Prediction!")
    text = "This sentiment is negative, as an example."
    text = st.text_area("Enter a sentence for sentiment testing:", value=text)
    if st.button('Predict'):
        response = requests.post("http://localhost:8000/predict_sentiment", json = {'text':text})
        st.write(response.status_code)
        st.write(response.text)
        result = response
        #sentiment = result['sentiment']
        st.write(result)

