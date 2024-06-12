import streamlit as st

st.header('Topic Modeling with Latent Dirichlet Allocation')

st.write('''This method uses Bayesian Inference to try to find groups of words that appear frequently together
         across different documents in the dataset. These groups of words can then be used to determine
         categories/topic areas that the documents could fall into. In this dataset, we set the number of categories to find to be 10.''')

topic_words = '''Topic 1:
worst minutes awful script stupid
Topic 2:
family mother father children girl
Topic 3:
american war dvd music tv
Topic 4:
human audience cinema art sense
Topic 5:
police guy car dead murder
Topic 6:
horror house sex girl woman
Topic 7:
role performance comedy actor performances
Topic 8:
series episode war episodes tv
Topic 9:
book version original read novel
Topic 10:
action fight guy guys cool'''

st.text_area('Top 5 most important words for each topic',topic_words,height = 400)
st.write('''Here you can come up with some potential topics based on these words. For example, topic 6 could 
         be generalized to "Horror Movies", and topic 5 could be generalized to murder/mystery movies or crime movies.''')
