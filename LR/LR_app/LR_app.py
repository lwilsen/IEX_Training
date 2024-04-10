import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
import streamlit as st
import pickle

st.write('---')
st.write("""
# Housing Price Linear Regression Project
Author: Luke Wilsen

This was data came from the assessor's office of Ames Iowa, and contains information used to computing 
        assessed values for individual residential properties sold in ames, IA from 2006 to 2010.
    """)     
st.write("""---""")
st.image("/Users/lukewilsen/Desktop/IEX/IEX_Training/LR/LR_app/Ames_Iowa.jpg",caption="Ames Iowa", use_column_width=True)
st.write("""        
I chose to look at a subsection of the entire dataset, selecting only the following columns:
""")

st.write("""
            
    | Column Name    | Description                                                                     |
    |----------------|---------------------------------------------------------------------------------|
    | Overall Qual   | Rates the overall material and finish of the house. (ordinal 1-10)              |
    | Overall Cond   | Rates the overall condition of the house. (ordinal 1-10)                        |
    | Gr Liv Area    | Above grade (ground) living area square feet. (continuous)                      |
    | Central Air    | Central air conditioning. (Nominal)                                             |
    | Total Bsmt SF  | Total square feet of basement area. (continuous)                                |
    | SalePrice      | Sale price of the house. (continuous)                                           |
    | Fireplaces     | The number of fireplaces. (discrete)                                            |
    | Exter Qual     | Evaluates the quality of the material on the exterior.                          |
    ---
    """)

columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area',
           'Central Air', 'Total Bsmt SF', 'SalePrice','Fireplaces', 'Exter Qual']

@st.cache_data
def load_data(url,columns = columns):
    df = pd.read_csv(str(url),sep = '\t',usecols=columns)
    return df

df = load_data('http://jse.amstat.org/v19n3/decock/AmesHousing.txt',)

df["Central Air"] = df['Central Air'].map({'Y':1,'N':0})
df = df.dropna(axis = 0)
df['Exter Qual'] = df['Exter Qual'].map({'TA': 2, 'Gd': 3, 'Ex': 4, 'Fa': 1})




with open('data.pkl','wb') as f:
    pickle.dump(df,f)

st.dataframe(df) 

slr = LinearRegression()
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=1.0)
elanet = ElasticNet(alpha=1.0, l1_ratio=0.5)
tree = DecisionTreeRegressor(max_depth=3)


with open('slr_mod.pkl', 'wb') as f:
    pickle.dump(slr, f) 

with open('ridge_mod.pkl', 'wb') as f:
    pickle.dump(ridge, f) 

with open('lasso_mod.pkl', 'wb') as f:
    pickle.dump(lasso, f) 

with open('elanet_mod.pkl', 'wb') as f:
    pickle.dump(elanet, f) 

with open('tree_mod.pkl', 'wb') as f: 
    pickle.dump(tree, f)