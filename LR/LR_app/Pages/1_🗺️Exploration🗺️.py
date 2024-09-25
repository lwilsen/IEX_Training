import streamlit as st
import pickle
import plotly.express as px
import numpy as np

st.write("# Data Exploration")

st.write(
    """In this section, I wanted to explore the distribution of some of the variables, to get a 
         sense of which ones could be useful to look at.
         
To start, looking at the distribution of SalesPrice helps me check for any unusual patterns"""
)

with open("data.pkl", "rb") as f:
    df = pickle.load(f)

df["Log_saleprice"] = np.log(df["SalePrice"])

fig = px.histogram(df, x="SalePrice", nbins=50, title="Histogram of SalePrice")
st.plotly_chart(fig)

st.write(
    """Clearly we're working with a right skewed dataset. This means that the linear regression assumption
         of a normally distributed dataset has been violated. In order to correct this, we can take the 
         natural logarithm of the SalePrice column, and use that to fit the model."""
)

fig = px.histogram(df, x="Log_saleprice", nbins=50, title="Histogram of Log(SalePrice)")
st.plotly_chart(fig)

fig = px.violin(df, x="Central Air", y="SalePrice")
st.plotly_chart(fig)

st.write(
    """Predictably, much of the right skew in SalePrice we observed earlier comes from homes with central
         air conditioning."""
)

fig = px.scatter(
    df,
    x="Overall Qual",
    y="SalePrice",
    color="Fireplaces",
    opacity=0.5,
    title="Quality vs SalePrice Accounting for Number of Fireplaces",
)
st.plotly_chart(fig)
