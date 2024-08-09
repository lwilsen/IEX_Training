import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots

st.write('# Conclusions')

with open('data.pkl','rb') as f:
    df = pickle.load(f)

df['Log_saleprice'] = np.log(df['SalePrice'])
df['sqrt_saleprice'] = np.sqrt(df['SalePrice'])

target = 'SalePrice'
sqrt_target = 'sqrt_saleprice'
log_target = 'Log_saleprice'
log_feats = df.columns[(df.columns != target) & (df.columns != log_target) & (df.columns != sqrt_target)]

X_log = df[log_feats]
y_log = df[log_target]

Xl_train, Xl_test,yl_train,yl_test = train_test_split(X_log,y_log,test_size=0.3,random_state=123)

mod_stats = {}

def reg_mod(model, X_train = Xl_train,X_test = Xl_test,y_train = yl_train,y_test = yl_test):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Create subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Test data', 'Training data'))

    # Add traces
    fig.add_trace(px.scatter(x=y_test_pred, y=y_test_pred - y_test, color_discrete_sequence=['limegreen'],opacity = 0.5).data[0], row=1, col=1)
    fig.add_trace(px.scatter(x=y_train_pred, y=y_train_pred - y_train, color_discrete_sequence=['steelblue'], opacity=0.5).data[0], row=1, col=2)


    fig.add_hline(y=0, line_dash='dash', line_color='white', line_width=1, row=1, col=1)
    fig.add_hline(y=0, line_dash='dash', line_color='white', line_width=1, row=1, col=2)


    # Update layout
    fig.update_layout(
        showlegend=False,
        xaxis=dict(title='Predicted values'),
        yaxis=dict(title='Residuals'),
        xaxis2=dict(title='Predicted values'),
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode='closest'
    )

    # Update subplots
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

    # Show the plot using Streamlit
    st.write(model.__class__.__name__)
    #st.plotly_chart(fig)

    #rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

    #mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    #r2_train = r2_score(y_train, y_train_pred)
    r2_test =r2_score(y_test, y_test_pred)

    model_key = f"{model.__class__.__name__}"

    mod_stats[model_key] = {"rmse":rmse_test,"mae":mae_test,"r2":r2_test}

    mod_dict = {'coefficients':model.coef_,
                'intercept':model.intercept_}
    
    df = pd.DataFrame(mod_dict).transpose()

    df.columns = X_train.columns

    df.iloc[0] = np.exp(df.iloc[0])

    st.dataframe(df)

    return

with open('slr_mod.pkl', 'rb') as f:
    slr = pickle.load(f)

reg_mod(slr, X_train = Xl_train,X_test = Xl_test,y_train = yl_train,y_test = yl_test)

st.write("""
### Interpretation of coefficients
- Because the model used data that was log transformed, the table above contains all the coefficients raised to the power e
- We can see that the most important predictors are:
    - Central Air
    - Exter Qual
    - Overall Qual
- Here, an increase by 1 in any of the parameters causes an increase by  $ ___ hundred thousand dollars in saleprice.
    - Ex. According to the model, an increase in central air corresponds to an increase in price by 0.243 hundred thousand dollars ($24,300).
""")