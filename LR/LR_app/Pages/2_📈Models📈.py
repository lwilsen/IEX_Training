import streamlit as st
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, validation_curve
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    make_scorer,
    accuracy_score,
)


st.write("# Models page")

st.write("")

with open("data.pkl", "rb") as f:
    df = pickle.load(f)

df["Log_saleprice"] = np.log(df["SalePrice"])
df["sqrt_saleprice"] = np.sqrt(df["SalePrice"])

target = "SalePrice"
log_target = "Log_saleprice"
sqrt_target = "sqrt_saleprice"
features = df.columns[
    (df.columns != target) & (df.columns != log_target) & (df.columns != sqrt_target)
]
log_feats = df.columns[
    (df.columns != target) & (df.columns != log_target) & (df.columns != sqrt_target)
]
sqrt_feats = df.columns[
    (df.columns != target) & (df.columns != log_target) & (df.columns != sqrt_target)
]

X = df[features].values
X_log = df[log_feats]
X_sqrt = df[sqrt_feats]
y = df[target].values
y_log = df[log_target]
y_sqrt = df[sqrt_target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123
)

Xl_train, Xl_test, yl_train, yl_test = train_test_split(
    X_log, y_log, test_size=0.3, random_state=123
)

Xsqrt_train, Xsqrt_test, ysqrt_train, ysqrt_test = train_test_split(
    X_sqrt, y_sqrt, test_size=0.3, random_state=123
)

mod_stats = {}


def reg_mod(model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Create subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Test data", "Training data"))

    # Add traces
    fig.add_trace(
        px.scatter(
            x=y_test_pred,
            y=y_test_pred - y_test,
            color_discrete_sequence=["limegreen"],
            opacity=0.5,
        ).data[0],
        row=1,
        col=1,
    )
    fig.add_trace(
        px.scatter(
            x=y_train_pred,
            y=y_train_pred - y_train,
            color_discrete_sequence=["steelblue"],
            opacity=0.5,
        ).data[0],
        row=1,
        col=2,
    )

    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1, row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1, row=1, col=2)

    # Update layout
    fig.update_layout(
        showlegend=False,
        xaxis=dict(title="Predicted values"),
        yaxis=dict(title="Residuals"),
        xaxis2=dict(title="Predicted values"),
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="closest",
    )

    # Update subplots
    fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=False)
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=False)

    # Show the plot using Streamlit
    st.write(model.__class__.__name__)
    st.plotly_chart(fig)

    # rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    # r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    model_key = f"{model.__class__.__name__}"

    mod_stats[model_key] = {"rmse": rmse_test, "mae": mae_test, "r2": r2_test}

    return


with open("slr_mod.pkl", "rb") as f:
    slr = pickle.load(f)

with open("ridge_mod.pkl", "rb") as f:
    ridge = pickle.load(f)

st.write(ridge.get_params())

with open("lasso_mod.pkl", "rb") as f:
    lasso = pickle.load(f)

with open("elanet_mod.pkl", "rb") as f:
    elanet = pickle.load(f)

with open("tree_mod.pkl", "rb") as f:
    tree = pickle.load(f)

reg_mod(slr)
st.write(
    """As you can see, the residuals of the regular sale price are very skewed. Transforming the data 
allows us to satisfy the assumptions of linear regression models """
)
st.write("### Log Transformation")
reg_mod(slr, X_train=Xl_train, X_test=Xl_test, y_train=yl_train, y_test=yl_test)
st.write("### Square Root Transformation")
reg_mod(slr, Xsqrt_train, Xsqrt_test, ysqrt_train, ysqrt_test)
st.write("Log transform not significantly different from the square root transform.")
st.write("- Choosing to use log transform for the rest of the models")
reg_mod(ridge, X_train=Xl_train, X_test=Xl_test, y_train=yl_train, y_test=yl_test)
reg_mod(lasso, X_train=Xl_train, X_test=Xl_test, y_train=yl_train, y_test=yl_test)
reg_mod(elanet, X_train=Xl_train, X_test=Xl_test, y_train=yl_train, y_test=yl_test)
reg_mod(tree, X_train=Xl_train, X_test=Xl_test, y_train=yl_train, y_test=yl_test)

restructured = []
for regressor, metric in mod_stats.items():
    row = {
        "Regressor": regressor,
        "RMSE": metric["rmse"],
        "MAE": metric["mae"],
        "R2": metric["r2"],
    }
    restructured.append(row)
df = pd.DataFrame(restructured)
st.dataframe(df, height=200, width=700)

pipeline = Pipeline([("standardscaler", StandardScaler()), ("ridge", ridge)])
scoring = make_scorer(mean_absolute_error, greater_is_better=True)

param_range = np.arange(0, 50, 0.5)
train_scores, test_scores = validation_curve(
    estimator=pipeline,
    X=Xl_train,
    y=yl_train,
    param_name="ridge__alpha",
    param_range=param_range,
    cv=5,
    scoring=scoring,
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Create a Matplotlib figure
fig, ax = plt.subplots()

# Plot the learning curves
ax.plot(
    param_range,
    train_mean,
    color="blue",
    marker="o",
    markersize=5,
    label="Training accuracy",
)
ax.fill_between(
    param_range,
    train_mean + train_std,
    train_mean - train_std,
    alpha=0.15,
    color="blue",
)

ax.plot(
    param_range,
    test_mean,
    color="green",
    linestyle="--",
    marker="s",
    markersize=5,
    label="Validation accuracy",
)

ax.fill_between(
    param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color="green"
)

ax.grid()
ax.legend(loc="upper right")
ax.set_xlabel("Parameter Alpha")
ax.set_ylabel("Mean Absolute Error")
fig.tight_layout()

# Display the plot using Streamlit
st.write("### Validation Curve for Ridge Regression based on the Alpha hyperparamter")
st.pyplot(fig)
