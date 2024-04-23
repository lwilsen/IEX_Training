# Housing Price Linear Regression Project

The goal of this project was to use data from the assessor's office of Ames Iowa to identify a relationship between certain variables and housing price. My secondary goal was to then use this data to predict the price of a house based on its characteristics. 

## About the Project

The specific variables I looked at are contained in the table below:

![Column key](https://github.com/lwilsen/IEX_Training/blob/main/LR/Images/Screenshot%202024-04-23%20at%208.20.55%E2%80%AFAM.png)

### Data Transformation

The outcome variable (SalePrice) was right skewed, so I used a log transformation to correct that. This affects the interpretation of my model, but allows the assumptions of linear regression to be satisfied.

![SalePrice Histogram](https://github.com/lwilsen/IEX_Training/blob/main/LR/Images/Screenshot%202024-04-23%20at%208.35.41%E2%80%AFAM.png)

### Validation Curve

I also explored the use of validation curves to test hyperparameter choices, specifically the alpha parameter of the sklearn ridge regression model. The alpha parameter corresponds to the regularization constant of the model.

![Alpha test](https://github.com/lwilsen/IEX_Training/blob/main/LR/Images/Screenshot%202024-04-23%20at%208.22.23%E2%80%AFAM.png)
