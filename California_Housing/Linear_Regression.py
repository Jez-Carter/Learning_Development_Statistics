# %% Importing Packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt
import seaborn as sns

# %% Load a sample dataset
from sklearn.datasets import fetch_california_housing
california = fetch_california_housing()
data = pd.DataFrame(california.data, columns=california.feature_names)
data['MedHouseVal'] = california.target

data = data.sample(n=len(data))

scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data),columns = data.columns)

# %% Exploring data relationships
sns.pairplot(data[:100])

# %% Split the data into training and testing sets
X = data_scaled.drop('MedHouseVal', axis=1)
y =  data_scaled['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% Simple Linear Regression
################# SLR 
# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
# Print the coefficients of the linear regression model
print("Linear Regression Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")

# %% Ridge Regression
################# Ridge Regression
# Create and train the Ridge Regression model
ridge_model = Ridge(alpha=1000.0)
ridge_model.fit(X_train, y_train)
# Make predictions
y_pred_ridge = ridge_model.predict(X_test)
# Evaluate the model
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
print(f"Ridge Regression Mean Squared Error: {mse_ridge}")
# Print the coefficients of the ridge regression model
print("Ridge Regression Coefficients:")
for feature, coef in zip(X.columns, ridge_model.coef_):
    print(f"{feature}: {coef}")

# %% Lasso Regression
################# Lasso Regression
# Create and train the Lasso Regression model
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
# Make predictions
y_pred_lasso = lasso_model.predict(X_test)
# Evaluate the model
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
print(f"Lasso Regression Mean Squared Error: {mse_lasso}")
# Print the coefficients of the ridge regression model
print("Lasso Regression Coefficients:")
for feature, coef in zip(X.columns, lasso_model.coef_):
    print(f"{feature}: {coef}")

# %% Polynomial Regression
################# Polynomial Regression
# Create and train the Polynomial Regression model
poly = PolynomialFeatures(degree=2)
model = LinearRegression()
# Combine the polynomial features transformer with the linear regression model
pipeline = make_pipeline(poly, model)
pipeline.fit(X_train, y_train)
# Make predictions
y_pred_poly = pipeline.predict(X_test)
# Evaluate the model
mse_poly = mean_squared_error(y_test, y_pred_poly)
print(f"Polynomial Regression Mean Squared Error: {mse_poly}")
# Print the coefficients of the ridge regression model
print("Polynomial Regression Coefficients:")
for feature, coef in zip(poly.get_feature_names_out(), model.coef_):
    print(f"{feature}: {coef:.4f}")

# %% Plotting Y test against predictions
plt.scatter(y_test,y_pred_poly, color='red', label='Data')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Polynomial Regression')
plt.legend()
plt.show()

# %% Plotting Distributions of Actual against Predicted
sns.kdeplot(y_test, bw_method=0.1)
sns.kdeplot(y_pred_poly, bw_method=0.1)

