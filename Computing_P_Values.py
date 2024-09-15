# %% Importing Packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from scipy import stats

import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler

# %% Load a sample dataset
from sklearn.datasets import fetch_california_housing
california = fetch_california_housing()
data = pd.DataFrame(california.data, columns=california.feature_names)
data['MedHouseVal'] = california.target

data = data.sample(n=len(data))

scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data),columns = data.columns)

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

# %% Computing P Values
# Calculate residuals and standard errors
residuals = y_test - y_pred
residual_sum_of_squares = np.sum(residuals**2)
degrees_of_freedom = X_test.shape[0] - X_test.shape[1]

# Variance of the residuals
residual_variance = residual_sum_of_squares / degrees_of_freedom

# Covariance matrix of the coefficients
X_transpose_X_inv = np.linalg.inv(np.dot(X_test.T, X_test))
covariance_matrix = residual_variance * X_transpose_X_inv

# Standard errors of the coefficients
standard_errors = np.sqrt(np.diag(covariance_matrix))

# Compute 95% confidence intervals
z = stats.norm.ppf(0.975)  # 95% confidence interval
confidence_intervals = np.array([model.coef_ - z * standard_errors, model.coef_ + z * standard_errors]).T

# Compute p-values
t_stats = model.coef_ / standard_errors
p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))

print("Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")

print("\nStandard Errors:")
for feature, stderr in zip(X.columns, standard_errors):
    print(f"{feature}: {stderr:.4f}")

print("\nConfidence Intervals:")
for feature, conint in zip(X.columns, confidence_intervals):
    print(f"{feature}: {conint[0]:.4f},{conint[1]:.4f}")

print("\nP Values:")
for feature, pval in zip(X.columns, p_values):
    print(f"{feature}: {pval:.4f}")

# %% Stats Models
    
model = sm.OLS(y_train, X_train)
results = model.fit()
print(results.summary())
