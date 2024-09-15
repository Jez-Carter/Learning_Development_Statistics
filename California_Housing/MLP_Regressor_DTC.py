# %% Importing Packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

import matplotlib.pyplot as plt
import seaborn as sns

# %% Load a sample dataset

data = pd.read_pickle('~/Machine_Learning_Revision/California_Housing/Auxiliary_Data/california_housing_dtc.pkl')

data = data.sample(n=len(data))

scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data),columns = data.columns)

# %% Exploring data relationships
sns.pairplot(data[:100])

# %% Split the data into training and testing sets
X = data_scaled.drop('MedHouseVal', axis=1)
y =  data_scaled['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% Initialize the MLPRegressor model
mlp = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu', solver='adam', max_iter=500, random_state=42)

# %% Fit the model to the training data
mlp.fit(X_train, y_train)

# %% Make predictions on the test data
y_pred = mlp.predict(X_test)

# %% Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# %% Plotting Y test against predictions
plt.scatter(y_test,y_pred, color='red', label='Data')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Polynomial Regression')
plt.legend()
plt.show()

# %% Plotting Distributions of Actual against Predicted
sns.kdeplot(y_test, bw_method=0.1)
sns.kdeplot(y_pred, bw_method=0.1)

# %%
