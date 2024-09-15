# %% Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
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

# %% Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer for regression
])

# %% Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# %% Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, verbose=1)

# %% Make predictions on the test data
y_pred = model.predict(X_test)

# %% Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# %% Optionally, you can plot the training history to see how the model converged

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Model Training History')
plt.legend()
plt.show()

# %% Plotting Y test against predictions
plt.scatter(y_test,y_pred, color='red', label='Data')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Polynomial Regression')
plt.legend()
plt.show()

# %% Plotting Distributions of Actual against Predicted
fig, ax = plt.subplots()
sns.kdeplot(y_test, bw_method=0.1,color='b',ax=ax,label='Actual')
sns.kdeplot(y_pred[:,0], bw_method=0.1,color='r',ax=ax,label='Predicted')
plt.legend()
# %%
y_pred.shape