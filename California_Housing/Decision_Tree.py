# %% Importing Packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
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

# %% Initialize the Decision Tree model
tree_model = DecisionTreeRegressor(random_state=42)

# %% Fit the model to the training data
tree_model.fit(X_train, y_train)

# %% Make predictions on the test data
y_pred = tree_model.predict(X_test)

# %% Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# %% Optionally, you can also check feature importance
feature_importance = pd.Series(tree_model.feature_importances_, index=X.columns)
print("\nFeature Importance:")
print(feature_importance.sort_values(ascending=False))

# %% Plot the decision tree (Takes a while)
plt.figure(figsize=(20,20))
plot_tree(tree_model, filled=True, feature_names=X.columns)#, class_names=data.target_names)
plt.show()

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
