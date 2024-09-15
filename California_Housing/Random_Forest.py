# %% Importing Packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree
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

# %% Initialize the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# %% Fit the model to the training data
rf_model.fit(X_train, y_train)

# %% Make predictions on the test data
y_pred = rf_model.predict(X_test)

# %% Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# %% Optionally, you can also check feature importance
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
print("\nFeature Importance:")
print(feature_importance.sort_values(ascending=False))

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

# %% Visualize individual trees in the forest
for index, tree_in_forest in enumerate(rf_model.estimators_):
    plt.figure(figsize=(20,10))
    plot_tree(tree_in_forest, filled=True, feature_names=X.columns)
    plt.title(f'Tree {index}')
    plt.show()










# # %% Importing Packages

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# # %% Initialize the Random Forest classifier
# forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# # %% Train the classifier
# forest_classifier.fit(X_train, y_train)

# # %% Make predictions
# y_pred_forest = forest_classifier.predict(X_test)

# # %% Evaluate the model
# accuracy_forest = accuracy_score(y_test, y_pred_forest)
# print(f"Accuracy of Random Forest: {accuracy_forest}")
