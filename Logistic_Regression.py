# %% Importing Packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# %% Load a sample dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# %% Split the data into training and testing sets
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% Create and train the Logistic Regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# %% Make predictions
y_pred = model.predict(X_test)

# %% Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

################# Ridge Regression

# %% Create and train the Ridge (L2) Regularized Logistic Regression model
ridge_model = LogisticRegression(penalty='l2', solver='liblinear', C=1.0, max_iter=10000)
ridge_model.fit(X_train, y_train)

# %% Make predictions
y_pred_ridge = ridge_model.predict(X_test)

# %% Evaluate the model
accuracy_ridge = accuracy_score(y_test, y_pred_ridge)
conf_matrix_ridge = confusion_matrix(y_test, y_pred_ridge)
class_report_ridge = classification_report(y_test, y_pred_ridge)

print(f"Ridge (L2) Regularization - Accuracy: {accuracy_ridge}")
print("Confusion Matrix:")
print(conf_matrix_ridge)
print("Classification Report:")
print(class_report_ridge)

################# Lasso Regression

# %% Create and train the Lasso (L1) Regularized Logistic Regression model
lasso_model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=10000)
lasso_model.fit(X_train, y_train)

# %% Make predictions
y_pred_lasso = lasso_model.predict(X_test)

# %% Evaluate the model
accuracy_lasso = accuracy_score(y_test, y_pred_lasso)
conf_matrix_lasso = confusion_matrix(y_test, y_pred_lasso)
class_report_lasso = classification_report(y_test, y_pred_lasso)

print(f"Lasso (L1) Regularization - Accuracy: {accuracy_lasso}")
print("Confusion Matrix:")
print(conf_matrix_lasso)
print("Classification Report:")
print(class_report_lasso)


# %%
