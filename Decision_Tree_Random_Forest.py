# %% Importing Packages
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

# %% Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# %% Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% Initialize the Decision Tree Classifier
tree_classifier = DecisionTreeClassifier(random_state=42)

# %% Train the classifier
tree_classifier.fit(X_train, y_train)

# %% Make predictions
y_pred = tree_classifier.predict(X_test)

# %% Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Decision Tree: {accuracy}")

# %% Plot the decision tree
plt.figure(figsize=(20,20))
plot_tree(tree_classifier, filled=True, feature_names=data.feature_names, class_names=data.target_names)
plt.show()

################# Random Forest

# %% Initialize the Random Forest classifier
forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# %% Train the classifier
forest_classifier.fit(X_train, y_train)

# %% Make predictions
y_pred_forest = forest_classifier.predict(X_test)

# %% Evaluate the model
accuracy_forest = accuracy_score(y_test, y_pred_forest)
print(f"Accuracy of Random Forest: {accuracy_forest}")

# %% Visualize individual trees in the forest
for index, tree_in_forest in enumerate(forest_classifier.estimators_):
    plt.figure(figsize=(20,10))
    plot_tree(tree_in_forest, filled=True, feature_names=data.feature_names, class_names=data.target_names)
    plt.title(f'Tree {index}')
    plt.show()