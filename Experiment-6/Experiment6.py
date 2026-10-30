# Experiment 6: Decision Tree Classifier on Iris dataset
# Aim: To build a decision tree (gini, max depth = 3) and visualize it

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# loading the dataset
data = load_iris()
x = data.data
y = data.target

# splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# creating the model using gini
model = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
model.fit(x_train, y_train)

# prediction and accuracy
y_pred = model.predict(x_test)
print("Accuracy Score:", round(accuracy_score(y_test, y_pred), 4))

# plotting the tree
plt.figure(figsize=(14, 8))
plot_tree(model, feature_names=data.feature_names, class_names=data.target_names,
          filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree Visualization (Max Depth = 3)")
plt.show()
