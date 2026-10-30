# Experiment 6.3: Overfitting analysis
# Aim: To plot training and testing accuracy against tree depth (1 to 10)

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = load_iris()
x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

train_acc = []
test_acc = []

for d in depths:
    model = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
    model.fit(x_train, y_train)

    # accuracy on training and testing data
    train_acc.append(accuracy_score(y_train, model.predict(x_train)))
    test_acc.append(accuracy_score(y_test, model.predict(x_test)))

# plotting the graph
plt.plot(depths, train_acc, marker="o", label="Training Accuracy")
plt.plot(depths, test_acc, marker="o", label="Testing Accuracy")

plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Training and Testing Accuracy vs Tree Depth")
plt.legend()
plt.grid()
plt.show()
