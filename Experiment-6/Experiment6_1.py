# Experiment 6.1: Effect of tree depth on accuracy
# Aim: To compare testing accuracy of decision tree for depths 2, 3, 5 and None

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = load_iris()
x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# different depths to try (None means no limit)
depths = [2, 3, 5, None]

for d in depths:
    model = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    print("Tree Depth:", d)
    print("Testing Accuracy:", acc)
    print()
