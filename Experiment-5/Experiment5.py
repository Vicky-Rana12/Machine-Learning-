# Experiment 5: Logistic Regression on Breast Cancer dataset
# Aim: To train a logistic regression model and find accuracy, precision, recall and F1 score

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# loading the dataset
data = load_breast_cancer()
x = data.data
y = data.target

# splitting data into training and testing (80% - 20%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# scaling the features
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# creating and training the model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(x_train, y_train)

# predicting on test data
y_pred = model.predict(x_test)

# calculating the metrics
print("--- Logistic Regression Performance ---")
print("Accuracy  :", round(accuracy_score(y_test, y_pred), 4))
print("Precision :", round(precision_score(y_test, y_pred), 4))
print("Recall    :", round(recall_score(y_test, y_pred), 4))
print("F1 Score  :", round(f1_score(y_test, y_pred), 4))
