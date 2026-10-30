# Experiment 5.3: ROC Curve and AUC
# Aim: To plot the ROC curve and find the AUC score of the logistic regression model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# each row = [study hours, attendance]
x = [
    [1, 35], [2, 40], [2, 50], [3, 45], [3, 55],
    [4, 40], [4, 50], [4, 60], [5, 45], [5, 55],
    [5, 60], [6, 45], [6, 50], [7, 45], [7, 50],
    [6, 60], [6, 65], [7, 60], [7, 70], [8, 50],
    [8, 60], [8, 70], [9, 55], [9, 65], [9, 75],
    [10, 55], [10, 65], [10, 75], [11, 60], [11, 80]
]

# 0 = fail, 1 = pass
y = [0] * 15 + [1] * 15

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(x_train, y_train)

# predicted probabilities for class 1
y_prob = model.predict_proba(x_test)[:, 1]

# finding fpr, tpr and auc
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

print("--- ROC Curve and AUC ---")
print("AUC Score:", auc)

# plotting the ROC curve
plt.plot(fpr, tpr, label="Logistic Regression")
plt.plot([0, 1], [0, 1], linestyle="--")   # random classifier line

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
