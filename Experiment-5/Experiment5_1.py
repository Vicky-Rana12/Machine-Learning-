# Experiment 5.1: Student Pass/Fail prediction using Logistic Regression
# Aim: To predict whether a student will pass or fail using study hours and attendance

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

# scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# training the model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(x_train, y_train)

# testing the model
y_pred = model.predict(x_test)

print("--- Student Pass/Fail Prediction ---")
print("Accuracy:", accuracy_score(y_test, y_pred))

# predicting for a new student (8 hours study, 75 attendance)
new_student = [[8, 75]]
new_student = sc.transform(new_student)
result = model.predict(new_student)

if result[0] == 1:
    print("Student will Pass")
else:
    print("Student will Fail")
