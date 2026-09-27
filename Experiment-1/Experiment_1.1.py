import numpy as np
import pandas as pd

# 1. NumPy Array - Internal Marks of 10 students
marks = np.array([72, 85, 91, 68, 77, 95, 60, 82, 74, 88])

print("Marks:", marks)
print("Mean:", np.mean(marks))
print("Median:", np.median(marks))
print("Standard Deviation:", np.std(marks))
print("Maximum:", np.max(marks))
print("Minimum:", np.min(marks))


# 2. Pandas DataFrame Creation
data = {
    "Student_Name": ["Amit", "Riya", "Sourav", "Neha", "Rahul", "Priya", "Karan", "Sneha", "Rohit", "Anjali"],
    "Roll_Number": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": marks,
    "Attendance": [88, 92, 76, 95, 81, 85, 70, 90, 78, 93]
}

df = pd.DataFrame(data)

print("\n--- Students Scoring Above 80 Marks ---")
print(df[df["Marks"] > 80])


# 3. Adding a dynamically calculated Grade column
def get_grade(m):
    if m >= 90:
        return "A"
    elif m >= 80:
        return "B"
    elif m >= 70:
        return "C"
    else:
        return "D"

df["Grade"] = df["Marks"].apply(get_grade)

print("\n--- DataFrame with Grade Column ---")
print(df)