import pandas as pd

# 2. Pandas DataFrame Creation
data = {
    "Student_Name": ["Amit", "Riya", "Sourav", "Neha", "Rahul", "Priya", "Karan", "Sneha", "Rohit", "Anjali"],
    "Roll_Number": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [72, 85, 91, 68, 77, 95, 60, 82, 74, 88],
    "Attendance": [88, 92, 76, 95, 81, 85, 70, 90, 78, 93]
}

df = pd.DataFrame(data)

print("--- Full DataFrame ---")
print(df)

print("\n--- Students Scoring Above 80 Marks ---")
print(df[df["Marks"] > 80])