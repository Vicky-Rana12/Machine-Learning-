import pandas as pd


data = {
    "Student_Name": ["Amit", "Riya", "Sourav", "Neha", "Rahul", "Priya", "Karan", "Sneha", "Rohit", "Anjali"],
    "Roll_Number": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [72, 85, 91, 68, 77, 95, 60, 82, 74, 88],
    "Attendance": [88, 92, 76, 95, 81, 85, 70, 90, 78, 93]
}

df = pd.DataFrame(data)

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

print("--- DataFrame with Grade Column ---")
print(df)