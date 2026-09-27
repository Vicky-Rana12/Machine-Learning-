import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
data = {
    "Age": [25, 30, None, 28, 35, 40, None, 22],
    "Salary": [50000, 60000, 55000, None, 72000, 80000, 45000, None],
    "Department": ["Sales", "IT", "IT", "HR", "Sales", "HR", "IT", "Sales"],
    "Years_of_Experience": [2, 5, 3, None, 8, 10, 1, 4]
}
df = pd.DataFrame(data)

X = df.drop("Department", axis=1)  # using Department as categorical feature, not target
y = df["Department"]

numeric_features = ["Age", "Salary", "Years_of_Experience"]
categorical_features = []

print("--- Original Dataset ---")
print(df)

print("\n--- Missing Values ---")
print(df.isnull().sum())