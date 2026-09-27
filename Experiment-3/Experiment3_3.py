import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
numeric_transformer_standard = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor_standard = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_standard, numeric_features)
    ]
)

X_standard = preprocessor_standard.fit_transform(df[numeric_features])

print("--- StandardScaler Transformed Array ---")
print(X_standard)
print("Range -> min:", X_standard.min(), " max:", X_standard.max())

print("\n--- MinMaxScaler Transformed Array ---")
print(X_minmax)
print("Range -> min:", X_minmax.min(), " max:", X_minmax.max())

print("\n--- Observation ---")
print("StandardScaler centers data around mean 0 with unit variance, so values can be negative and can exceed 1 or -1.")
print("MinMaxScaler squeezes all values strictly into the range [0, 1].")