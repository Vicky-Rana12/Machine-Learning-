import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
numeric_transformer_minmax = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())
])

preprocessor_minmax = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_minmax, numeric_features)
    ]
)

X_minmax = preprocessor_minmax.fit_transform(df[numeric_features])

print("--- MinMaxScaler Transformed Array ---")
print(X_minmax)