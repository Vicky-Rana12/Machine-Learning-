import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Synthetic real-estate dataset
area = np.array([500, 750, 1000, 1200, 1500, 1800, 2000, 2300, 2500, 2800])
bedrooms = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 5])
price = np.array([25, 35, 45, 55, 65, 75, 85, 95, 105, 115])  # in lakhs
X_multi = np.column_stack((area, bedrooms))
y = price

X_train, X_test, y_train, y_test = train_test_split(
    X_multi, y, test_size=0.25, random_state=42
)

model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

y_pred_multi = model_multi.predict(X_test)

mae_multi = mean_absolute_error(y_test, y_pred_multi)
mse_multi = mean_squared_error(y_test, y_pred_multi)
rmse_multi = np.sqrt(mse_multi)
r2_multi = r2_score(y_test, y_pred_multi)

print("Coefficients (Area, Bedrooms):", model_multi.coef_)
print("Intercept:", model_multi.intercept_)
print("\n--- Evaluation Metrics (Multiple Linear Regression) ---")
print(f"MAE : {mae_multi:.2f}")
print(f"MSE : {mse_multi:.2f}")
print(f"RMSE : {rmse_multi:.2f}")
print(f"R2 : {r2_multi:.4f}")