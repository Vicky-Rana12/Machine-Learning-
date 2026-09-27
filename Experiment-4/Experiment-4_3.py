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
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(area.reshape(-1, 1))
y = price

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_poly, y, test_size=0.25, random_state=42
)

model_poly = LinearRegression()
model_poly.fit(X_train_p, y_train_p)

y_pred_poly = model_poly.predict(X_test_p)
r2_poly = r2_score(y_test_p, y_pred_poly)

print("--- Comparison: Linear vs Polynomial Regression ---")
print(f"Linear Regression R2    : {r2:.4f}")
print(f"Polynomial Regression R2: {r2_poly:.4f}")

# Plot polynomial curve
X_range = np.linspace(area.min(), area.max(), 100).reshape(-1, 1)
X_range_poly = poly.transform(X_range)
y_range_pred = model_poly.predict(X_range_poly)

plt.figure(figsize=(7, 5))
plt.scatter(area, price, color="blue", label="Actual Data")
plt.plot(X_range, y_range_pred, color="green", linewidth=2, label="Polynomial Regression")
plt.xlabel("House Area (sq. ft)")
plt.ylabel("Price (Lakhs)")
plt.title("Polynomial Regression: Area vs Price")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()