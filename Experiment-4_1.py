import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Synthetic real-estate dataset
area = np.array([500, 750, 1000, 1200, 1500, 1800, 2000, 2300, 2500, 2800])
bedrooms = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 5])
price = np.array([25, 35, 45, 55, 65, 75, 85, 95, 105, 115])  
X = area.reshape(-1, 1)
y = price

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Slope (b1):", model.coef_[0])
print("Intercept (b0):", model.intercept_)
print("\n--- Evaluation Metrics (Simple Linear Regression) ---")
print(f"MAE : {mae:.2f}")
print(f"MSE : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2 : {r2:.4f}")

plt.figure(figsize=(7, 5))
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", linewidth=2, label="Regression Line")
plt.xlabel("House Area (sq. ft)")
plt.ylabel("Price (Lakhs)")
plt.title("House Area vs Price")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()