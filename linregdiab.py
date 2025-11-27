import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
diabetes = load_diabetes()

# Select one feature (BMI = 3rd column in dataset)
X = diabetes.data[:, np.newaxis, 2]
y = diabetes.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Regression line equation
print("Intercept (b0):", model.intercept_)
print("Coefficient (b1):", model.coef_[0])
print(f"Regression Line Equation: y = {model.intercept_:.2f} + {model.coef_[0]:.2f}*x")

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Visualization
plt.scatter(X_test, y_test, label="Actual Data")
plt.plot(X_test, y_pred, linewidth=2, label="Regression Line")
plt.xlabel("BMI (Body Mass Index)")
plt.ylabel("Disease Progression")
plt.title("Simple Linear Regression - Diabetes Dataset")
plt.legend()
plt.show()
