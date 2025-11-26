import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
df = pd.read_csv("linear_regression_bike.csv")

# Step 2: Select features and target
X = df[['Age']]          # Predictor (independent variable)
y = df['Selling_Price']  # Target (dependent variable)

# Step 3: Split data into train and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 4: Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict on test data
y_pred = model.predict(X_test)

# Step 6: Print the regression equation and performance metrics
print("Equation: y =", round(model.intercept_, 2), "+", round(model.coef_[0], 2), "* x")
print("Mean Squared Error (MSE):", round(mean_squared_error(y_test, y_pred), 2))
print("R² Score:", round(r2_score(y_test, y_pred), 2))

# Step 7: Visualize results
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.title("Simple Linear Regression: Age vs Selling Price")
plt.xlabel("Age (Years)")
plt.ylabel("Selling Price")
plt.legend()
plt.grid(True)
plt.savefig("linear_regression_bike.png")  # Save the figure as PNG
plt.show()
