import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Load dataset
df = pd.read_csv("decision_tree_salary.csv")

# Step 2: Select features and target
X = df[['Experience', 'Test_Score', 'Interview_Score']]
y = df['Salary']

# Step 3: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 4: Create Decision Tree Regressor
model = DecisionTreeRegressor(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Step 5: Predict on test data
y_pred = model.predict(X_test)

# Step 6: Evaluate model
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Step 7: Visualize decision tree
plt.figure(figsize=(12,8))
plot_tree(model, feature_names=X.columns, filled=True)
plt.title("Decision Tree - Salary Prediction")
plt.savefig("decision_tree_salary.png")  # Save figure as PNG
plt.show()
