KNN

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load Dataset
df = pd.read_csv("knn_sample_data.csv")
print(df.head())
print(df.isnull().sum())

# 2. Boxplot and Histograms

# Boxplot for Salary
plt.figure()
plt.boxplot(df['Salary'])
plt.title("Boxplot of Salary")
plt.ylabel("Salary")
plt.savefig('knn_boxplot_salary.png')

# Histograms for Age & Salary
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(df['Age'], bins=10, color='skyblue', edgecolor='black')
plt.title("Age Distribution")

plt.subplot(1,2,2)
plt.hist(df['Salary'], bins=10, color='lightgreen', edgecolor='black')
plt.title("Salary Distribution")

plt.tight_layout()
plt.savefig('knn_hist_age_salary.png')

# -------------------------------
# 3. Scatter Plot (Color = Purchased)
# -------------------------------

# Ensure numeric data
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')

plt.figure(figsize=(10,4))
sns.scatterplot(x='Age', y='Salary', hue='Purchased', data=df, palette='cool', s=20)
plt.title("Age vs Salary")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.savefig('knn_scatter_age_salary.png')

# 4. Model Creation and Training

X = df[['Age', 'Salary']]
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# 5. Model Evaluation

y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 6. Finding the Optimal K

k_values = range(1, 21)
scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

plt.figure()
plt.plot(k_values, scores, marker='o')
plt.title("K Value vs Model Accuracy")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy")
plt.savefig('knn_optimal_k.png')


DECISION TREE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import plot_tree

# Load dataset
california = fetch_california_housing(as_frame=True)
X = california.data
y = california.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42, max_depth=5)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Visualization: Plotting tree structure
plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X.columns, fontsize=10)
plt.title("Decision Tree Regression - California Housing Dataset")
plt.savefig('decisiontree')
plt.close()

# Visualization: Predicted vs Actual
plt.scatter(y_test, y_pred,s=5)
plt.xlabel("Actual Prices (in $100,000s)")
plt.ylabel("Predicted Prices (in $100,000s)")
plt.title("Decision Tree Regression - Predicted vs Actual")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red")
plt.savefig('scatterplotDecisiontree')

dtreee - without dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("csv/decision_tree_salary.csv")

X = df[['Experience', 'Test_Score', 'Interview_Score']]
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeRegressor(max_depth=4, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

plot_tree(model, feature_names=X.columns, filled=True)
plt.title("Decision Tree - Salary Prediction")
plt.savefig("decision_tree_salary.png")


REGRESSION

lin_reg

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("csv/linear_regression_bike.csv")

X = df[['Age']]
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Equation: y =", model.intercept_, "+", model.coef_[0], "*x")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.title("Simple Linear Regression")
plt.xlabel("Age")
plt.ylabel("Selling Price")
plt.savefig("linear_regression_bike.png")


LINEAR

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
plt.scatter(X_test, y_test, s=20, label="Actual Data")
plt.plot(X_test, y_pred, color="red",label="Regression Line")
plt.xlabel("BMI (Body Mass Index)")
plt.ylabel("Disease Progression")
plt.title("Simple Linear Regression - Diabetes Dataset")
plt.legend()
plt.savefig("linear_regression_diabetes.png")


MULTIPLE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
california = fetch_california_housing(as_frame=True)
X = california.data
y = california.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Regression equation (coefficients for each feature)
print("Intercept (b0):", model.intercept_)
print("Coefficients (b1, b2, ..., bn):")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")

# Evaluation
print("\nMean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Visualization: Predicted vs Actual
plt.scatter(y_test, y_pred, s=5, label="Predicted Points")
plt.xlabel("Actual Prices (in $100,000s)")
plt.ylabel("Predicted Prices (in $100,000s)")
plt.title("Multiple Linear Regression - California Housing Dataset")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", label="Regression line")
plt.legend()
plt.savefig("multiple_linear_regression_california.png")


mul_reg

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("csv/multiple_regression_house.csv")

X = df[['Area', 'Bedrooms', 'Age']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Intercept:", model.intercept_)
print("Coefficients:", dict(zip(X.columns, model.coef_)))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

plt.scatter(y_test, y_pred, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Multiple Linear Regression - House Price")
plt.savefig("multiple_regression_house.png")
