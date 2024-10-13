from sklearn.datasets import fetch_california_housing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import scipy.stats as stats

data = fetch_california_housing()
X = data.data
y = data.target
feature_names = data.feature_names

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the RandomForestRegressor
random_forest = RandomForestRegressor(max_depth=45, min_samples_leaf=4, min_samples_split=7, n_estimators=195, random_state=42)
random_forest.fit(X_train, y_train)

# Best estimator found by RandomizedSearch
print("\033[92m========================= Random Forest ===========================\033[0m")
# Predict using the best model
y_pred = random_forest.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# log output
print(f"Optimized Random Forest: MSE = {mse:.4f}, R2 = {r2:.4f}")
print("\033[92m===================================================================\033[0m")

# Plot the results
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, edgecolors='k', s=15, label='Random Forest Regressor Predictions')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.title(f'Optimized Random Forest: Predicted vs Actual, MSE = {mse:.4f}, R2 = {r2:.4f}')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
# !Uncomment the following lines to save the plot
# if not os.path.exists("images/"):
#     os.makedirs("images/")
# plt.savefig("images/Ex3_RF_Prediction.png")
plt.show()
