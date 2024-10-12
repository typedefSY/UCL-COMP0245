import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import scipy.stats as stats

# Generate or reuse synthetic data
# Generate synthetic data
x1 = np.arange(0, 10, 0.1)
x2 = np.arange(0, 10, 0.1)
x1, x2 = np.meshgrid(x1, x2)
y = np.sin(x1) * np.cos(x2) + np.random.normal(scale=0.1, size=x1.shape)

# Flatten the arrays
x1 = x1.flatten()
x2 = x2.flatten()
y = y.flatten()
X = np.vstack((x1, x2)).T

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

param_dist = {
    'n_estimators': stats.randint(50, 200),
    'max_depth': [None, 20, 25, 30, 35, 40],
    'min_samples_split': stats.randint(2, 10),
    'min_samples_leaf': stats.randint(1, 10)
}

# Initialize the RandomForestRegressor
random_forest = RandomForestRegressor(random_state=42)

# Initialize the RandomizedSearchCV object
random_search = RandomizedSearchCV(estimator=random_forest, param_distributions=param_dist, n_iter=100, cv=5, verbose=1, random_state=42, scoring='neg_mean_squared_error')

# Fit the model
random_search.fit(X_train, y_train)

# Best estimator found by RandomizedSearch
print("\033[92m========================= Random Forest ===========================\033[0m")
print("Best parameters found: ", random_search.best_params_)
best_rf = random_search.best_estimator_

# Predict using the best model
y_pred = best_rf.predict(X_test)
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
