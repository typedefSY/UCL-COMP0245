import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

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

mse_list = []
r2_list = []
max_depth_list = []
best_mse = 1
best_max_depth = 0
best_r2 = 0
for i in range(3, 20):
      max_depth = i
      bagging_regressor = BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=max_depth), n_estimators=50, random_state=42)
      bagging_regressor.fit(X_train, y_train)
      y_pred = bagging_regressor.predict(X_test)
      r2 = r2_score(y_test, y_pred)
      mse = mean_squared_error(y_test, y_pred)
      if mse < best_mse:
            best_mse = mse
            best_r2 = r2
            best_max_depth = max_depth
      mse_list.append(mse)
      r2_list.append(r2)
      max_depth_list.append(max_depth)
plt.figure(figsize=(12, 6))
# fig 1
plt.subplot(1, 2, 1)
plt.plot(max_depth_list, mse_list, color='r')
plt.title('Bagging regressor: MSE vs Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('MSE')
plt.xlim(left=3)
plt.ylim(bottom=0)

# fig 2
plt.subplot(1, 2, 2)
plt.plot(max_depth_list, r2_list, color='b')
plt.title('Bagging regressor: R2 vs Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('R2')
plt.xlim(left=3)
plt.ylim(bottom=0)

plt.tight_layout()
# !Uncomment the following lines to save the plot
# if not os.path.exists("images/"):
#       os.makedirs("images/")
# plt.savefig(f"images/Ex2_Bagging_Max_Depth.png")
plt.show()

print("\033[92m======================= Bagging Regressor =========================\033[0m")
print(f"Bagging Regressor with Decision Tree (max_depth={best_max_depth}) base estimator\nMinimum MSE: {best_mse}, Maximum R2 score: {best_r2}")
print("\033[92m===================================================================\033[0m")

bagging_regressor = BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=best_max_depth), n_estimators=50, random_state=42)
bagging_regressor.fit(X_train, y_train)
y_pred = bagging_regressor.predict(X_test)

# Plot the results
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, edgecolors='k', s=15, label='Bagging Decision Tree Regressor')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.title(f'Bagging Regressor: Predicted vs Actual, MSE = {best_mse:.4f}, R2 = {best_r2:.4f}')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
# !Uncomment the following lines to save the plot
# plt.savefig(f"images/Ex2_Bagging_Prediction.png")
plt.show()
