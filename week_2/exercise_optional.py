from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
import os

data = fetch_california_housing()
X = data.data
y = data.target
feature_names = data.feature_names

print(feature_names)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the RandomForestRegressor
random_forest = RandomForestRegressor(max_depth=30, random_state=42)
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
plt.title(f'Random Forest: Predicted vs Actual, MSE = {mse:.4f}, R2 = {r2:.4f}')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
# !Uncomment the following lines to save the plot
if not os.path.exists("images/"):
    os.makedirs("images/")
plt.savefig("images/Ex5_RF_Prediction.png")
plt.show()

# Polynomial Regression, find the best degree
min_mse = 1
max_r2 = 0
degrees_list = []
mse_list = []
r2_list = []
for i in range(1, 35):
      # print current degree, optional
      print(f"Current Degree: {i}")
      poly = PolynomialFeatures(degree=i)
      poly.fit(X_train)
      X_train_poly = poly.transform(X_train)
      X_test_poly = poly.transform(X_test)
      linear = linear_model.LinearRegression()
      linear.fit(X_train_poly, y_train)
      y_pred_poly = linear.predict(X_test_poly)
      r2_poly = r2_score(y_test, y_pred_poly)
      mse_poly = mean_squared_error(y_test, y_pred_poly)
      if mse_poly < min_mse:
            min_mse = mse_poly
            max_r2 = r2_poly
            best_degree = i
      degrees_list.append(i)
      mse_list.append(mse_poly)
      r2_list.append(r2_poly)

print("\033[92m=============================== PR ================================\033[0m")
print(f"Polynomial Regressor with best degree: {best_degree}\nMinimum MSE: {min_mse}, Maximum R2: {max_r2}")
print("\033[92m===================================================================\033[0m")
poly = PolynomialFeatures(degree=best_degree)
poly.fit(X_train)
X_train_poly = poly.transform(X_train)
X_test_poly = poly.transform(X_test)
linear = linear_model.LinearRegression()
linear.fit(X_train_poly, y_train)
y_pred_poly = linear.predict(X_test_poly)

# plot the MSE and R2 scores for different polynomial degrees
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(degrees_list, mse_list, color='r')
plt.xlim(left=1)
plt.ylim(bottom=0)
plt.scatter([best_degree], [min_mse], color='g', s=20, label='Best Degree', zorder=2)
plt.vlines(best_degree, 0, min_mse, color='g', linestyle='--', linewidth=1)
plt.hlines(min_mse, 0, best_degree, color='g', linestyle='--', linewidth=1)
plt.text(best_degree, min_mse + 0.005, f' ({best_degree}, {min_mse:.2f})', color='k',
         fontsize=9, ha='center', va='bottom')
plt.title('Polynomial Regression: MSE vs Degree')
plt.xlabel('Degree')
plt.ylabel('MSE')

plt.subplot(1, 2, 2)
plt.plot(degrees_list, r2_list, color='b')
plt.xlim(left=1)
plt.ylim(bottom=0)
plt.scatter([best_degree], [max_r2], color='g', s=20, label='Best Degree', zorder=2)
plt.vlines(best_degree, 0, max_r2, color='g', linestyle='--', linewidth=1)
plt.hlines(max_r2, 0, best_degree, color='g', linestyle='--', linewidth=1)
plt.text(best_degree, max_r2 + 0.005, f' ({best_degree}, {max_r2:.2f})', color='k',
         fontsize=9, ha='left', va='bottom')
plt.title('Polynomial Regression: R2 vs Degree')
plt.xlabel('Degree')
plt.ylabel('R2')
plt.tight_layout()
# !Uncomment the following lines to save the plot
plt.savefig(f"images/Ex5_PR_Degrees.png")
plt.show()
