import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

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

tree = DecisionTreeRegressor(max_depth=5)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R2 score: {r2}")
print(f"MSE: {mse}")

# Polynomial Regression, find the best degree
min_mse = 1
max_r2 = 0
degrees_list = []
mse_list = []
r2_list = []
for i in range(1, 50):
      # print current degree, optional
      # print(f"Current Degree: {i}")
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
            
print(f"Minimum MSE: {min_mse}, maximum R2: {max_r2}, best degree: {best_degree}")

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
plt.show()

# Plot the both Polynomial and Decision Tree regressions
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, edgecolors='k', label='Decision Tree')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.title('Decision Tree: Predicted vs Actual')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_poly, edgecolors='k', color='r', label='Polynomial Regression')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.title('Polynomial Regression: Predicted vs Actual')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()

plt.tight_layout()
plt.show()
