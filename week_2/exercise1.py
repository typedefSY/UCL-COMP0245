import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import os

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

# Decision Tree, find the best max_depth with spliter 'random'
spliter = 'random'
mse_list_2 = []
r2_list_2 = []
max_depth_list_2 = []
best_mse_random = 1
best_max_depth_random = 0
for i in range(3, 20):
      max_depth = i
      tree = DecisionTreeRegressor(max_depth=max_depth, splitter=spliter)
      tree.fit(X_train, y_train)
      y_pred = tree.predict(X_test)
      r2 = r2_score(y_test, y_pred)
      mse = mean_squared_error(y_test, y_pred)
      if mse < best_mse_random:
            best_mse_random = mse
            best_max_depth_random = max_depth
      mse_list_2.append(mse)
      r2_list_2.append(r2)
      max_depth_list_2.append(max_depth)
      # print(f"Decision Tree with max_depth: {max_depth} and spliter: {spliter}\n R2 score: {r2}, MSE: {mse}")

# Decision Tree, find the best max_depth with spliter 'best
spliter = 'best'
mse_list = []
r2_list = []
max_depth_list = []
best_mse_best = 1
best_max_depth_best = 0
for i in range(3, 20):
      max_depth = i
      tree = DecisionTreeRegressor(max_depth=max_depth, splitter=spliter)
      tree.fit(X_train, y_train)
      y_pred = tree.predict(X_test)
      r2 = r2_score(y_test, y_pred)
      mse = mean_squared_error(y_test, y_pred)
      if mse < best_mse_best:
            best_mse_best = mse
            best_max_depth_best = max_depth
      mse_list.append(mse)
      r2_list.append(r2)
      max_depth_list.append(max_depth)
      # print(f"Decision Tree with max_depth: {max_depth} and spliter: {spliter}\n R2 score: {r2}, MSE: {mse}")

# plot the MSE and R2 scores for different max_depths and splitters
plt.figure(figsize=(12, 6))
# fig 1
plt.subplot(2, 2, 1)
plt.plot(max_depth_list, mse_list, color='r')
plt.title('Decision tree regressor: MSE vs Max Depth with splitter: best')
plt.xlabel('Max Depth')
plt.ylabel('MSE')
plt.xlim(left=3)
plt.ylim(bottom=0)

# fig 2
plt.subplot(2, 2, 2)
plt.plot(max_depth_list, r2_list, color='b')
plt.title('Decision tree regressor: R2 vs Max Depth with spliter: best')
plt.xlabel('Max Depth')
plt.ylabel('R2')
plt.xlim(left=3)
plt.ylim(bottom=0)

# fig 3
plt.subplot(2, 2, 3)
plt.plot(max_depth_list_2, mse_list_2, color='r')
plt.title('Decision tree regressor: MSE vs Max Depth with spliter: random')
plt.xlabel('Max Depth')
plt.ylabel('MSE')
plt.xlim(left=3)
plt.ylim(bottom=0)

# fig 4
plt.subplot(2, 2, 4)
plt.plot(max_depth_list_2, r2_list_2, color='b')
plt.title('Decision tree regressor: R2 vs Max Depth with spliter: random')
plt.xlabel('Max Depth')
plt.ylabel('R2')
plt.xlim(left=3)
plt.ylim(bottom=0)

plt.tight_layout()
if not os.path.exists("images/"):
      os.makedirs("images/")
plt.savefig(f"images/Ex1_DT_Hyper_Parameters.png")
plt.show()

if best_mse_best < best_mse_random:
      best_max_depth = best_max_depth_best
      best_spliter = 'best'
else:
      best_max_depth = best_max_depth_random
      best_spliter = 'random'

tree = DecisionTreeRegressor(max_depth=best_max_depth, splitter=best_spliter)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("\033[92m=============================== DT ================================\033[0m")
print(f"Decision Tree with best max_depth: {best_max_depth} and spliter: {best_spliter}.\nMinimum MSE:: {mse}, Maximum R2: {r2}")
print("\033[92m===================================================================\033[0m")

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
plt.savefig(f"images/Ex1_PR_Degrees.png")
plt.show()

# Plot the both Polynomial and Decision Tree regressions
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, edgecolors='k', s=15, label='Decision Tree')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.title('Decision Tree: Predicted vs Actual')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_poly, edgecolors='k', color='r', s=15, label='Polynomial Regression')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.title('Polynomial Regression: Predicted vs Actual')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()

plt.tight_layout()
plt.savefig(f"images/Ex1_DT_PR_Comparison.png")
plt.show()
