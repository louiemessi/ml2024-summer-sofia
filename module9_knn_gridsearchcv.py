import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

N = int(input("Enter the number of training points (N): "))

train_x = np.zeros((N, 1), dtype=float)  # Reshaped for scikit-learn compatibility
train_y = np.zeros(N, dtype=int)

print("Enter training points:")
for i in range(N):
    train_x[i] = float(input(f"Training Point {i + 1} - x (real number): "))
    train_y[i] = int(input(f"Training Point {i + 1} - y (non-negative integer): "))

M = int(input("\nEnter the number of test points (M): "))

test_x = np.zeros((M, 1), dtype=float)  # Reshaped for scikit-learn compatibility
test_y = np.zeros(M, dtype=int)

print("Enter test points:")
for i in range(M):
    test_x[i] = float(input(f"Test Point {i + 1} - x (real number): "))
    test_y[i] = int(input(f"Test Point {i + 1} - y (non-negative integer): "))

print("\nPerforming Grid Search for optimal k...")
param_grid = {'n_neighbors': np.arange(1, 11)}  # Test k values from 1 to 10
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5)  # 5-fold cross-validation
grid_search.fit(train_x, train_y)

best_k = grid_search.best_params_['n_neighbors']
print(f"Best k found: {best_k}")

best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(train_x, train_y)

predictions = best_knn.predict(test_x)
accuracy = accuracy_score(test_y, predictions)
print(f"\nTest Accuracy with best k ({best_k}): {accuracy:.2f}")
