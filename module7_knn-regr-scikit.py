import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Get input N
N = int(input("Enter the number of points (N): "))

# Get input k
k = int(input("Enter the number of neighbors (k): "))

# Initialize arrays to store points
X = np.zeros((N, 1))
y = np.zeros(N)

# Get N (x, y) points from user
for i in range(N):
    X[i] = float(input(f"Enter x value for point {i+1}: "))
    y[i] = float(input(f"Enter y value for point {i+1}: "))

# Calculate variance of labels in the training dataset
variance = np.var(y)
print(f"Variance of labels in the training dataset: {variance}")

# Check if k <= N
if k <= N:
    # Create and train the KNN regressor
    knn_regressor = KNeighborsRegressor(n_neighbors=k)
    knn_regressor.fit(X, y)

    # Get input X for prediction
    X_pred = float(input("Enter X value for prediction: "))

    # Reshape X_pred for prediction
    X_pred = np.array([[X_pred]])

    # Make prediction
    y_pred = knn_regressor.predict(X_pred)

    print(f"The predicted Y value is: {y_pred[0]}")
else:
    print("Error: k must be less than or equal to N")
